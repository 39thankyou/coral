"""Download SW and official MeshGraphNets Airfoil (small complete-record prefix by default).

Full IVP downloads are resumable and opt-in via --full-ivp; a manifest distinguishes
prefix data from the full official splits. Existing completed files are preserved.
"""
import argparse
import json
import shutil
import struct
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://storage.googleapis.com/dm-meshgraphnets/airfoil/"
SW = {
    "shallow_water_2_160_128_256_test.h5": ("1JU5sQxKyx5dH8BNQ8s49BbsB3DS5RTzc", 167774208),
    "shallow_water_16_160_128_256_train.h5": ("1g3sZv-SQDOnbUMjAqOpU5SrIH_HLlm7e", 1342179328),
}


def download_sw(directory):
    import gdown
    directory.mkdir(parents=True, exist_ok=True)
    for name, (file_id, size) in SW.items():
        target = directory / name
        if target.exists() and target.stat().st_size == size:
            continue
        partial = target.with_suffix(".h5.part")
        print(f"Downloading {name} ({size} bytes)", flush=True)
        result = gdown.download(id=file_id, output=str(partial), resume=True, quiet=True)
        if result is None or partial.stat().st_size != size:
            raise RuntimeError(f"SW download incomplete: {partial}; expected {size} bytes")
        partial.replace(target)


def exact(stream, size):
    result = stream.read(size)
    if len(result) != size:
        raise EOFError("Truncated TFRecord download")
    return result


def download_ivp(directory, train_count=8, test_count=4, full=False):
    directory.mkdir(parents=True, exist_ok=True)
    meta = requests.get(BASE + "meta.json", timeout=60)
    meta.raise_for_status()
    (directory / "meta.json").write_bytes(meta.content)
    manifest_path = directory / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"source": BASE, "splits": {}}
    for split, count in (("test", test_count), ("valid", test_count), ("train", train_count)):
        target = directory / f"{split}.tfrecord"
        previous = manifest["splits"].get(split, {})
        if target.exists() and (previous.get("complete") or (not full and previous.get("records", 0) >= count)):
            continue
        partial = target.with_suffix(".tfrecord.part")
        if full:
            total = int(requests.head(BASE + target.name, timeout=60).headers["Content-Length"])
            start = partial.stat().st_size if partial.exists() else 0
            if shutil.disk_usage(directory).free < total - start + 5 * 1024**3:
                raise RuntimeError(f"Insufficient disk space for {split}: {total-start} additional bytes")
            if start < total:
                with requests.get(BASE + target.name, headers={"Range": f"bytes={start}-"}, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    append = start > 0 and response.status_code == 206
                    with partial.open("ab" if append else "wb") as handle:
                        for chunk in response.iter_content(8 * 1024**2):
                            handle.write(chunk)
            if partial.stat().st_size != total:
                raise RuntimeError(f"Incomplete download: {partial}")
            entry = {"complete": True, "bytes": total, "records": 1000 if split == "train" else 100}
        else:
            # TFRecord = uint64 length + CRC + payload + CRC. Keep complete
            # trajectories rather than truncating frames or writing synthetic data.
            with requests.get(BASE + target.name, stream=True, timeout=60) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for index in range(count):
                        header = exact(response.raw, 12)
                        length = struct.unpack("<Q", header[:8])[0]
                        handle.write(header)
                        remaining = length + 4
                        while remaining:
                            chunk = exact(response.raw, min(remaining, 4 * 1024**2))
                            handle.write(chunk)
                            remaining -= len(chunk)
                        print(f"{split}: {index+1}/{count} complete trajectories", flush=True)
            entry = {"complete": False, "bytes": partial.stat().st_size, "records": count}
        partial.replace(target)
        manifest["splits"][split] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sw", "airfoil-ivp", "all"), default="all")
    parser.add_argument("--full-ivp", action="store_true")
    parser.add_argument("--ntrain", type=int, default=8)
    parser.add_argument("--ntest", type=int, default=4)
    args = parser.parse_args()
    if args.ntrain < 1 or args.ntest < 1:
        parser.error("counts must be positive")
    if args.dataset in ("sw", "all"):
        download_sw(ROOT / "datasets/dino")
    if args.dataset in ("airfoil-ivp", "all"):
        download_ivp(ROOT / "datasets/airfoil_flow", args.ntrain, args.ntest, args.full_ivp)
