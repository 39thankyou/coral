"""Shared Airfoil analysis: author-compatible encoding and frozen decoding."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from coral.mlp import ResNet
from coral.utils.data.load_data import get_operator_data
from coral.utils.models.load_inr import create_inr_instance
from run_adaptor.viz_script._modulation_tsne import (
    ROOT, absolute, encode_batch, load_codes, load_context, plt, save_codes,
    save_figure, save_metadata,
)


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(ROOT / "run_adaptor/config.yaml"))
    parser.add_argument("--mode", choices=("author", "smoke"), default="author")
    parser.add_argument("--inr-checkpoint")
    parser.add_argument("--data-dir")
    parser.add_argument("--codes-file", help="Reuse codes.npz from viz_airfoil_task_ltt.py or these analyses")
    parser.add_argument("--output-dir", help="Parent directory; each analysis creates its own subdirectory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ntrain", type=int, help="Training prefix; mapper comparison requires its full training set")
    parser.add_argument("--ntest", type=int, help="Test prefix (original IDs start at 1000)")
    parser.add_argument("--seed", type=int, default=123)
    return parser


def output_directory(args, name):
    if args.output_dir:
        parent = absolute(args.output_dir)
    else:
        cfg = OmegaConf.load(absolute(args.config))
        parent = absolute(cfg.common.output_root) / "airfoil" / "latent_analysis"
    directory = parent / (name + ("-smoke" if args.mode == "smoke" else ""))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_inr(checkpoint, kind, device, output_dim):
    cfg = OmegaConf.create(OmegaConf.to_container(checkpoint["cfg"], resolve=True))
    cfg.inr = cfg[f"inr_{kind}"]
    inr = create_inr_instance(cfg, input_dim=2, output_dim=output_dim, device=device)
    inr.load_state_dict(checkpoint[f"inr_{kind}"])
    return inr.eval().requires_grad_(False)


def load_analysis(args, require_mapper=False):
    """Keep codes, case IDs, raw fields and checkpoint provenance aligned."""
    for key in ("batch_size", "ntrain", "ntest"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.codes_file:
        arrays, source = load_codes(args.codes_file, "airfoil")
        inr_path = absolute(args.inr_checkpoint or source["inr_checkpoint"])
        if hashlib.sha256(inr_path.read_bytes()).hexdigest() != source["inr_sha256"]:
            raise ValueError("Codes were encoded with a different INR checkpoint")
        checkpoint = torch.load(inr_path, map_location="cpu", weights_only=False)
        source = dict(source, inr_checkpoint=str(inr_path), codes_file=str(absolute(args.codes_file)))
        data_dir = absolute(args.data_dir or source["data_dir"])
        for split in ("train", "test"):
            available = len(arrays[f"case_id_{split}"])
            count = getattr(args, f"n{split}") or available
            if count > available:
                raise ValueError(f"Requested {split} count exceeds cached codes ({available})")
            for key in (f"case_id_{split}", f"z_in_{split}", f"z_out_{split}"):
                if len(arrays[key]) != available:
                    raise ValueError("Cached codes and case IDs are not paired")
                arrays[key] = arrays[key][:count]
            source[f"n{split}"] = count
    else:
        checkpoint, _, _, data_dir, _, _, source = load_context(args, "airfoil")
        arrays = None
    if checkpoint["cfg"].data.dataset_name != "airfoil":
        raise ValueError("Expected an Airfoil checkpoint")
    for kind in ("in", "out"):
        if int(checkpoint["cfg"][f"inr_{kind}"].latent_dim) != 128:
            raise ValueError("These analyses require the original 128-dimensional Airfoil latent spaces")
    mapper_checkpoint = None
    if require_mapper:
        path = getattr(args, "mapper_checkpoint", None) or source.get("downstream_checkpoint")
        if not path:
            raise ValueError("Mapper comparison needs --mapper-checkpoint or the saved downstream checkpoint")
        path = absolute(path)
        mapper_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = mapper_checkpoint["cfg"]
        if cfg.data.dataset_name != "airfoil" or cfg.inr.run_name != Path(source["inr_checkpoint"]).stem:
            raise ValueError("Mapper must reference the same Airfoil INR checkpoint as the codes")
        if int(cfg.inr.inner_steps) != int(source["inner_steps"]):
            raise ValueError("Code fitting steps must match the mapper's training setup")
        if int(cfg.data.ntrain) != source["ntrain"] or source["ntrain"] < 2:
            raise ValueError("Mapper normalization requires all of its training codes; omit --ntrain")
        source.update(downstream_checkpoint=str(path),
                      downstream_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      downstream_epoch_zero_based=int(mapper_checkpoint["epoch"]))
    data = get_operator_data(data_dir, "airfoil", source["ntrain"], source["ntest"],
                             sub_tr=1, sub_te=1, same_grid=True)
    x_train, y_train, x_test, y_test, grid_train, grid_test = data
    if arrays is None:
        arrays = {"case_id_train": np.arange(len(x_train)),
                  "case_id_test": np.arange(1000, 1000 + len(x_test))}
        # Same zero initialization, inner-loop steps and encode_batch as the t-SNE script.
        for kind, train, test in (("in", x_train, x_test), ("out", y_train, y_test)):
            inr = make_inr(checkpoint, kind, args.device, train.shape[-1])
            alpha = torch.as_tensor(checkpoint[f"alpha_{kind}"]).detach().to(args.device)
            for split, values, coords in (("train", train, grid_train), ("test", test, grid_test)):
                codes = []
                for start in range(0, len(values), args.batch_size):
                    stop = min(start + args.batch_size, len(values))
                    codes.append(encode_batch(inr, values[start:stop], coords[start:stop],
                                              128, alpha, source["inner_steps"], args.device))
                    if start // args.batch_size % 25 == 0 or stop == len(values):
                        print(f"Airfoil {kind}/{split}: {stop}/{len(values)}", flush=True)
                arrays[f"z_{kind}_{split}"] = torch.cat(codes).numpy()
            del inr
    for split, values, offset in (("train", y_train, 0), ("test", y_test, 1000)):
        if not np.array_equal(arrays[f"case_id_{split}"], np.arange(offset, offset + len(values))):
            raise ValueError("Codes must follow the official train/test prefix ordering")
        for kind in ("in", "out"):
            z = arrays[f"z_{kind}_{split}"]
            if z.shape != (len(values), 128) or not np.isfinite(z).all():
                raise ValueError("Expected finite, paired [cases, 128] latent codes")
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite target fields")
    source.update(data_dir=str(data_dir), analysis_device=args.device,
                  analysis_batch_size=args.batch_size,
                  output_code_source="Observed target encoded with frozen output INR; not mapper prediction")
    return arrays, source, data, checkpoint, mapper_checkpoint


def nearest_neighbors(train, query, k):
    """Raw Euclidean distance: no t-SNE, feature normalization or test fitting."""
    if not 1 <= k <= len(train):
        raise ValueError(f"neighbors must be between 1 and {len(train)}")
    train, query = np.asarray(train, dtype=np.float64), np.asarray(query, dtype=np.float64)
    indices, distances = [], []
    # Direct differences avoid cancellation for identical/tiny latent vectors.
    for row in query:
        distance = np.linalg.norm(train - row, axis=1)
        index = np.argsort(distance, kind="stable")[:k]
        indices.append(index)
        distances.append(distance[index])
    return np.asarray(indices), np.asarray(distances)


def inverse_distance_weights(distances):
    distances = np.asarray(distances, dtype=np.float64)
    weights = np.zeros_like(distances)
    for index, row in enumerate(distances):
        exact = row == 0
        if exact.any():
            weights[index] = exact / exact.sum()
        else:
            # Algebraically 1/d, rescaled to avoid overflow for tiny distances.
            weights[index] = row.min() / row
            weights[index] /= weights[index].sum()
    return weights


def field_errors(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    difference = (prediction - target).reshape(len(prediction), -1)
    norm = np.linalg.norm(target.reshape(len(target), -1), axis=1)
    mse = np.mean(difference ** 2, axis=1)
    relative_l2 = np.linalg.norm(difference, axis=1) / np.maximum(norm, np.finfo(np.float64).eps)
    if not np.isfinite(mse).all() or not np.isfinite(relative_l2).all():
        raise ValueError("Non-finite field errors")
    return mse, relative_l2


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()), "count": int(values.size)}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def decode(inr, codes, coordinates, args):
    fields = []
    for start in range(0, len(codes), args.batch_size):
        stop = min(start + args.batch_size, len(codes))
        z = torch.as_tensor(codes[start:stop], dtype=torch.float32, device=args.device)
        field = inr.modulated_forward(coordinates[start:stop].to(args.device).contiguous(), z)
        if not torch.isfinite(field).all():
            raise ValueError("Decoder produced non-finite fields")
        fields.append(field.cpu())
    return torch.cat(fields).numpy()


@torch.no_grad()
def mapper_codes(checkpoint, arrays, args):
    cfg = checkpoint["cfg"]
    model = ResNet(input_dim=128, hidden_dim=cfg.model.width, output_dim=128,
                   depth=cfg.model.depth, dropout=cfg.model.dropout).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    train = torch.as_tensor(arrays["z_in_train"], dtype=torch.float32)
    mean, std = train.mean(0), train.std(0)  # author's sample std, correction=1
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid full-training input-code normalization")
    query = (torch.as_tensor(arrays["z_in_test"], dtype=torch.float32) - mean) / std
    codes = torch.cat([model(batch.to(args.device)).cpu()
                       for batch in query.split(args.batch_size)]).numpy()
    return codes, mean.numpy(), std.numpy()
