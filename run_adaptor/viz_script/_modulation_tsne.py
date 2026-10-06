"""Shared checkpoint, encoding and plotting utilities for modulation inspection."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from coral.metalearning import outer_step
from coral.utils.data.load_data import set_seed

ROOT = Path(__file__).resolve().parents[2]


def absolute(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def make_parser(description, batch_size):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(ROOT / "run_adaptor/config.yaml"))
    parser.add_argument("--mode", choices=("author", "smoke"), default="author")
    parser.add_argument("--inr-checkpoint", help="Explicit INR checkpoint; use its own config")
    parser.add_argument("--data-dir", help="Override the dataset location saved in the checkpoint")
    parser.add_argument("--output-dir", help="Directory for PNG, PDF, codes, CSV and metadata")
    parser.add_argument("--codes-file", help="Replot this script's codes.npz without encoding again")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="Cases per batch (Airfoil), frames per batch (NS)")
    parser.add_argument("--ntrain", type=int, help="Use a prefix of training cases/trajectories")
    parser.add_argument("--ntest", type=int, help="Use a prefix of test cases/trajectories")
    parser.add_argument("--seed", type=int, default=123, help="t-SNE seed; data uses checkpoint seed")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--standardize", action="store_true",
                        help="Standardize each latent feature using training codes only")
    return parser


def validate_args(args):
    for key in ("batch_size", "ntrain", "ntest"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    if not np.isfinite(args.perplexity) or args.perplexity <= 0:
        raise ValueError("perplexity must be finite and positive")
    if args.max_iter < 300:
        raise ValueError("max-iter must be >= 300, including iterations after early exaggeration")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.codes_file and any((args.ntrain, args.ntest, args.inr_checkpoint, args.data_dir)):
        raise ValueError("codes-file already fixes the data/checkpoint; omit extraction overrides")


def output_directory(args, key):
    if args.output_dir:
        directory = absolute(args.output_dir)
    else:
        adaptor = OmegaConf.load(absolute(args.config))
        suffix = "-smoke" if args.mode == "smoke" else ""
        directory = (absolute(adaptor.common.output_root) / adaptor.datasets[key].author_name
                     / "visualization" / f"modulations_tsne{suffix}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_context(args, key):
    """Prefer the INR/config referenced by the saved downstream task, if present."""
    adaptor = OmegaConf.load(absolute(args.config))
    dataset = adaptor.datasets[key]
    run_dir = absolute(adaptor.common.output_root) / dataset.author_name
    suffix = "-smoke" if args.mode == "smoke" else ""
    downstream = None
    downstream_path = run_dir / "model" / f"{dataset.downstream_run_name}{suffix}.pt"
    if args.inr_checkpoint:
        inr_path = absolute(args.inr_checkpoint)
    elif downstream_path.exists():
        downstream = torch.load(downstream_path, map_location="cpu", weights_only=False)
        inr_path = run_dir / "inr" / f"{downstream['cfg'].inr.run_name}.pt"
    else:
        inr_path = run_dir / "inr" / f"{dataset.inr_run_name}{suffix}.pt"
    checkpoint = torch.load(inr_path, map_location="cpu", weights_only=False)
    inr_cfg = checkpoint["cfg"]
    task_cfg = downstream["cfg"] if downstream is not None else inr_cfg
    if any(cfg.data.dataset_name != dataset.author_name for cfg in (inr_cfg, task_cfg)):
        raise ValueError(f"Checkpoint dataset must be {dataset.author_name}")
    counts = {}
    for name in ("ntrain", "ntest"):
        saved = int(task_cfg.data[name])
        requested = getattr(args, name)
        counts[name] = saved if requested is None else requested
        if not 0 < counts[name] <= saved:
            raise ValueError(f"{name} must be between 1 and saved count {saved}")
    seed = int(inr_cfg.data.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    data_dir = absolute(args.data_dir or task_cfg.data.dir)
    steps = int(task_cfg.inr.inner_steps if downstream is not None else inr_cfg.optim.inner_steps)
    if steps <= 0:
        raise ValueError("Checkpoint inner_steps must be positive")
    metadata = {
        "dataset": dataset.author_name,
        "inr_checkpoint": str(inr_path),
        "inr_sha256": hashlib.sha256(inr_path.read_bytes()).hexdigest(),
        "inr_epoch_zero_based": int(checkpoint["epoch"]),
        "downstream_checkpoint": str(downstream_path) if downstream is not None else None,
        "downstream_epoch_zero_based": int(downstream["epoch"]) if downstream is not None else None,
        "inr_config": OmegaConf.to_container(inr_cfg, resolve=True),
        "task_config": OmegaConf.to_container(task_cfg, resolve=True),
        "data_dir": str(data_dir), "data_seed": seed, "inner_steps": steps,
        "batch_size": args.batch_size, "device": args.device, **counts,
        "code_definition": "Zero-initialized latent z fitted to observed fields with frozen INR weights",
        "model_weights_updated": False,
    }
    print(f"INR: {inr_path} (epoch {checkpoint['epoch']}, zero-based)", flush=True)
    return checkpoint, downstream, task_cfg, data_dir, counts, steps, metadata


def encode_batch(inr, values, coordinates, latent_dim, alpha, steps, device):
    values, coordinates = values.to(device), coordinates.to(device)
    # Do not use inference_mode: fitting codes needs gradients with respect to z.
    result = outer_step(
        inr, coordinates, values, steps, alpha,
        is_train=False, gradient_checkpointing=False, loss_type="mse",
        modulations=values.new_zeros(len(values), latent_dim),
    )
    codes = result["modulations"].detach().cpu()
    if not torch.isfinite(codes).all():
        raise ValueError("Non-finite modulations from INR encoding")
    return codes


def save_codes(path, arrays, metadata):
    np.savez_compressed(path, **arrays, source_metadata_json=json.dumps(metadata, allow_nan=False))


def load_codes(path, dataset):
    with np.load(absolute(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["source_metadata_json"]))
        arrays = {key: archive[key].copy() for key in archive.files if key != "source_metadata_json"}
    if metadata["dataset"] != dataset:
        raise ValueError(f"Expected a codes archive for {dataset}")
    return arrays, metadata


def project_joint(train, test, args):
    """Fit ONE t-SNE to train+test, preserving their common coordinate system."""
    try:
        import sklearn
        from sklearn.manifold import TSNE
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError("Install plotting dependencies: python -m pip install 'scikit-learn>=1.5,<2'") from exc
    train, test = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Expected train/test arrays [samples, shared latent dimension]")
    if min(len(train), len(test)) == 0 or len(train) + len(test) < 3 or train.shape[1] < 2:
        raise ValueError("t-SNE requires both splits, >= 3 total samples and >= 2 latent features")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Modulations contain NaN or infinity")
    center = train.mean(0)
    if args.standardize:
        scale = train.std(0)
        scale[scale == 0] = 1
    else:
        # A uniform rescaling conditions tiny latent values without reweighting features.
        scale = np.full(train.shape[1], np.sqrt(np.mean((train - center) ** 2)))
        if scale[0] == 0:
            scale[:] = 1
    features = (np.concatenate((train, test)) - center) / scale
    if np.max(np.std(features, axis=0)) == 0:
        raise ValueError("All modulations are identical; no meaningful t-SNE embedding exists")
    perplexity = min(args.perplexity, len(features) - 1.0)
    if perplexity != args.perplexity:
        print(f"perplexity reduced to {perplexity:g} for {len(features)} samples", flush=True)
    model = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca",
                 random_state=args.seed, max_iter=args.max_iter, method="barnes_hut", n_jobs=1)
    print(f"t-SNE: {len(train)} train + {len(test)} test, latent_dim={train.shape[1]}", flush=True)
    with threadpool_limits(limits=4):
        embedding = model.fit_transform(features.astype(np.float32))
    if not np.isfinite(embedding).all() or not np.isfinite(model.kl_divergence_):
        raise ValueError("Non-finite t-SNE result")
    return embedding, {
        "seed": args.seed, "requested_perplexity": args.perplexity,
        "effective_perplexity": perplexity, "max_iter": args.max_iter,
        "iterations": int(model.n_iter_), "kl_divergence": float(model.kl_divergence_),
        "sklearn_version": sklearn.__version__, "joint_train_test_fit": True,
        "preprocessing": "train_feature_standardization" if args.standardize else "uniform_scaling_only",
        "center": center.tolist(), "scale": scale.tolist(),
    }


def scatter_splits(ax, embedding, ntrain):
    ax.scatter(*embedding[:ntrain].T, s=14, marker="o", color="#2878B5", alpha=0.45,
               linewidths=0, label=f"Train ({ntrain:,})", rasterized=True)
    ax.scatter(*embedding[ntrain:].T, s=36, marker="^", color="#D95319", alpha=0.85,
               edgecolors="white", linewidths=0.35, label=f"Test ({len(embedding)-ntrain:,})",
               rasterized=True)
    ax.legend(frameon=True, fontsize=10)
    style_axis(ax)


def style_axis(ax):
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.12)
    ax.set_axisbelow(True)


def save_figure(fig, output_dir, stem):
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"figure={path}", flush=True)
    plt.close(fig)


def save_metadata(output_dir, metadata):
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
