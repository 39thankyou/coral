"""Plot per-frame Navier-Stokes state modulations and selected trajectories."""

# 修改这里：额外图片展示的轨迹总数（train + test），默认各取 3 条。
K = 1

# Run from the repository root:
#     python run_adaptor/viz_script/viz_ns_task_ltt.py
# Uses actual training/test frames and grids from the checkpoint. Each code is
# fitted to an observed state, not predicted by the ODE. --test-frames in-t
# compares matching time horizons. Train/test share ONE t-SNE embedding.
# The extra K-trajectory plot selects train/test alternately in index order and
# keeps the full-data embedding coordinates (no independent t-SNE fit).

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from coral.utils.data.load_data import get_dynamics_data
from coral.utils.models.load_inr import create_inr_instance
from run_adaptor.viz_script._modulation_tsne import (
    encode_batch, load_codes, load_context, make_parser, output_directory, plt,
    project_joint, save_codes, save_figure, save_metadata, scatter_splits, style_axis, validate_args,
)


def check_grid(actual, checkpoint, key, label):
    if key not in checkpoint:
        raise ValueError(f"{label} checkpoint has no saved {key} to verify sampling")
    saved = checkpoint[key].cpu()
    if actual.shape != saved.shape or not torch.equal(actual, saved):
        raise ValueError(f"{label} {key} differs from regenerated data. Check the loader, data and checkpoint pair.")


def extract_codes(args):
    checkpoint, downstream, cfg, data_dir, counts, steps, metadata = load_context(args, "navier_stokes")
    inr_cfg = checkpoint["cfg"]
    inter, extra = int(cfg.data.seq_inter_len), int(cfg.data.seq_extra_len)
    sampling = {"sub_from": inr_cfg.data.get("sub_from", 1), "sub_tr": inr_cfg.data.sub_tr,
                "sub_te": cfg.data.sub_te, "same_grid": cfg.data.same_grid}
    # Load the saved counts BEFORE selecting prefixes, to reproduce random grids.
    train, train_extra, test, grid_train, grid_extra, grid_test = get_dynamics_data(
        data_dir, "navier-stokes-dino", int(cfg.data.ntrain), int(cfg.data.ntest),
        seq_inter_len=inter, seq_extra_len=extra, **sampling,
    )
    del train_extra, grid_extra
    for label, ckpt in (("INR", checkpoint), ("ODE", downstream)):
        if ckpt is not None:
            check_grid(grid_train, ckpt, "grid_tr", label)
            check_grid(grid_test, ckpt, "grid_te", label)
    if downstream is not None and not torch.equal(checkpoint["alpha"], downstream["alpha"]):
        raise ValueError("INR and ODE checkpoint alpha values differ")
    if len(train) != int(cfg.data.ntrain) or len(test) != int(cfg.data.ntest):
        raise ValueError("Expected one temporal window per NS trajectory")
    if train.shape[-1] != inter or test.shape[-1] != inter + extra:
        raise ValueError("Frame counts do not match the saved training/evaluation horizons")
    train, grid_train = train[:counts["ntrain"]], grid_train[:counts["ntrain"]]
    test, grid_test = test[:counts["ntest"]], grid_test[:counts["ntest"]]
    latent_dim = int(inr_cfg.inr.latent_dim)
    inr = create_inr_instance(inr_cfg, input_dim=grid_train.shape[-2],
                              output_dim=train.shape[-2], device=args.device)
    inr.load_state_dict(checkpoint["inr"])
    inr.eval().requires_grad_(False)
    alpha = torch.as_tensor(checkpoint["alpha"]).detach().to(args.device)
    arrays = {}
    for split, values, coords in (("train", train, grid_train), ("test", test, grid_test)):
        nframes = values.shape[-1]
        codes = []
        # Batch individual frames, with a consistent trajectory-major ordering.
        # This avoids putting every frame of a trajectory on the GPU at once.
        for start in range(0, len(values) * nframes, args.batch_size):
            stop = min(start + args.batch_size, len(values) * nframes)
            pairs = [divmod(index, nframes) for index in range(start, stop)]
            frame_values = torch.stack([values[traj, ..., frame] for traj, frame in pairs])
            frame_coords = torch.stack([coords[traj, ..., frame] for traj, frame in pairs])
            codes.append(encode_batch(inr, frame_values, frame_coords,
                                      latent_dim, alpha, steps, args.device))
            if start // args.batch_size % 50 == 0 or stop == len(values) * nframes:
                print(f"NS {split}: {stop}/{len(values)*nframes} frames", flush=True)
        arrays[f"z_{split}"] = torch.cat(codes).reshape(len(values), nframes, latent_dim).numpy()
    metadata.update({"sampling": sampling, "checkpoint_grids_verified": True,
                     "training_points_per_frame": int(train[0, ..., 0, 0].numel()),
                     "test_points_per_frame": int(test[0, ..., 0, 0].numel()),
                     "train_frames": inter, "test_frames": inter + extra,
                     "code_array_layout": "trajectory, frame, latent_feature",
                     "trajectory_id_definition": "Zero-based index within split; loader sorts shelve keys numerically",
                     "state_code_source": "Each observed frame encoded independently; no ODE rollout"})
    return arrays, metadata


def plot_selected_trajectories(embedding, split, trajectory, frame, output_dir, k):
    """Show at most k trajectories TOTAL, alternating train/test without replacement."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("K at the top of this file must be a positive integer")
    train_ids = np.unique(trajectory[split == "train"])
    test_ids = np.unique(trajectory[split == "test"])
    candidates = []
    for index in range(max(len(train_ids), len(test_ids))):
        for label, ids in (("train", train_ids), ("test", test_ids)):
            if index < len(ids):
                candidates.append((label, int(ids[index])))
    selected = candidates[:k]
    if not selected:
        raise ValueError("No trajectories available for the K-trajectory plot")
    if len(selected) < k:
        print(f"K={k}: only {len(selected)} trajectories available; showing all", flush=True)
    mask = np.zeros(len(embedding), dtype=bool)
    for label, traj in selected:
        mask |= (split == label) & (trajectory == traj)
    max_frame = max(int(frame[mask].max()), 1)
    palette = plt.get_cmap("tab10" if len(selected) <= 10 else "turbo", len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), layout="constrained", sharex=True, sharey=True)
    for index, (label, traj) in enumerate(selected):
        indices = np.flatnonzero((split == label) & (trajectory == traj))
        indices = indices[np.argsort(frame[indices])]
        xy, times = embedding[indices], frame[indices]
        marker = "o" if label == "train" else "^"
        color = palette(index)
        axes[0].scatter(*xy.T, color=color, marker=marker, s=34, alpha=0.9,
                        edgecolors="white", linewidths=0.3, label=f"{label.capitalize()} #{traj}")
        points = axes[1].scatter(*xy.T, c=times, cmap="viridis", vmin=0, vmax=max_frame,
                                 marker=marker, s=34, edgecolors="white", linewidths=0.3)
        for ax, line_color in ((axes[0], color), (axes[1], "#777777")):
            ax.plot(*xy.T, color=line_color, alpha=0.45, linewidth=1, zorder=1)
            if len(xy) > 1:
                middle = (len(xy) - 1) // 2
                ax.annotate("", xy=xy[middle + 1], xytext=xy[middle],
                            arrowprops={"arrowstyle": "->", "color": line_color, "lw": 1.3})
        for pos in dict.fromkeys((0, len(xy) - 1)):
            axes[0].annotate(f"t={times[pos]}", xy[pos], xytext=(4, 5),
                             textcoords="offset points", fontsize=8, color=color)
    axes[0].legend(fontsize=9, ncol=max(1, (len(selected) + 9) // 10))
    axes[0].set_title("Color = trajectory; lines follow frame order")
    axes[1].set_title("Same trajectories, colored by frame")
    for ax in axes:
        style_axis(ax)
    fig.colorbar(points, ax=axes[1], label="Frame index (zero-based)")
    fig.suptitle(f"Navier-Stokes: {len(selected)} selected trajectories (K={k})\n"
                 "Train = circles; test = triangles | Coordinates from the full t-SNE embedding",
                 fontsize=12)
    stem = f"ns_state_modulations_tsne_k{k}"
    save_figure(fig, output_dir, stem)
    with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "trajectory_id", "frame_id", "tsne_1", "tsne_2"))
        for i in np.flatnonzero(mask):
            writer.writerow((split[i], int(trajectory[i]), int(frame[i]), *embedding[i]))
    return {"requested_k": k, "actual_k": len(selected),
            "selection": [{"split": label, "trajectory_id": traj} for label, traj in selected],
            "selection_rule": "Alternate train/test in ascending trajectory index order",
            "embedding": "Subset of the full joint embedding, no refitting",
            "figure_stem": stem}


def visualize(args):
    validate_args(args)
    if isinstance(K, bool) or not isinstance(K, int) or K <= 0:
        raise ValueError("K at the top of this file must be a positive integer")
    output_dir = output_directory(args, "navier_stokes")
    arrays, source = (load_codes(args.codes_file, "navier-stokes-dino") if args.codes_file else extract_codes(args))
    train, test = arrays["z_train"], arrays["z_test"]
    if train.ndim != 3 or test.ndim != 3 or train.shape[-1] != test.shape[-1]:
        raise ValueError("NS codes must have shape [trajectory, frame, latent_feature]")
    save_codes(output_dir / "codes.npz", arrays, source)
    if args.test_frames == "in-t":
        test = test[:, :train.shape[1]]
    ntrain = train.shape[0] * train.shape[1]
    embedding, tsne_metadata = project_joint(train.reshape(-1, train.shape[-1]),
                                             test.reshape(-1, test.shape[-1]), args)
    split = np.array(["train"] * ntrain + ["test"] * (len(embedding) - ntrain))
    trajectory = np.concatenate([np.repeat(np.arange(len(z)), z.shape[1]) for z in (train, test)])
    frame = np.concatenate([np.tile(np.arange(z.shape[1]), len(z)) for z in (train, test)])
    phase = np.where(frame < train.shape[1], "In-t", "Out-t")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), layout="constrained", sharex=True, sharey=True)
    scatter_splits(axes[0], embedding, ntrain)
    axes[0].set_title("State modulation: train / test")
    for indices, marker, size, label in ((slice(0, ntrain), "o", 14, "Train"),
                                        (slice(ntrain, None), "^", 36, "Test")):
        points = axes[1].scatter(*embedding[indices].T, c=frame[indices], cmap="viridis",
                                 vmin=0, vmax=max(int(frame.max()), 1), marker=marker,
                                 s=size, alpha=0.7, linewidths=0.25,
                                 edgecolors="white" if label == "Test" else "none",
                                 label=label, rasterized=True)
    axes[1].legend()
    axes[1].set_title("Same embedding, colored by frame")
    style_axis(axes[1])
    fig.colorbar(points, ax=axes[1], label="Frame index (zero-based)")
    fig.suptitle(f"Navier-Stokes: one point per observed frame\n"
                 f"Train: {len(train)} trajectories x {train.shape[1]} frames; "
                 f"test: {len(test)} trajectories x {test.shape[1]} frames", fontsize=12)
    save_figure(fig, output_dir, "ns_state_modulations_tsne")
    selected_metadata = plot_selected_trajectories(embedding, split, trajectory, frame, output_dir, K)
    np.savez_compressed(output_dir / "embedding.npz", xy=embedding, split=split,
                        trajectory_id=trajectory, frame_id=frame, phase=phase)
    with (output_dir / "embedding.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "trajectory_id", "frame_id", "phase", "tsne_1", "tsne_2"))
        for label, traj, time, interval, xy in zip(split, trajectory, frame, phase, embedding):
            writer.writerow((label, int(traj), int(time), interval, *xy))
    save_metadata(output_dir, {"source": source, "tsne": tsne_metadata, "test_frame_selection": args.test_frames,
                              "plotted_train_frames": ntrain, "plotted_test_frames": len(embedding) - ntrain,
                              "selected_trajectories": selected_metadata,
                              "interpretation": "Both panels show the same joint embedding; color differs"})
    return output_dir


if __name__ == "__main__":
    parser = make_parser(__doc__, batch_size=16)
    parser.add_argument("--test-frames", choices=("all", "in-t"), default="all")
    visualize(parser.parse_args())
