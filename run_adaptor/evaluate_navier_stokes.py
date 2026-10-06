"""Evaluate saved DINo Navier–Stokes INR/ODE checkpoints from the first frame."""

import argparse
import json
import os
import sys
from pathlib import Path

import einops
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coral.losses import batch_mse_fn, batch_mse_rel_fn
from coral.metalearning import outer_step
from coral.mlp import Derivative
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from coral.utils.models.scheduling import ode_scheduling


def absolute(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def encode_trajectories(inr, values, coordinates, latent_dim, alpha, inner_steps):
    """Fit independent frame codes with the author's outer_step, freezing INR."""
    steps = values.shape[-1]
    values = einops.rearrange(values, "b ... t -> (b t) ...")
    coordinates = einops.rearrange(coordinates, "b ... t -> (b t) ...")
    result = outer_step(
        inr, coordinates, values, inner_steps, alpha,
        is_train=False, return_reconstructions=False,
        gradient_checkpointing=False, loss_type="mse",
        modulations=values.new_zeros(len(values), latent_dim),
    )
    return einops.rearrange(
        result["modulations"].detach(), "(b t) l -> b l t", t=steps
    )


def predict_trajectory(inr, model, initial_code, coordinates, mean, std, timestamps):
    """Roll out without teacher forcing; only the initial code enters the ODE."""
    with torch.no_grad():
        codes = ode_scheduling(
            odeint, model, (initial_code - mean) / std, timestamps,
            epsilon=0, method="rk4",
        )
        codes = einops.rearrange(codes * std + mean, "b l t -> (b t) l")
        flat_coords = einops.rearrange(coordinates, "b ... t -> (b t) ...")
        prediction = inr.modulated_forward(flat_coords, codes)
    return einops.rearrange(
        prediction, "(b t) ... -> b ... t", t=len(timestamps)
    )


def check_grid(actual, saved, name):
    if actual.shape != saved.shape or not torch.equal(actual, saved.cpu()):
        raise ValueError(
            f"Regenerated {name} grid differs from the ODE checkpoint. "
            "Check the data files, loader, seed and sampling configuration."
        )


def summarize_errors(mse, relative_l2):
    mse, relative_l2 = torch.cat(mse), torch.cat(relative_l2)
    if not torch.isfinite(mse).all() or not torch.isfinite(relative_l2).all():
        raise ValueError("Non-finite predictions or test metrics")
    return {
        "mse": mse.mean().item(),
        "relative_l2": relative_l2.mean().item(),
        "relative_l2_percent": relative_l2.mean().item() * 100,
        "per_trajectory_mse": mse.tolist(),
        "per_trajectory_relative_l2": relative_l2.tolist(),
    }


def evaluate(args):
    if args.batch_size <= 0 or (args.ntest is not None and args.ntest <= 0):
        raise ValueError("batch-size and ntest must be positive")
    device = torch.device(args.device)
    adaptor = OmegaConf.load(absolute(args.config))
    dataset = adaptor.datasets.navier_stokes
    output_dir = absolute(getattr(args, "output_root", None) or adaptor.common.output_root) / dataset.author_name
    suffix = "-smoke" if args.mode == "smoke" else ""
    ode_path = output_dir / "model" / f"{dataset.downstream_run_name}{suffix}.pt"
    if getattr(args, "ode_checkpoint", None):
        ode_path = absolute(args.ode_checkpoint)
        if not getattr(args, "output_root", None):
            output_dir = ode_path.parent.parent
    ode_checkpoint = torch.load(ode_path, map_location="cpu", weights_only=False)
    cfg = ode_checkpoint["cfg"]
    if cfg.inr.run_name is None or cfg.data.data_to_encode is not None:
        raise ValueError("Expected a shared-INR Navier–Stokes checkpoint")
    inr_path = output_dir / "inr" / f"{cfg.inr.run_name}.pt"
    if getattr(args, "inr_checkpoint", None):
        inr_path = absolute(args.inr_checkpoint)
    inr_checkpoint = torch.load(inr_path, map_location="cpu", weights_only=False)
    inr_cfg = inr_checkpoint["cfg"]
    if any(c.data.dataset_name != "navier-stokes-dino" for c in (cfg, inr_cfg)):
        raise ValueError("Both checkpoints must belong to navier-stokes-dino")
    if cfg.dynamics.model_type != "ode":
        raise ValueError("Expected a latent ODE checkpoint")
    if not torch.equal(inr_checkpoint["alpha"], ode_checkpoint["alpha"]):
        raise ValueError("INR and ODE saved code step sizes differ; check the checkpoint pair")

    ntrain, saved_ntest = int(cfg.data.ntrain), int(cfg.data.ntest)
    ntest = int(args.ntest if args.ntest is not None else saved_ntest)
    if not 0 < ntest <= saved_ntest:
        raise ValueError(f"ntest must be between 1 and the checkpoint's {saved_ntest}")
    inter_steps, extra_steps = int(cfg.data.seq_inter_len), int(cfg.data.seq_extra_len)
    if min(inter_steps, extra_steps) <= 0:
        raise ValueError("Both temporal evaluation intervals must be positive")
    total_steps = inter_steps + extra_steps

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = int(inr_cfg.data.seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Match dynamics_modeling/train.py: sampling and seed come partly from INR.
    # Load the original counts before selecting a test prefix so the random
    # grids can be checked against the full grids saved during ODE training.
    data_dir = absolute(getattr(args, "data_dir", None) or dataset.data_dir)
    sampling = {
        "sub_from": inr_cfg.data.get("sub_from", 1),
        "sub_tr": inr_cfg.data.sub_tr,
        "sub_te": cfg.data.sub_te,
        "same_grid": cfg.data.same_grid,
    }
    u_train, _, u_test, grid_train, _, grid_test = get_dynamics_data(
        data_dir, "navier-stokes-dino", ntrain, saved_ntest,
        seq_inter_len=inter_steps, seq_extra_len=extra_steps, **sampling,
    )
    check_grid(grid_train, ode_checkpoint["grid_tr"], "training")
    check_grid(grid_test, ode_checkpoint["grid_te"], "test")
    if u_train.shape[-1] != inter_steps or u_test.shape[-1] != total_steps:
        raise ValueError("Data sequence lengths differ from the checkpoint configuration")
    if len(u_train) != ntrain or len(u_test) != saved_ntest:
        raise ValueError("Expected one temporal window per DINo trajectory")
    u_test, grid_test = u_test[:ntest], grid_test[:ntest]

    latent_dim = int(inr_cfg.inr.latent_dim)
    inr = create_inr_instance(
        inr_cfg, input_dim=grid_train.shape[-2], output_dim=u_train.shape[-2],
        device=device,
    )
    inr.load_state_dict(inr_checkpoint["inr"])
    model = Derivative(1, latent_dim, cfg.dynamics.width, cfg.dynamics.depth).to(device)
    model.load_state_dict(ode_checkpoint["model"])
    for network in (inr, model):
        network.eval()
        network.requires_grad_(False)
    alpha = inr_checkpoint["alpha"].detach().to(device)
    inner_steps = int(cfg.inr.inner_steps)
    print(
        f"Loaded INR epoch={inr_checkpoint['epoch']} ODE epoch={ode_checkpoint['epoch']} "
        "(zero-based)", flush=True,
    )

    # Author ODE checkpoints omit z_mean/z_std. Recompute them using only the
    # training interval, not the extra training frames or any test trajectories.
    # Do not overwrite or trust the unversioned modulation cache.
    train_loader = DataLoader(
        TensorDataset(u_train, grid_train), batch_size=2, shuffle=False
    )
    codes = []
    print(
        f"Recovering normalization: {ntrain} trajectories x {inter_steps} frames", flush=True
    )
    for index, (values, coordinates) in enumerate(train_loader):
        codes.append(encode_trajectories(
            inr, values.to(device), coordinates.to(device), latent_dim, alpha, inner_steps
        ).cpu())
        if (index + 1) % 32 == 0 or index + 1 == len(train_loader):
            print(f"normalization batches={index + 1}/{len(train_loader)}", flush=True)
    codes = einops.rearrange(torch.cat(codes), "b l t -> (b t) l")
    if len(codes) < 2 or not torch.isfinite(codes).all():
        raise ValueError("Invalid training codes for normalization")
    mean = codes.mean(0).reshape(1, latent_dim, 1).to(device)
    std = codes.std(0).reshape(1, latent_dim, 1).to(device)
    if not torch.isfinite(std).all() or (std == 0).any():
        raise ValueError("Invalid training-code normalization statistics")

    timestamps = torch.arange(total_steps, dtype=torch.float32, device=device)
    intervals = {"in_t": (0, inter_steps), "out_t": (inter_steps, total_steps), "all": (0, total_steps)}
    errors = {key: {"mse": [], "relative_l2": []} for key in intervals}
    frame_mse, frame_l2 = [], []
    test_loader = DataLoader(
        TensorDataset(u_test, grid_test), batch_size=args.batch_size, shuffle=False
    )
    print(f"Evaluating {ntest} test trajectories, first-frame-only RK4 rollout", flush=True)
    for index, (values, coordinates) in enumerate(test_loader):
        coordinates = coordinates.to(device)
        # Future target values do not enter encoding or ODE integration.
        initial_code = encode_trajectories(
            inr, values[..., :1].to(device), coordinates[..., :1],
            latent_dim, alpha, inner_steps,
        )
        prediction = predict_trajectory(
            inr, model, initial_code, coordinates, mean, std, timestamps
        )
        target = values.to(device)
        for key, (start, stop) in intervals.items():
            pred_part = prediction[..., start:stop].contiguous()
            true_part = target[..., start:stop].contiguous()
            errors[key]["mse"].append(batch_mse_fn(pred_part, true_part).cpu())
            errors[key]["relative_l2"].append(batch_mse_rel_fn(pred_part, true_part).cpu())
        pred_frames = einops.rearrange(prediction, "b ... t -> (b t) ...").contiguous()
        true_frames = einops.rearrange(target, "b ... t -> (b t) ...").contiguous()
        frame_mse.append(batch_mse_fn(pred_frames, true_frames).reshape(-1, total_steps).cpu())
        frame_l2.append(batch_mse_rel_fn(pred_frames, true_frames).reshape(-1, total_steps).cpu())
        print(f"test batches={index + 1}/{len(test_loader)}", flush=True)

    metrics = {key: summarize_errors(**error) for key, error in errors.items()}
    frame_mse, frame_l2 = torch.cat(frame_mse), torch.cat(frame_l2)
    if not torch.isfinite(frame_mse).all() or not torch.isfinite(frame_l2).all():
        raise ValueError("Non-finite per-frame test metrics")
    summary = {
        "dataset": "navier-stokes-dino", "mode": args.mode,
        "inr_checkpoint": str(inr_path), "ode_checkpoint": str(ode_path),
        "inr_epoch_zero_based": int(inr_checkpoint["epoch"]),
        "ode_epoch_zero_based": int(ode_checkpoint["epoch"]),
        "data_dir": str(data_dir), "normalization_ntrain": ntrain, "ntest": ntest,
        "test_trajectory_indices": list(range(ntest)),
        "normalization_frames": [0, inter_steps], "frame_intervals": intervals,
        "sampling": sampling, "checkpoint_grids_verified": True,
        "training_points_per_frame": int(u_train[0, ..., 0, 0].numel()),
        "test_points_per_frame": int(u_test[0, ..., 0, 0].numel()),
        "seed": seed, "inner_steps": inner_steps,
        "batch_size": args.batch_size, "device": str(device),
        "solver": "rk4", "dt": 1, "teacher_forcing": False,
        "initial_frame": 0,
        "metric_definition": {
            "mse": "Mean squared field error over trajectories, spatial points, channels and interval frames",
            "relative_l2": "Mean over trajectories of ||prediction-target||_2 / ||target||_2 over the entire interval",
            "per_frame_relative_l2": "Mean over trajectories of spatial relative L2 at each frame",
            "intervals": "Zero-based half-open intervals; in_t includes the initial-frame reconstruction",
        },
        "metrics": metrics,
        "per_frame_mse": frame_mse.mean(0).tolist(),
        "per_frame_relative_l2": frame_l2.mean(0).tolist(),
    }
    result_path = absolute(args.output) if args.output else output_dir / f"test_metrics{suffix}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for key, result in metrics.items():
        print(f"{key}: mse={result['mse']:.8g} relative_l2={result['relative_l2']:.8f} ({result['relative_l2_percent']:.4f}%)")
    print(f"metrics={result_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--mode", choices=("author", "smoke"), default="author")
    for option in ("ode-checkpoint", "inr-checkpoint", "output-root", "data-dir"):
        parser.add_argument(f"--{option}")
    parser.add_argument("--ntest", type=int, default=None, help="Test prefix size (default: checkpoint count)")
    parser.add_argument("--batch-size", type=int, default=2, help="Trajectories per batch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
