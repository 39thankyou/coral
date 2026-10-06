"""Evaluate the author's separate height/vorticity INRs and joint SW latent ODE."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import einops
import torch
from omegaconf import OmegaConf
from torchdiffeq import odeint
from coral.losses import batch_mse_fn, batch_mse_rel_fn
from coral.mlp import Derivative
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from run_adaptor.evaluate_navier_stokes import absolute, encode_trajectories, check_grid, summarize_errors, parse_args


def evaluate(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    config = OmegaConf.load(absolute(args.config))
    dataset = config.datasets.shallow_water
    suffix = "-smoke" if args.mode == "smoke" else ""
    root = absolute(args.output_root or config.common.output_root) / dataset.author_name
    path = absolute(args.ode_checkpoint) if args.ode_checkpoint else root / "model" / f"{dataset.downstream_run_name}{suffix}.pt"
    if args.ode_checkpoint and not args.output_root:
        root = path.parent.parent
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = saved["cfg"]
    if cfg.data.dataset_name != dataset.author_name or cfg.inr.run_name is not None:
        raise ValueError("Expected a separate-channel SW ODE checkpoint")
    channels = ("height", "vorticity")
    snapshots, paths = {}, {}
    for channel in channels:
        paths[channel] = root / channel / "inr" / f"{cfg.inr.run_dict[channel]}.pt"
        snapshots[channel] = torch.load(paths[channel], map_location="cpu", weights_only=False)
        if snapshots[channel]["cfg"].data.dataset_name != dataset.author_name or snapshots[channel]["cfg"].data.data_to_encode != channel:
            raise ValueError(f"Wrong SW {channel} INR checkpoint")
    ntrain, saved_ntest = int(cfg.data.ntrain), int(cfg.data.ntest)
    ntest = int(args.ntest if args.ntest is not None else saved_ntest)
    if not 1 <= ntest <= saved_ntest:
        raise ValueError("ntest must be within the saved SW test split")
    base = snapshots["height"]["cfg"]
    inter, extra = int(cfg.data.seq_inter_len), int(cfg.data.seq_extra_len)
    set_seed(int(base.data.seed))
    data_dir = absolute(args.data_dir or dataset.data_dir)
    data = get_dynamics_data(data_dir, dataset.author_name, ntrain, saved_ntest,
            seq_inter_len=inter, seq_extra_len=extra, sub_from=base.data.sub_from,
            sub_tr=base.data.sub_tr, sub_te=cfg.data.sub_te, same_grid=cfg.data.same_grid)
    check_grid(data[3], saved["grid_tr"], "training")
    check_grid(data[5], saved["grid_te"], "test")
    device = torch.device(args.device)
    inrs, means, stds, alpha = {}, {}, {}, {}
    latent = int(base.inr.latent_dim)
    steps = int(cfg.inr.inner_steps)
    for index, channel in enumerate(channels):
        snapshot = snapshots[channel]
        inr = create_inr_instance(snapshot["cfg"], data[3].shape[-2], 1, device)
        inr.load_state_dict(snapshot["inr"])
        inrs[channel] = inr.eval().requires_grad_(False)
        alpha[channel] = snapshot["alpha"].detach().to(device)
        codes = []
        # Match the author's fixed two-trajectory encoding batch size.
        for start in range(0, ntrain, 2):
            codes.append(encode_trajectories(inr, data[0][start:start+2, ..., index:index+1, :].to(device),
                         data[3][start:start+2].to(device), latent, alpha[channel], steps).cpu())
        flat = einops.rearrange(torch.cat(codes), "b l t -> (b t) l")
        means[channel], stds[channel] = flat.mean(0).to(device), flat.std(0).to(device)
        if not torch.isfinite(stds[channel]).all() or (stds[channel] <= 0).any():
            raise ValueError("Invalid SW training-code normalization")
    model = Derivative(2, latent, cfg.dynamics.width, cfg.dynamics.depth).to(device)
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    total = inter+extra
    times = torch.arange(total, dtype=torch.float32, device=device)
    intervals = {"in_t": (0, inter), "out_t": (inter, total), "all": (0, total)}
    errors = {k: {"mse": [], "relative_l2": []} for k in intervals}
    frame_mse, frame_l2 = [], []
    for start in range(0, ntest, args.batch_size):
        target = data[2][start:min(start+args.batch_size, ntest)].to(device)
        coords = data[5][start:min(start+args.batch_size, ntest)].to(device)
        initial = []
        for index, channel in enumerate(channels):
            z = encode_trajectories(inrs[channel], target[..., index:index+1, :1], coords[..., :1], latent, alpha[channel], steps)[..., 0]
            initial.append((z-means[channel])/stds[channel])
        with torch.no_grad():
            initial = torch.stack(initial, dim=-1).flatten(1)
            z = odeint(model, initial, times, method="rk4")
            z = einops.rearrange(z, "t b (l c) -> b l c t", l=latent, c=2)
            fields = []
            flat_coords = einops.rearrange(coords, "b ... t -> (b t) ...")
            for index, channel in enumerate(channels):
                native = z[:, :, index]*stds[channel][:, None]+means[channel][:, None]
                field = inrs[channel].modulated_forward(flat_coords, einops.rearrange(native, "b l t -> (b t) l"))
                fields.append(einops.rearrange(field, "(b t) ... -> b ... t", t=total))
            prediction = torch.cat(fields, dim=-2)
        for key, (a, b) in intervals.items():
            errors[key]["mse"].append(batch_mse_fn(prediction[..., a:b].contiguous(), target[..., a:b].contiguous()).cpu())
            errors[key]["relative_l2"].append(batch_mse_rel_fn(prediction[..., a:b].contiguous(), target[..., a:b].contiguous()).cpu())
        pred_frames = einops.rearrange(prediction, "b ... t -> (b t) ...").contiguous()
        true_frames = einops.rearrange(target, "b ... t -> (b t) ...").contiguous()
        frame_mse.append(batch_mse_fn(pred_frames, true_frames).reshape(-1, total).cpu())
        frame_l2.append(batch_mse_rel_fn(pred_frames, true_frames).reshape(-1, total).cpu())
        print(f"evaluate SW: {min(start+args.batch_size, ntest)}/{ntest}", flush=True)
    summary = {"dataset": dataset.author_name, "mode": args.mode, "ntest": ntest,
               "ode_checkpoint": str(path), "inr_checkpoints": {k: str(v) for k,v in paths.items()},
               "normalization_ntrain": ntrain, "normalization_frames": [0, inter],
               "test_window_indices": [0, ntest], "frame_intervals": intervals,
               "checkpoint_grids_verified": True, "solver": "rk4", "teacher_forcing": False,
               "initial_frame": 0, "metrics": {k: summarize_errors(**v) for k,v in errors.items()},
               "per_frame_mse": torch.cat(frame_mse).mean(0).tolist(),
               "per_frame_relative_l2": torch.cat(frame_l2).mean(0).tolist()}
    destination = absolute(args.output) if args.output else root / f"test_metrics{suffix}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"metrics={destination}", flush=True)
    return summary


if __name__ == "__main__":
    evaluate(parse_args())
