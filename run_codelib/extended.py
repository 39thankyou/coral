"""Shared-INR stages for ragged endpoint IVPs and shared-INR latent dynamics.

IVPs adapt each mesh independently, then average case losses within each optimizer
batch. Dynamics fits one joint-field code per frame and evolves that code with
RK4 using the existing training-code normalization and scheduled sampling.
"""
import argparse
import json
import os
from pathlib import Path

import einops
import torch
from omegaconf import OmegaConf
from torch import nn
from torchdiffeq import odeint

from coral.losses import batch_mse_fn, batch_mse_rel_fn
from coral.metalearning import outer_step
from coral.mlp import Derivative
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from coral.utils.models.scheduling import ode_scheduling
from run_adaptor.raw_cylinder_dataset import RawCylinderFlowDataset
from run_adaptor.evaluate_navier_stokes import summarize_errors, check_grid
from run_codelib.pipeline import absolute, sha256
from run_codelib.eval_airfoil import restore_mapper, read_checkpoint
from run_codelib.checkpoints import validate_checkpoint, validate_config
from static.design_regression_shared import (create_mapper, decode_field, fit_latent,
                                             input_normalization, load_inrs, train_batch)


def save_state(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def graph_fields(cfg, split, count):
    dataset = RawCylinderFlowDataset(cfg.data.dir, split, count, dataset_name=cfg.data.dataset_name)
    result = []
    for graph, _ in dataset:
        coords = graph.pos[..., 0].unsqueeze(0)
        result.append((coords, graph.input[:, 2:].unsqueeze(0), graph.images.unsqueeze(0)))
    return result


def temporal_data(cfg):
    set_seed(int(cfg.data.get("sampling_seed", cfg.data.seed)))
    data = get_dynamics_data(cfg.data.dir, cfg.data.dataset_name,
                             int(cfg.data.get("sampling_ntrain", cfg.data.ntrain)),
                             int(cfg.data.get("sampling_ntest", cfg.data.ntest)),
                             seq_inter_len=int(cfg.data.seq_inter_len), seq_extra_len=int(cfg.data.seq_extra_len),
                             sub_from=cfg.data.sub_from, sub_tr=cfg.data.sub_tr,
                             sub_te=cfg.data.sub_te, same_grid=cfg.data.same_grid)
    counts = (cfg.data.ntrain, cfg.data.ntrain, cfg.data.ntest,
              cfg.data.ntrain, cfg.data.ntrain, cfg.data.ntest)
    return tuple(value[:int(count)] for value, count in zip(data, counts))


def flat_frames(values):
    return einops.rearrange(values, "b ... t -> (b t) ...")


def adapt(bundle, coords, values, steps, training=False):
    kwargs = dict(is_train=training, modulations=values.new_zeros(len(values), bundle["model"].modulation_net.latent_dim))
    return outer_step(bundle["model"], coords, values, steps, bundle["alpha"],
                      gradient_checkpointing=False, loss_type="mse", **kwargs)


def train_inr(cfg, family, device, root):
    set_seed(int(cfg.data.seed))
    if family == "graph":
        train = graph_fields(cfg, "train", int(cfg.data.ntrain))
        test = graph_fields(cfg, "test", int(cfg.data.ntest))
        models = (("in", "a", cfg.inr_in, 1), ("out", "u", cfg.inr_out, 2))
    else:
        data = temporal_data(cfg)
        train = [(c.unsqueeze(0), v.unsqueeze(0)) for c, v in zip(flat_frames(data[3]), flat_frames(data[0]))]
        test = [(c.unsqueeze(0), v.unsqueeze(0)) for c, v in zip(flat_frames(data[5]), flat_frames(data[2]))]
        models = (("state", "state", cfg.inr, 1),)
    bundles, optimizers = {}, {}
    for kind, suffix, architecture, value_index in models:
        current = OmegaConf.create({"inr": OmegaConf.to_container(architecture)})
        net = create_inr_instance(current, train[0][0].shape[-1], train[0][value_index].shape[-1], device)
        alpha = nn.Parameter(torch.tensor([float(cfg.optim.lr_code)], device=device))
        bundles[kind] = dict(model=net, alpha=alpha)
        optimizers[kind] = torch.optim.AdamW([{"params": net.parameters()},
                              {"params": [alpha], "lr": float(cfg.optim.meta_lr_code)}],
                              lr=float(cfg.optim.lr_inr), weight_decay=0)
    best = float("inf")
    history = []
    for epoch in range(int(cfg.optim.epochs)):
        totals = {kind: 0.0 for kind, *_ in models}
        order = torch.randperm(len(train)).tolist()
        for start in range(0, len(train), int(cfg.optim.batch_size)):
            indices = order[start:start+int(cfg.optim.batch_size)]
            for kind, suffix, architecture, value_index in models:
                bundle = bundles[kind]
                bundle["model"].train()
                losses = []
                for idx in indices:
                    coords, values = train[idx][0].to(device), train[idx][value_index].to(device)
                    out = adapt(bundle, coords, values, int(cfg.optim.inner_steps), training=True)
                    losses.append(out["loss"])
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite INR training loss")
                optimizers[kind].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_value_(bundle["model"].parameters(), 1.0)
                optimizers[kind].step()
                totals[kind] += float(loss.detach()) * len(indices)
        metrics = {f"train_{k}_mse": v/len(train) for k, v in totals.items()}
        # Evaluate each test case with independently fitted latent codes.
        if epoch == 0 or epoch == int(cfg.optim.epochs)-1 or epoch % 20 == 0:
            for kind, _, _, value_index in models:
                bundles[kind]["model"].eval()
                losses = [float(adapt(bundles[kind], item[0].to(device), item[value_index].to(device),
                                     int(cfg.optim.test_inner_steps))["loss"]) for item in test]
                metrics[f"test_{kind}_mse"] = sum(losses)/len(losses)
        selection = metrics[f"train_{models[-1][0]}_mse"]
        if selection < best:
            best = selection
            saved = {"cfg": cfg, "epoch": epoch, "loss": selection, "metrics": metrics,
                     "family": family}
            for kind, suffix, _, _ in models:
                key = "inr" if kind == "state" else f"inr_{kind}"
                alpha_key = "alpha" if kind == "state" else f"alpha_{kind}"
                saved[key] = bundles[kind]["model"].state_dict()
                saved[alpha_key] = bundles[kind]["alpha"].detach().cpu()
                saved[f"optimizer_{kind}"] = optimizers[kind].state_dict()
            if family == "dynamics":
                saved.update(grid_tr=data[3], grid_te=data[5])
            save_state(saved, root / "inr" / f"{cfg.wandb.name}.pt")
        history.append({"epoch": epoch, **metrics})
        print(f"inr epoch={epoch+1}/{cfg.optim.epochs} {metrics}", flush=True)
    write_json(history, root / "inr" / f"{cfg.wandb.name}_metrics.json")
    return best


def validate_temporal_inr(checkpoint, name):
    validate_checkpoint(checkpoint)
    cfg = checkpoint["cfg"]
    if cfg.data.dataset_name != name or cfg.inr.model_type != "siren":
        raise ValueError("Temporal INR dataset or model type mismatch")
    if not torch.isfinite(torch.as_tensor(checkpoint["alpha"])).all():
        raise ValueError("Non-finite temporal adaptation step size")


def restore_temporal(checkpoint, device, coordinate_dim, channels):
    cfg = checkpoint["cfg"]
    validate_temporal_inr(checkpoint, cfg.data.dataset_name)
    net = create_inr_instance(cfg, coordinate_dim, channels, device)
    net.load_state_dict(checkpoint["inr"])
    return dict(model=net.eval().requires_grad_(False), alpha=checkpoint["alpha"].detach().to(device))


def encode_frames(bundle, values, coordinates, steps, device, batch_size=4):
    frames, coords = flat_frames(values), flat_frames(coordinates)
    fitted = []
    for start in range(0, len(frames), batch_size):
        out = fit_latent(bundle, coords[start:start+batch_size].to(device), frames[start:start+batch_size].to(device), steps)
        fitted.append(out["modulations"].detach().cpu())
    return einops.rearrange(torch.cat(fitted), "(b t) l -> b l t", t=values.shape[-1])


def graph_predict(model, state, fields, inrs, steps, device):
    errors, mses, recon_in, recon_out = [], [], [], []
    model.eval()
    print(f"reconstruction adaptation: steps={steps} "
          f"lr_in={inrs['in']['alpha'].tolist()} lr_out={inrs['out']['alpha'].tolist()}", flush=True)
    for coords, inputs, targets in fields:
        coords, inputs, targets = coords.to(device), inputs.to(device), targets.to(device)
        fit = fit_latent(inrs["in"], coords, inputs, steps, use_rel_loss=True)
        normalized = (fit["modulations"].detach()-state["mu_a"])/state["sigma_a"]
        with torch.no_grad():
            code = model(normalized)
            prediction = decode_field(inrs["out"], coords, code*state["sigma_u"]+state["mu_u"])
            errors.append(batch_mse_rel_fn(prediction, targets).item())
            mses.append(batch_mse_fn(prediction, targets).item())
        recon_in.append(fit["rel_loss"].item())
        recon_out.append(fit_latent(inrs["out"], coords, targets, steps, use_rel_loss=True)["rel_loss"].item())
    return {"relative_l2": sum(errors)/len(errors), "relative_l2_percent": 100*sum(errors)/len(errors),
            "mse": sum(mses)/len(mses), "per_sample_relative_l2": errors, "per_sample_mse": mses,
            "inr_in_relative_l2": sum(recon_in)/len(recon_in), "inr_out_relative_l2": sum(recon_out)/len(recon_out)}


def train_graph_mapper(cfg, checkpoint, path, device, root):
    train, test = graph_fields(cfg, "train", int(cfg.data.ntrain)), graph_fields(cfg, "test", int(cfg.data.ntest))
    inrs = load_inrs(checkpoint, device, 2, train[0][1].shape[-1], train[0][2].shape[-1])
    codes = {}
    for kind, index in (("in", 1), ("out", 2)):
        for split, fields in (("train", train), ("test", test)):
            codes[f"z_{kind}_{split}"] = torch.cat([fit_latent(inrs[kind], item[0].to(device), item[index].to(device), int(cfg.inr.inner_steps))["modulations"].detach().cpu() for item in fields])
    mu_a, sigma_a = input_normalization(codes["z_in_train"])
    mu_u, sigma_u = input_normalization(codes["z_out_train"])
    if cfg.data.dataset_name == "airfoil-flow":
        mu_a, sigma_a = codes["z_in_train"].mean().expand_as(mu_a), codes["z_in_train"].std().expand_as(sigma_a)
        mu_u, sigma_u = codes["z_out_train"].mean().expand_as(mu_u), codes["z_out_train"].std().expand_as(sigma_u)
    inputs, targets = (codes["z_in_train"]-mu_a)/sigma_a, (codes["z_out_train"]-mu_u)/sigma_u
    model = create_mapper(cfg.model, inputs.shape[1], targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=float(cfg.optim.gamma_step), patience=250)
    state = {k: v.to(device) for k, v in dict(mu_a=mu_a, sigma_a=sigma_a, mu_u=mu_u, sigma_u=sigma_u).items()}
    best = float("inf")
    for epoch in range(int(cfg.optim.epochs)):
        order, total = torch.randperm(len(train)), 0.0
        for start in range(0, len(train), int(cfg.optim.batch_size)):
            idx = order[start:start+int(cfg.optim.batch_size)]
            total += train_batch(model, optimizer, inputs[idx].to(device), targets[idx].to(device))*len(idx)
        loss = total/len(train)
        scheduler.step(loss)
        if loss < best:
            best = loss
            metrics = graph_predict(model, state, test, inrs, int(cfg.inr.inner_steps), device)
            saved = {"cfg": cfg, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "inr_checkpoint": str(path), "inr_sha256": sha256(path),
                     "family": "graph", "metrics": metrics,
                     **{k: v.detach().cpu() for k, v in state.items()}}
            save_state(saved, root / "model" / f"{cfg.wandb.name}.pt")
        print(f"regression epoch={epoch+1}/{cfg.optim.epochs} code_mse={loss:.8g}", flush=True)
    save_state(codes, root / "modulations" / f"{cfg.wandb.name}_codes.pt")
    return best


def dynamics_predict(model, bundle, values, coordinates, mean, std, steps, inter_steps, device, batch_size):
    total_steps = values.shape[-1]
    times = torch.arange(total_steps, dtype=torch.float32, device=device)
    intervals = dict(in_t=(0, inter_steps), out_t=(inter_steps, total_steps), all=(0, total_steps))
    errors = {k: {"mse": [], "relative_l2": []} for k in intervals}
    frame_mse, frame_l2 = [], []
    model.eval()
    for start in range(0, len(values), batch_size):
        target = values[start:start+batch_size].to(device)
        coords = coordinates[start:start+batch_size].to(device)
        code = encode_frames(bundle, target[..., :1], coords[..., :1], steps, device).to(device)[..., 0]
        with torch.no_grad():
            latent = odeint(model, (code-mean)/std, times, method="rk4").movedim(0, -1)
            decoded = decode_field(bundle, flat_frames(coords), flat_frames(latent*std[..., None]+mean[..., None]))
            prediction = einops.rearrange(decoded, "(b t) ... -> b ... t", t=total_steps)
        for key, (a, b) in intervals.items():
            errors[key]["mse"].append(batch_mse_fn(prediction[..., a:b].contiguous(), target[..., a:b].contiguous()).cpu())
            errors[key]["relative_l2"].append(batch_mse_rel_fn(prediction[..., a:b].contiguous(), target[..., a:b].contiguous()).cpu())
        frame_mse.append(batch_mse_fn(flat_frames(prediction).contiguous(), flat_frames(target).contiguous()).reshape(-1, total_steps).cpu())
        frame_l2.append(batch_mse_rel_fn(flat_frames(prediction).contiguous(), flat_frames(target).contiguous()).reshape(-1, total_steps).cpu())
    return {"metrics": {key: summarize_errors(**item) for key, item in errors.items()},
            "per_frame_mse": torch.cat(frame_mse).mean(0).tolist(),
            "per_frame_relative_l2": torch.cat(frame_l2).mean(0).tolist(),
            "frame_intervals": intervals, "teacher_forcing": False, "initial_frame": 0, "solver": "rk4", "dt": 1}


def train_ode(cfg, checkpoint, path, device, root):
    # Regenerate the exact INR sampling before inferring any training frame code.
    original = checkpoint["cfg"].data
    cfg.data.sampling_ntrain = original.get("sampling_ntrain", original.ntrain)
    cfg.data.sampling_ntest = original.get("sampling_ntest", original.ntest)
    cfg.data.sampling_seed = original.get("sampling_seed", original.seed)
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    data_cfg.data.seed = checkpoint["cfg"].data.seed
    data = temporal_data(data_cfg)
    check_grid(data[3], checkpoint["grid_tr"][:int(cfg.data.ntrain)], "training")
    check_grid(data[5], checkpoint["grid_te"][:int(cfg.data.ntest)], "test")
    bundle = restore_temporal(checkpoint, device, data[3].shape[-2], data[0].shape[-2])
    codes = encode_frames(bundle, data[0], data[3], int(cfg.inr.inner_steps), device)
    mean, std = input_normalization(flat_frames(codes))
    normalized = (codes-mean[:, None])/std[:, None]
    model = Derivative(1, len(mean), cfg.dynamics.width, cfg.dynamics.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=float(cfg.optim.gamma_step), patience=250)
    times = torch.arange(codes.shape[-1], dtype=torch.float32, device=device)
    best, epsilon = float("inf"), float(cfg.dynamics.teacher_forcing_decay)
    mean, std = mean.to(device), std.to(device)
    for epoch in range(int(cfg.optim.epochs)):
        if epoch % int(cfg.dynamics.teacher_forcing_update) == 0:
            epsilon *= float(cfg.dynamics.teacher_forcing_init)
        model.train()
        order, total = torch.randperm(len(codes)), 0.0
        for start in range(0, len(codes), int(cfg.optim.batch_size)):
            idx = order[start:start+int(cfg.optim.batch_size)]
            target = normalized[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = ode_scheduling(odeint, model, target, times, epsilon, method="rk4")
            loss = (prediction-target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite ODE training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(idx)
        loss = total/len(codes)
        scheduler.step(loss)
        if loss < best:
            best = loss
            saved = {"cfg": cfg, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                     "inr_checkpoint": str(path), "inr_sha256": sha256(path), "family": "dynamics",
                     "grid_tr": data[3], "grid_te": data[5]}
            save_state(saved, root / "model" / f"{cfg.wandb.name}.pt")
        print(f"ode epoch={epoch+1}/{cfg.optim.epochs} code_mse={loss:.8g} epsilon={epsilon:.6g}", flush=True)
    save_state({"z_train": codes, "mean": mean.cpu(), "std": std.cpu()}, root / "modulations" / f"{cfg.wandb.name}_codes.pt")
    return best


def run_stage(cfg, stage, family):
    validate_config(cfg)
    root = Path(os.environ["WANDB_DIR"]) / cfg.data.dataset_name
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if stage == "inr":
        return train_inr(cfg, family, device, root)
    path = absolute(cfg.inr.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint)
    if checkpoint["cfg"].data.dataset_name != cfg.data.dataset_name:
        raise ValueError("INR checkpoint belongs to another dataset")
    if cfg.data.ntrain > checkpoint["cfg"].data.ntrain or cfg.data.ntest > checkpoint["cfg"].data.ntest:
        raise ValueError("Requested cases exceed the saved INR split")
    set_seed(int(cfg.data.seed))
    if family == "graph":
        return train_graph_mapper(cfg, checkpoint, path, device, root)
    return train_ode(cfg, checkpoint, path, device, root)


def parse_args(dataset):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--mode", choices=("full", "smoke"))
    for name in ("regression-checkpoint", "inr-checkpoint", "output-root", "data-dir", "output"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--ntest", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.dataset = dataset
    return args


def evaluate(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    config = OmegaConf.load(absolute(args.config))
    dataset = config.datasets[args.dataset]
    mode = args.mode or config.mode
    suffix = "-smoke" if mode == "smoke" else ""
    root = absolute(args.output_root or config.common.output_root) / dataset.author_name
    path = absolute(args.regression_checkpoint) if args.regression_checkpoint else root / "model" / f"{dataset.regression_run_name}{suffix}.pt"
    if args.regression_checkpoint and not args.output_root:
        root = path.parent.parent
    saved, digest = read_checkpoint(path)
    validate_checkpoint(saved)
    cfg = OmegaConf.create(OmegaConf.to_container(saved["cfg"]))
    if cfg.data.dataset_name != dataset.author_name:
        raise ValueError("Checkpoint dataset mismatch")
    inr_path = absolute(args.inr_checkpoint) if args.inr_checkpoint else root / "inr" / Path(saved["inr_checkpoint"]).name
    if not inr_path.exists():
        inr_path = absolute(saved["inr_checkpoint"])
    checkpoint, inr_digest = read_checkpoint(inr_path)
    validate_checkpoint(checkpoint)
    if inr_digest != saved["inr_sha256"]:
        raise ValueError("INR hash does not match the downstream training snapshot")
    ntest = int(args.ntest if args.ntest is not None else cfg.data.ntest)
    if not 1 <= ntest <= min(int(cfg.data.ntest), int(checkpoint["cfg"].data.ntest)):
        raise ValueError("ntest must be a prefix within both saved splits")
    device = torch.device(args.device)
    cfg.data.dir = str(absolute(args.data_dir or dataset.data_dir))
    if dataset.family == "graph":
        fields = graph_fields(cfg, "test", ntest)
        inrs = load_inrs(checkpoint, device, 2, fields[0][1].shape[-1], fields[0][2].shape[-1])
        model, state = restore_mapper(saved, checkpoint, device)
        metrics = graph_predict(model, state, fields, inrs, int(cfg.inr.inner_steps), device)
    else:
        cfg.data.seed = checkpoint["cfg"].data.seed
        data = temporal_data(cfg)
        check_grid(data[3], saved["grid_tr"], "training")
        check_grid(data[5], saved["grid_te"], "test")
        bundle = restore_temporal(checkpoint, device, data[3].shape[-2], data[0].shape[-2])
        model = Derivative(1, int(checkpoint["cfg"].inr.latent_dim), cfg.dynamics.width, cfg.dynamics.depth).to(device)
        state = saved["model"]
        # Older plain ODE checkpoints wrapped the unchanged derivative once.
        if state and all(key.startswith("derivative.") for key in state):
            state = {key.removeprefix("derivative."): value for key, value in state.items()}
        model.load_state_dict(state)
        model.eval().requires_grad_(False)
        mean, std = saved["mean"].to(device), saved["std"].to(device)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid saved normalization")
        metrics = dynamics_predict(model, bundle, data[2][:ntest], data[5][:ntest], mean, std, int(cfg.inr.inner_steps), int(cfg.data.seq_inter_len), device, args.batch_size)
    summary = {"dataset": dataset.author_name, "mode": mode, "ntest": ntest, "test_indices": [0, ntest],
               "inr_checkpoint": str(inr_path), "regression_checkpoint": str(path),
               "inr_sha256": inr_digest, "regression_sha256": digest,
               "inr_epoch_zero_based": int(checkpoint["epoch"]), "regression_epoch_zero_based": int(saved["epoch"]),
               "data_dir": str(cfg.data.dir), "normalization_ntrain": int(cfg.data.ntrain),
               "normalization_source": "Saved training-code statistics", "device": str(device),
               "model_weights_updated": False, **metrics}
    destination = absolute(args.output) if args.output else root / f"test_metrics{suffix}.json"
    write_json(summary, destination)
    print(f"metrics={destination}", flush=True)
    return summary
