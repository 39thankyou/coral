"""Evaluate shared-INR checkpoints and save per-case test_metrics.json.

Prediction encodes only the input geometry. Separate INR reconstruction
diagnostics also fit observed output codes and report aggregate relative L2.
All model parameters, normalization statistics remain fixed.
"""

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from coral.losses import batch_mse_fn, batch_mse_rel_fn
from coral.utils.data.load_data import get_operator_data, set_seed
from run_codelib.pipeline import absolute
from run_codelib.checkpoints import validate_checkpoint
from static.design_regression_shared import create_mapper, decode_field, fit_latent, load_inrs
from run_codelib.representation import configure_request, suffix as representation_suffix, validate_overrides


def read_checkpoint(path):
    # Hash and deserialize the same snapshot even if a training job saves again.
    contents = path.read_bytes()
    return (torch.load(io.BytesIO(contents), map_location="cpu", weights_only=False),
            hashlib.sha256(contents).hexdigest())


def restore_mapper(checkpoint, inr_checkpoint, device):
    validate_checkpoint(checkpoint)
    validate_checkpoint(inr_checkpoint)
    required = {"mu_a", "sigma_a", "mu_u", "sigma_u"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Regression checkpoint lacks inference state: {sorted(required - checkpoint.keys())}")
    cfg = checkpoint["cfg"]
    input_dim = int(inr_checkpoint["cfg"].inr_in.latent_dim)
    output_dim = int(inr_checkpoint["cfg"].inr_out.latent_dim)
    state = {key: checkpoint[key].detach().to(device) for key in required}
    for key in ("mu_a", "sigma_a"):
        if state[key].shape != (input_dim,):
            raise ValueError(f"Invalid saved {key} dimensions")
    for key in ("mu_u", "sigma_u"):
        if state[key].shape not in (torch.Size([1]), torch.Size([output_dim])):
            raise ValueError(f"Invalid saved {key} dimensions")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Non-finite saved normalization")
    if (state["sigma_a"] <= 0).any() or (state["sigma_u"] <= 0).any():
        raise ValueError("Saved normalization scales must be positive")
    model = create_mapper(cfg.model, input_dim, output_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False), state


def evaluate(args):
    dataset_name = getattr(args, "dataset", "airfoil")
    if args.batch_size <= 0 or (args.ntest is not None and args.ntest <= 0):
        raise ValueError("batch-size and ntest must be positive")
    config = OmegaConf.load(absolute(args.config))
    from copy import copy
    naming_args = copy(args)
    configure_request(config, config.datasets[dataset_name], naming_args)
    mode = args.mode or config.mode
    if mode not in ("full", "smoke"):
        raise ValueError("mode must be full or smoke")
    suffix = "-smoke" if mode == "smoke" else ""
    output_dir = absolute(args.output_root or config.common.output_root) / dataset_name
    regression_path = (absolute(args.regression_checkpoint) if args.regression_checkpoint else
                       output_dir / "model" / f"{config.datasets[dataset_name].regression_run_name}{representation_suffix(naming_args)}{suffix}.pt")
    if args.regression_checkpoint and not args.output_root:
        output_dir = regression_path.parent.parent
    print(f"selected regression checkpoint: {regression_path}", flush=True)
    if not regression_path.is_file():
        raise FileNotFoundError(
            f"Regression checkpoint not found: {regression_path}\n"
            "Evaluation selects the configured experiment, not the latest file. "
            "For AnchorMix use --representation anchormix (and the training --mode/--output-root), "
            "or select the trained processor with --regression-checkpoint."
        )
    checkpoint, regression_hash = read_checkpoint(regression_path)
    cfg = checkpoint["cfg"]
    validate_checkpoint(checkpoint)
    if cfg.data.dataset_name != dataset_name:
        raise ValueError(f"Expected a {dataset_name} regression checkpoint")
    if args.inr_checkpoint:
        inr_path = absolute(args.inr_checkpoint)
    else:
        # Prefer the managed copy, allowing the whole output tree to be relocated.
        inr_path = output_dir / "inr" / Path(checkpoint["inr_checkpoint"]).name
        if not inr_path.exists():
            inr_path = absolute(checkpoint["inr_checkpoint"])
    inr_checkpoint, inr_hash = read_checkpoint(inr_path)
    validate_checkpoint(inr_checkpoint)
    if inr_hash != checkpoint["inr_sha256"]:
        raise ValueError(
            "INR SHA-256 does not match the regression training checkpoint.\n"
            f"Processor: {regression_path}\nINR: {inr_path}\n"
            f"Expected: {checkpoint['inr_sha256']}\nActual:   {inr_hash}\n"
            "The selected INR is not the snapshot used to train this processor; "
            "it may have been overwritten by another training run. "
            "Restore the matching INR using --inr-checkpoint, or retrain the processor for this INR.\n"
            "Evaluation does not automatically select the latest experiment. "
            "If you intended to evaluate AnchorMix, use --representation anchormix "
            "or --regression-checkpoint pointing to its processor."
        )
    if inr_checkpoint["cfg"].data.dataset_name != dataset_name:
        raise ValueError(f"Expected a {dataset_name} INR checkpoint")
    validate_overrides(args, inr_checkpoint["cfg"])
    ntest = int(args.ntest if args.ntest is not None else cfg.data.ntest)
    if not 1 <= ntest <= min(200, int(cfg.data.ntest), int(inr_checkpoint["cfg"].data.ntest)):
        raise ValueError("ntest must be a prefix within both checkpoints' saved test split")
    inner_steps = int(cfg.inr.inner_steps)
    if inner_steps < 1:
        raise ValueError("Saved inner_steps must be positive")
    device = torch.device(args.device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(int(inr_checkpoint["cfg"].data.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, state = restore_mapper(checkpoint, inr_checkpoint, device)
    data_dir = absolute(args.data_dir or config.datasets[dataset_name].data_dir)
    # Elasticity coordinates are the mean geometry of the regression training
    # partition. Its loader takes the last ntest cases, so load the saved split
    # before selecting a prefix for --ntest.
    saved_ntest = int(cfg.data.ntest)
    load_ntrain = int(cfg.data.ntrain) if dataset_name == "elasticity" else 1
    load_ntest = saved_ntest if dataset_name == "elasticity" else ntest
    _, _, inputs, targets, _, coordinates = get_operator_data(
        data_dir, dataset_name, load_ntrain, load_ntest,
        sub_tr=1, sub_te=1, same_grid=True,
    )
    inputs, targets, coordinates = inputs[:ntest], targets[:ntest], coordinates[:ntest]
    test_start = 1000
    if dataset_name == "elasticity":
        import numpy as np
        sigma = np.load(data_dir / "Meshes" / "Random_UnitCell_sigma_10.npy", mmap_mode="r")
        test_start = int(sigma.shape[1]) - saved_ntest
    inrs = load_inrs(inr_checkpoint, device, coordinates.shape[-1], inputs.shape[-1], targets.shape[-1])
    chunk_size = getattr(args, "query_chunk_size", None)
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        for bundle in inrs.values():
            if hasattr(bundle["model"], "query_chunk_size"):
                bundle["model"].query_chunk_size = chunk_size
    supplied_steps = getattr(args, "reconstruction_steps", None)
    reconstruction_steps = inner_steps if supplied_steps is None else supplied_steps
    reconstruction_lr = getattr(args, "reconstruction_lr", None)
    if reconstruction_steps < 1 or (reconstruction_lr is not None and
            (not torch.isfinite(torch.tensor(reconstruction_lr)) or reconstruction_lr <= 0)):
        raise ValueError("Diagnostic adaptation steps and learning rate must be positive and finite")
    diagnostic_inrs = {kind: dict(bundle) for kind, bundle in inrs.items()}
    if reconstruction_lr is not None:
        for bundle in diagnostic_inrs.values():
            bundle["alpha"] = torch.tensor([reconstruction_lr], device=device)
    adaptation = {kind: {"steps": reconstruction_steps, "lr": bundle["alpha"].detach().cpu().tolist()}
                  for kind, bundle in diagnostic_inrs.items()}
    print(f"prediction input adaptation: steps={inner_steps} lr={inrs['in']['alpha'].tolist()}\n"
          f"reconstruction adaptation: {json.dumps(adaptation)}", flush=True)
    errors_l2, errors_mse = [], []
    inr_in_relative_l2_sum, inr_out_relative_l2_sum = 0.0, 0.0
    print(f"regression={regression_path}\ninr={inr_path}", flush=True)
    for start in range(0, ntest, args.batch_size):
        stop = min(start + args.batch_size, ntest)
        values = inputs[start:stop].to(device)
        coords = coordinates[start:stop].to(device)
        # Reuse the input-code fit's final reconstruction for the INR-in metric.
        input_fit = fit_latent(inrs["in"], coords, values, inner_steps,
                               use_rel_loss=True)
        encoded = input_fit["modulations"].detach()
        diagnostic_fit = input_fit
        if reconstruction_steps != inner_steps or reconstruction_lr is not None:
            diagnostic_fit = fit_latent(diagnostic_inrs["in"], coords, values, reconstruction_steps,
                                        use_rel_loss=True)
        inr_in_relative_l2_sum += diagnostic_fit["rel_loss"].item() * len(values)
        with torch.no_grad():
            normalized = (encoded - state["mu_a"]) / state["sigma_a"]
            z_output = model(normalized)
            prediction = decode_field(inrs["out"], coords, z_output * state["sigma_u"] + state["mu_u"])
            target = targets[start:stop].to(device)
            errors_l2.append(batch_mse_rel_fn(prediction, target).cpu())
            errors_mse.append(batch_mse_fn(prediction, target).cpu())
        # Independently fit the observed output for reconstruction diagnostics.
        # Never feed these target-derived codes to the mapper.
        output_fit = fit_latent(diagnostic_inrs["out"], coords, target, reconstruction_steps,
                                use_rel_loss=True)
        inr_out_relative_l2_sum += output_fit["rel_loss"].item() * len(values)
        if start // args.batch_size % 25 == 0 or stop == ntest:
            print(f"evaluate test cases: {stop}/{ntest}", flush=True)
    errors_l2, errors_mse = torch.cat(errors_l2), torch.cat(errors_mse)
    if not torch.isfinite(errors_l2).all() or not torch.isfinite(errors_mse).all():
        raise ValueError("Non-finite test metrics")
    relative_l2 = errors_l2.mean().item()
    inr_in_relative_l2 = inr_in_relative_l2_sum / ntest
    inr_out_relative_l2 = inr_out_relative_l2_sum / ntest
    if not torch.isfinite(torch.tensor([inr_in_relative_l2, inr_out_relative_l2])).all():
        raise ValueError("Non-finite INR reconstruction metrics")
    summary = {
        "dataset": dataset_name, "method": ("anchormix" if any(inr_checkpoint["cfg"][f"inr_{kind}"].model_type == "anchormix" for kind in ("in", "out"))
                                          else inr_checkpoint["cfg"].inr_out.model_type), "mode": mode,
        "inr_checkpoint": str(inr_path), "regression_checkpoint": str(regression_path),
        "inr_sha256": inr_hash, "regression_sha256": regression_hash,
        "inr_epoch_zero_based": int(inr_checkpoint["epoch"]),
        "regression_epoch_zero_based": int(checkpoint["epoch"]),
        "data_dir": str(data_dir), "normalization_ntrain": int(cfg.data.ntrain),
        "normalization_source": "Saved regression checkpoint; no test statistics",
        "ntest": ntest, "test_indices": [test_start, test_start + ntest],
        "inner_steps": inner_steps, "batch_size": args.batch_size, "device": str(device),
        "reconstruction_adaptation": adaptation,
        "representation": inr_checkpoint.get("representation"),
        "query_chunk_size_override": chunk_size,
        "relative_l2": relative_l2, "relative_l2_percent": relative_l2 * 100,
        "mse": errors_mse.mean().item(),
        "inr_in_relative_l2": inr_in_relative_l2,
        "inr_out_relative_l2": inr_out_relative_l2,
        "inr_reconstruction_metric_definition": (
            "Mean over test cases of ||INR(coords, fitted_code)-observed_field||_2 / "
            "||observed_field||_2; codes fitted from zero using reconstruction_adaptation steps and lr. "
            "INR-out fits observed test outputs independently of the mapper."
        ),
        "metric_definition": "Mean of per-sample ||prediction-target||_2 / ||target||_2",
        "per_sample_relative_l2": errors_l2.tolist(), "per_sample_mse": errors_mse.tolist(),
        "model_weights_updated": False,
    }
    saved_rep = inr_checkpoint["cfg"].get("representation_branch", "out")
    rep_suffix = f"-anchormix-{saved_rep}" if summary["method"] == "anchormix" else representation_suffix(naming_args)
    path = absolute(args.output) if args.output else output_dir / f"test_metrics{rep_suffix}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"test_relative_l2={relative_l2:.8f} ({relative_l2 * 100:.4f}%)\n"
          f"test_mse={summary['mse']:.8g}\n"
          f"test_inr_in_relative_l2={inr_in_relative_l2:.8f}\n"
          f"test_inr_out_relative_l2={inr_out_relative_l2:.8f}\nmetrics={path}", flush=True)
    return summary


def parse_args(dataset_name="airfoil"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--mode", choices=("full", "smoke"))
    parser.add_argument("--regression-checkpoint", help="Exact trained processor; otherwise use the config/representation/mode filename, not the latest checkpoint")
    parser.add_argument("--inr-checkpoint", help="Relocated copy of the exact INR used for regression")
    parser.add_argument("--output-root")
    parser.add_argument("--data-dir")
    parser.add_argument("--ntest", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Override the test_metrics.json output path")
    parser.add_argument("--dataset", choices=("airfoil", "elasticity", "pipe"), default=dataset_name)
    parser.add_argument("--reconstruction-steps", type=int, help="Diagnostic adaptation only; prediction input encoding is unchanged")
    parser.add_argument("--reconstruction-lr", type=float, help="Diagnostic adaptation only; default is each INR's saved learned alpha")
    from anchormix.config import add_arguments
    add_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    evaluate(parse_args())
