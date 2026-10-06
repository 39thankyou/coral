"""Evaluate author Airfoil/Elasticity/Pipe models and visualize INR errors."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coral.losses import batch_mse_fn, batch_mse_rel_fn
from coral.metalearning import outer_step
from coral.mlp import ResNet
from coral.utils.data.load_data import get_operator_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from run_adaptor.visualize_error import save_inr_error_case, select_test_indices


def absolute(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def encode(inr, values, coordinates, latent_dim, alpha, inner_steps, return_relative_l2=False,
           return_reconstructions=False):
    # The inner loop fits sample codes; all model weights remain frozen.
    result = outer_step(
        inr, coordinates, values, inner_steps, alpha,
        is_train=False, gradient_checkpointing=False, loss_type="mse",
        modulations=values.new_zeros(len(values), latent_dim),
        use_rel_loss=return_relative_l2,
        return_reconstructions=return_reconstructions,
    )
    outputs = [result["modulations"].detach()]
    if return_relative_l2:
        outputs.append(result["rel_loss"].item())
    if return_reconstructions:
        outputs.append(result["reconstructions"].detach())
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def evaluate(args):
    dataset_name = getattr(args, "dataset", "airfoil")
    if args.batch_size <= 0 or (args.ntest is not None and args.ntest <= 0):
        raise ValueError("batch-size and ntest must be positive")
    visualization_k = getattr(args, "visualization_k", 10)
    if visualization_k <= 0:
        raise ValueError("visualization-k must be positive")
    device = torch.device(args.device)
    adaptor = OmegaConf.load(absolute(args.config))
    dataset = adaptor.datasets[dataset_name]
    output_dir = absolute(getattr(args, "output_root", None) or adaptor.common.output_root) / dataset.author_name
    suffix = "-smoke" if args.mode == "smoke" else ""
    regression_path = output_dir / "model" / f"{dataset.downstream_run_name}{suffix}.pt"
    if getattr(args, "regression_checkpoint", None):
        regression_path = absolute(args.regression_checkpoint)
        if not getattr(args, "output_root", None):
            output_dir = regression_path.parent.parent
    regression_checkpoint = torch.load(
        regression_path, map_location="cpu", weights_only=False
    )
    cfg = regression_checkpoint["cfg"]
    inr_path = output_dir / "inr" / f"{cfg.inr.run_name}.pt"
    if getattr(args, "inr_checkpoint", None):
        inr_path = absolute(args.inr_checkpoint)
    inr_checkpoint = torch.load(inr_path, map_location="cpu", weights_only=False)
    inr_cfg = inr_checkpoint["cfg"]
    if cfg.data.dataset_name != dataset_name or inr_cfg.data.dataset_name != dataset_name:
        raise ValueError(f"Both checkpoints must belong to {dataset_name}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(int(inr_cfg.data.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    ntrain = int(cfg.data.ntrain)
    ntest = int(args.ntest if args.ntest is not None else cfg.data.ntest)
    if ntrain < 2:
        raise ValueError("At least two training samples are needed for code normalization")
    saved_ntest = int(cfg.data.ntest)
    if not 1 <= ntest <= min(saved_ntest, int(inr_cfg.data.ntest)):
        raise ValueError("ntest must be a prefix within both saved test splits")
    if ntrain > int(inr_cfg.data.ntrain):
        raise ValueError("Regression training split exceeds the INR training split")
    data_dir = absolute(getattr(args, "data_dir", None) or dataset.data_dir)
    load_ntest = saved_ntest if dataset_name == "elasticity" else ntest
    x_train, _, x_test, y_test, grid_train, grid_test = get_operator_data(
        data_dir, dataset_name, ntrain, load_ntest,
        sub_tr=1, sub_te=1, same_grid=True,
    )
    x_test, y_test, grid_test = x_test[:ntest], y_test[:ntest], grid_test[:ntest]
    test_start = 1000
    if dataset_name == "elasticity":
        import numpy as np
        sigma = np.load(data_dir / "Meshes" / "Random_UnitCell_sigma_10.npy", mmap_mode="r")
        test_start = int(sigma.shape[1]) - saved_ntest
    latent_dim = int(inr_cfg.inr_in.latent_dim)
    inr_cfg.inr = inr_cfg.inr_in
    inr_in = create_inr_instance(inr_cfg, input_dim=grid_train.shape[-1], output_dim=x_train.shape[-1], device=device)
    inr_in.load_state_dict(inr_checkpoint["inr_in"])
    inr_cfg.inr = inr_cfg.inr_out
    inr_out = create_inr_instance(inr_cfg, input_dim=grid_test.shape[-1], output_dim=y_test.shape[-1], device=device)
    inr_out.load_state_dict(inr_checkpoint["inr_out"])
    model = ResNet(
        input_dim=latent_dim, hidden_dim=cfg.model.width,
        output_dim=inr_cfg.inr_out.latent_dim, depth=cfg.model.depth,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(regression_checkpoint["model"])
    for network in (inr_in, inr_out, model):
        network.eval()
        network.requires_grad_(False)
    alpha = inr_checkpoint["alpha_in"].detach().to(device)
    alpha_out = inr_checkpoint["alpha_out"].detach().to(device)
    inner_steps = int(cfg.inr.inner_steps)

    # Author regression checkpoints omit mu_a/sigma_a. Recover exactly the
    # training-input code statistics used by design_regression.py (no fitting
    # of model weights and no statistics from the test set). Encoding uses the
    # author's fixed batch size of 4, independent of the test batch size.
    train_loader = DataLoader(
        TensorDataset(x_train, grid_train), batch_size=4, shuffle=False
    )
    codes = []
    print(f"Recovering input-code normalization from {ntrain} training samples", flush=True)
    for index, (values, coordinates) in enumerate(train_loader):
        codes.append(encode(
            inr_in, values.to(device), coordinates.to(device),
            latent_dim, alpha, inner_steps,
        ).cpu())
        if (index + 1) % 50 == 0 or index + 1 == len(train_loader):
            print(f"normalization batches={index + 1}/{len(train_loader)}", flush=True)
    codes = torch.cat(codes)
    mean = codes.mean(0).to(device)
    std = codes.std(0).to(device)
    if not torch.isfinite(codes).all() or not torch.isfinite(std).all() or (std == 0).any():
        raise ValueError("Invalid training-code normalization statistics")

    test_loader = DataLoader(
        TensorDataset(x_test, y_test, grid_test),
        batch_size=args.batch_size, shuffle=False,
    )
    errors_l2, errors_mse = [], []
    inr_in_relative_l2_sum, inr_out_relative_l2_sum = 0.0, 0.0
    selected_indices = select_test_indices(ntest, visualization_k)
    visualization_dir = output_dir / "visualization_error"
    visualization = {
        "directory": str(visualization_dir), "requested_k": visualization_k,
        "case_count": len(selected_indices), "test_indices": selected_indices,
        "case_ids": [test_start + index for index in selected_indices],
        "layout": "2 rows (INR-in, INR-out) x 3 columns (GT, INR reconstruction, absolute error)",
        "channel_handling": "One figure per channel pair; a scalar output is repeated for each input channel",
        "error_definition": "Absolute pointwise difference |INR reconstruction - observed field|",
        "field_units": "Dataset-loader units, including its training-data normalization",
        "output_reconstruction_source": "Observed test output fitted independently; not the mapper prediction",
        "rendering": "pixels" if x_test.ndim == 4 else "scatter on observed geometry",
        "cases": [],
    }
    test_offset = 0
    print(f"Evaluating saved checkpoints on {ntest} test samples", flush=True)
    for values, target, coordinates in test_loader:
        coordinates = coordinates.to(device)
        z_input, reconstruction_relative_l2, input_reconstruction = encode(
            inr_in, values.to(device), coordinates, latent_dim, alpha, inner_steps,
            return_relative_l2=True, return_reconstructions=True,
        )
        inr_in_relative_l2_sum += reconstruction_relative_l2 * len(values)
        with torch.no_grad():
            z_output = model((z_input - mean) / std)
            # The author uses mu_u=0 and sigma_u=1 for output codes.
            prediction = inr_out.modulated_forward(coordinates, z_output)
            target = target.to(device)
            errors_l2.append(batch_mse_rel_fn(prediction, target).cpu())
            errors_mse.append(batch_mse_fn(prediction, target).cpu())
        # Diagnostic only: fit the observed output from zero. This code is never
        # supplied to the mapper or used to compute the predictive field metrics.
        _, reconstruction_relative_l2, output_reconstruction = encode(
            inr_out, target, coordinates, int(inr_cfg.inr_out.latent_dim), alpha_out,
            inner_steps, return_relative_l2=True, return_reconstructions=True,
        )
        inr_out_relative_l2_sum += reconstruction_relative_l2 * len(values)
        for test_index in selected_indices:
            if test_offset <= test_index < test_offset + len(values):
                local_index = test_index - test_offset
                visualization["cases"].append(save_inr_error_case(
                    dataset_name, test_index, test_start + test_index,
                    values[local_index], input_reconstruction[local_index],
                    target[local_index], output_reconstruction[local_index],
                    coordinates[local_index], visualization_dir, suffix=suffix,
                ))
                print(f"visualized INR errors: test index {test_index}, case {test_start + test_index}", flush=True)
        test_offset += len(values)
    errors_l2 = torch.cat(errors_l2)
    errors_mse = torch.cat(errors_mse)
    if not torch.isfinite(errors_l2).all() or not torch.isfinite(errors_mse).all():
        raise ValueError("Non-finite test metrics")
    relative_l2 = errors_l2.mean().item()
    inr_in_relative_l2 = inr_in_relative_l2_sum / ntest
    inr_out_relative_l2 = inr_out_relative_l2_sum / ntest
    if not torch.isfinite(torch.tensor([inr_in_relative_l2, inr_out_relative_l2])).all():
        raise ValueError("Non-finite INR reconstruction metrics")
    summary = {
        "dataset": dataset_name,
        "mode": args.mode,
        "inr_checkpoint": str(inr_path),
        "regression_checkpoint": str(regression_path),
        "inr_epoch_zero_based": int(inr_checkpoint["epoch"]),
        "regression_epoch_zero_based": int(regression_checkpoint["epoch"]),
        "data_dir": str(data_dir),
        "normalization_ntrain": ntrain,
        "ntest": ntest,
        "test_indices": [test_start, test_start + ntest],
        "inner_steps": inner_steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "relative_l2": relative_l2,
        "relative_l2_percent": relative_l2 * 100,
        "mse": errors_mse.mean().item(),
        "inr_in_relative_l2": inr_in_relative_l2,
        "inr_out_relative_l2": inr_out_relative_l2,
        "inr_reconstruction_metric_definition": (
            "Mean over test cases of ||INR(coords, fitted_code)-observed_field||_2 / "
            "||observed_field||_2; codes fitted from zero for inner_steps. "
            "INR-out fits observed test outputs independently of the mapper."
        ),
        "metric_definition": "Mean of per-sample ||prediction-target||_2 / ||target||_2",
        "per_sample_relative_l2": errors_l2.tolist(),
        "per_sample_mse": errors_mse.tolist(),
        "visualization_error": visualization,
    }
    manifest_path = visualization_dir / f"manifest{suffix}.json"
    visualization["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps({
        "dataset": dataset_name, "mode": args.mode,
        "inr_checkpoint": str(inr_path), "regression_checkpoint": str(regression_path),
        "inr_epoch_zero_based": int(inr_checkpoint["epoch"]),
        "inner_steps": inner_steps, **visualization,
    }, indent=2, allow_nan=False) + "\n")
    result_path = absolute(args.output) if args.output else output_dir / f"test_metrics{suffix}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"test_relative_l2={relative_l2:.8f} ({relative_l2 * 100:.4f}%)")
    print(f"test_mse={summary['mse']:.8g}")
    print(f"test_inr_in_relative_l2={inr_in_relative_l2:.8f}")
    print(f"test_inr_out_relative_l2={inr_out_relative_l2:.8f}")
    print(f"metrics={result_path}")
    print(f"INR error visualizations={visualization_dir} ({len(selected_indices)} test cases)")
    return summary


def parse_args(dataset_name="airfoil"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--mode", choices=("author", "smoke"), default="author")
    parser.add_argument("--regression-checkpoint")
    parser.add_argument("--inr-checkpoint")
    parser.add_argument("--output-root")
    parser.add_argument("--data-dir")
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--visualization-k", type=int, default=10,
                        help="Number of evenly spaced test cases for INR error plots (default 10)")
    args = parser.parse_args()
    args.dataset = dataset_name
    return args


if __name__ == "__main__":
    evaluate(parse_args())
