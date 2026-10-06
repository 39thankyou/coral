"""Train the original ResNet processor on freshly adapted shared-INR codes."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from coral.losses import batch_mse_fn, batch_mse_rel_fn
from coral.mlp import ResNet
from coral.metalearning import outer_step
from coral.utils.data.load_data import get_operator_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from run_codelib.checkpoints import validate_checkpoint, validate_config


def absolute(path):
    path = Path(path).expanduser()
    base = Path(get_original_cwd()) if HydraConfig.initialized() else ROOT
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_inrs(checkpoint, device, coordinate_dim, input_channels, output_channels):
    """Restore exact decoder weights and detached adaptation step sizes."""
    validate_checkpoint(checkpoint)
    cfg = OmegaConf.create(OmegaConf.to_container(checkpoint["cfg"], resolve=True))
    if any(cfg[f"inr_{kind}"].model_type in ("anchormix", "siren_GridMix", "gridmix") for kind in ("in", "out")):
        cfg.network_package = "anchormix"
    result = {}
    for kind, suffix, channels in (
        ("in", "a", input_channels),
        ("out", "u", output_channels),
    ):
        cfg.inr = cfg[f"inr_{kind}"]
        if not cfg.inr.use_latent:
            raise ValueError("The processor requires a latent-enabled INR checkpoint")
        inr = create_inr_instance(cfg, coordinate_dim, channels, device)
        inr.load_state_dict(checkpoint[f"inr_{kind}"])
        inr.eval().requires_grad_(False)
        alpha = torch.as_tensor(checkpoint[f"alpha_{kind}"]).detach().to(device)
        if not torch.isfinite(alpha).all():
            raise ValueError("INR adaptation step sizes must be finite")
        result[kind] = {
            "model": inr,
            "alpha": alpha,
        }
    return result


def fit_latent(bundle, coordinates, features, steps, use_rel_loss=False):
    """Zero initialization and inner-loop loss match the original outer_step."""
    kwargs = {"is_train": False, "use_rel_loss": use_rel_loss,
              "modulations": features.new_zeros(len(features), bundle["model"].modulation_net.latent_dim)}
    adapt = outer_step
    if getattr(bundle["model"], "network_package", None) == "anchormix":
        from anchormix.metalearning import outer_step as adapt
    return adapt(bundle["model"], coordinates, features, steps, bundle["alpha"],
                      gradient_checkpointing=False, loss_type="mse", **kwargs)


def decode_field(bundle, coordinates, codes):
    return bundle["model"].modulated_forward(coordinates, codes)


def create_mapper(cfg, input_dim, output_dim):
    plain_class = ResNet
    if cfg.get("network_package") == "anchormix":
        from anchormix.mlp import ResNet as plain_class
    kwargs = dict(input_dim=input_dim, output_dim=output_dim, hidden_dim=int(cfg.width),
                  depth=int(cfg.depth), dropout=float(cfg.dropout))
    # design_regression.py uses ResNet's default Swish activation.
    return plain_class(**kwargs)


def encode_codes(inrs, data, steps, batch_size, device):
    """Fit every code from zero using the frozen representation."""
    x_train, y_train, x_test, y_test, grid_train, grid_test = data
    codes = {}
    for kind, train, test in (("in", x_train, x_test), ("out", y_train, y_test)):
        bundle = inrs[kind]
        for split, values, coordinates in (
            ("train", train, grid_train),
            ("test", test, grid_test),
        ):
            fitted = []
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                features = values[start:stop].to(device)
                output = fit_latent(
                    bundle,
                    coordinates[start:stop].to(device),
                    features,
                    steps,
                )
                z = output["modulations"].detach().cpu()
                if not torch.isfinite(z).all():
                    raise ValueError(f"Non-finite {kind}/{split} codes")
                fitted.append(z)
                if start // batch_size % 25 == 0 or stop == len(values):
                    print(f"encode {kind}/{split}: {stop}/{len(values)}", flush=True)
            codes[f"z_{kind}_{split}"] = torch.cat(fitted)
    return codes


def input_normalization(train_codes):
    if len(train_codes) < 2 or not torch.isfinite(train_codes).all():
        raise ValueError(
            "Normalization requires at least two finite training input codes"
        )
    mean, std = train_codes.mean(0), train_codes.std(0)
    # A constant feature stays zero after centering, without dividing by zero.
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def train_batch(model, optimizer, inputs, targets):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = (prediction - targets).square().mean()
    if not torch.isfinite(loss):
        raise ValueError("Non-finite regression training loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(
    model,
    inputs,
    target_codes,
    targets,
    coordinates,
    output_inr,
    batch_size,
    device,
):
    model.eval()
    code_mse, field_mse, relative_l2 = 0.0, 0.0, 0.0
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        prediction = model(inputs[start:stop].to(device))
        target_z = target_codes[start:stop].to(device)
        field = decode_field(
            output_inr,
            coordinates[start:stop].to(device),
            prediction,
        )
        target = targets[start:stop].to(device)
        code_mse += float((prediction - target_z).square().flatten(1).mean(1).sum())
        field_mse += float(batch_mse_fn(field, target).sum())
        relative_l2 += float(batch_mse_rel_fn(field, target).sum())
    metrics = {
        "code_test_mse": code_mse / len(inputs),
        "test_field_mse": field_mse / len(inputs),
        "test_relative_l2": relative_l2 / len(inputs),
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise ValueError("Non-finite regression evaluation metrics")
    return metrics


@hydra.main(
    version_base=None, config_path="config/static/", config_name="regression_shared.yaml"
)
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    validate_config(cfg)
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    if cfg.data.dataset_name not in ("airfoil", "elasticity", "pipe"):
        raise ValueError(
            "This regression entrypoint supports Airfoil, Elasticity and Pipe"
        )
    ntrain, ntest = int(cfg.data.ntrain), int(cfg.data.ntest)
    batch_size = int(cfg.optim.batch_size)
    batch_size_val = int(cfg.optim.batch_size_val or batch_size)
    encode_batch_size = int(cfg.inr.get("encode_batch_size", 4))
    epochs, steps = int(cfg.optim.epochs), int(cfg.inr.inner_steps)
    eval_every = int(cfg.optim.get("eval_every", 20))
    if (
        ntrain < 2
        or min(
            ntest,
            batch_size,
            batch_size_val,
            encode_batch_size,
            epochs,
            steps,
            eval_every,
        )
        < 1
    ):
        raise ValueError(
            "Counts/steps/batch sizes must be positive; ntrain must be >=2"
        )
    if cfg.model.model_type != "resnet":
        raise ValueError("model.model_type must be resnet")
    output_root = absolute(
        os.getenv("WANDB_DIR") or cfg.get("output_root", "run_codelib/outputs")
    )
    root = output_root / cfg.data.dataset_name
    for name in ("inr", "model", "modulations"):
        (root / name).mkdir(parents=True, exist_ok=True)
    explicit = cfg.inr.get("checkpoint")
    checkpoint_path = (
        absolute(explicit) if explicit else root / "inr" / f"{cfg.inr.run_name}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint)
    saved_cfg = checkpoint["cfg"]
    if saved_cfg.get("network_package") == "anchormix" or any(
            saved_cfg[f"inr_{kind}"].model_type in ("anchormix", "siren_GridMix", "gridmix") for kind in ("in", "out")):
        cfg.model.network_package = "anchormix"
        if ntrain != int(saved_cfg.data.ntrain) or ntest != int(saved_cfg.data.ntest):
            raise ValueError("Representation/processor must use the saved train/test split sizes (and identical elasticity mean grid)")
    if saved_cfg.data.dataset_name != cfg.data.dataset_name:
        raise ValueError("INR checkpoint belongs to a different dataset")
    if ntrain > int(saved_cfg.data.ntrain) or ntest > int(saved_cfg.data.ntest):
        raise ValueError("Requested cases exceed the saved INR train/test splits")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    cfg.inr.run_name = checkpoint_path.stem
    cfg.inr.checkpoint = str(checkpoint_path)
    cfg.data.dir = str(absolute(cfg.data.dir))
    set_seed(int(saved_cfg.data.seed))
    # Preserve each dataset's author split and full spatial grids.
    data = get_operator_data(
        cfg.data.dir, cfg.data.dataset_name, ntrain, ntest, sub_tr=1, sub_te=1, same_grid=True
    )
    inrs = load_inrs(
        checkpoint, device, data[4].shape[-1], data[0].shape[-1], data[1].shape[-1]
    )
    codes = encode_codes(inrs, data, steps, encode_batch_size, device)
    mu_a, sigma_a = input_normalization(codes["z_in_train"])
    train_inputs = (codes["z_in_train"] - mu_a) / sigma_a
    test_inputs = (codes["z_in_test"] - mu_a) / sigma_a
    # Output codes stay in the native output-INR latent space (mu_u=0, sigma_u=1).
    set_seed(int(cfg.data.seed))
    model = create_mapper(cfg.model, train_inputs.shape[1], codes["z_out_train"].shape[1]).to(device)

    loader = DataLoader(
        TensorDataset(train_inputs, codes["z_out_train"]),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.optim.gamma_step,
        patience=250,
        threshold=0.01,
        min_lr=min(1e-5, float(cfg.optim.lr)),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        id=cfg.wandb.id,
        dir=str(output_root),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = cfg.wandb.name or run.name or "airfoil-shared-regression"
    checkpoint_out = root / "model" / f"{run_name}.pt"
    cache_path = root / "modulations" / f"{run_name}_codes.pt"
    test_offset = 1000
    if cfg.data.dataset_name == "elasticity":
        import numpy as np
        sigma = np.load(Path(cfg.data.dir) / "Meshes" / "Random_UnitCell_sigma_10.npy", mmap_mode="r")
        test_offset = int(sigma.shape[1]) - ntest
    torch.save(
        {
            **codes,
            "mu_a": mu_a,
            "sigma_a": sigma_a,
            "inr_checkpoint": str(checkpoint_path),
            "inr_sha256": checkpoint_hash,
            "inner_steps": steps,
            "case_id_train": torch.arange(ntrain),
            "case_id_test": torch.arange(test_offset, test_offset + ntest),
            "representation": checkpoint.get("representation"),
        },
        cache_path,
    )
    history = root / "model" / f"{run_name}_metrics.jsonl"
    best_loss, best_metrics = float("inf"), None
    try:
        with history.open("w") as handle:
            for epoch in range(epochs):
                total = 0.0
                for inputs, targets in loader:
                    total += train_batch(
                        model,
                        optimizer,
                        inputs.to(device),
                        targets.to(device),
                    ) * len(inputs)
                train_mse = total / ntrain
                scheduler.step(train_mse)
                improved = train_mse < best_loss
                metrics = {
                    "epoch": epoch,
                    "code_train_mse": train_mse,
                }
                # A saved checkpoint always has freshly evaluated test metrics.
                if improved or epoch % eval_every == 0 or epoch == epochs - 1:
                    metrics.update(
                        evaluate(
                            model,
                            test_inputs,
                            codes["z_out_test"],
                            data[3],
                            data[5],
                            inrs["out"],
                            batch_size_val,
                            device,
                        )
                    )
                run.log(metrics, step=epoch)
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                handle.flush()
                print(
                    f"regression epoch={epoch + 1}/{epochs} train_code_mse={train_mse:.8g} "
                    + (
                        f" test_rel_l2={metrics['test_relative_l2']:.8g}"
                        if "test_relative_l2" in metrics
                        else ""
                    ),
                    flush=True,
                )
                if improved:
                    best_loss, best_metrics = train_mse, metrics.copy()
                    state = {
                        "cfg": cfg,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "loss": metrics["code_test_mse"],
                        "metrics": metrics,
                        "best_train_loss": best_loss,
                        "mu_a": mu_a,
                        "sigma_a": sigma_a,
                        "mu_u": torch.tensor([0.0]),
                        "sigma_u": torch.tensor([1.0]),
                        "inr_checkpoint": str(checkpoint_path),
                        "inr_sha256": checkpoint_hash,
                        "inr_cfg": saved_cfg,
                        "codes_file": str(cache_path),
                        "representation": checkpoint.get("representation"),
                    }
                    temporary = checkpoint_out.with_suffix(".tmp")
                    torch.save(state, temporary)
                    temporary.replace(checkpoint_out)
        summary = {
            "checkpoint": str(checkpoint_out),
            "best": best_metrics,
            "final": metrics,
            "inr_checkpoint": str(checkpoint_path),
            "inr_sha256": checkpoint_hash,
        }
        (root / "model" / f"{run_name}_metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
    finally:
        run.finish()
    return float(best_metrics["code_test_mse"])


if __name__ == "__main__":
    main()
