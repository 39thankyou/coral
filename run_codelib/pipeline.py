"""Unified shared-INR training entrypoint; delegates to the existing stage loops."""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import wandb
from omegaconf import OmegaConf

from run_codelib.checkpoints import validate_checkpoint, validate_config
from run_codelib.representation import apply_representation, configure_request, resume_config, suffix as representation_suffix

def absolute(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_checkpoint(source, dataset_dir, dataset_name="airfoil"):
    """Copy a complete INR snapshot, preserving the original and its provenance."""
    source = absolute(source)
    digest = sha256(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    temporal = "inr" in checkpoint and "inr_in" not in checkpoint
    required = {"cfg", "inr", "alpha"} if temporal else {"cfg", "inr_in", "inr_out", "alpha_in", "alpha_out"}
    validate_checkpoint(checkpoint)
    if not required.issubset(checkpoint):
        raise ValueError(f"INR checkpoint lacks required state: {sorted(required - checkpoint.keys())}")
    if checkpoint["cfg"].data.dataset_name != dataset_name:
        raise ValueError(f"Expected a {dataset_name} INR checkpoint")
    if temporal:
        from run_codelib.extended import validate_temporal_inr
        validate_temporal_inr(checkpoint, dataset_name)
    destination = dataset_dir / "inr" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) != digest:
        destination = destination.with_name(f"{source.stem}-{digest[:12]}.pt")
    if source != destination.resolve():
        temporary = destination.with_suffix(".importing")
        shutil.copy2(source, temporary)
        if sha256(temporary) != digest or sha256(source) != digest:
            temporary.unlink()
            raise RuntimeError("Source checkpoint changed during import; retry after its save completes")
        temporary.replace(destination)
    manifest = {"source": str(source), "checkpoint": str(destination), "sha256": digest,
                "epoch_zero_based": int(checkpoint.get("epoch", -1)),
                "ntrain": int(checkpoint["cfg"].data.ntrain), "ntest": int(checkpoint["cfg"].data.ntest)}
    # Repeated imports of the managed copy preserve the original source record.
    manifest_path = destination.with_suffix(".import.json")
    if source != destination.resolve() or not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    configs = dataset_dir / "resolved_configs"
    configs.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(checkpoint["cfg"], configs / f"{destination.stem}_imported_inr.yaml")
    print(f"imported_inr={destination} sha256={digest}", flush=True)
    return destination


def stage_config(config, stage, args, checkpoint=None):
    dataset_name = getattr(args, "dataset", "airfoil")
    dataset = config.datasets[dataset_name]
    prefix = "regression" if stage == "ode" else stage
    cfg = OmegaConf.load(absolute(dataset[f"{prefix}_base_config"]))
    for key, value in dataset[f"{prefix}_overrides"].items():
        OmegaConf.update(cfg, key, value, force_add=True)
    smoke = args.mode == "smoke"
    suffix = "-smoke" if smoke else ""
    cfg.data.dir = str(absolute(dataset.data_dir))
    if getattr(args, "data_dir", None):
        cfg.data.dir = str(absolute(args.data_dir))
    cfg.data.dataset_name = dataset.get("author_name", dataset_name)
    cfg.data.ntrain = args.ntrain if args.ntrain is not None else int(config.common.smoke_ntrain if smoke else dataset.ntrain)
    cfg.data.ntest = args.ntest if args.ntest is not None else int(config.common.smoke_ntest if smoke else dataset.ntest)
    cfg.optim.epochs = args.epochs if args.epochs is not None else int(config.common.smoke_epochs if smoke else cfg.optim.epochs)
    if args.batch_size is not None:
        cfg.optim.batch_size = args.batch_size
    if smoke and args.batch_size is None:
        cfg.optim.batch_size = min(int(cfg.optim.batch_size), 4)
    if args.device:
        cfg.device = args.device
    cfg.wandb.name = dataset[f"{prefix}_run_name"] + representation_suffix(args) + suffix
    cfg.wandb.id = None
    cfg.wandb.dir = None
    cfg.wandb.entity = os.getenv("WANDB_ENTITY", cfg.wandb.entity)
    cfg.wandb.project = os.getenv("WANDB_PROJECT", cfg.wandb.project)
    if stage in ("regression", "ode"):
        cfg.wandb.name = dataset.regression_run_name + representation_suffix(args) + suffix
        cfg.inr.checkpoint = str(checkpoint)
        cfg.inr.run_name = checkpoint.stem
        if args.encode_batch_size is not None:
            cfg.inr.encode_batch_size = args.encode_batch_size
        if smoke:
            cfg.optim.eval_every = 1
    elif smoke:
        # Smoke still uses full spatial grids and the requested architecture.
        cfg.optim.visualize_every = 0
    apply_representation(cfg, config, args, stage)
    validate_config(cfg)
    return cfg


def run_pipeline(args):
    config = OmegaConf.load(absolute(args.config))
    validate_config(config)
    args.mode = args.mode or config.mode
    if args.mode not in ("full", "smoke"):
        raise ValueError("mode must be full or smoke")
    for key in ("epochs", "ntrain", "ntest", "batch_size", "encode_batch_size"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    output_root = absolute(args.output_root or config.common.output_root)
    dataset_name = getattr(args, "dataset", "airfoil")
    dataset = config.datasets[dataset_name]
    if getattr(args, "resume", None):
        args.resume = str(absolute(args.resume))
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint(resume_state)
        saved_resume = resume_state["cfg"]
        from run_codelib.representation import TYPES, validate_overrides
        validate_overrides(args, saved_resume)
        if getattr(args, "representation_branch", None) is None:
            args.representation_branch = saved_resume.get("representation_branch", "out")
        if getattr(args, "representation", None) is None:
            kind = "out" if args.representation_branch == "both" else args.representation_branch
            reverse_types = {value: key for key, value in TYPES.items()}
            args.representation = reverse_types.get(saved_resume[f"inr_{kind}"].model_type)
    configure_request(config, dataset, args)
    family = dataset.get("family", "static")
    downstream_stage = "ode" if family == "dynamics" else "regression"
    if args.stage not in ("import", "inr", downstream_stage, "all"):
        raise ValueError(f"{dataset_name} supports inr, {downstream_stage}, all, import")
    if getattr(args, "resume", None) and args.stage not in ("inr", "all"):
        raise ValueError("--resume resumes the INR stage; use --stage inr/all")
    dataset_dir = output_root / dataset.get("author_name", dataset_name)
    for name in ("inr", "model", "modulations", "resolved_configs", "visualization"):
        (dataset_dir / name).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(output_root)
    os.environ["WANDB_MODE"] = args.wandb_mode or str(config.common.wandb_mode)
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset_name = getattr(args, "dataset", "airfoil")
    dataset = config.datasets[dataset_name]
    suffix = "-smoke" if args.mode == "smoke" else ""
    checkpoint = None
    if args.stage in ("import", "regression", "ode"):
        inr_name = dataset.inr_run_name
        local_inr = dataset_dir / "inr" / f"{inr_name}{representation_suffix(args)}{suffix}.pt"
        source = args.inr_checkpoint or (local_inr if local_inr.exists() else None if args.representation else
                                          dataset.pretrained_inr)
        if not source:
            raise ValueError("No matching INR checkpoint; specify --inr-checkpoint or train it with --stage inr/all")
        checkpoint = import_checkpoint(source, dataset_dir, dataset_name=dataset.get("author_name", dataset_name))
        saved_inr_cfg = torch.load(checkpoint, map_location="cpu", weights_only=False)["cfg"]
        if args.representation is None and saved_inr_cfg.get("network_package") == "anchormix":
            from run_codelib.representation import TYPES
            args.representation_branch = saved_inr_cfg.get("representation_branch", "out")
            kind = "out" if args.representation_branch == "both" else args.representation_branch
            args.representation = {value: key for key, value in TYPES.items()}.get(saved_inr_cfg[f"inr_{kind}"].model_type)
        if args.representation:
            from run_codelib.representation import validate_overrides
            validate_overrides(args, saved_inr_cfg)
    elif args.inr_checkpoint:
        raise ValueError("Use --stage regression/import to reuse an INR; inr/all trains a fresh INR")
    stages = ["inr", downstream_stage] if args.stage == "all" else ([] if args.stage == "import" else [args.stage])
    results = {}
    for stage in stages:
        cfg = stage_config(config, stage, args, checkpoint)
        if stage == "inr":
            cfg = resume_config(args, cfg)
        elif getattr(args, "resume", None):
            saved_inr_cfg = torch.load(checkpoint, map_location="cpu", weights_only=False)["cfg"]
            cfg.data.ntrain, cfg.data.ntest = saved_inr_cfg.data.ntrain, saved_inr_cfg.data.ntest
        config_path = dataset_dir / "resolved_configs" / f"{stage}{representation_suffix(args)}{suffix}.yaml"
        OmegaConf.save(cfg, config_path)
        # These imports resolve to this repository via run_airfoil.py's sys.path.
        module_name = "static.design_inr_shared" if stage == "inr" else "static.design_regression_shared"
        module = importlib.import_module(module_name) if family == "static" else None
        print(f"stage={stage} mode={args.mode} epochs={cfg.optim.epochs} "
              f"ntrain={cfg.data.ntrain} ntest={cfg.data.ntest}", flush=True)
        started = time.monotonic()
        try:
            if family == "static":
                result = module.main.__wrapped__(cfg)
            else:
                from run_codelib.extended import run_stage
                result = run_stage(cfg, stage, family)
        finally:
            wandb.finish()
        if family != "static":
            OmegaConf.save(cfg, config_path)
        if stage == "inr":
            checkpoint = dataset_dir / "inr" / f"{cfg.wandb.name}.pt"
        results[stage] = {"loss": float(result), "seconds": time.monotonic() - started,
                          "config": str(config_path)}
    summary = {"status": "ok", "method": args.representation or "siren",
               "dataset": dataset_name, "mode": args.mode,
               "stage": args.stage, "inr_checkpoint": str(checkpoint), "results": results}
    path = dataset_dir / f"{args.stage}_metrics{representation_suffix(args)}{suffix}.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"metrics={path}", flush=True)
    return summary


def main(dataset_name="airfoil", default_stage="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--mode", choices=("full", "smoke"))
    parser.add_argument("--stage", choices=("import", "inr", "regression", "ode", "all"), default=default_stage)
    parser.add_argument("--inr-checkpoint")
    parser.add_argument("--resume", help="Resume an INR checkpoint; --epochs is the total epoch budget")
    parser.add_argument("--data-dir", help="Override the dataset directory")
    parser.add_argument("--dataset", choices=("airfoil", "elasticity", "pipe", "cylinder_flow", "airfoil_flow", "navier_stokes", "shallow_water"), default=dataset_name)
    from anchormix.config import add_arguments
    add_arguments(parser)
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--wandb-mode", choices=("offline", "disabled", "online"))
    for option in ("epochs", "ntrain", "ntest", "batch-size", "encode-batch-size"):
        parser.add_argument(f"--{option}", type=int)
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    main(default_stage="all")
