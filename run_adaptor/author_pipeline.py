"""Thin launcher around CORAL's original, separate training entrypoints.

No training loop lives here. Each stage calls the undecorated function behind
the author's Hydra entrypoint with a resolved author config and bash overrides.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import threading
import copy
import time
from pathlib import Path

import torch
import wandb
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_adaptor.raw_cylinder_dataset import bind_raw_cylinder_dataset, bind_raw_airfoil_dataset
from run_adaptor.visualize_data import visualize_navier_stokes, visualize_static


AUTHOR_MODULES = {
    "airfoil": ("static.design_inr", "static.design_regression"),
    "pipe": ("static.design_inr", "static.design_regression"),
    "airfoil_flow": ("static.static_inr", "static.static_regression"),
    "shallow_water": ("inr.inr", "dynamics_modeling.train"),
    "elasticity": ("static.design_inr", "static.design_regression"),
    "cylinder_flow": ("static.static_inr", "static.static_regression"),
    "navier_stokes": ("inr.inr", "dynamics_modeling.train"),
}


def _absolute(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _set_reproducible_runtime(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _install_pytorch_compatibility():
    """Bridge removed logging-only arguments used by the 2023 code."""

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    if "verbose" not in inspect.signature(scheduler).parameters:
        def compatible_reduce_on_plateau(*args, **kwargs):
            kwargs.pop("verbose", None)
            return scheduler(*args, **kwargs)

        torch.optim.lr_scheduler.ReduceLROnPlateau = compatible_reduce_on_plateau


def _apply_overrides(cfg, overrides):
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)


def _apply_wandb_config(cfg, adaptor_cfg, run_name, mode, cli=None):
    """Merge the adaptor's wandb section into the resolved author config.

    Precedence (highest first): ``WANDB_ENTITY`` / ``WANDB_PROJECT`` env vars,
    CLI flags, the adaptor's ``wandb`` section, then the author base config (or
    an auto-generated run name for ``name``). Defaults preserve the existing
    logging project and account.
    """
    cli = cli or {}
    wandb_cfg = adaptor_cfg.get("wandb", {})
    cfg.wandb.entity = (
        os.environ.get("WANDB_ENTITY")
        or cli.get("entity")
        or wandb_cfg.get("entity")
        or cfg.wandb.entity
    )
    cfg.wandb.project = (
        os.environ.get("WANDB_PROJECT")
        or cli.get("project")
        or wandb_cfg.get("project")
        or cfg.wandb.project
    )
    cfg.wandb.name = cli.get("name") or wandb_cfg.get("name") or (
        run_name if mode == "author" else f"{run_name}-smoke"
    )
    cfg.wandb.id = cli.get("id") or wandb_cfg.get("id") or None
    cfg.wandb.dir = cli.get("dir") or wandb_cfg.get("dir") or None
    cfg.wandb.sweep_id = cli.get("sweep_id") or wandb_cfg.get("sweep_id") or None


def _build_stage_config(
    adaptor_cfg: DictConfig,
    dataset_key: str,
    stage: str,
    mode: str,
    epochs: int | None,
    ntrain: int | None,
    ntest: int | None,
    wandb_cli: dict | None = None,
):
    dataset_cfg = adaptor_cfg.datasets[dataset_key]
    downstream = stage in {"regression", "ode"}
    base_key = "downstream_base_config" if downstream else "inr_base_config"
    override_key = "downstream_overrides" if downstream else "inr_overrides"
    run_key = "downstream_run_name" if downstream else "inr_run_name"
    cfg = OmegaConf.load(_absolute(dataset_cfg[base_key]))
    _apply_overrides(cfg, dataset_cfg[override_key])

    if mode == "smoke":
        resolved_ntrain = adaptor_cfg.common.smoke_ntrain
        resolved_ntest = adaptor_cfg.common.smoke_ntest
        resolved_epochs = adaptor_cfg.common.smoke_epochs
    else:
        resolved_ntrain = dataset_cfg.author_ntrain
        resolved_ntest = dataset_cfg.author_ntest
        resolved_epochs = cfg.optim.epochs

    cfg.data.dir = str(_absolute(dataset_cfg.data_dir))
    cfg.data.dataset_name = dataset_cfg.author_name
    cfg.data.ntrain = int(ntrain if ntrain is not None else resolved_ntrain)
    cfg.data.ntest = int(ntest if ntest is not None else resolved_ntest)
    cfg.optim.epochs = int(epochs if epochs is not None else resolved_epochs)
    run_name = dataset_cfg[run_key]
    _apply_wandb_config(cfg, adaptor_cfg, run_name, mode, wandb_cli)
    if downstream:
        cfg.inr.run_name = (
            dataset_cfg.inr_run_name
            if mode == "author"
            else f"{dataset_cfg.inr_run_name}-smoke"
        )
    if dataset_key == "shallow_water" and downstream:
        cfg.inr.run_name = None
        suffix = "-smoke" if mode == "smoke" else ""
        cfg.inr.run_dict = {channel: f"{dataset_cfg.inr_run_name}-{channel}{suffix}"
                            for channel in ("height", "vorticity")}
    return cfg


def _prepare_runtime(adaptor_cfg):
    output_root = _absolute(adaptor_cfg.common.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(output_root)
    os.environ.setdefault("WANDB_MODE", "online")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    _install_pytorch_compatibility()
    return output_root


def _bind_cylinder(module, cfg):
    module.CylinderFlowDataset = bind_raw_cylinder_dataset(
        cfg.data.dir, int(cfg.data.ntrain), int(cfg.data.ntest)
    )


def _run_author_stage(dataset_key, stage, cfg):
    module_name = AUTHOR_MODULES[dataset_key][0 if stage == "inr" else 1]
    module = importlib.import_module(module_name)
    if dataset_key == "cylinder_flow":
        _bind_cylinder(module, cfg)
    elif dataset_key == "airfoil_flow":
        module.AirfoilFlowDataset = bind_raw_airfoil_dataset(cfg.data.dir, int(cfg.data.ntrain), int(cfg.data.ntest))
    _set_reproducible_runtime(int(cfg.data.seed))
    print(
        f"author_entrypoint={module.__file__} stage={stage} "
        f"dataset={cfg.data.dataset_name} epochs={cfg.optim.epochs} "
        f"ntrain={cfg.data.ntrain} ntest={cfg.data.ntest}",
        flush=True,
    )
    started_at = time.monotonic()
    stop_heartbeat = threading.Event()
    from wandb.sdk.wandb_run import Run

    original_run_log = Run.log
    original_wandb_init = wandb.init

    def init_with_requested_name(*args, **kwargs):
        run = original_wandb_init(*args, **kwargs)
        if kwargs.get("name") and os.environ.get("WANDB_MODE") == "disabled":
            run.name = kwargs["name"]
        return run

    wandb.init = init_with_requested_name
    completed_epochs = 0
    last_epoch_report = 0.0
    progress_keys = {
        "inr": ("train_loss",) if dataset_key in ("navier_stokes", "shallow_water") else ("train_loss_in",),
        "ode": ("code_train_mse", "code_train_inter_mse"),
        "regression": (
            ("code_train_mse",)
            if dataset_key not in ("cylinder_flow", "airfoil_flow")
            else ("code_train_loss",)
        ),
    }

    def run_log_with_epoch_progress(run, data, *args, **kwargs):
        nonlocal completed_epochs, last_epoch_report
        result = original_run_log(run, data, *args, **kwargs)
        signals = progress_keys.get(stage, ())
        if any(signal in data for signal in signals):
            completed_epochs += 1
            now = time.monotonic()
            total_epochs = int(cfg.optim.epochs)
            if (
                completed_epochs == 1
                or completed_epochs == total_epochs
                or now - last_epoch_report >= 30.0
            ):
                print(
                    f"author_epoch_progress stage={stage} "
                    f"epoch={completed_epochs}/{total_epochs} "
                    f"elapsed_seconds={now - started_at:.1f}",
                    flush=True,
                )
                last_epoch_report = now
        return result

    Run.log = run_log_with_epoch_progress

    def report_stage_running():
        while not stop_heartbeat.wait(30.0):
            if completed_epochs == 0:
                print(
                    f"author_stage_preparing stage={stage} "
                    f"elapsed_seconds={time.monotonic() - started_at:.1f}",
                    flush=True,
                )

    heartbeat = threading.Thread(target=report_stage_running, daemon=True)
    heartbeat.start()
    try:
        result = module.main.__wrapped__(cfg)
    finally:
        Run.log = original_run_log
        wandb.init = original_wandb_init
        stop_heartbeat.set()
        heartbeat.join(timeout=1.0)
        wandb.finish()
    print(
        f"author_stage_complete stage={stage} "
        f"elapsed_seconds={time.monotonic() - started_at:.2f}",
        flush=True,
    )
    return float(result) if isinstance(result, (float, int)) else float(result.item())


def _save_resolved_config(output_root, author_name, stage, cfg):
    config_dir = output_root / author_name / "resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{stage}.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def run_pipeline(
    dataset_key,
    config_path,
    mode=None,
    stage="all",
    epochs=None,
    ntrain=None,
    ntest=None,
    visualization_steps=None,
    wandb_cli=None,
):
    adaptor_cfg = OmegaConf.load(_absolute(config_path))
    mode = mode or adaptor_cfg.mode
    if mode not in {"smoke", "author"}:
        raise ValueError("mode must be 'smoke' or 'author'")
    output_root = _prepare_runtime(adaptor_cfg)
    dataset_cfg = adaptor_cfg.datasets[dataset_key]

    probe_cfg = _build_stage_config(
        adaptor_cfg, dataset_key, "inr", mode, epochs, ntrain, ntest, wandb_cli
    )
    visualization_cfg = OmegaConf.create(
        OmegaConf.to_container(dataset_cfg, resolve=True)
    )
    visualization_cfg.data_dir = probe_cfg.data.dir
    visualization_dir = output_root / dataset_cfg.author_name / "visualization"
    if dataset_key in ("navier_stokes", "shallow_water"):
        visualization = visualize_navier_stokes(
            visualization_cfg,
            int(probe_cfg.data.ntrain),
            int(probe_cfg.data.ntest),
            int(visualization_steps or adaptor_cfg.common.visualization_steps),
            visualization_dir,
        )
        downstream_stage = "ode"
    else:
        visualization = visualize_static(
            dataset_key,
            visualization_cfg,
            int(probe_cfg.data.ntrain),
            visualization_dir,
        )
        downstream_stage = "regression"

    requested_stages = (
        ["inr", downstream_stage]
        if stage == "all"
        else [stage]
    )
    results = {}
    resolved_configs = {}
    for current_stage in requested_stages:
        if current_stage not in {"inr", downstream_stage}:
            raise ValueError(
                f"Dataset {dataset_key} supports stages: inr, {downstream_stage}, all"
            )
        cfg = _build_stage_config(
            adaptor_cfg,
            dataset_key,
            current_stage,
            mode,
            epochs,
            ntrain,
            ntest,
            wandb_cli,
        )
        resolved_configs[current_stage] = _save_resolved_config(
            output_root, dataset_cfg.author_name, current_stage, cfg
        )
        if dataset_key == "shallow_water" and current_stage == "inr":
            for channel in ("height", "vorticity"):
                channel_cfg = copy.deepcopy(cfg)
                channel_cfg.data.data_to_encode = channel
                suffix = "-smoke" if mode == "smoke" else ""
                channel_cfg.wandb.name = f"{dataset_cfg.inr_run_name}-{channel}{suffix}"
                resolved_configs[f"inr_{channel}"] = _save_resolved_config(output_root, dataset_cfg.author_name, f"inr_{channel}", channel_cfg)
                results[f"inr_{channel}"] = _run_author_stage(dataset_key, current_stage, channel_cfg)
        else:
            results[current_stage] = _run_author_stage(dataset_key, current_stage, cfg)

    summary = {
        "status": "ok",
        "alignment": "author entrypoints and stage separation",
        "dataset": dataset_key,
        "author_dataset_name": dataset_cfg.author_name,
        "mode": mode,
        "stages": requested_stages,
        "stage_status": {name: "ok" for name in requested_stages},
        "author_return_values": results,
        "resolved_configs": resolved_configs,
        "visualization": visualization,
    }
    summary_path = output_root / dataset_cfg.author_name / "metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"status=ok metrics={summary_path}", flush=True)
    return summary


def parse_args(default_dataset=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=tuple(AUTHOR_MODULES),
        default=default_dataset,
        required=default_dataset is None,
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config.yaml"))
    )
    parser.add_argument("--mode", choices=("smoke", "author"), default=None)
    parser.add_argument("--stage", choices=("inr", "regression", "ode", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--visualization-steps", type=int, default=None)
    parser.add_argument(
        "--wandb-entity", default="39thankyou-",
        help="WandB entity (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb-project", default="latent_lib",
        help="WandB project (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb-name", default="",
        help="run name; empty uses the auto-generated dataset/stage name",
    )
    parser.add_argument(
        "--wandb-id", default="",
        help="run id; empty lets WandB assign one",
    )
    parser.add_argument(
        "--wandb-dir", default="",
        help="extra run dir; empty disables",
    )
    parser.add_argument(
        "--wandb-sweep-id", default="",
        help="sweep id; empty disables",
    )
    return parser.parse_args()


def main(default_dataset=None):
    args = parse_args(default_dataset)
    wandb_cli = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": args.wandb_name,
        "id": args.wandb_id,
        "dir": args.wandb_dir,
        "sweep_id": args.wandb_sweep_id,
    }
    return run_pipeline(
        dataset_key=args.dataset,
        config_path=args.config,
        mode=args.mode,
        stage=args.stage,
        epochs=args.epochs,
        ntrain=args.ntrain,
        ntest=args.ntest,
        visualization_steps=args.visualization_steps,
        wandb_cli=wandb_cli,
    )


if __name__ == "__main__":
    main()
