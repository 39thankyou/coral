"""Representation configuration glue; training loops remain in static/."""
from pathlib import Path

import torch
from omegaconf import OmegaConf

from anchormix.config import AnchorMixConfig, DEFAULTS, explicit_options


TYPES = {"anchormix": "anchormix", "gridmix": "siren_GridMix", "siren": "siren"}


def configure_request(config, dataset, args):
    explicit_rep = getattr(args, "representation", None)
    args.representation = explicit_rep or dataset.get("representation", config.get("representation"))
    if args.representation is None and explicit_options(args):
        args.representation = "anchormix"
    args.representation_branch = (getattr(args, "representation_branch", None)
                                  or dataset.get("representation_branch", config.get("representation_branch", "out")))
    if args.representation not in (None, *TYPES):
        raise ValueError(f"Unknown representation: {args.representation}")
    if args.representation_branch not in ("in", "out", "both"):
        raise ValueError("representation_branch must be in, out or both")
    if args.representation is not None and getattr(args, "dataset", "airfoil") not in ("airfoil", "elasticity"):
        raise ValueError("The first AnchorMix integration supports airfoil and elasticity")


def suffix(args):
    return f"-{args.representation}-{args.representation_branch}" if getattr(args, "representation", None) else ""


def selected_branches(args):
    branch = getattr(args, "representation_branch", None) or "out"
    return ("in", "out") if branch == "both" else (branch,)


def apply_representation(cfg, config, args, stage):
    if not getattr(args, "representation", None):
        return
    cfg.network_package = "anchormix"
    cfg.model.network_package = "anchormix"
    cfg.representation_branch = args.representation_branch
    if stage != "inr":
        return
    for kind in selected_branches(args):
        settings = cfg[f"inr_{kind}"]
        settings.model_type = TYPES[args.representation]
        settings.grid_base = (getattr(args, "grid_base", None) if getattr(args, "grid_base", None) is not None
                              else settings.get("grid_base", 64))
        if isinstance(settings.grid_base, bool) or not isinstance(settings.grid_base, int) or settings.grid_base < 1:
            raise ValueError("grid_base must be a positive integer")
        if args.representation == "anchormix":
            options = {**DEFAULTS, **dict(config.get("anchormix", {})),
                       **dict(config.datasets[args.dataset].get("anchormix", {})),
                       **dict(settings.get("anchormix", {})), **explicit_options(args)}
            AnchorMixConfig(**options).validate()
            if options["anchor_path"]:
                from run_codelib.pipeline import absolute
                options["anchor_path"] = str(absolute(options["anchor_path"]))
            settings.anchormix = options


def validate_overrides(args, saved_cfg):
    """Evaluation/resume structure comes from the checkpoint, never current YAML."""
    branch = getattr(args, "representation_branch", None) or saved_cfg.get("representation_branch", "out")
    kinds = ("in", "out") if branch == "both" else (branch,)
    options = explicit_options(args)
    options.pop("query_chunk_size", None)
    for kind in kinds:
        settings = saved_cfg[f"inr_{kind}"]
        representation = getattr(args, "representation", None)
        if representation is not None and TYPES[representation] != settings.model_type:
            raise ValueError(f"Incompatible CLI representation for {kind}: checkpoint has {settings.model_type}")
        grid_base = getattr(args, "grid_base", None)
        if grid_base is not None and grid_base != settings.get("grid_base", 64):
            raise ValueError(f"Incompatible CLI grid_base for {kind}")
        if options and settings.model_type != "anchormix":
            raise ValueError(f"AnchorMix options supplied for {kind}, a {settings.model_type} checkpoint")
        stored = {**DEFAULTS, **dict(settings.get("anchormix", {}))}
        for key, value in options.items():
            if key == "anchor_path" and value is not None and stored[key] is not None:
                from run_codelib.pipeline import absolute
                value, expected = str(absolute(value)), str(absolute(stored[key]))
            else:
                expected = stored[key]
            if value != expected:
                raise ValueError(f"Incompatible CLI {key}={value!r}; checkpoint {kind} has {expected!r}")


def resume_config(args, current):
    path = getattr(args, "resume", None)
    if not path:
        return current
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from run_codelib.checkpoints import validate_checkpoint
    validate_checkpoint(checkpoint)
    cfg = OmegaConf.create(OmegaConf.to_container(checkpoint["cfg"], resolve=True))
    if cfg.get("network_package") != "anchormix":
        raise ValueError("Resume is supported for the independent representation checkpoints")
    validate_overrides(args, cfg)
    if cfg.data.dataset_name != current.data.dataset_name:
        raise ValueError("Resume checkpoint belongs to a different dataset")
    for key in ("ntrain", "ntest"):
        if getattr(args, key, None) is not None and getattr(args, key) != cfg.data[key]:
            raise ValueError(f"Cannot change {key} when resuming")
    for key in ("epochs", "batch_size"):
        if getattr(args, key, None) is not None:
            cfg.optim[key] = getattr(args, key)
    if getattr(args, "device", None):
        cfg.device = args.device
    if getattr(args, "data_dir", None):
        cfg.data.dir = current.data.dir
    chunk = getattr(args, "query_chunk_size", None)
    if chunk is not None:
        for kind in ("in", "out"):
            if cfg[f"inr_{kind}"].model_type == "anchormix":
                cfg[f"inr_{kind}"].anchormix.query_chunk_size = chunk
    cfg.resume = str(Path(path).resolve())
    return cfg
