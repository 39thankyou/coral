"""Reject retired retrieval models before creating any replacement network."""
from collections.abc import Mapping

from omegaconf import OmegaConf


def validate_checkpoint(checkpoint):
    cfg = checkpoint["cfg"]
    for section in ("inr", "inr_in", "inr_out", "model"):
        settings = cfg.get(section, {})
        if settings.get("model_type") in ("siren_codelib", "resnet_lib"):
            raise ValueError("Codelib retrieval models have been removed; retrain the representation and processor")
    for section in ("inr", "inr_in", "inr_out", "model"):
        state = checkpoint.get(section, {})
        if isinstance(state, Mapping) and any(
                part in ("retrieve", "memory_gate") for key in state for part in key.split(".")):
            raise ValueError("Checkpoint contains removed Codelib retrieval weights; retraining is required")


def validate_config(config):
    """Fail on obsolete YAML switches instead of silently ignoring an old experiment."""
    values = OmegaConf.to_container(config, resolve=False) if OmegaConf.is_config(config) else config
    removed = {"use_codelib", "inr_lib", "model_lib", "top_k", "num_head", "exclude_self"}

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key.split(".")[-1] in removed:
                    raise ValueError(f"Removed Codelib configuration option: {path}")
                if key.split(".")[-1] == "model_type" and item in ("siren_codelib", "resnet_lib"):
                    raise ValueError(f"Removed Codelib model type at {path}; choose siren, anchormix or gridmix")
                walk(item, path)
        elif isinstance(value, list):
            for item in value:
                walk(item, prefix)
    walk(values)
