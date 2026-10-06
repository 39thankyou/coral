"""Additional representation/RNG metadata in the existing checkpoint schema."""
import random

import numpy as np
import torch

from .factory import representation_metadata


def training_metadata(models, cfg):
    return {"representation": representation_metadata(models, cfg),
            "rng_state": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                          "python": random.getstate(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def restore_training_state(checkpoint, alphas, optimizers):
    for kind, alpha, optimizer in zip(("in", "out"), alphas, optimizers):
        with torch.no_grad():
            alpha.copy_(checkpoint[f"alpha_{kind}"])
        optimizer.load_state_dict(checkpoint[f"optimizer_inr_{kind}"])
    if "rng_state" not in checkpoint:
        raise ValueError("Resume requires saved RNG state; legacy checkpoints support evaluate/import instead")
    state = checkpoint["rng_state"]
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_train_loss"])
