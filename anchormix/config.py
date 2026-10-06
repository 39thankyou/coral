"""AnchorMix options shared by model construction and the existing CLI."""
import argparse
import math
from dataclasses import asdict, dataclass


@dataclass
class AnchorMixConfig:
    num_anchors: int = 64
    anchor_init: str = "fps"
    anchor_path: str = None
    learnable_pos: bool = False
    pos_lr_scale: float = 0.1
    init_jitter: float = 0.0
    pos_constraint: str = "none"
    read_mode: str = "knn"
    num_neighbors: int = 8
    weight_mode: str = "distance"
    kernel_sigma: float = 0.2
    attention_power: float = 1.0
    gate_hidden_dim: int = 64
    fusion: str = "anchor"
    fusion_gate_hidden_dim: int = 64
    fusion_gate_init: float = 0.05
    global_alpha_init: float = 0.1
    learnable_global_alpha: bool = True
    query_chunk_size: int = 4096

    def validate(self):
        for key in ("num_anchors", "num_neighbors", "gate_hidden_dim", "fusion_gate_hidden_dim", "query_chunk_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key, choices in CHOICES.items():
            if getattr(self, key) not in choices:
                raise ValueError(f"{key} must be one of {choices}")
        for key in ("learnable_pos", "learnable_global_alpha"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a boolean")
        for key in ("kernel_sigma", "attention_power", "pos_lr_scale", "init_jitter", "global_alpha_init", "fusion_gate_init"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if isinstance(self.attention_power, bool) or self.attention_power <= 0:
            raise ValueError("attention_power must be >0 (1 preserves the original weights)")
        if isinstance(self.fusion_gate_init, bool) or not 0 < self.fusion_gate_init < 1:
            raise ValueError("fusion_gate_init must be strictly between 0 and 1")
        if self.kernel_sigma <= 0 or self.pos_lr_scale < 0 or self.init_jitter < 0:
            raise ValueError("kernel_sigma must be >0; pos_lr_scale and init_jitter must be >=0")
        if self.read_mode == "knn" and self.num_neighbors > self.num_anchors:
            raise ValueError("num_neighbors must not exceed num_anchors in knn mode")
        return self


CHOICES = {
    "anchor_init": ("fps", "zero", "file"),
    "pos_constraint": ("none", "bbox"),
    "read_mode": ("knn", "all"),
    "weight_mode": ("distance", "distance_gate"),
    "fusion": ("anchor", "global", "hybrid", "gated"),
}
DEFAULTS = asdict(AnchorMixConfig())


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("expected true or false")


def add_arguments(parser):
    group = parser.add_argument_group("AnchorMix (explicit CLI > config > defaults)")
    help_text = {
        "kernel_sigma": "Gaussian sigma in existing model coordinate units: NACA [0,1] query grid; elasticity training-mean geometry. No extra normalization.",
        "attention_power": "Normalize attention weights raised to this positive power (>1 sharpens, <1 softens). Applies to knn/all; global fusion ignores it. Stable equivalent: softmax(power * scores). For distance only, effective sigma = kernel_sigma / sqrt(power).",
        "init_jitter": "Optional Gaussian position jitter std in the same model coordinate units; zero means no perturbation.",
        "anchor_init": "Initialize once from training coordinates (fps), coincident zeros (zero), or anchor_path (file).",
        "anchor_path": "NPY or tensor-only PT file of [K,d] positions in model coordinates; only read for fresh initialization.",
        "pos_constraint": "bbox projects after the OUTER optimizer step into the saved training bounds.",
        "read_mode": "knn selects nearest q anchors; all reads every anchor (still spatial, unlike fusion=global).",
        "fusion": "anchor: spatial only; global: broadcast latent modulation only; hybrid: anchor + alpha * global; gated: (1-g) * global + g * anchor, with a per-layer/channel sigmoid MLP.",
        "fusion_gate_hidden_dim": "Hidden width of each fusion=gated MLP; input is concatenated global and anchor modulation. Independent of the spatial distance_gate MLP.",
        "fusion_gate_init": "Initial anchor fraction for fusion=gated, strictly between 0 and 1; sigmoid bias=logit(value), with small nonzero final weights. global_alpha is unused in gated mode.",
        "query_chunk_size": "Maximum query points per distance/decoder chunk, per batch item.",
        "pos_lr_scale": "Position learning rate multiplier relative to lr_inr.",
    }
    for key, default in DEFAULTS.items():
        value_type = boolean if isinstance(default, bool) else str if default is None else type(default)
        flags = ["--" + key.replace("_", "-"), "--" + key]
        group.add_argument(*dict.fromkeys(flags), dest=key, type=value_type, default=None,
                           choices=CHOICES.get(key),
                           help=help_text.get(key, key.replace("_", " ")) + f" (default: {default})")
    group.add_argument("--representation", choices=("anchormix", "siren", "gridmix"),
                       help="Representation for selected branches; omitted preserves existing config")
    group.add_argument("--representation-branch", choices=("in", "out", "both"),
                       help="Branch to replace (default: out); other branch keeps its configured model")
    group.add_argument("--grid-base", "--num-bases", dest="grid_base", type=int,
                       help="M, inherited from inr branch grid_base (fallback: original GridMix default 64)")


def explicit_options(args):
    return {key: getattr(args, key) for key in DEFAULTS if getattr(args, key, None) is not None}
