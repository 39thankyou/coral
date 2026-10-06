"""Shared basis mixing followed by a position-dependent, layer-shared read."""
from dataclasses import asdict
import warnings

import torch
from torch import nn

from .config import AnchorMixConfig
from .fusion import LayerwiseFusionGate
from .positions import farthest_points, read_positions, training_candidates
from .siren import LatentToModulation, ModulatedSiren


class AnchorMix(ModulatedSiren):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, *, grid_base=64,
                 anchor_config=None, **kwargs):
        if not kwargs.get("use_latent", True):
            raise ValueError("AnchorMix requires use_latent=true")
        kwargs["use_latent"] = True
        if min(dim_in, dim_hidden, dim_out, grid_base) < 1 or num_layers < 2:
            raise ValueError("Model dimensions/M must be positive and depth >=2")
        self.options = AnchorMixConfig(**(anchor_config or {})).validate()
        super().__init__(dim_in, dim_hidden, dim_out, num_layers, **kwargs)
        self.num_bases = grid_base
        self.modulated_layers = num_layers - 1
        self.modulation_channels = dim_hidden * (int(self.modulate_scale) + int(self.modulate_shift))
        self.query_chunk_size = self.options.query_chunk_size
        # Keep CORAL's complete global latent->modulation map (including its
        # scale/shift ordering), without introducing any coefficient softmax.
        global_net = self.modulation_net
        if self.options.fusion != "global":
            self.modulation_net = LatentToModulation(
                global_net.latent_dim, self.modulated_layers * grid_base,
                global_net.dim_hidden, global_net.num_layers)
            self.B = nn.Parameter(torch.randn(grid_base, self.modulated_layers,
                                             self.modulation_channels, self.options.num_anchors))
            # Random Linear biases and nonidentical random B keep z=0 values,
            # latent derivatives, and dictionary/position gradients alive.
        if self.options.fusion in ("hybrid", "gated"):
            self.global_modulation_net = global_net
        if self.options.fusion == "gated":
            self.fusion_gate = LayerwiseFusionGate(
                self.modulated_layers, self.modulation_channels,
                self.options.fusion_gate_hidden_dim, self.options.fusion_gate_init)
        if self.options.fusion == "hybrid":
            alpha = torch.tensor(float(self.options.global_alpha_init))
            if self.options.learnable_global_alpha:
                self.global_alpha = nn.Parameter(alpha)
            else:
                self.register_buffer("global_alpha", alpha)
        position = torch.zeros(self.options.num_anchors, dim_in)
        if self.options.learnable_pos:
            self.positions = nn.Parameter(position)
        else:
            self.register_buffer("positions", position)
        self.register_buffer("initial_positions", position.clone())
        self.register_buffer("bbox_min", torch.zeros(dim_in))
        self.register_buffer("bbox_max", torch.ones(dim_in))
        # Identity transform: coordinates already have the loader's global scale.
        self.register_buffer("coordinate_offset", torch.zeros(dim_in))
        self.register_buffer("coordinate_scale", torch.ones(dim_in))
        self.register_buffer("positions_initialized", torch.tensor(False))
        # The checkpoint buffer remains authoritative on load. Its Python mirror
        # avoids a CUDA -> CPU .item() synchronization on every inner-loop forward.
        self._positions_ready = False
        if self.options.weight_mode == "distance_gate" and self.options.fusion != "global":
            self.gate = nn.Sequential(nn.Linear(2 * dim_in, self.options.gate_hidden_dim),
                                      nn.SiLU(), nn.Linear(self.options.gate_hidden_dim, 1))

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._positions_ready = bool(self.positions_initialized.item())

    @torch.no_grad()
    def initialize_positions(self, training_coordinates):
        if self._positions_ready:
            raise RuntimeError("Anchors are already initialized; resume/load must reuse checkpoint positions")
        candidates, lo, hi = training_candidates(training_coordinates)
        if self.options.anchor_init == "fps":
            positions = farthest_points(candidates, self.options.num_anchors)
        elif self.options.anchor_init == "file":
            positions = read_positions(self.options.anchor_path, self.positions.shape)
        else:
            positions = torch.zeros_like(self.positions, device="cpu")
            if self.options.init_jitter == 0 and self.options.fusion != "global":
                warnings.warn("Zero anchors coincide: knn has tied neighbors and can starve unselected tokens; "
                              "all has identical coordinate scores initially. Prefer fusion=global for the zero "
                              "experiment, or explicitly request init_jitter. No jitter was added.", UserWarning)
        if self.options.init_jitter:
            positions += torch.randn_like(positions) * self.options.init_jitter
        self.positions.copy_(positions)
        self.bbox_min.copy_(lo)
        self.bbox_max.copy_(hi)
        if self.options.pos_constraint == "bbox" and not self.options.learnable_pos:
            if ((self.positions < self.bbox_min) | (self.positions > self.bbox_max)).any():
                raise ValueError("Fixed initial anchors are outside the training bbox; use pos_constraint=none "
                                 "or supply positions inside the bounds (initialization is never silently moved)")
        # Preserve requested zero/file/jitter initialization exactly. A bbox
        # constraint is applied after outer updates, never during a forward.
        self.initial_positions.copy_(self.positions)
        self.positions_initialized.fill_(True)
        self._positions_ready = True

    @torch.no_grad()
    def project_positions(self):
        """Call only AFTER an outer optimizer step, never during latent adaptation."""
        if self.options.pos_constraint == "bbox" and isinstance(self.positions, nn.Parameter):
            self.positions.copy_(torch.maximum(self.bbox_min, torch.minimum(self.bbox_max, self.positions)))

    def parameter_groups(self, lr):
        position = self.positions if isinstance(self.positions, nn.Parameter) else None
        groups = [{"params": [p for p in self.parameters() if p is not position], "lr": lr}]
        if position is not None:
            groups.append({"params": [position], "lr": lr * self.options.pos_lr_scale,
                           "weight_decay": 0, "name": "anchor_positions"})
        return groups

    def _weights(self, queries):
        indices = None
        if self.options.read_mode == "knn":
            # Discrete neighbor selection is nondifferentiable; recompute selected
            # distances below WITH gradients, including second derivatives.
            with torch.no_grad():
                distances = (queries.unsqueeze(-2) - self.positions).square().sum(-1)
                indices = distances.topk(self.options.num_neighbors, largest=False, sorted=True).indices
            positions = self.positions[indices]
        else:
            positions = self.positions
        relative = queries.unsqueeze(-2) - positions
        scores = -relative.square().sum(-1) / (2 * self.options.kernel_sigma ** 2)
        if self.options.weight_mode == "distance_gate":
            coords = queries.unsqueeze(-2).expand_as(relative)
            scores = scores + self.gate(torch.cat((coords, relative), dim=-1)).squeeze(-1)
        # softmax(scores)**power / sum(...) == softmax(power * scores).
        # Work in logit space to avoid underflow from powering small weights.
        # This sharpens spatial reads only; basis coefficients c(z) are unchanged.
        if self.options.attention_power != 1.0:
            scores = (scores - scores.amax(dim=-1, keepdim=True)) * self.options.attention_power
        return scores.softmax(-1), indices

    def _read(self, values, weights, indices):
        # values [b,K,L*C] is prepared ONCE per forward. Scatter only the small
        # [b,chunk,K] weight matrix, with exact zeros for unselected neighbors.
        # bmm reads all layers/channels together without a [b,chunk,q,L*C] gather
        # or a Python loop per neighbor. Both scatter and bmm support grad-grad.
        if indices is not None:
            weights = weights.new_zeros(*weights.shape[:2], values.shape[1]).scatter(-1, indices, weights)
        result = torch.bmm(weights, values)
        return result.reshape(*weights.shape[:2], self.modulated_layers, self.modulation_channels)

    def _global_values(self, latent):
        net = self.modulation_net if self.options.fusion == "global" else self.global_modulation_net
        values = net(latent)
        if self.modulate_scale and self.modulate_shift:
            scale, shift = values.chunk(2, dim=-1)
            return torch.cat((scale.reshape(-1, self.modulated_layers, self.dim_hidden),
                              shift.reshape(-1, self.modulated_layers, self.dim_hidden)), dim=-1)
        return values.reshape(-1, self.modulated_layers, self.dim_hidden)

    def modulated_forward(self, x, latent):
        if not self._positions_ready:
            raise RuntimeError("Initialize anchors from TRAINING coordinates once, or load a checkpoint first")
        if x.shape[-1] != self.dim_in or latent.shape != (x.shape[0], self.modulation_net.latent_dim):
            raise ValueError("Coordinates/latent do not match the saved model dimensions")
        spatial_shape = x.shape[:-1]
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        values = global_values = None
        if self.options.fusion != "global":
            coefficients = self.modulation_net(latent).reshape(len(x), self.modulated_layers, self.num_bases)
            values = torch.einsum("blm,mlck->bklc", coefficients, self.B).flatten(2)
        if self.options.fusion != "anchor":
            global_values = self._global_values(latent).unsqueeze(1)
        predictions = []
        for queries in x.split(self.query_chunk_size, dim=1):
            if values is not None:
                weights, indices = self._weights(queries)
                phi = self._read(values, weights, indices)
                if global_values is not None:
                    if self.options.fusion == "gated":
                        gate = self.fusion_gate(global_values, phi)
                        phi = (1 - gate) * global_values + gate * phi
                    else:
                        phi = phi + self.global_alpha * global_values
            else:
                phi = global_values
            hidden = queries
            for layer, module in enumerate(self.net):
                modulation = phi[:, :, layer]
                scale = 1 + modulation[..., :self.dim_hidden] if self.modulate_scale else 1.0
                shift = modulation[..., -self.dim_hidden:] if self.modulate_shift else 0.0
                hidden = module.activation(scale * module.linear(hidden) + shift)
            predictions.append(self.last_activation(self.last_layer(hidden)) * self.sigma + self.mu)
        return torch.cat(predictions, dim=1).reshape(*spatial_shape, self.dim_out)

    forward = modulated_forward

    def anchor_metadata(self):
        return {"options": asdict(self.options), "coordinate_transform": "identity_after_existing_loader",
                "coordinate_dim": self.dim_in, "dictionary_shape": list(self.B.shape) if hasattr(self, "B") else None}
