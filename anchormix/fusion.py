"""Independent per-layer fusion MLPs evaluated together with batched matmul."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LayerwiseFusionGate(nn.Module):
    def __init__(self, layers, channels, hidden_dim, initial_fraction):
        super().__init__()
        self.weight_in = nn.Parameter(torch.empty(layers, 2 * channels, hidden_dim))
        self.bias_in = nn.Parameter(torch.empty(layers, hidden_dim))
        self.weight_out = nn.Parameter(torch.empty(layers, hidden_dim, channels))
        self.bias_out = nn.Parameter(torch.empty(layers, channels))
        bound = 1 / math.sqrt(2 * channels)
        nn.init.uniform_(self.weight_in, -bound, bound)
        nn.init.uniform_(self.bias_in, -bound, bound)
        # Small nonzero weights keep every gate layer's gradient path active
        # while starting close to the configured anchor contribution.
        bound_out = 1e-3 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.weight_out, -bound_out, bound_out)
        nn.init.constant_(self.bias_out, math.log(initial_fraction / (1 - initial_fraction)))

    def forward(self, global_values, anchor_values):
        """[B,1,L,C], [B,N,L,C] -> sigmoid gate [B,N,L,C]."""
        batch, points, layers, channels = anchor_values.shape
        # Linear([global, anchor]) = global @ W_global + anchor @ W_anchor.
        # Project the spatially constant global input once per case/layer,
        # avoiding its repeated projection and a [B,N,L,2C] concatenation.
        global_projection = torch.bmm(global_values[:, 0].transpose(0, 1), self.weight_in[:, :channels])
        features = anchor_values.permute(2, 0, 1, 3).reshape(layers, batch * points, channels)
        anchor_projection = torch.bmm(features, self.weight_in[:, channels:]).reshape(layers, batch, points, -1)
        hidden = F.silu(anchor_projection + global_projection[:, :, None] + self.bias_in[:, None, None])
        hidden = hidden.reshape(layers, batch * points, -1)
        logits = torch.bmm(hidden, self.weight_out) + self.bias_out[:, None]
        return logits.sigmoid().reshape(layers, batch, points, channels).permute(1, 2, 0, 3)
