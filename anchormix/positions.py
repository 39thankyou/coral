"""One-time, bounded-memory FPS of the UNION of training query coordinates."""
import math
from pathlib import Path

import numpy as np
import torch


def _blocks(coordinates, block_size=65536):
    # Iterate cases BEFORE flattening, so an expanded common grid is never copied
    # into a full [cases * points,d] array just to initialize anchors.
    previous = None
    for case in coordinates:
        case = case.detach().reshape(-1, coordinates.shape[-1])
        if previous is not None and torch.equal(case, previous):
            continue
        previous = case
        for block in case.split(block_size):
            block = block.to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(block).all():
                raise ValueError("Training coordinates must be finite")
            yield block


def training_candidates(coordinates, max_candidates=65536):
    if coordinates.ndim < 3 or coordinates.numel() == 0:
        raise ValueError("Expected nonempty training coordinates [cases,...,d]")
    d = coordinates.shape[-1]
    lo, hi = torch.full((d,), float("inf")), torch.full((d,), -float("inf"))
    for block in _blocks(coordinates):
        lo = torch.minimum(lo, block.amin(0))
        hi = torch.maximum(hi, block.amax(0))
    # At most bins**d occupied cells. Retain an ACTUAL training query per cell.
    # These bounds are used only for downsampling; model coordinates stay intact.
    bins = max(1, int(math.floor(max_candidates ** (1.0 / d))))
    span = (hi - lo).clamp_min(torch.finfo(lo.dtype).eps)
    representatives = {}
    for block in _blocks(coordinates):
        cells = (((block - lo) / span * bins).floor().long().clamp(0, bins - 1)).numpy()
        _, indices = np.unique(cells, axis=0, return_index=True)
        for i in indices:
            representatives.setdefault(tuple(cells[i]), block[i].clone())
    return torch.stack(list(representatives.values())), lo, hi


def farthest_points(candidates, count):
    if len(candidates) < count:
        raise ValueError(f"FPS needs {count} distinct spatial candidates; got {len(candidates)}")
    distances = torch.full((len(candidates),), float("inf"))
    selected = []
    index = 0
    for _ in range(count):
        selected.append(index)
        distances = torch.minimum(distances, (candidates - candidates[index]).square().sum(-1))
        distances[selected] = -1
        index = int(distances.argmax())
    return candidates[selected].clone()


def read_positions(path, shape):
    if not path:
        raise ValueError("anchor_init=file requires anchor_path")
    path = Path(path).expanduser()
    values = (torch.from_numpy(np.load(path, allow_pickle=False)) if path.suffix == ".npy"
              else torch.load(path, map_location="cpu", weights_only=True))
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != tuple(shape):
        raise ValueError(f"Anchor file must contain a tensor of shape {tuple(shape)}")
    if not torch.isfinite(values).all():
        raise ValueError("Anchor file contains non-finite positions")
    return values.float()
