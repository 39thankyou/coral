"""Raw-TFRecord replacement for the author's cache-backed cylinder dataset.

It preserves the graph fields consumed by static_inr.py/static_regression.py,
but streams only the requested smoke-test trajectories and never creates HDF5.
"""

import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from coral.utils.data.graph_dataset import load_dataset, AirfoilFlowDataset


class RawCylinderFlowDataset(Dataset):
    def __init__(self, path, split, n_samples, latent_dim=128, noise=0.02, task="static", dataset_name="cylinder-flow"):
        super().__init__(None, None, None)
        if task != "static":
            raise NotImplementedError("The author cylinder experiment is static endpoint IVP")
        self.dataset_name = dataset_name
        self.path = Path(path)
        self.split = "valid" if split == "val" else split
        manifest = self.path / "download_manifest.json"
        if manifest.exists():
            available = json.loads(manifest.read_text())["splits"].get(self.split, {}).get("records", 0)
            if n_samples > available:
                raise ValueError(f"{self.split} has only {available} downloaded trajectories; use --mode smoke or download_missing_data.py --full-ivp")
        self.latent_dim = latent_dim
        self.noise = noise
        self.T = 2
        self._graphs = []

        for index, record in enumerate(islice(load_dataset(self.path, self.split), n_samples)):
            if index >= n_samples:
                break

            def as_numpy(value):
                return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

            pos = as_numpy(record["mesh_pos"])[[0, -1]].transpose(1, 2, 0)
            velocity = as_numpy(record["velocity"])[[0, -1]].transpose(1, 2, 0)
            pressure = as_numpy(record["pressure"])[[0, -1]].transpose(1, 2, 0)
            graph = Data(
                pos=torch.from_numpy(np.ascontiguousarray(pos)),
                v=torch.from_numpy(np.ascontiguousarray(velocity)),
                p=torch.from_numpy(np.ascontiguousarray(pressure)),
            )
            if dataset_name == "airfoil-flow":
                density = as_numpy(record["density"])[[0, -1]].transpose(1, 2, 0)
                graph.rho = torch.from_numpy(np.ascontiguousarray(density))
                graph.z_rho = torch.zeros(1, latent_dim, 2)
            graph.z_v = torch.zeros(1, latent_dim, 2)
            graph.z_vx = torch.zeros(1, latent_dim, 2)
            graph.z_vy = torch.zeros(1, latent_dim, 2)
            graph.z_p = torch.zeros(1, latent_dim, 2)
            graph.z_geo = torch.zeros(1, latent_dim, 2)
            self._graphs.append(graph)

        if len(self._graphs) != n_samples:
            raise ValueError(
                f"Cylinder split {self.split!r} yielded {len(self._graphs)}, "
                f"expected {n_samples}"
            )

    def len(self):
        return len(self._graphs)

    def get(self, key):
        graph = self._graphs[key].clone()
        if self.dataset_name == "airfoil-flow":
            graph = AirfoilFlowDataset.normalize(self, graph)
        graph.z_input = torch.cat(
            [graph.z_p[..., 0], graph.z_vx[..., 0], graph.z_vy[..., 0]], dim=-1
        )
        graph.z_output = torch.cat(
            [graph.z_p[..., 1], graph.z_vx[..., 1], graph.z_vy[..., 1]], dim=-1
        )
        graph.input = torch.cat(
            [graph.pos[..., 0], graph.p[..., 0], graph.v[..., 0]], dim=-1
        ).float()
        graph.images = torch.cat(
            [graph.p[..., 1], graph.v[..., 1]], dim=-1
        ).float()
        if self.dataset_name == "airfoil-flow":
            graph.input = torch.cat([graph.pos[..., 0], graph.p[..., 0], graph.rho[..., 0], graph.v[..., 0]], dim=-1).float()
            graph.images = torch.cat([graph.p[..., 1], graph.rho[..., 1], graph.v[..., 1]], dim=-1).float()
            graph.z_input = torch.cat([graph.z_p[..., 0], graph.z_rho[..., 0], graph.z_vx[..., 0], graph.z_vy[..., 0]], dim=-1)
            graph.z_output = torch.cat([graph.z_p[..., 1], graph.z_rho[..., 1], graph.z_vx[..., 1], graph.z_vy[..., 1]], dim=-1)
        return graph, key


def bind_raw_cylinder_dataset(path, ntrain, ntest):
    """Return the author's constructor signature bound to local raw data."""

    counts = {"train": ntrain, "val": ntest, "valid": ntest, "test": ntest}

    class BoundRawCylinderFlowDataset(RawCylinderFlowDataset):
        def __init__(self, split="train", latent_dim=128, noise=0.02, task="static"):
            super().__init__(
                path=path,
                split=split,
                n_samples=counts[split],
                latent_dim=latent_dim,
                noise=noise,
                task=task,
            )

    return BoundRawCylinderFlowDataset


def bind_raw_airfoil_dataset(path, ntrain, ntest):
    counts = {"train": ntrain, "val": ntest, "valid": ntest, "test": ntest}

    class BoundRawAirfoilFlowDataset(RawCylinderFlowDataset):
        def __init__(self, split="train", latent_dim=128, noise=0.02, task="static"):
            super().__init__(path, split, counts[split], latent_dim, noise, task,
                             dataset_name="airfoil-flow")

    return BoundRawAirfoilFlowDataset
