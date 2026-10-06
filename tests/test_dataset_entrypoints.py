"""Split/grid and temporal-model invariants of the seven dataset entrypoints."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from coral.utils.data.load_data import get_pipe, get_shallow_water_dino
from run_adaptor.author_pipeline import _build_stage_config
from run_codelib.pipeline import stage_config
from run_codelib.extended import validate_temporal_inr
from coral.mlp import Derivative


class DatasetEntrypointTests(unittest.TestCase):
    def test_pipe_smoke_preserves_official_test_offset_and_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xy = np.arange(1200, dtype=np.float32)[:, None, None]*np.ones((1, 2, 2), dtype=np.float32)
            np.save(root/"Pipe_X.npy", xy)
            np.save(root/"Pipe_Y.npy", xy+1)
            np.save(root/"Pipe_Q.npy", xy[:, None])
            a = get_pipe(root, 1000, 4)
            b = get_pipe(root, 2, 1)
            torch.testing.assert_close(a[0][:2], b[0])
            torch.testing.assert_close(a[2][:1], b[2])
            self.assertEqual(b[3][0, 0, 0], 1000)

    def test_sw_counts_select_temporal_window_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"dino"
            root.mkdir()
            for split, marker in (("16_160_128_256_train", 0), ("2_160_128_256_test", 100)):
                values = np.arange(2*8*3*4, dtype=np.float32).reshape(2, 8, 3, 4)+marker
                with h5py.File(root/f"shallow_water_{split}.h5", "w") as f:
                    f["height"] = values
                    f["vorticity"] = values*0.1
            all_data = get_shallow_water_dino(root.parent, 2, 2)
            prefix = get_shallow_water_dino(root.parent, 2, 2, ntrain=2, ntest=1)
            self.assertEqual(prefix[0].shape[-1], 2)
            self.assertEqual(prefix[2].shape[-1], 4)
            torch.testing.assert_close(prefix[0], all_data[0][:2])
            torch.testing.assert_close(prefix[2], all_data[2][:1])

    def test_all_configurations_resolve_correct_dataset_and_stage(self):
        root = Path(__file__).resolve().parents[1]
        author = OmegaConf.load(root/"run_adaptor/config.yaml")
        shared = OmegaConf.load(root/"run_codelib/config.yaml")
        expected = {"airfoil", "elasticity", "pipe", "cylinder_flow", "airfoil_flow", "navier_stokes", "shallow_water"}
        self.assertEqual(set(author.datasets), expected)
        self.assertEqual(set(shared.datasets), expected)
        for key in expected:
            downstream = "ode" if key in ("navier_stokes", "shallow_water") else "regression"
            for stage in ("inr", downstream):
                original = _build_stage_config(author, key, stage, "smoke", 1, 2, 1)
                args = SimpleNamespace(dataset=key, mode="smoke", ntrain=2,
                                       ntest=1, epochs=1, batch_size=None, device="cpu", encode_batch_size=None)
                custom = stage_config(shared, stage, args, Path("/tmp/inr.pt"))
                self.assertEqual(original.data.dataset_name, custom.data.dataset_name)
                self.assertEqual(custom.optim.epochs, 1)
                self.assertTrue((root/f"run_codelib/run_{key}.py").exists())
                self.assertTrue((root/f"run_codelib/eval_{key}.py").exists())
                self.assertTrue((root/f"run_adaptor/evaluate_{key}.py").exists())

    def test_ode_derivative_has_finite_gradients(self):
        model = Derivative(1, 8, 12, 2)
        output = model(0, torch.randn(2, 8))
        self.assertEqual(output.shape, (2, 8))
        output.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_temporal_inr_rejects_wrong_dataset_and_nonfinite_alpha(self):
        cfg = OmegaConf.create({"data": {"dataset_name": "navier-stokes-dino"},
                                "inr": {"model_type": "siren", "latent_dim": 8}})
        checkpoint = {"cfg": cfg, "alpha": torch.tensor([0.01])}
        validate_temporal_inr(checkpoint, "navier-stokes-dino")
        with self.assertRaises(ValueError):
            validate_temporal_inr(checkpoint, "shallow-water-dino")
        checkpoint["alpha"] = torch.tensor([float("nan")])
        with self.assertRaises(ValueError):
            validate_temporal_inr(checkpoint, "navier-stokes-dino")


if __name__ == "__main__":
    unittest.main()
