"""Exercise graph and temporal stage handoffs with small synthetic datasets."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from run_codelib import extended
from run_codelib.pipeline import stage_config


class SharedStageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def run_stages(self, dataset, loader, data):
        config = OmegaConf.load("run_codelib/config.yaml")
        family = config.datasets[dataset].family
        args = SimpleNamespace(dataset=dataset, mode="smoke", epochs=1, ntrain=2,
                               ntest=1, batch_size=2, encode_batch_size=2, device="cpu")
        cfg = stage_config(config, "inr", args)
        sections = ("inr_in", "inr_out") if family == "graph" else ("inr",)
        for section in sections:
            cfg[section].hidden_dim, cfg[section].depth, cfg[section].latent_dim = 12, 3, 5
        if family == "dynamics":
            cfg.data.seq_inter_len, cfg.data.seq_extra_len = 3, 2
        cfg.optim.inner_steps = cfg.optim.test_inner_steps = 2
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, WANDB_DIR=directory), patch(
                f"run_codelib.extended.{loader}", side_effect=data if callable(data) else None,
                return_value=None if callable(data) else data):
            root = Path(directory) / cfg.data.dataset_name
            extended.run_stage(cfg, "inr", family)
            inr_path = root / "inr" / f"{cfg.wandb.name}.pt"
            stage = "regression" if family == "graph" else "ode"
            downstream = stage_config(config, stage, args, inr_path)
            if family == "graph":
                downstream.model.width, downstream.model.depth = 12, 2
            downstream.inr.inner_steps = 2
            if family == "dynamics":
                downstream.data.seq_inter_len, downstream.data.seq_extra_len = 3, 2
                downstream.dynamics.width, downstream.dynamics.depth = 12, 2
            extended.run_stage(downstream, stage, family)
            path = root / "model" / f"{downstream.wandb.name}.pt"
            saved = torch.load(path, weights_only=False)
            self.assertFalse(any("memory" in key or "codelib" in key for key in saved))
            self.assertFalse(any("retrieve" in key for key in saved["model"]))
            eval_args = SimpleNamespace(dataset=dataset, config="run_codelib/config.yaml", mode="smoke",
                                        regression_checkpoint=str(path), inr_checkpoint=None,
                                        output_root=None, data_dir=None, ntest=1, batch_size=1,
                                        device="cpu", output=None)
            if family == "graph":
                result = extended.evaluate(eval_args)
                self.assertEqual(result["relative_l2"], saved["metrics"]["relative_l2"])
            else:
                with patch("run_codelib.extended.encode_frames", wraps=extended.encode_frames) as encode:
                    result = extended.evaluate(eval_args)
                # Evaluation must never adapt the future target frames.
                self.assertTrue(all(call.args[1].shape[-1] == 1 for call in encode.call_args_list))
                self.assertFalse(result["teacher_forcing"])
                self.assertEqual(len(result["per_frame_mse"]), 5)
                # Old ordinary ODE checkpoints had an extra wrapper prefix.
                saved["model"] = {f"derivative.{key}": value for key, value in saved["model"].items()}
                torch.save(saved, path)
                legacy = extended.evaluate(eval_args)
                self.assertEqual(result["per_frame_mse"], legacy["per_frame_mse"])

    def test_ragged_graph_train_restore_and_evaluate(self):
        fields = [(torch.rand(1, n, 2), torch.rand(1, n, 3), torch.rand(1, n, 3)) for n in (7, 9, 6)]
        self.run_stages("cylinder_flow", "graph_fields",
                        lambda cfg, split, count: fields[:count] if split == "train" else fields[2:2+count])

    def test_temporal_train_restore_and_initial_frame_only_evaluation(self):
        train, test = torch.rand(2, 7, 1, 3), torch.rand(1, 7, 1, 5)
        train_grid = torch.rand(2, 7, 2, 1).expand(-1, -1, -1, 3).clone()
        test_grid = torch.rand(1, 7, 2, 1).expand(-1, -1, -1, 5).clone()
        data = (train, train, test, train_grid, train_grid, test_grid)
        self.run_stages("navier_stokes", "temporal_data", data)


if __name__ == "__main__":
    unittest.main()
