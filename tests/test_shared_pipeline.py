"""Original adaptation/processor equivalence and retired-checkpoint rejection."""

import io
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from coral.mlp import ResNet
from coral.metalearning import outer_step
from coral.siren import ModulatedSiren
from run_codelib.checkpoints import validate_checkpoint, validate_config
from run_codelib.pipeline import main
from static.design_regression_shared import (
    create_mapper, decode_field, evaluate, fit_latent, train_batch,
)


class SharedPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)


    def test_removed_cli_options_are_rejected(self):
        for argv in (("--inr-lib", "false"), ("--model-lib", "true"),
                     ("--representation", "codelib")):
            with self.subTest(argv=argv), patch("sys.argv", ["pipeline.py", *argv]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_matches_original_encoding_and_decoding(self):
        inr = ModulatedSiren(2, 12, 2, 3, use_latent=True, latent_dim=8)
        inr.eval().requires_grad_(False)
        bundle = {"model": inr, "alpha": torch.tensor([0.01])}
        coords, fields = torch.rand(3, 5, 2), torch.rand(3, 5, 2)
        expected = outer_step(inr, coords, fields, 2, bundle["alpha"], is_train=False,
                              modulations=torch.zeros(3, 8), use_rel_loss=True)
        actual = fit_latent(bundle, coords, fields, 2, use_rel_loss=True)
        torch.testing.assert_close(actual["modulations"], expected["modulations"], rtol=0, atol=0)
        torch.testing.assert_close(actual["rel_loss"], expected["rel_loss"], rtol=0, atol=0)
        torch.testing.assert_close(decode_field(bundle, coords, actual["modulations"]),
                                   inr.modulated_forward(coords, expected["modulations"]))
        cfg = OmegaConf.create({"width": 12, "depth": 2, "dropout": 0.0})
        metrics = evaluate(create_mapper(cfg, 8, 8), actual["modulations"],
                           expected["modulations"], fields, coords, bundle, 2, "cpu")
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_matches_original_training_step(self):
        cfg = OmegaConf.create({"width": 12, "depth": 2, "dropout": 0.0})
        model = create_mapper(cfg, 8, 6)
        self.assertIs(type(model), ResNet)
        self.assertFalse(any("retrieve" in key or "memory" in key for key in model.state_dict()))
        reference = deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0.0)
        inputs, targets = torch.randn(3, 8), torch.randn(3, 6)
        expected_loss = ((reference(inputs) - targets) ** 2).mean()
        reference_optimizer.zero_grad()
        expected_loss.backward()
        reference_optimizer.step()
        actual_loss = train_batch(model, optimizer, inputs, targets)
        self.assertEqual(actual_loss, float(expected_loss.detach()))
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_retired_options_and_checkpoints_fail_explicitly(self):
        for key in ("use_codelib", "top_k", "num_head", "exclude_self", "inr_lib", "model_lib"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Removed Codelib"):
                validate_config(OmegaConf.create({"model": {key: False}}))
        for checkpoint in (
            {"cfg": {"inr_in": {"model_type": "siren_codelib"}}},
            {"cfg": {}, "model": {"retrieve.attn.in_proj_weight": torch.ones(1)}},
        ):
            with self.assertRaisesRegex(ValueError, "Codelib"):
                validate_checkpoint(checkpoint)
        # Old ordinary checkpoints can still carry disabled legacy metadata.
        validate_checkpoint({"cfg": {"inr": {"model_type": "siren", "use_codelib": False}},
                             "inr": {"net.0.linear.weight": torch.ones(1)}})


if __name__ == "__main__":
    unittest.main()
