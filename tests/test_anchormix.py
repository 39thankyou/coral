"""Representation math, high-order gradients, replay and CLI invariants."""
import argparse
import io
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from anchormix.config import AnchorMixConfig, add_arguments
from anchormix.factory import create_inr_instance
from anchormix.fusion import LayerwiseFusionGate
from anchormix.model import AnchorMix
from anchormix.positions import training_candidates
from anchormix.siren import ModulatedSiren
from coral.metalearning import outer_step
from run_codelib.pipeline import stage_config
from run_codelib.representation import configure_request, validate_overrides
from static.design_regression_shared import load_inrs


class AnchorMixTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def model(self, d=2, scale=False, **options):
        return AnchorMix(d, 12, 2, 3, grid_base=4, latent_dim=5, modulate_scale=scale,
                         anchor_config={"num_anchors": 6, "num_neighbors": 3,
                                        "query_chunk_size": 4, **options})

    def test_all_modes_support_original_second_order_adaptation(self):
        for read in ("knn", "all"):
            for fusion in ("anchor", "global", "hybrid", "gated"):
                for weight in ("distance", "distance_gate"):
                    with self.subTest(read=read, fusion=fusion, weight=weight):
                        model = self.model(d=3, read_mode=read, fusion=fusion, weight_mode=weight,
                                           learnable_pos=True, attention_power=2.0)
                        coords, fields = torch.rand(2, 3, 3, 3), torch.rand(2, 3, 3, 2)
                        model.initialize_positions(coords)
                        alpha = torch.tensor([0.01], requires_grad=True)
                        output = outer_step(model, coords, fields, 2, alpha, is_train=True,
                                            modulations=torch.zeros(2, 5), return_reconstructions=True)
                        self.assertEqual(output["reconstructions"].shape, fields.shape)
                        output["loss"].backward()
                        self.assertTrue(torch.isfinite(alpha.grad).all())
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                self.assertTrue(torch.isfinite(parameter.grad).all())
                        if fusion != "global":
                            for gradient in (model.positions.grad, model.B.grad, model.modulation_net.net.weight.grad):
                                self.assertIsNotNone(gradient)
                                self.assertGreater(gradient.norm().item(), 0)
                        if fusion == "gated":
                            for parameter in (*model.global_modulation_net.parameters(), *model.fusion_gate.parameters()):
                                self.assertIsNotNone(parameter.grad)
                                self.assertGreater(parameter.grad.norm().item(), 0)

    def test_parallel_fusion_gate_matches_independent_layer_mlps(self):
        gate = LayerwiseFusionGate(3, 5, 7, 0.05).double()
        global_values = torch.randn(2, 1, 3, 5, dtype=torch.float64, requires_grad=True)
        anchor_values = torch.randn(2, 11, 3, 5, dtype=torch.float64, requires_grad=True)
        actual = gate(global_values, anchor_values)
        reference = []
        for layer in range(3):
            features = torch.cat((global_values[:, :, layer].expand(-1, 11, -1), anchor_values[:, :, layer]), -1)
            hidden = torch.nn.functional.silu(torch.nn.functional.linear(
                features, gate.weight_in[layer].T, gate.bias_in[layer]))
            reference.append(torch.nn.functional.linear(hidden, gate.weight_out[layer].T, gate.bias_out[layer]).sigmoid())
        expected = torch.stack(reference, dim=2)
        self.assertEqual(actual.shape, (2, 11, 3, 5))
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(actual, torch.full_like(actual, 0.05), rtol=0, atol=1e-4)
        parameters = (global_values, anchor_values, *gate.parameters())
        for value in (actual, expected):
            gradients = torch.autograd.grad(value.square().sum(), parameters, retain_graph=True)
            if value is actual:
                first_gradients = gradients
            else:
                for a, b in zip(first_gradients, gradients):
                    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)

    def test_gated_fusion_uses_convex_mixture_and_correct_endpoints(self):
        for scale in (False, True):
            model = self.model(read_mode="all", attention_power=2, fusion="gated", scale=scale).double()
            model.query_chunk_size = 100
            x, z = torch.rand(2, 7, 2, dtype=torch.float64), torch.rand(2, 5, dtype=torch.float64)
            model.initialize_positions(x)
            self.assertFalse(hasattr(model, "global_alpha"))
            global_values = model._global_values(z).unsqueeze(1)
            coeff = model.modulation_net(z).reshape(2, 2, 4)
            values = torch.einsum("blm,mlck->bklc", coeff, model.B).flatten(2)
            weights, indices = model._weights(x)
            anchor_values = model._read(values, weights, indices)
            for fraction in (0.0, 0.25, 1.0):
                with self.subTest(scale=scale, fraction=fraction):
                    phi = torch.lerp(global_values.expand_as(anchor_values), anchor_values, fraction)
                    hidden = x
                    for layer, module in enumerate(model.net):
                        modulation = phi[:, :, layer]
                        gain = 1 + modulation[..., :12] if scale else 1.0
                        hidden = module.activation(gain * module.linear(hidden) + modulation[..., -12:])
                    expected = model.last_activation(model.last_layer(hidden)) * model.sigma + model.mu
                    with patch.object(model.fusion_gate, "forward", return_value=torch.full_like(anchor_values, fraction)):
                        torch.testing.assert_close(model(x, z), expected, rtol=1e-11, atol=1e-13)

    def test_gated_chunks_preserve_outputs_and_second_order_gradients(self):
        model = self.model(read_mode="all", attention_power=2, fusion="gated", learnable_pos=True).double()
        x = torch.rand(2, 9, 2, dtype=torch.float64)
        fields = torch.rand(2, 9, 2, dtype=torch.float64)
        model.initialize_positions(x)
        other = deepcopy(model)
        other.query_chunk_size = 100
        results, gradients = [], []
        for net in (model, other):
            result = outer_step(net, x, fields, 2, .01, is_train=True,
                                modulations=torch.zeros(2, 5, dtype=torch.float64), return_reconstructions=True)
            results.append(result)
            gradients.append(torch.autograd.grad(result["loss"], tuple(net.parameters())))
        torch.testing.assert_close(results[0]["reconstructions"], results[1]["reconstructions"], rtol=1e-10, atol=1e-12)
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)

    def test_read_matches_direct_definition_and_shares_weights(self):
        model = self.model(read_mode="all")
        x = torch.rand(2, 9, 2)
        model.initialize_positions(x)
        z = torch.randn(2, 5)
        coeff = model.modulation_net(z).reshape(2, 2, 4)
        values = torch.einsum("blm,mlck->blck", coeff, model.B)
        weights, indices = model._weights(x)
        expected = torch.einsum("bnk,blck->bnlc", weights, values)
        torch.testing.assert_close(model._read(values.permute(0, 3, 1, 2).flatten(2), weights, indices), expected)
        knn = deepcopy(model)
        knn.options.read_mode, knn.options.num_neighbors = "knn", 6
        torch.testing.assert_close(knn(x, z), model(x, z), atol=1e-6, rtol=1e-5)

    def test_attention_power_matches_post_softmax_normalization(self):
        for read in ("all", "knn"):
            for mode in ("distance", "distance_gate"):
                with self.subTest(read=read, mode=mode):
                    model = self.model(read_mode=read, weight_mode=mode).double()
                    x = torch.rand(2, 9, 2, dtype=torch.float64)
                    model.initialize_positions(x)
                    original, indices = model._weights(x)
                    for power in (0.5, 1.0, 2.0, 4.0):
                        model.options.attention_power = power
                        actual, selected = model._weights(x)
                        expected = original.pow(power)
                        expected = expected / expected.sum(-1, keepdim=True)
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
                        torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[..., 0]))
                        if indices is not None:
                            torch.testing.assert_close(selected, indices)
                        if power > 1:
                            self.assertTrue((actual.max(-1).values >= original.max(-1).values).all())
                    model.options.attention_power = 1.0
                    torch.testing.assert_close(model._weights(x)[0], original, rtol=0, atol=0)

    def test_all_power_equals_narrower_gaussian_and_preserves_continuity(self):
        model = self.model(d=1, num_anchors=2, num_neighbors=1, read_mode="all",
                           attention_power=4.0).double()
        model.initialize_positions(torch.tensor([[[0.], [1.]]]))
        other = deepcopy(model)
        other.options.attention_power = 1.0
        other.options.kernel_sigma /= 2
        x = torch.tensor([[[0.5 - 1e-6], [0.5 + 1e-6]]], dtype=torch.float64)
        weights, indices = model._weights(x)
        self.assertIsNone(indices)
        torch.testing.assert_close(weights, other._weights(x)[0], rtol=1e-12, atol=1e-14)
        self.assertLess((weights[:, 0] - weights[:, 1]).abs().max().item(), 1e-3)
        # Explicitly raising already tiny softmax values can underflow to all
        # zeros. The logit-space implementation must remain normalized.
        dense = self.model(read_mode="all", kernel_sigma=10.0, attention_power=10000.0)
        coords = torch.rand(2, 9, 2)
        dense.initialize_positions(coords)
        sharp, _ = dense._weights(coords)
        self.assertTrue(torch.isfinite(sharp).all())
        torch.testing.assert_close(sharp.sum(-1), torch.ones_like(sharp[..., 0]))

    def test_attention_power_checkpoint_restore_and_legacy_default(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        for power, fusion in ((None, "anchor"), (2.0, "anchor"), (2.0, "gated")):
            with self.subTest(power=power, fusion=fusion):
                options = dict(num_anchors=6, num_neighbors=3, read_mode="all", fusion=fusion)
                if power is not None:
                    options["attention_power"] = power
                settings = dict(model_type="anchormix", use_latent=True, latent_dim=5, depth=3,
                                hidden_dim=12, w0=10, modulate_scale=False, modulate_shift=True,
                                hypernet_width=16, hypernet_depth=1, last_activation=None,
                                grid_base=4, anchormix=options)
                cfg = OmegaConf.create(dict(inr=settings, inr_in=settings, inr_out=settings))
                model = create_inr_instance(cfg, 2, 2, "cpu")
                x, z = torch.rand(2, 9, 2), torch.rand(2, 5)
                model.initialize_positions(x)
                checkpoint = {"cfg": cfg, "inr_in": model.state_dict(), "inr_out": model.state_dict(),
                              "alpha_in": torch.tensor([.01]), "alpha_out": torch.tensor([.01])}
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)
                checkpoint = torch.load(buffer, weights_only=False)
                with patch("anchormix.model.farthest_points", side_effect=AssertionError("FPS on restore")):
                    restored = load_inrs(checkpoint, "cpu", 2, 2, 2)["out"]["model"]
                self.assertEqual(restored.options.attention_power, power or 1.0)
                self.assertEqual(restored.options.fusion, fusion)
                torch.testing.assert_close(restored(x, z), model(x, z), rtol=0, atol=0)
                for flag in ("--attention-power", "--attention_power"):
                    args = parser.parse_args([flag, str(power or 1.0)])
                    validate_overrides(args, checkpoint["cfg"])
                    args.attention_power = 3.0
                    with self.assertRaisesRegex(ValueError, "Incompatible CLI attention_power"):
                        validate_overrides(args, checkpoint["cfg"])
                for name, value in (("fusion_gate_init", 0.25), ("fusion_gate_hidden_dim", 32)):
                    args = parser.parse_args(["--" + name.replace("_", "-"), str(value)])
                    with self.assertRaisesRegex(ValueError, "Incompatible CLI " + name):
                        validate_overrides(args, checkpoint["cfg"])

    def test_chunking_preserves_outputs_and_gradients(self):
        model = self.model(weight_mode="distance_gate", fusion="hybrid", learnable_pos=True, scale=True)
        x, z = torch.rand(2, 11, 2), torch.randn(2, 5)
        model.initialize_positions(x)
        other = deepcopy(model)
        other.query_chunk_size = 100
        first, second = model(x, z), other(x, z)
        torch.testing.assert_close(first, second)
        first.square().sum().backward()
        second.square().sum().backward()
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)

    def test_vectorized_knn_matches_neighbor_loop_through_adaptation(self):
        for neighbors in (1, 3, 6):
            for weight in ("distance", "distance_gate"):
                with self.subTest(neighbors=neighbors, weight=weight):
                    model = self.model(num_neighbors=neighbors, weight_mode=weight,
                                       fusion="hybrid", learnable_pos=True, scale=True).double()
                    coords = torch.rand(2, 7, 2, dtype=torch.float64)
                    fields = torch.rand(2, 7, 2, dtype=torch.float64)
                    model.initialize_positions(coords)
                    reference = deepcopy(model)

                    def neighbor_loop(values, weights, indices):
                        # The previous q separate gathers are an independent
                        # reference for the new scatter/batched-matmul read.
                        result = 0
                        for neighbor in range(indices.shape[-1]):
                            selected = values.gather(1, indices[..., neighbor, None].expand(-1, -1, values.shape[-1]))
                            result = result + weights[..., neighbor, None] * selected
                        return result.reshape(*weights.shape[:2], model.modulated_layers, model.modulation_channels)

                    alphas = [torch.tensor([0.01], dtype=torch.float64, requires_grad=True) for _ in range(2)]
                    outputs, gradients = [], []
                    with patch.object(reference, "_read", side_effect=neighbor_loop):
                        for net, alpha in zip((model, reference), alphas):
                            output = outer_step(net, coords, fields, 2, alpha, is_train=True,
                                                modulations=torch.zeros(2, 5, dtype=torch.float64),
                                                return_reconstructions=True)
                            outputs.append(output)
                            gradients.append(torch.autograd.grad(output["loss"], (*net.parameters(), alpha)))
                    for key in ("loss", "modulations", "reconstructions"):
                        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=1e-9, atol=1e-11)
                    for actual, expected in zip(*gradients):
                        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)

    def test_initialization_guard_survives_nested_load_without_forward_sync(self):
        coords, latent = torch.rand(2, 7, 2), torch.rand(2, 5)
        model = self.model()
        uninitialized = deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "Initialize anchors"):
            model(coords, latent)
        model.initialize_positions(coords)
        parent = torch.nn.ModuleDict({"representation": self.model()})
        # Loading through a parent must also restore the Python initialization flag.
        parent.load_state_dict({f"representation.{key}": value for key, value in model.state_dict().items()})
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("Tensor.item in forward")):
            result = parent["representation"](coords, latent)
        torch.testing.assert_close(result, model(coords, latent), rtol=0, atol=0)
        parent["representation"].load_state_dict(uninitialized)
        with self.assertRaisesRegex(RuntimeError, "Initialize anchors"):
            parent["representation"](coords, latent)

    def test_global_fusion_is_exact_copied_global_modulation(self):
        for scale in (False, True):
            model = self.model(fusion="global", scale=scale)
            x, z = torch.rand(2, 9, 2), torch.randn(2, 5)
            model.initialize_positions(x)
            expected = ModulatedSiren.modulated_forward(model, x, z)
            torch.testing.assert_close(model(x, z), expected)

    def test_zero_latent_does_not_disable_values_or_key_gradients(self):
        model = self.model(learnable_pos=True)
        x = torch.rand(2, 9, 2)
        model.initialize_positions(x)
        z = torch.zeros(2, 5, requires_grad=True)
        self.assertGreater(model.modulation_net(z).norm().item(), 0)
        self.assertGreater(model.B.var(dim=-1).min().item(), 0)
        model(x, z).square().sum().backward()
        for gradient in (z.grad, model.B.grad, model.positions.grad, model.modulation_net.net.bias.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0)

    def test_fixed_and_learned_positions_optimizer_projection(self):
        for learnable in (False, True):
            model = self.model(learnable_pos=learnable, pos_lr_scale=0.2, pos_constraint="bbox")
            x = torch.rand(2, 9, 2)
            model.initialize_positions(x)
            before = model.positions.detach().clone()
            optimizer = torch.optim.AdamW(model.parameter_groups(0.01), weight_decay=0)
            model(x, torch.rand(2, 5)).square().mean().backward()
            optimizer.step()
            if learnable:
                self.assertEqual(optimizer.param_groups[1]["lr"], 0.002)
                self.assertFalse(torch.equal(before, model.positions))
                with torch.no_grad():
                    model.positions[0] = 100
                model.project_positions()
                self.assertTrue((model.positions <= model.bbox_max).all())
                self.assertTrue((model.positions >= model.bbox_min).all())
            else:
                self.assertIn("positions", dict(model.named_buffers()))
                torch.testing.assert_close(before, model.positions, rtol=0, atol=0)

    def test_fps_uses_union_once_and_keeps_loader_units(self):
        coordinates = torch.stack((torch.linspace(2, 3, 12), torch.linspace(7, 9, 12)))[..., None]
        model = self.model(d=1)
        model.initialize_positions(coordinates)
        self.assertEqual(model(coordinates, torch.zeros(2, 5)).shape, (2, 12, 2))
        self.assertTrue((model.positions > 3).any())
        torch.testing.assert_close(model.bbox_min, torch.tensor([2.]))
        torch.testing.assert_close(model.bbox_max, torch.tensor([9.]))
        self.assertTrue(all(torch.any(point == coordinates) for point in model.positions))
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            model.initialize_positions(coordinates * 100)
        # Repeated common grids remain one bounded candidate set.
        candidates, _, _ = training_candidates(coordinates[:1].expand(100, -1, -1), max_candidates=8)
        self.assertLessEqual(len(candidates), 8)

    def test_zero_initialization_is_exact_without_jitter(self):
        model = self.model(anchor_init="zero")
        with self.assertWarnsRegex(UserWarning, "coincide"):
            model.initialize_positions(torch.rand(2, 9, 2))
        self.assertEqual(model.positions.count_nonzero().item(), 0)
        jittered = self.model(anchor_init="zero", init_jitter=0.01)
        jittered.initialize_positions(torch.rand(2, 9, 2))
        self.assertGreater(jittered.positions.count_nonzero().item(), 0)

    def test_file_init_checkpoint_replay_does_not_require_file_or_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.npy"
            positions = np.arange(12, dtype=np.float32).reshape(6, 2) / 12
            np.save(path, positions)
            model = self.model(anchor_init="file", anchor_path=str(path), learnable_pos=True,
                               weight_mode="distance_gate", fusion="hybrid")
            x, z = torch.rand(2, 9, 2), torch.rand(2, 5)
            model.initialize_positions(x)
            torch.testing.assert_close(model.initial_positions, torch.from_numpy(positions))
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            path.unlink()
            with patch("anchormix.model.farthest_points", side_effect=AssertionError("FPS on restore")), patch(
                    "anchormix.model.read_positions", side_effect=AssertionError("file on restore")):
                restored = self.model(anchor_init="file", anchor_path=str(path), learnable_pos=True,
                                      weight_mode="distance_gate", fusion="hybrid")
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                model.eval(); restored.eval()
                torch.testing.assert_close(restored(x, z), model(x, z), rtol=0, atol=0)

    def test_cli_booleans_precedence_and_checkpoint_structure(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(["--representation", "anchormix", "--num-anchors", "9", "--learnable-pos", "False",
                                  "--attention-power", "3", "--fusion", "gated",
                                  "--fusion-gate-init", "0.1", "--fusion_gate_hidden_dim", "16"])
        for key, value in dict(dataset="airfoil", mode="smoke",
                               ntrain=2, ntest=1, epochs=1, batch_size=None, device="cpu",
                               encode_batch_size=None).items():
            setattr(args, key, value)
        config = OmegaConf.load("run_codelib/config.yaml")
        config.anchormix.num_anchors = 12
        config.anchormix.learnable_pos = True
        config.anchormix.kernel_sigma = 0.7
        config.anchormix.attention_power = 2.0
        configure_request(config, config.datasets.airfoil, args)
        cfg = stage_config(config, "inr", args)
        self.assertEqual(cfg.inr_out.anchormix.num_anchors, 9)
        self.assertFalse(cfg.inr_out.anchormix.learnable_pos)
        self.assertEqual(cfg.inr_out.anchormix.kernel_sigma, 0.7)
        self.assertEqual(cfg.inr_out.anchormix.attention_power, 3.0)
        self.assertEqual(cfg.inr_out.anchormix.fusion, "gated")
        self.assertEqual(cfg.inr_out.anchormix.fusion_gate_init, 0.1)
        self.assertEqual(cfg.inr_out.anchormix.fusion_gate_hidden_dim, 16)
        self.assertEqual(cfg.inr_in.model_type, "siren")
        validate_overrides(args, cfg)
        args.num_anchors = 10
        with self.assertRaisesRegex(ValueError, "Incompatible CLI num_anchors"):
            validate_overrides(args, cfg)
        args.num_anchors = None
        args.query_chunk_size = 2
        validate_overrides(args, cfg)
        for options in ({"kernel_sigma": 0}, {"num_neighbors": 65}, {"learnable_pos": "False"},
                        {"pos_lr_scale": -1}, {"init_jitter": float("nan")}, {"num_anchors": 0},
                        {"attention_power": 0}, {"attention_power": -1}, {"attention_power": True},
                        {"attention_power": float("inf")}, {"attention_power": float("nan")},
                        {"fusion_gate_init": 0}, {"fusion_gate_init": 1}, {"fusion_gate_init": True},
                        {"fusion_gate_init": float("nan")}, {"fusion_gate_hidden_dim": 0}):
            with self.assertRaises(ValueError):
                AnchorMixConfig(**options).validate()

    def test_legacy_gridmix_checkpoint_still_loads_and_differentiates(self):
        settings = dict(model_type="siren_GridMix", use_latent=True, latent_dim=5, depth=3,
                        hidden_dim=12, w0=10, modulate_scale=False, modulate_shift=True,
                        hypernet_width=16, hypernet_depth=1, last_activation=None, grid_base=4, grid_size=3)
        cfg = OmegaConf.create(dict(inr=settings, inr_in=settings, inr_out=settings,
                                    data={"ntrain": 2}))
        model = create_inr_instance(cfg, 2, 2, "cpu")
        checkpoint = {"cfg": cfg, "inr_in": model.state_dict(), "inr_out": model.state_dict(),
                      "alpha_in": torch.tensor([.01]), "alpha_out": torch.tensor([.01])}
        restored = load_inrs(checkpoint, "cpu", 2, 2, 2)
        x, z = torch.rand(2, 7, 2), torch.rand(2, 5)
        torch.testing.assert_close(model.modulated_forward(x, z), restored["out"]["model"].modulated_forward(x, z))
        result = outer_step(model, x, torch.rand(2, 7, 2), 2, .01, is_train=True,
                            modulations=torch.zeros(2, 5))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(model.grid_bases.grad).all())

    def test_resume_reuses_anchors_rng_and_optimizer_exactly(self):
        import wandb
        from static.design_inr_shared import main
        from run_codelib.representation import resume_config
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(["--representation", "anchormix", "--learnable-pos", "true",
                                  "--num-anchors", "4", "--num-neighbors", "2",
                                  "--read-mode", "all", "--attention-power", "2", "--fusion", "gated"])
        for key, value in dict(dataset="airfoil", mode="smoke",
                               ntrain=3, ntest=2, epochs=1, batch_size=2, device="cpu",
                               encode_batch_size=2).items():
            setattr(args, key, value)
        config = OmegaConf.load("run_codelib/config.yaml")
        configure_request(config, config.datasets.airfoil, args)
        cfg = stage_config(config, "inr", args)
        for kind in ("in", "out"):
            settings = cfg[f"inr_{kind}"]
            settings.hidden_dim, settings.depth, settings.latent_dim = 8, 3, 5
            settings.grid_base = 3
        train_x, test_x = torch.rand(3, 8, 2), torch.rand(2, 8, 2)
        data = (torch.rand(3, 8, 2), torch.rand(3, 8, 1),
                torch.rand(2, 8, 2), torch.rand(2, 8, 1), train_x, test_x)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, WANDB_MODE="disabled", WANDB_DIR=directory), patch(
                "static.design_inr_shared.get_operator_data", return_value=data):
            try:
                cfg.wandb.name = "split"
                main.__wrapped__(cfg)
                wandb.finish()
                path = Path(directory) / "airfoil/inr/split.last.pt"
                args.resume, args.epochs = str(path), 2
                resumed_cfg = resume_config(args, cfg)
                resumed_cfg.wandb.name = "resumed"
                with patch("anchormix.model.farthest_points", side_effect=AssertionError("FPS on resume")):
                    main.__wrapped__(resumed_cfg)
                wandb.finish()
                cfg.optim.epochs = 2
                cfg.wandb.name = "continuous"
                main.__wrapped__(cfg)
                resumed = torch.load(Path(directory) / "airfoil/inr/resumed.last.pt", weights_only=False)
                continuous = torch.load(Path(directory) / "airfoil/inr/continuous.last.pt", weights_only=False)
                for kind in ("in", "out"):
                    for key, value in resumed[f"inr_{kind}"].items():
                        torch.testing.assert_close(value, continuous[f"inr_{kind}"][key], rtol=0, atol=0)
                    torch.testing.assert_close(resumed[f"alpha_{kind}"], continuous[f"alpha_{kind}"], rtol=0, atol=0)
            finally:
                wandb.finish()


if __name__ == "__main__":
    unittest.main()
