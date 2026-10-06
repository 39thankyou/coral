"""Small CUDA benchmark; no dataset loading, optimizer updates or full training.

By default compare the vectorized read with the former neighbor loop inside the
same decoder. --baseline-source can instead load a saved pre-change model.py to
measure all forward changes together, including layout copies and CPU syncs.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from anchormix.metalearning import outer_step
from anchormix.model import AnchorMix


class NeighborLoopReference(AnchorMix):
    def _read(self, values, weights, indices):
        if indices is None:
            result = weights @ values
        else:
            result = 0
            for neighbor in range(indices.shape[-1]):
                selected = values.gather(1, indices[..., neighbor, None].expand(-1, -1, values.shape[-1]))
                result = result + weights[..., neighbor, None] * selected
        return result.reshape(*weights.shape[:2], self.modulated_layers, self.modulation_channels)


def measure_pair(functions, warmup, repeats):
    for _ in range(warmup):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    measurements = {name: {"cuda_ms": [], "wall_ms": [], "extra_peak_mib": []} for name in functions}
    # Alternate the order to reduce GPU clock / warmup bias.
    for repeat in range(repeats):
        names = list(functions) if repeat % 2 == 0 else list(reversed(functions))
        for name in names:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            functions[name]()
            stop.record()
            stop.synchronize()
            measurements[name]["wall_ms"].append((time.perf_counter() - wall) * 1000)
            measurements[name]["cuda_ms"].append(start.elapsed_time(stop))
            measurements[name]["extra_peak_mib"].append((torch.cuda.max_memory_allocated() - allocated) / 2**20)
    return {name: {key: statistics.median(values) for key, values in metrics.items()}
            for name, metrics in measurements.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--points", type=int, default=972)
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--num-neighbors", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--read-mode", choices=("knn", "all"), default="knn")
    parser.add_argument("--weight-mode", choices=("distance", "distance_gate"), default="distance")
    parser.add_argument("--fusion", choices=("anchor", "global", "hybrid"), default="anchor")
    parser.add_argument("--learnable-pos", action="store_true")
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.points, args.inner_steps, args.warmup, args.repeats) < 1:
        parser.error("counts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this benchmark")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(17)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    baseline_class = NeighborLoopReference
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location("anchormix._benchmark_baseline", args.baseline_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline_class = module.AnchorMix
    options = {key: getattr(args, key) for key in ("num_anchors", "num_neighbors", "query_chunk_size",
                                                 "read_mode", "weight_mode", "fusion", "learnable_pos")}
    kwargs = dict(dim_in=2, dim_hidden=256, dim_out=1, num_layers=4, grid_base=64,
                  use_latent=True, latent_dim=128, w0=15, w0_initial=15, anchor_config=options)
    coords = torch.rand(args.batch_size, args.points, 2)
    current = AnchorMix(**kwargs)
    current.initialize_positions(coords)
    baseline = baseline_class(**kwargs)
    baseline.load_state_dict(current.state_dict())
    models = {"before": baseline.cuda(), "after": current.cuda()}
    coords = coords.cuda()
    latent = torch.randn(args.batch_size, 128, device="cuda") * 0.01
    target = torch.rand(args.batch_size, args.points, 1, device="cuda")
    with torch.no_grad():
        expected, actual = (model(coords, latent) for model in models.values())
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        forward_error = (actual - expected).abs().max().item()

    def forward(model):
        with torch.no_grad():
            model(coords, latent)

    def meta_step(model):
        model.zero_grad(set_to_none=True)
        alpha = torch.tensor([0.01], device="cuda", requires_grad=True)
        result = outer_step(model, coords, target, args.inner_steps, alpha, is_train=True,
                            modulations=torch.zeros_like(latent))
        result["loss"].backward()
        model.zero_grad(set_to_none=True)

    # Validate CUDA higher-order gradients outside the timed region.
    meta_results = []
    for model in models.values():
        model.zero_grad(set_to_none=True)
        alpha = torch.tensor([0.01], device="cuda", requires_grad=True)
        result = outer_step(model, coords, target, args.inner_steps, alpha, is_train=True,
                            modulations=torch.zeros_like(latent))
        result["loss"].backward()
        gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters() if value.grad is not None}
        gradients["inner_lr"] = alpha.grad.detach().clone()
        meta_results.append((result["loss"].detach(), gradients))
        model.zero_grad(set_to_none=True)
        del result
    torch.testing.assert_close(meta_results[0][0], meta_results[1][0], rtol=1e-4, atol=1e-5)
    assert meta_results[0][1].keys() == meta_results[1][1].keys()
    gradient_error = 0.0
    for name, expected in meta_results[0][1].items():
        actual = meta_results[1][1][name]
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5, msg=lambda msg: f"{name}: {msg}")
        gradient_error = max(gradient_error, (actual - expected).abs().max().item())
    del meta_results

    results = {}
    for name, operation in (("forward", forward), ("adaptation_and_backward", meta_step)):
        results[name] = measure_pair({key: lambda model=model: operation(model) for key, model in models.items()},
                                     args.warmup, args.repeats)
        results[name]["speedup_wall"] = results[name]["before"]["wall_ms"] / results[name]["after"]["wall_ms"]
        print(f"{name}: {json.dumps(results[name])}", flush=True)
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "baseline": str(args.baseline_source) if args.baseline_source else "neighbor loop in the same decoder",
              "shape": {"batch": args.batch_size, "points": args.points, "M": 64, "L": 3, "C": 256, "latent": 128},
              "options": options, "inner_steps": args.inner_steps, "warmup": args.warmup, "repeats": args.repeats,
              "max_abs_forward_difference": forward_error, "max_abs_meta_gradient_difference": gradient_error,
              "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
