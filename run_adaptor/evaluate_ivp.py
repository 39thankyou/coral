"""Frozen first-to-last-frame inference on variable-size MeshGraphNets meshes."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from coral.losses import batch_mse_fn, batch_mse_rel_fn
from coral.metalearning import graph_outer_step
from coral.mlp import ResNet
from coral.utils.data.load_data import set_seed
from coral.utils.models.load_inr import create_inr_instance
from run_adaptor.evaluate_airfoil import absolute
from run_adaptor.raw_cylinder_dataset import RawCylinderFlowDataset


def parse_args(dataset="cylinder_flow"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--mode", choices=("author", "smoke"), default="author")
    for name in ("regression-checkpoint", "inr-checkpoint", "output-root", "data-dir", "output"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--ntest", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.dataset = dataset
    return args


def fit_graph(inr, alpha, graph, values, latent_dim, steps):
    work = graph.clone()
    work.pos = graph.pos[..., 0]
    work.images = values
    work.modulations = values.new_zeros(graph.num_graphs, latent_dim)
    return graph_outer_step(inr, work, steps, alpha, is_train=False,
                            gradient_checkpointing=False, loss_type="mse")


def evaluate(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    config = OmegaConf.load(absolute(args.config))
    dataset = config.datasets[args.dataset]
    name = dataset.author_name
    suffix = "-smoke" if args.mode == "smoke" else ""
    output_dir = absolute(args.output_root or config.common.output_root) / name
    regression_path = absolute(args.regression_checkpoint) if args.regression_checkpoint else output_dir / "model" / f"{dataset.downstream_run_name}{suffix}.pt"
    # Old static_regression.py saved its model in inr/ by mistake.
    if not regression_path.exists() and not args.regression_checkpoint:
        regression_path = output_dir / "inr" / regression_path.name
    if args.regression_checkpoint and not args.output_root:
        output_dir = regression_path.parent.parent
    saved = torch.load(regression_path, map_location="cpu", weights_only=False)
    cfg = saved["cfg"]
    inr_path = absolute(args.inr_checkpoint) if args.inr_checkpoint else output_dir / "inr" / f"{cfg.inr.run_name}.pt"
    checkpoint = torch.load(inr_path, map_location="cpu", weights_only=False)
    inr_cfg = checkpoint["cfg"]
    if cfg.data.dataset_name != name or inr_cfg.data.dataset_name != name:
        raise ValueError("Checkpoints belong to a different IVP dataset")
    ntest = int(args.ntest if args.ntest is not None else cfg.data.ntest)
    if not 1 <= ntest <= min(int(cfg.data.ntest), int(inr_cfg.data.ntest)):
        raise ValueError("ntest must be a prefix within both saved test splits")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(int(inr_cfg.data.seed))
    device = torch.device(args.device)
    data_dir = absolute(args.data_dir or dataset.data_dir)
    test = RawCylinderFlowDataset(data_dir, "test", ntest, dataset_name=name)
    channels = test[0][0].images.shape[-1]
    inrs = {}
    for kind in ("in", "out"):
        inr_cfg.inr = inr_cfg[f"inr_{kind}"]
        net = create_inr_instance(inr_cfg, 2, channels, device)
        net.load_state_dict(checkpoint[f"inr_{kind}"])
        inrs[kind] = net.eval().requires_grad_(False)
    model = ResNet(input_dim=inr_cfg.inr_in.latent_dim, output_dim=inr_cfg.inr_out.latent_dim,
                   hidden_dim=cfg.model.width, depth=cfg.model.depth,
                   dropout=cfg.model.dropout, activation=cfg.model.activation).to(device)
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    stats = {key: saved[key].detach().to(device) for key in ("mu", "sigma", "mu_u", "sigma_u")}
    if any(not torch.isfinite(v).all() for v in stats.values()) or (stats["sigma"] <= 0).any() or (stats["sigma_u"] <= 0).any():
        raise ValueError("Invalid saved code normalization")
    alpha = {kind: checkpoint[f"alpha_{kind}"].detach().to(device) for kind in inrs}
    errors, mse, recon_in, recon_out = [], [], [], []
    for graph, _ in DataLoader(test, batch_size=args.batch_size, shuffle=False):
        graph = graph.to(device)
        fitted = fit_graph(inrs["in"], alpha["in"], graph, graph.input[:, 2:], int(inr_cfg.inr_in.latent_dim), int(cfg.inr.inner_steps))
        with torch.no_grad():
            code = model((fitted["modulations"] - stats["mu"]) / stats["sigma"])
            prediction = inrs["out"].modulated_forward(graph.pos[..., 0], (code*stats["sigma_u"]+stats["mu_u"])[graph.batch])
            reconstruction_in = inrs["in"].modulated_forward(graph.pos[..., 0], fitted["modulations"][graph.batch])
        output_fit = fit_graph(inrs["out"], alpha["out"], graph, graph.images, int(inr_cfg.inr_out.latent_dim), int(cfg.inr.inner_steps))
        with torch.no_grad():
            reconstruction_out = inrs["out"].modulated_forward(graph.pos[..., 0], output_fit["modulations"][graph.batch])
        for start, stop in zip(graph.ptr[:-1], graph.ptr[1:]):
            section = slice(int(start), int(stop))
            target = graph.images[section].unsqueeze(0)
            errors.append(batch_mse_rel_fn(prediction[section].unsqueeze(0), target).item())
            mse.append(batch_mse_fn(prediction[section].unsqueeze(0), target).item())
            recon_in.append(batch_mse_rel_fn(reconstruction_in[section].unsqueeze(0), graph.input[section, 2:].unsqueeze(0)).item())
            recon_out.append(batch_mse_rel_fn(reconstruction_out[section].unsqueeze(0), target).item())
        print(f"evaluate IVP: {len(errors)}/{ntest}", flush=True)
    summary = {"dataset": name, "mode": args.mode, "ntest": ntest, "test_indices": [0, ntest],
               "inr_checkpoint": str(inr_path), "regression_checkpoint": str(regression_path),
               "inr_epoch_zero_based": int(checkpoint["epoch"]), "regression_epoch_zero_based": int(saved["epoch"]),
               "relative_l2": sum(errors)/ntest, "relative_l2_percent": 100*sum(errors)/ntest,
               "mse": sum(mse)/ntest, "per_sample_relative_l2": errors, "per_sample_mse": mse,
               "inr_in_relative_l2": sum(recon_in)/ntest, "inr_out_relative_l2": sum(recon_out)/ntest,
               "data_dir": str(data_dir), "prediction": "First observed frame to last frame; no target-derived code enters prediction",
               "normalization_source": "Saved regression training-code statistics", "model_weights_updated": False}
    path = absolute(args.output) if args.output else output_dir / f"test_metrics{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"test_relative_l2={summary['relative_l2']:.8g} metrics={path}", flush=True)
    return summary
