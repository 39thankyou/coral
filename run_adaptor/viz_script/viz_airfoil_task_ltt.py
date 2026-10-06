"""Plot Airfoil input/output latent modulations in two separate t-SNE panels.

Run from the repository root:
    python run_adaptor/viz_script/viz_airfoil_task_ltt.py

Each point is one case. Both panels jointly embed train and test, but the input
and output latent spaces are fitted independently. Output codes encode observed
target fields; they are not predictions of the regression network.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from coral.utils.data.load_data import get_operator_data
from coral.utils.models.load_inr import create_inr_instance
from run_adaptor.viz_script._modulation_tsne import (
    encode_batch, load_codes, load_context, make_parser, output_directory, plt,
    project_joint, save_codes, save_figure, save_metadata, scatter_splits, validate_args,
)


def extract_codes(args):
    checkpoint, _, _, data_dir, counts, steps, metadata = load_context(args, "airfoil")
    cfg = OmegaConf.create(OmegaConf.to_container(checkpoint["cfg"], resolve=True))
    x_train, y_train, x_test, y_test, grid_train, grid_test = get_operator_data(
        data_dir, "airfoil", counts["ntrain"], counts["ntest"],
        sub_tr=1, sub_te=1, same_grid=True,
    )
    arrays = {}
    # Process networks sequentially to keep memory use small.
    for kind, train, test in (("in", x_train, x_test), ("out", y_train, y_test)):
        cfg.inr = cfg[f"inr_{kind}"]
        latent_dim = int(cfg.inr.latent_dim)
        inr = create_inr_instance(cfg, input_dim=grid_train.shape[-1],
                                  output_dim=train.shape[-1], device=args.device)
        inr.load_state_dict(checkpoint[f"inr_{kind}"])
        inr.eval().requires_grad_(False)
        alpha = torch.as_tensor(checkpoint[f"alpha_{kind}"]).detach().to(args.device)
        for split, values, coords in (("train", train, grid_train), ("test", test, grid_test)):
            if len(values) != counts[f"n{split}"]:
                raise ValueError(f"Unexpected {split} case count")
            codes = []
            for start in range(0, len(values), args.batch_size):
                stop = min(start + args.batch_size, len(values))
                codes.append(encode_batch(inr, values[start:stop], coords[start:stop],
                                          latent_dim, alpha, steps, args.device))
                if start // args.batch_size % 25 == 0 or stop == len(values):
                    print(f"Airfoil {kind}/{split}: {stop}/{len(values)} cases", flush=True)
            arrays[f"z_{kind}_{split}"] = torch.cat(codes).numpy()
        del inr
    # The author loader uses the fixed [0:1000] / [1000:1200] dataset split.
    arrays["case_id_train"] = np.arange(counts["ntrain"])
    arrays["case_id_test"] = np.arange(1000, 1000 + counts["ntest"])
    metadata["output_code_source"] = "Observed output field encoded with output INR (not regressor prediction)"
    metadata["case_id_definition"] = "Zero-based index in the original NACA arrays"
    return arrays, metadata


def visualize(args):
    validate_args(args)
    output_dir = output_directory(args, "airfoil")
    arrays, source = (load_codes(args.codes_file, "airfoil") if args.codes_file else extract_codes(args))
    ntrain, ntest = len(arrays["case_id_train"]), len(arrays["case_id_test"])
    for kind in ("in", "out"):
        if len(arrays[f"z_{kind}_train"]) != ntrain or len(arrays[f"z_{kind}_test"]) != ntest:
            raise ValueError("Input/output codes and case IDs must remain paired")
    save_codes(output_dir / "codes.npz", arrays, source)
    embeddings, tsne_metadata = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, kind, title in zip(axes, ("in", "out"), ("Input modulation", "Output modulation")):
        embedding, info = project_joint(arrays[f"z_{kind}_train"], arrays[f"z_{kind}_test"], args)
        embeddings[kind], tsne_metadata[kind] = embedding, info
        scatter_splits(ax, embedding, ntrain)
        ax.set_title(f"{title}  |  dim={arrays[f'z_{kind}_train'].shape[1]}")
    fig.suptitle("Airfoil: one point per case\nIndependent input/output embeddings; train and test fitted jointly per panel",
                 fontsize=12)
    save_figure(fig, output_dir, "airfoil_modulations_tsne")
    split = np.array(["train"] * ntrain + ["test"] * ntest)
    case_ids = np.concatenate((arrays["case_id_train"], arrays["case_id_test"]))
    np.savez_compressed(output_dir / "embedding.npz", split=split, case_id=case_ids,
                        input_xy=embeddings["in"], output_xy=embeddings["out"])
    with (output_dir / "embedding.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "case_id", "input_tsne_1", "input_tsne_2", "output_tsne_1", "output_tsne_2"))
        for label, case_id, xy_in, xy_out in zip(split, case_ids, embeddings["in"], embeddings["out"]):
            writer.writerow((label, int(case_id), *xy_in, *xy_out))
    save_metadata(output_dir, {"source": source, "tsne": tsne_metadata,
                              "interpretation": "Axes of the two independently fitted latent spaces are not aligned"})
    return output_dir


if __name__ == "__main__":
    visualize(make_parser(__doc__, batch_size=4).parse_args())
