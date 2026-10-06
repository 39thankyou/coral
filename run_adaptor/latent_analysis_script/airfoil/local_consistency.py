"""Compare flow differences for raw 128-D latent neighbors and random pairs.

Both input and observed-output latent spaces are evaluated independently.
Queries are test cases; neighbors/random controls are drawn from training only.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from run_adaptor.latent_analysis_script.airfoil._common import (
    field_errors, load_analysis, make_parser, nearest_neighbors, output_directory,
    plt, save_codes, save_figure, save_metadata, summarize, write_csv,
)


def analyze(args):
    arrays, source, data, _, _ = load_analysis(args)
    directory = output_directory(args, "local_consistency")
    train, test = data[1].numpy(), data[3].numpy()
    rng = np.random.default_rng(args.seed)
    if not 1 <= args.neighbors <= len(train):
        raise ValueError(f"neighbors must be between 1 and {len(train)}")
    # Match the number of pairs for every query; use the same controls in both spaces.
    # Random sampling may include a true neighbor: it is an unbiased train-pair baseline.
    random_indices = np.stack([rng.choice(len(train), args.neighbors, replace=False)
                               for _ in test])
    rows, summary = [], {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for axis_row, kind in zip(axes, ("in", "out")):
        neighbors, distances = nearest_neighbors(arrays[f"z_{kind}_train"],
                                                  arrays[f"z_{kind}_test"], args.neighbors)
        groups = {}
        for pairing, indices in (("nearest", neighbors), ("random", random_indices)):
            group_rows = []
            for query, selected in enumerate(indices):
                targets = np.broadcast_to(test[query], train[selected].shape)
                mse, relative_l2 = field_errors(train[selected], targets)
                latent_distances = (distances[query] if pairing == "nearest" else
                                    np.linalg.norm(arrays[f"z_{kind}_train"][selected].astype(np.float64)
                                                   - arrays[f"z_{kind}_test"][query], axis=1))
                for rank, (neighbor, distance, error, relative) in enumerate(
                        zip(selected, latent_distances, mse, relative_l2), 1):
                    row = {"latent_space": kind, "pairing": pairing,
                           "query_case_id": int(arrays["case_id_test"][query]),
                           "train_case_id": int(arrays["case_id_train"][neighbor]),
                           "pair_index": rank, "latent_distance": float(distance),
                           "field_mse": float(error), "field_relative_l2": float(relative)}
                    rows.append(row)
                    group_rows.append(row)
            groups[pairing] = group_rows
        summary[kind] = {
            pairing: {metric: summarize([r[metric] for r in group])
                      for metric in ("latent_distance", "field_mse", "field_relative_l2")}
            for pairing, group in groups.items()
        }
        nearest_mean = np.array([r["field_relative_l2"] for r in groups["nearest"]]).reshape(len(test), -1).mean(1)
        random_mean = np.array([r["field_relative_l2"] for r in groups["random"]]).reshape(len(test), -1).mean(1)
        summary[kind]["paired_query_random_minus_nearest_relative_l2"] = summarize(random_mean - nearest_mean)
        summary[kind]["fraction_queries_nearest_better"] = float(np.mean(nearest_mean < random_mean))
        axis_row[0].boxplot([nearest_mean, random_mean], tick_labels=["Nearest", "Random"], showfliers=False)
        axis_row[0].set_ylabel("Mean relative L2 to query field (per test case)")
        axis_row[0].set_title(f"{kind} latent: {args.neighbors} pairs per query")
        for pairing, color in (("random", "#999999"), ("nearest", "#2878B5")):
            group = groups[pairing]
            axis_row[1].scatter([r["latent_distance"] for r in group],
                                [r["field_relative_l2"] for r in group],
                                s=9, alpha=0.35, color=color, label=pairing, rasterized=True)
        axis_row[1].set(xlabel="Euclidean distance in original 128-D latent space",
                        ylabel="Flow relative L2 to query")
        axis_row[1].legend()
    save_codes(directory / "codes.npz", arrays, source)
    write_csv(directory / "pairs.csv", rows)
    save_figure(fig, directory, "neighbor_vs_random")
    save_metadata(directory, {
        "source": source, "seed": args.seed, "neighbors": args.neighbors,
        "search": "test queries -> training pool; raw 128-D Euclidean; no normalization/projection",
        "control": "same number of uniform random training pairs without replacement per query; neighbors allowed",
        "field_comparison": "Observed flow values aligned by common computational grid index, not physical-coordinate resampling",
        "relative_l2": "||train_field-test_field||_2 / max(||test_field||_2, float64 epsilon)",
        "interpretation": "Output codes encode observed targets; output-space consistency is a reconstruction diagnostic, not predictive evidence",
        "summary": summary,
    })
    print(f"results={directory}", flush=True)
    return directory


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument("--neighbors", type=int, default=5)
    analyze(parser.parse_args())
