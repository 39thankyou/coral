"""Compare input-neighbor output-code interpolation with the trained mapper.

Weights are inverse raw input-latent distances. Only training output codes are
used by the baseline. Observed test-output reconstruction is a separate reference.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from run_adaptor.latent_analysis_script.airfoil._common import (
    decode, field_errors, inverse_distance_weights, load_analysis, make_inr,
    make_parser, mapper_codes, nearest_neighbors, output_directory, plt,
    save_codes, save_figure, save_metadata, summarize, write_csv,
)


def analyze(args):
    arrays, source, data, checkpoint, mapper = load_analysis(args, require_mapper=True)
    directory = output_directory(args, "local_interpolation")
    neighbors, distances = nearest_neighbors(arrays["z_in_train"], arrays["z_in_test"], args.neighbors)
    weights = inverse_distance_weights(distances)
    interpolated = np.einsum("nk,nkl->nl", weights, arrays["z_out_train"][neighbors]).astype(np.float32)
    mapped, mean, std = mapper_codes(mapper, arrays, args)
    inr = make_inr(checkpoint, "out", args.device, data[3].shape[-1])
    target, coordinates = data[3].numpy(), data[5]
    predictions = {"local_interpolation": interpolated, "mapper": mapped,
                   "observed_output_reconstruction": arrays["z_out_test"]}
    rows, summary, errors = [], {}, {}
    for method, codes in predictions.items():
        prediction = decode(inr, codes, coordinates, args)
        mse, relative_l2 = field_errors(prediction, target)
        errors[method] = relative_l2
        summary[method] = {"mse": summarize(mse), "relative_l2": summarize(relative_l2)}
        rows.extend({"case_id": int(case_id), "method": method,
                     "mse": float(m), "relative_l2": float(r)}
                    for case_id, m, r in zip(arrays["case_id_test"], mse, relative_l2))
        print(f"{method}: relative_l2={relative_l2.mean():.8f}, mse={mse.mean():.8g}", flush=True)
    summary["paired_interpolation_minus_mapper_relative_l2"] = summarize(
        errors["local_interpolation"] - errors["mapper"])
    neighbor_rows = [
        {"query_case_id": int(arrays["case_id_test"][query]), "rank": rank + 1,
         "train_case_id": int(arrays["case_id_train"][neighbor]),
         "input_latent_distance": float(distances[query, rank]), "weight": float(weights[query, rank])}
        for query, selected in enumerate(neighbors) for rank, neighbor in enumerate(selected)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    axes[0].boxplot(list(errors.values()), tick_labels=["Local interpolation", "Mapper", "Observed-output\nreconstruction"],
                    showfliers=False)
    axes[0].set_ylabel("Per-case relative L2")
    axes[1].scatter(errors["mapper"], errors["local_interpolation"], s=18, alpha=0.65)
    limit = max(float(errors["mapper"].max()), float(errors["local_interpolation"].max()), 1e-12) * 1.05
    axes[1].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[1].set(xlabel="Mapper relative L2", ylabel="Local interpolation relative L2",
                xlim=(0, limit), ylim=(0, limit))
    fig.suptitle(f"Airfoil test predictions | {args.neighbors} training neighbors | inverse-distance weights")
    save_codes(directory / "codes.npz", arrays, source)
    np.savez_compressed(directory / "predicted_codes.npz", case_id=arrays["case_id_test"],
                        z_interpolated=interpolated, z_mapper=mapped,
                        neighbor_case_id=arrays["case_id_train"][neighbors], distances=distances,
                        weights=weights, mapper_input_mean=mean, mapper_input_std=std)
    write_csv(directory / "per_case_metrics.csv", rows)
    write_csv(directory / "neighbors.csv", neighbor_rows)
    save_figure(fig, directory, "interpolation_vs_mapper")
    save_metadata(directory, {
        "source": source, "neighbors": args.neighbors,
        "search": "Raw 128-D input latent Euclidean distance; training candidates only",
        "weights": "Normalized 1/d; exact matches share all weight equally",
        "mapper_normalization": "Full training input codes: torch.mean / torch.std(correction=1); output mean=0, std=1",
        "metrics": "Mean per-case MSE and ||prediction-target||_2 / max(||target||_2, float64 epsilon)",
        "observed_output_reconstruction": "Diagnostic reference using encoded test targets; excluded from both predictive methods",
        "summary": summary,
    })
    print(f"results={directory}", flush=True)
    return directory


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--mapper-checkpoint", help="Override the saved downstream mapper checkpoint")
    analyze(parser.parse_args())
