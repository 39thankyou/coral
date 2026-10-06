"""Interpolate Airfoil input/output codes with frozen decoders on a fixed grid.

Each PNG has two rows (INR-in geometry / INR-out flow) and five columns:
origin case, target case, decoded z, abs from origin, abs from target.
ref=10 saves exactly 10 endpoint-inclusive frames. No scalar error evaluation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from run_adaptor.latent_analysis_script.airfoil._common import (
    decode, load_analysis, make_inr, make_parser, nearest_neighbors,
    output_directory, plt, save_metadata,
)


def geometry_errors(prediction, reference):
    """Pointwise vector magnitude; retain both physical x and y components."""
    return np.linalg.norm(prediction - reference, axis=-1)


def plot_geometry(axis, xy):
    # Draw a sparse curvilinear mesh and its inner boundary, including the wake.
    for index in range(0, xy.shape[0], 10):
        axis.plot(xy[index, :, 0], xy[index, :, 1], color="#2878B5", lw=0.45, alpha=0.65)
    for index in range(0, xy.shape[1], 4):
        axis.plot(xy[:, index, 0], xy[:, index, 1], color="#2878B5", lw=0.45, alpha=0.65)
    axis.plot(xy[:, 0, 0], xy[:, 0, 1], color="#172c44", lw=1.2)


def plot_field(axis, xy, field, limits, cmap):
    artist = axis.pcolormesh(xy[..., 0], xy[..., 1], field, shading="gouraud",
                            cmap=cmap, vmin=limits[0], vmax=limits[1], rasterized=True)
    axis.figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)


def plot_frame(origin, target, path_data, index, scales, view_bounds, title, path):
    fig, axes = plt.subplots(2, 5, figsize=(24, 9), layout="constrained")
    column_titles = ("Origin case", "Target case", "Decoded z", "Abs from org", "Abs from target")
    origin_xy, target_xy = origin["in"], target["in"]
    for axis, xy in zip(axes[0, :3], (origin_xy, target_xy, path_data["in"][index])):
        plot_geometry(axis, xy)
    for column, reference in ((3, "org"), (4, "target")):
        plot_field(axes[0, column], origin_xy, path_data[f"in_abs_{reference}"][index],
                   (0.0, scales["in_error_max"]), "magma")
    # True endpoint cases use their own geometry. Decoded fields and both error
    # maps use the fixed origin mesh; differences are aligned by computational index.
    for column, xy, field in ((0, origin_xy, origin["out"]),
                              (1, target_xy, target["out"]),
                              (2, origin_xy, path_data["out"][index])):
        plot_field(axes[1, column], xy, field, scales["out_limits"], "viridis")
    for column, reference in ((3, "org"), (4, "target")):
        plot_field(axes[1, column], origin_xy, path_data[f"out_abs_{reference}"][index],
                   (0.0, scales["out_error_max"]), "magma")
    for row, label in enumerate(("INR-in: geometry", "INR-out: flow")):
        for column, axis in enumerate(axes[row]):
            subtitle = (" | vector distance" if row == 0 and column >= 3 else "")
            axis.set(title=column_titles[column] + subtitle, xlabel="x", aspect="equal")
            axis.set_ylabel(f"{label}\ny" if column == 0 else "y")
            if view_bounds is not None:
                axis.set_xlim(view_bounds[:2])
                axis.set_ylim(view_bounds[2:])
    fig.suptitle(title + "\nFrozen INR-in / INR-out; common alpha and fixed computational queries", fontsize=14)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def visualize(args):
    if args.ref < 2 or args.num_cases <= 0:
        raise ValueError("ref must be >= 2 to include both endpoints; num-cases must be positive")
    view_bounds = None if args.full_domain else args.view_bounds
    if view_bounds is not None and (not np.isfinite(view_bounds).all()
                                   or view_bounds[0] >= view_bounds[1]
                                   or view_bounds[2] >= view_bounds[3]):
        raise ValueError("view-bounds must be finite xmin < xmax and ymin < ymax")
    arrays, source, data, checkpoint, _ = load_analysis(args)
    directory = output_directory(args, "ltt_replace")
    case_ids = arrays["case_id_test"]
    selected_ids = args.case_ids if args.case_ids is not None else case_ids[:args.num_cases].tolist()
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids).issubset(set(case_ids.tolist())):
        raise ValueError("case-ids must be unique original test IDs present in the loaded codes")
    if len(arrays["case_id_train"]) < 2:
        raise ValueError("Replacement requires at least two training donors")
    inrs = {kind: make_inr(checkpoint, kind, args.device, channels)
            for kind, channels in (("in", 2), ("out", 1))}
    neighbors, _ = nearest_neighbors(arrays[f"z_{args.neighbor_space}_train"],
                                      arrays[f"z_{args.neighbor_space}_test"], 1)
    rng = np.random.default_rng(args.seed)
    alphas = np.linspace(0.0, 1.0, args.ref)
    data_dir = Path(source["data_dir"]) / "naca"
    mesh_x = np.load(data_dir / "NACA_Cylinder_X.npy", mmap_mode="r")
    mesh_y = np.load(data_dir / "NACA_Cylinder_Y.npy", mmap_mode="r")
    # Invert exactly the official loader's normalization (fixed first 1000 cases),
    # so both geometry components and their errors are shown in physical units.
    lower = np.array([mesh_x[:1000].min(), mesh_y[:1000].min()])
    upper = np.array([mesh_x[:1000].max(), mesh_y[:1000].max()])
    extent = upper - lower
    if not np.isfinite(extent).all() or (extent <= 0).any():
        raise ValueError("Invalid training geometry normalization bounds")
    sequences = []
    for case_id in selected_ids:
        query = int(np.flatnonzero(case_ids == case_id)[0])
        nearest = int(neighbors[query, 0])
        random = int(rng.choice(np.delete(np.arange(len(arrays["case_id_train"])), nearest)))
        origin = {"in": data[2][query].numpy().astype(np.float64) * extent + lower,
                  "out": data[3][query, ..., 0].numpy()}
        coordinates = data[5][query:query + 1].expand(len(alphas), -1, -1, -1)
        paths = []
        for pairing, donor in (("nearest", nearest), ("random", random)):
            target = {"in": data[0][donor].numpy().astype(np.float64) * extent + lower,
                      "out": data[1][donor, ..., 0].numpy()}
            path_data = {"pairing": pairing, "donor": donor, "target": target}
            for kind in ("in", "out"):
                original_z = arrays[f"z_{kind}_test"][query]
                target_z = arrays[f"z_{kind}_train"][donor]
                codes = ((1 - alphas[:, None]) * original_z + alphas[:, None] * target_z).astype(np.float32)
                decoded = decode(inrs[kind], codes, coordinates, args)
                fields = decoded.astype(np.float64) * extent + lower if kind == "in" else decoded[..., 0]
                path_data[kind] = fields
                for reference, endpoint in (("org", origin), ("target", target)):
                    path_data[f"{kind}_abs_{reference}"] = (
                        geometry_errors(fields, endpoint[kind]) if kind == "in"
                        else np.abs(fields - endpoint[kind]))
            paths.append(path_data)
        # Per-row color ranges are fixed across both donor types and every alpha.
        flow_fields = [origin["out"]]
        for path_data in paths:
            flow_fields.extend((path_data["out"], path_data["target"]["out"]))
        field_min = min(float(field.min()) for field in flow_fields)
        field_max = max(float(field.max()) for field in flow_fields)
        scales = {"out_limits": (field_min, max(field_max, field_min + 1e-8))}
        for kind in ("in", "out"):
            scales[f"{kind}_error_max"] = max(
                1e-12, *(float(p[f"{kind}_abs_{reference}"].max())
                         for p in paths for reference in ("org", "target")))
        for path_data in paths:
            pairing, donor = path_data["pairing"], path_data["donor"]
            donor_id = int(arrays["case_id_train"][donor])
            subdirectory = directory / f"case_{case_id}" / f"{pairing}_train_{donor_id}" / f"ref_{args.ref}"
            subdirectory.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            for index, alpha in enumerate(alphas):
                path = subdirectory / f"frame_{index:03d}.png"
                plot_frame(origin, path_data["target"], path_data, index, scales, view_bounds,
                           f"Origin: test {case_id} -> Target: {pairing} train {donor_id} | "
                           f"z = (1 - alpha) z_org + alpha z_target | alpha={alpha:.3f}", path)
                frame_paths.append(str(path.relative_to(directory)))
            # Remove only obsolete generated frames in this regenerated sequence,
            # including frame_010.png left by the old ref+1 interpretation.
            expected = {Path(frame).name for frame in frame_paths}
            for stale in subdirectory.glob("frame_*.png"):
                if stale.stem.removeprefix("frame_").isdigit() and stale.name not in expected:
                    stale.unlink()
            sequences.append({"original_case_id": int(case_id), "donor_case_id": donor_id,
                              "pairing": pairing, "frames": frame_paths})
            print(f"sequence={subdirectory} ({len(alphas)} frames, 2 x 5 panels)", flush=True)
    save_metadata(directory, {
        "source": source, "seed": args.seed, "ref": args.ref, "alphas": alphas.tolist(),
        "ref_definition": "Number of frames including alpha=0 and alpha=1",
        "frames_per_sequence": len(alphas), "neighbor_space": args.neighbor_space,
        "rows": ["INR-in geometry", "INR-out flow"],
        "columns": ["origin case", "target case", "decoded z", "abs from org", "abs from target"],
        "search": "Nearest training case by raw 128-D Euclidean distance; same donor for both rows",
        "random_donor": "Uniform training case excluding the selected nearest donor",
        "interpolation": "For each of in/out: (1-alpha)*z_original + alpha*z_target; same alpha and paired cases",
        "fixed": "Both decoder weights and original computational query grid",
        "display_mesh": "Endpoint flow panels use each case's own mesh; decoded flow and all error maps use the fixed origin mesh",
        "color_scales": "Separate geometry/flow error scales; shared across both reference errors, all frames and both donors per origin",
        "physical_view_bounds": view_bounds,
        "geometry_normalization": {"min": lower.tolist(), "max": upper.tolist(), "training_cases": 1000},
        "error_images": {"in": "sqrt((decoded_x-reference_x)^2 + (decoded_y-reference_y)^2), physical coordinates",
                         "out": "abs(decoded_flow-reference_flow)",
                         "alignment": "Pointwise common computational grid index; no scalar error metrics"},
        "interpretation": "Alpha=0 reconstructs origin and alpha=1 reconstructs target in each latent space",
        "sequences": sequences,
    })
    print(f"results={directory}", flush=True)
    return directory


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument("--ref", type=int, default=10, help="Number of PNG frames including both endpoints (>=2)")
    parser.add_argument("--num-cases", type=int, default=3, help="Number of leading test cases when case-ids is omitted")
    parser.add_argument("--case-ids", type=int, nargs="+", help="Original test IDs, e.g. 1000 1005; overrides num-cases")
    parser.add_argument("--neighbor-space", choices=("in", "out"), default="out",
                        help="Latent space for donor selection; both input and output codes are interpolated")
    parser.add_argument("--view-bounds", type=float, nargs=4, default=[-0.5, 1.5, -0.75, 0.75],
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="Physical display window around the airfoil")
    parser.add_argument("--full-domain", action="store_true", help="Show the entire far-field mesh instead of the airfoil window")
    visualize(parser.parse_args())
