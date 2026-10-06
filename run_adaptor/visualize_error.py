"""Plot static INR reconstructions and absolute errors from the eval loop."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def select_test_indices(ntest, k=10):
    """Select distinct, evenly spaced test cases, including both endpoints."""
    if ntest < 1 or k < 1:
        raise ValueError("ntest and visualization-k must be positive")
    return np.rint(np.linspace(0, ntest - 1, min(k, ntest))).astype(int).tolist()


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _draw(axis, field, coordinates, shape, title, cmap, vmin, vmax):
    if shape is not None:
        artist = axis.imshow(
            field.reshape(shape).T, origin="lower", interpolation="nearest",
            aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        )
    else:
        artist = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=field,
            s=max(2.0, min(18.0, 6000.0 / len(field))), linewidths=0,
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)


def save_inr_error_case(dataset, test_index, case_id, input_gt, input_reconstruction,
                        output_gt, output_reconstruction, coordinates, output_dir,
                        suffix=""):
    """Save one 2x3 figure per channel pair; output codes fit observed targets.

    Structured fields use the full computational grid. Unstructured fields
    use the observed input geometry for every panel. GT/reconstruction share
    a color scale within each row; errors use |reconstruction - GT|.
    """
    input_gt, input_reconstruction, output_gt, output_reconstruction = map(
        _numpy, (input_gt, input_reconstruction, output_gt, output_reconstruction)
    )
    if input_gt.shape != input_reconstruction.shape or output_gt.shape != output_reconstruction.shape:
        raise ValueError("Reconstruction dimensions must match their observed fields")
    fields = (input_gt, input_reconstruction, output_gt, output_reconstruction)
    if not all(np.isfinite(field).all() for field in fields):
        raise ValueError("Cannot visualize non-finite INR fields")
    shape = tuple(input_gt.shape[:-1]) if input_gt.ndim == 3 else None
    coordinates = _numpy(coordinates).reshape(-1, 2)
    if shape is None and input_gt.shape[-1] == 2:
        coordinates = input_gt.reshape(-1, 2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for index in range(max(input_gt.shape[-1], output_gt.shape[-1])):
        input_channel = min(index, input_gt.shape[-1] - 1)
        output_channel = min(index, output_gt.shape[-1] - 1)
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        relative_errors = {}
        try:
            for row, name, gt, reconstruction, channel in (
                (0, "INR-in", input_gt, input_reconstruction, input_channel),
                (1, "INR-out", output_gt, output_reconstruction, output_channel),
            ):
                observed = gt[..., channel].reshape(-1)
                fitted = reconstruction[..., channel].reshape(-1)
                error = np.abs(fitted - observed)
                vmin = float(min(observed.min(), fitted.min()))
                vmax = float(max(observed.max(), fitted.max()))
                if vmin == vmax:
                    vmax = vmin + max(abs(vmin) * 1e-6, 1e-8)
                denominator = float(np.linalg.norm(observed))
                relative_error = float(np.linalg.norm(error) / denominator) if denominator else None
                relative_errors[name] = relative_error
                label = f"{name} | channel {channel + 1}"
                _draw(axes[row, 0], observed, coordinates, shape,
                      f"{label} | GT", "viridis", vmin, vmax)
                _draw(axes[row, 1], fitted, coordinates, shape,
                      f"{label} | INR reconstruction", "viridis", vmin, vmax)
                metric = f" | L2 {100 * relative_error:.4f}%" if relative_error is not None else ""
                _draw(axes[row, 2], error, coordinates, shape,
                      f"{name} | absolute error{metric}", "inferno", 0.0,
                      max(float(error.max()), 1e-12))
            figure.suptitle(f"{dataset} | test index {test_index} | original case {case_id}")
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            path = output_dir / (
                f"test_{test_index:04d}_case_{case_id:04d}{suffix}__"
                f"inr_in_channel_{input_channel + 1}__inr_out_channel_{output_channel + 1}.png"
            )
            figure.savefig(path, dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
        figures.append({
            "path": str(path), "input_channel": input_channel + 1,
            "output_channel": output_channel + 1,
            "input_relative_l2": relative_errors["INR-in"],
            "output_relative_l2": relative_errors["INR-out"],
        })
    return {"test_index": test_index, "case_id": case_id, "figures": figures}
