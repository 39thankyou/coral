"""Pure raw-data visualization for the author-aligned adaptor pipelines."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from coral.utils.data.graph_dataset import load_cylinder_data
from coral.utils.data.load_data import get_dynamics_data, get_operator_data


def _colorbar(axis, artist):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)


def _pixel(axis, values, shape, title, vmin=None, vmax=None):
    artist = axis.imshow(
        np.asarray(values).reshape(shape).T,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title)
    _colorbar(axis, artist)


def _scatter(axis, coordinates, values, title):
    values = np.asarray(values).reshape(-1)
    size = max(2.0, min(18.0, 6000.0 / len(values)))
    artist = axis.scatter(
        coordinates[:, 0], coordinates[:, 1], c=values, s=size,
        cmap="viridis", linewidths=0,
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    _colorbar(axis, artist)


def _save_static_pairs(coordinates, input_field, output_field, output_dir, shape=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    input_field = np.asarray(input_field)
    output_field = np.asarray(output_field)
    figures = {}
    for index in range(max(input_field.shape[-1], output_field.shape[-1])):
        input_channel = min(index, input_field.shape[-1] - 1)
        output_channel = min(index, output_field.shape[-1] - 1)
        figure, axes = plt.subplots(1, 2, figsize=(9, 4), squeeze=False)
        if shape is None:
            _scatter(
                axes[0, 0], coordinates, input_field[:, input_channel],
                f"INR-in data · channel {input_channel + 1}",
            )
            _scatter(
                axes[0, 1], coordinates, output_field[:, output_channel],
                f"INR-out data · channel {output_channel + 1}",
            )
        else:
            _pixel(
                axes[0, 0], input_field[:, input_channel], shape,
                f"INR-in data · channel {input_channel + 1}",
            )
            _pixel(
                axes[0, 1], output_field[:, output_channel], shape,
                f"INR-out data · channel {output_channel + 1}",
            )
        figure.tight_layout()
        path = output_dir / (
            f"inr_in_channel_{input_channel + 1}__"
            f"inr_out_channel_{output_channel + 1}.png"
        )
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        figures[f"input_{input_channel + 1}_output_{output_channel + 1}"] = str(path)
    return {
        "figures": figures,
        "point_count": int(len(coordinates)),
        "spatial_shape": list(shape) if shape else None,
        "rendering": "pixels" if shape else "scatter",
        "content": "dataset input/output only",
    }


def visualize_static(dataset_key, dataset_cfg, ntrain, output_dir):
    data_dir = Path(dataset_cfg["data_dir"])
    if dataset_key in ("cylinder_flow", "airfoil_flow"):
        from run_adaptor.raw_cylinder_dataset import RawCylinderFlowDataset
        graph, _ = RawCylinderFlowDataset(data_dir, "test", 1, dataset_name=dataset_cfg["author_name"])[0]
        inputs = [graph.input[:, 2:].numpy()]
        outputs = [graph.images.numpy()]
        coordinates = [graph.pos[..., 0].numpy()]
        return _save_static_pairs(
            coordinates[0], inputs[0], outputs[0], output_dir, shape=None
        )

    reader = dataset_cfg["author_name"]
    reader_ntrain = 1 if dataset_key == "airfoil" else ntrain
    _, _, x_test, y_test, _, grid_test = get_operator_data(
        data_dir, reader, reader_ntrain, 1,
        sub_tr=1, sub_te=1, same_grid=True,
    )
    shape = tuple(x_test.shape[1:-1]) if dataset_key == "airfoil" else None
    coordinates = grid_test[0].reshape(-1, grid_test.shape[-1]).cpu().numpy()
    input_field = x_test[0].reshape(-1, x_test.shape[-1]).cpu().numpy()
    output_field = y_test[0].reshape(-1, y_test.shape[-1]).cpu().numpy()
    return _save_static_pairs(
        coordinates, input_field, output_field, output_dir, shape=shape
    )


def visualize_navier_stokes(dataset_cfg, ntrain, ntest, steps_to_show, output_dir):
    data = get_dynamics_data(
        dataset_cfg["data_dir"], dataset_cfg["author_name"], ntrain, ntest,
        seq_inter_len=20, seq_extra_len=20, sub_from=1, sub_tr=1, sub_te=1,
        same_grid=True,
    )
    values = data[2]
    shape = tuple(values.shape[1:-2])
    values = values[0].reshape(-1, values.shape[-2], values.shape[-1])
    relative_steps = np.linspace(0, 19, min(steps_to_show, 20), dtype=np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for channel in range(values.shape[-2]):
        channel_values = values[:, channel].cpu().numpy()
        vmin, vmax = float(channel_values.min()), float(channel_values.max())
        figure, axes = plt.subplots(
            len(relative_steps), 2,
            figsize=(9, max(3.5, 3.3 * len(relative_steps))), squeeze=False,
        )
        for row, step in enumerate(relative_steps):
            _pixel(
                axes[row, 0], channel_values[:, step], shape,
                f"INR-in data · t={step} · channel {channel + 1}", vmin, vmax,
            )
            _pixel(
                axes[row, 1], channel_values[:, step + 20], shape,
                f"INR-out data · t={step + 20} · channel {channel + 1}", vmin, vmax,
            )
        figure.tight_layout()
        path = output_dir / f"inr_in_out_channel_{channel + 1}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        figures[f"channel_{channel + 1}"] = str(path)
    return {
        "figures": figures,
        "point_count": int(np.prod(shape)),
        "spatial_shape": list(shape),
        "rendering": "pixels",
        "time_pairs": [[int(t), int(t + 20)] for t in relative_steps],
        "content": "dataset input/output only",
    }
