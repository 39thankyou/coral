七个数据集的 run/eval 入口、完整参数与下载顺序见
[DATASETS.md](DATASETS.md)。`airfoil` 表示 Geo-FNO NACA-Euler，
`airfoil_flow` 表示 MeshGraphNets Airfoil IVP。

# CORAL author-entrypoint adaptor

`run_adaptor/` is the strict reproduction path. It does not contain a unified
replacement training loop. Instead, it resolves the author's YAML plus the
overrides from the repository bash launchers and calls the original, separate
Hydra entrypoint functions.

| Dataset | Stage 1 | Stage 2 |
| --- | --- | --- |
| Airfoil | `static/design_inr.py` | `static/design_regression.py` |
| Elasticity | `static/design_inr.py` | `static/design_regression.py` |
| Cylinder Flow | `static/static_inr.py` | `static/static_regression.py` |
| Navier–Stokes | `inr/inr.py` | `dynamics_modeling/train.py` |

The Navier–Stokes stages are intentionally separate. Stage 1 learns one shared
INR for individual frames. Stage 2 reloads that checkpoint, infers trajectory
modulations with `load_dynamics_modulations`, and trains the author's latent
ODE with `ode_scheduling`.

## Compatibility layer

The adaptor changes no optimizer step, loss, scheduled-sampling rule, model, or
checkpoint schema. It only supplies the following runtime compatibility:

- fixes all Python/NumPy/PyTorch/CUDA random seeds and deterministic settings;
- runs W&B offline and gives each stage a deterministic checkpoint name;
- removes the obsolete logging-only `verbose` argument when the installed
  PyTorch no longer accepts it in `ReduceLROnPlateau`;
- permits loading the author's `DictConfig` checkpoints with current PyTorch;
- binds author data paths to this workspace;
- reads Cylinder Flow directly from TFRecord and builds the same first/last
  graph fields in memory, avoiding the author's generated `static_*.h5` cache.

The raw Cylinder adaptor is a storage adapter, not a different task: the author
training functions still receive `pos`, `p`, `v`, latent placeholders, graph
batch indices, and the first/last frame fields in their original layout.

## Configuration

[`config.yaml`](config.yaml) records:

- each original entrypoint and base YAML;
- the effective overrides from `bash_static/*` and
  `bash_dynamics/navier-stokes/{inr,ode}.sh`;
- deterministic run names that connect stage 1 to stage 2;
- explicit smoke settings of 8/4 samples and 5 epochs. The default mode is
  `author`; smoke mode must be requested with `--mode smoke`.

Resolved author configurations used by an actual run are saved under
`outputs/<author-dataset-name>/resolved_configs/`.

## Usage

From the repository root, using the requested environment:

```bash
~/miniconda3/envs/marble/bin/python run_adaptor/run_airfoil.py
~/miniconda3/envs/marble/bin/python run_adaptor/run_elasticity.py
~/miniconda3/envs/marble/bin/python run_adaptor/run_cylinder_flow.py
~/miniconda3/envs/marble/bin/python run_adaptor/run_navier_stokes.py
```

These commands use the full author sample counts and epoch counts by default.
For the earlier five-epoch connectivity check, request smoke mode explicitly:

```bash
~/miniconda3/envs/marble/bin/python run_adaptor/run_navier_stokes.py --mode smoke
```

Run stages separately when inspecting the author pipeline:

```bash
~/miniconda3/envs/marble/bin/python run_adaptor/run_navier_stokes.py --stage inr
~/miniconda3/envs/marble/bin/python run_adaptor/run_navier_stokes.py --stage ode
```

The downstream stage requires the deterministic stage-1 checkpoint. CLI
`--epochs`, `--ntrain`, and `--ntest` are explicit diagnostic overrides.

The launcher's automatic visualization contains raw data only. The independent
modulation scripts below inspect the learned representation after training.
Airfoil (`221×51`) and Navier–Stokes (`64×64`) use pixel images; Elasticity and
Cylinder use scatter plots. These launcher plots show raw data. Static eval
scripts additionally save INR reconstruction/error plots as described below.

## Evaluate saved Airfoil models

After both stages finish, evaluate the saved INR and regression checkpoints:

```bash
~/miniconda3/envs/marble/bin/python run_adaptor/evaluate_airfoil.py
```

The default `--mode author` uses the regression checkpoint's test sample count
(normally 200). Use `--mode smoke` for smoke checkpoints. `--ntest` can select a
smaller prefix of the fixed test split, and `--batch-size` controls evaluation
memory usage (default 4, matching the author's code-inference batch size).

The script freezes all model weights, recovers input-code normalization from
the regression training samples (the author checkpoint does not store these
statistics), and predicts each test field using only its input geometry. Test
targets are used for error measurement and separate output-INR reconstruction
diagnostics. It prints mean per-sample relative L2
error and MSE, and writes checkpoint provenance and per-sample errors to
`outputs/airfoil/test_metrics.json` (`test_metrics-smoke.json` in smoke mode).
It does not start a W&B run.

Airfoil, Elasticity and Pipe evaluators also select 10 evenly spaced test
cases, including the first and last, and save PNGs under the corresponding
`outputs/<dataset>/visualization_error/`. Each figure has two rows (INR-in,
INR-out) and three columns (GT, INR reconstruction, absolute pointwise error).
GT and reconstruction share a color scale within each row; errors have a
separate scale starting at zero. The input geometry's x/y channels produce
two figures per case, repeating the scalar output row. Airfoil/Pipe use full
computational-grid images; Elasticity uses scatter plots on the observed
case geometry. Values retain the dataset loader's normalization.

The INR-out row reconstructs the observed output from its independently
fitted latent code, using the saved inner-loop steps. It is a reconstruction
diagnostic, separate from the mapper's field prediction. Plotting reuses the
reconstructions already computed for the INR metrics. `manifest.json` and
the metrics JSON record selected zero-based test indices, original case IDs,
per-channel relative L2 and image paths. Smoke images and manifests include
`-smoke` in their filenames. Fewer than 10 test cases means all cases are
plotted; use `--visualization-k` to change the count:

```bash
~/miniconda3/envs/marble/bin/python run_adaptor/evaluate_elasticity.py --visualization-k 10
~/miniconda3/envs/marble/bin/python run_adaptor/evaluate_pipe.py --mode smoke
```

Do not interpret `metrics.json`'s `author_return_values.regression` or the
regression checkpoint's `loss` as final field prediction performance: the
author trains on code MSE, evaluates every 100 epochs starting at epoch index
0, and resets the test metric to zero on other epochs. Its W&B metric
`pred_test_mse` is actually mean relative L2 error, despite the name. The
independent evaluator measures the saved checkpoints directly.

## Evaluate saved Navier–Stokes models

After the INR and ODE stages finish, run:

```bash
~/miniconda3/envs/marble/bin/python run_adaptor/evaluate_navier_stokes.py
```

The default `--mode author` loads the saved checkpoints from
`outputs/navier-stokes-dino/{inr,model}/` and evaluates the checkpoint's test
trajectory count (normally 16). Use `--mode smoke` for smoke checkpoints, or
`--ntest 4` for a prefix of the saved test split. `--batch-size` controls the
number of trajectories per batch (default 2). `--device` and `--output` override
the evaluation device and result JSON path.

Evaluation freezes the INR and ODE weights, recovers latent normalization from
the training trajectories' training interval, encodes **only the initial test
frame**, and integrates the latent ODE with RK4, dt=1 and no teacher forcing.
The remaining test frames are used only as targets for error measurement.
Regenerated training and test grids must match the ODE checkpoint's saved grids.
This preserves the author's spatial sampling, including the current loader's
repeated `sub_from` operation on training data; it is not a full 64×64-grid test.
The script neither modifies the modulation cache nor starts a W&B run.

Results are printed and written to
`outputs/navier-stokes-dino/test_metrics.json`
(`test_metrics-smoke.json` for smoke mode):

- `metrics.in_t`: frames `[0, 20)`, including the initial-frame reconstruction.
- `metrics.out_t`: frames `[20, 40)`, the time-extrapolation interval.
- `metrics.all`: all 40 frames.

These intervals follow the checkpoint's `seq_inter_len`/`seq_extra_len` if they
differ. Each interval reports MSE, mean per-trajectory relative L2 over the
entire interval, and per-trajectory errors. Per-frame mean MSE and relative L2
are also included. The interval MSE values correspond to the author's
`pred_test_mse_inter`, `pred_test_mse_extra` and `pred_test_mse` metrics.
Relative L2 is an additional metric, not latent-code MSE.

The JSON records actual checkpoint epochs (zero-based), sampling and test size.
Saved checkpoints follow the author's selection rule and need not come from
the final training epoch. The training summary's `author_return_values.ode`
is a training error, not final test performance.

## W&B dashboard

The adaptor defaults to offline logging. Existing runs under `outputs/wandb/`
can be uploaded with `wandb sync`. For live dashboard updates, authenticate and
override the author team's logging destination without changing training:

```bash
~/miniconda3/envs/marble/bin/wandb login --verify
WANDB_MODE=online \
WANDB_SILENT=false \
WANDB_ENTITY=<your-user-or-team> \
WANDB_PROJECT=coral-reproduction \
~/miniconda3/envs/marble/bin/python run_adaptor/run_navier_stokes.py
```

INR and ODE are separate author stages and therefore appear as two W&B runs.

## Learned modulation t-SNE

Run these scripts after training (no model weights are updated and no W&B run is
created):

```bash
python -m pip install 'scikit-learn>=1.5,<2'
python run_adaptor/viz_script/viz_airfoil_task_ltt.py
python run_adaptor/viz_script/viz_ns_task_ltt.py
```

By default they use the INR referenced by the saved regression/ODE checkpoint,
and its saved architecture, seed and encoding step size. Without a downstream
checkpoint they use the INR checkpoint directly. `--inr-checkpoint PATH` selects
another INR explicitly; `--data-dir PATH` relocates its dataset. NS regenerates
and verifies the sampled grids against the checkpoints before encoding.

- **Airfoil:** one row with two panels, input modulation and output modulation.
  Each case has a pair of codes; output codes are inferred from observed output
  fields, not predicted by the regressor. Each panel fits t-SNE jointly on train
  and test, using blue circles for train and orange triangles for test. The two
  panels have independent embeddings, so their axes are not aligned.
- **NS:** one point per observed frame. The first panel distinguishes train/test;
  the second shows the same embedding colored by frame index, preserving the
  different marker shapes. By default train has its actual In-t frames (20) and
  test has all evaluation frames (40). `--test-frames in-t` selects matching time
  horizons for the plot. These are encoded state codes, not ODE rollout codes.

Outputs go to `outputs/<dataset>/visualization/modulations_tsne/` (or
`modulations_tsne-smoke/` with `--mode smoke`):

- `airfoil_modulations_tsne.{png,pdf}` or `ns_state_modulations_tsne.{png,pdf}`;
- `codes.npz`: raw latent vectors plus checkpoint/config provenance;
- `embedding.csv` and `embedding.npz`: 2D coordinates, split and original case
  IDs (Airfoil), or split, trajectory indices, frame indices and In-t/Out-t (NS);
- `metadata.json`: checkpoint fingerprint, encoding settings, t-SNE settings,
  preprocessing and final KL divergence.

`--ntrain` / `--ntest` select prefixes of cases/trajectories for a quick check;
default is the full saved dataset. `--batch-size` counts cases for Airfoil and
individual frames for NS. These settings do not change model parameters.

For another seed/perplexity, reuse the exported codes rather than encoding again:

```bash
python run_adaptor/viz_script/viz_ns_task_ltt.py \
  --codes-file run_adaptor/outputs/navier-stokes-dino/visualization/modulations_tsne/codes.npz \
  --test-frames in-t --perplexity 50 --seed 42 \
  --output-dir run_adaptor/outputs/navier-stokes-dino/visualization/tsne_in_t_p50
```

The default preserves the raw latent Euclidean geometry (a uniform rescaling
only improves numerical conditioning). `--standardize` instead normalizes each
feature using training codes only; it changes the distance metric. Both splits
participate in the t-SNE fit for exploratory visualization; this is not a held-out
performance evaluation. Cluster sizes and distances between distant groups in
t-SNE are not quantitative evidence of generalization or overfitting. Compare
multiple seeds/perplexities and the reconstruction/prediction errors.

## Elasticity training and independent evaluation

```bash
PYTHON=~/miniconda3/envs/marble/bin/python
WANDB_MODE=offline $PYTHON run_adaptor/run_elasticity.py
$PYTHON run_adaptor/evaluate_elasticity.py

# Smoke mode: 8 training / 4 test cases, 5 epochs per stage, full 972-point meshes
WANDB_MODE=offline $PYTHON run_adaptor/run_elasticity.py --mode smoke
$PYTHON run_adaptor/evaluate_elasticity.py --mode smoke

# Evaluate a prefix of the saved test partition
$PYTHON run_adaptor/evaluate_elasticity.py --mode smoke --ntest 2
```

Training calls the original `static.design_inr` and `static.design_regression`
entrypoints and requires CUDA. Defaults in `config.yaml` reproduce the author's
Elasticity bash launchers: 1000/200 cases, INR 5000 epochs, batch size 64,
INR/meta-code learning rates 1e-4, input/output w0 10/15, regression 10000 epochs.
Results are saved under `run_adaptor/outputs/elasticity/`.

The evaluator freezes all networks, reconstructs training input codes with the
author's encoding batch size of 4 to recover mean/sample standard deviation,
and reports predictive relative L2/MSE, per-case errors and both INR reconstruction
relative L2 metrics. It rebuilds the mean-geometry grid using the saved regression
training sample count. `--ntest` selects a prefix of the saved test interval;
Elasticity's loader otherwise selects the last requested cases of the raw array.
The reported `test_indices` use an exclusive end index.

Evaluation supports `--regression-checkpoint`, `--inr-checkpoint`, `--output-root`,
`--data-dir`, `--output`, `--batch-size`, `--device` and `--ntest`.
`--batch-size` controls test evaluation; normalization recovery always uses 4.
