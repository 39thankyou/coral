# CORAL 项目代码索引与复现约定

本仓库是 NeurIPS 2023 CORAL（Coordinate-based neural operator）的官方
PyTorch 实现。CORAL 先用共享的调制 INR 把连续场压缩成 latent code，再在
latent space 中学习静态映射或时间动力学。

## 严格对齐作者的复现实验入口

严格复现使用 `run_adaptor/`。它不重写训练循环，而是直接调用作者分开的阶段入口：

- Airfoil / Elasticity：`static/design_inr.py` → `static/design_regression.py`。
- Cylinder Flow：`static/static_inr.py` → `static/static_regression.py`。
- Navier–Stokes：`inr/inr.py` → `dynamics_modeling/train.py`。

`run_adaptor/config.yaml` 从作者 YAML 和 bash launcher 合并实际参数，默认使用
作者完整样本数和 epoch；只有显式传入 `--mode smoke` 才把样本设为 8/4、两个
阶段各设为 5 epoch。第一阶段以确定名称保存作者格式
checkpoint，第二阶段通过作者原有的 loader 重新推断 modulations/codes 后读取它。
兼容层仅处理本地数据路径、随机种子、离线 W&B、当前 PyTorch 移除的日志参数，
以及 Cylinder 原始 TFRecord 到作者首末帧图字段的内存适配；不修改 loss、优化器
步骤、scheduled sampling、网络或 checkpoint schema。完整说明见
`run_adaptor/README.md`。

运行入口：

```bash
~/miniconda3/envs/marble/bin/python run_adaptor/run_airfoil.py
~/miniconda3/envs/marble/bin/python run_adaptor/run_elasticity.py
~/miniconda3/envs/marble/bin/python run_adaptor/run_cylinder_flow.py
~/miniconda3/envs/marble/bin/python run_adaptor/run_navier_stokes.py
```

早期 `run/coral_experiment.py` 是统一重写的训练流程，只保留作对照，不作为严格
复现实验入口。

## Elasticity 当前入口与独立评估

原版方法使用 `run_adaptor/run_elasticity.py`，默认依次调用
`static/design_inr.py` 和 `static/design_regression.py`；独立评估入口是
`run_adaptor/evaluate_elasticity.py`。配置见 `run_adaptor/config.yaml`。

```bash
PYTHON=~/miniconda3/envs/marble/bin/python
WANDB_MODE=offline $PYTHON run_adaptor/run_elasticity.py
$PYTHON run_adaptor/evaluate_elasticity.py

# 小规模验证，保留完整空间网格与模型结构
WANDB_MODE=offline $PYTHON run_adaptor/run_elasticity.py --mode smoke
$PYTHON run_adaptor/evaluate_elasticity.py --mode smoke
```

原版训练使用 CUDA，默认 1000/200 案例，INR 5000 epoch、batch size 64、
lr_inr/meta_lr_code=1e-4、w0_in/out=10/15，regression 10000 epoch。
参数来自 `bash_static/elasticity/` 的启动器及作者 YAML。输出位于
`run_adaptor/outputs/elasticity/`。评估冻结模型，从训练输入重新推断 code，
按原版编码 batch size=4 恢复 input-code 均值与样本标准差；output code
不标准化。测试仅适配 latent，报告 relative L2、MSE、逐案例误差，以及
input/output INR 的重构 relative L2。真实测试输出仅用于指标和重构诊断。

Shared INR 与 AnchorMix 使用 `run_codelib/run_elasticity.py` 和
`run_codelib/eval_elasticity.py`，配置见 `run_codelib/config.yaml`，结果位于
`run_codelib/outputs/elasticity/`。先训练 INR，再重新提取 codes 并训练 regression。
训练循环来自 `static/design_inr_shared.py` 与 `static/design_regression_shared.py`。
Codelib 检索模型、记忆库和参数已移除；历史目录名保留，旧检索 checkpoint 需重训。
普通 SIREN 和 AnchorMix 均使用原 ResNet processor；评估恢复保存的训练统计量。

```bash
$PYTHON run_codelib/run_elasticity.py --mode smoke
$PYTHON run_codelib/eval_elasticity.py --mode smoke
```

两个方法均读取 `datasets/elasticity/Meshes/Random_UnitCell_XY_10.npy`
和 `Random_UnitCell_sigma_10.npy`，应力使用完整官方训练分区的标准差。
坐标网格来自当前 regression 训练分区输入几何的均值，独立评估需使用相同的
训练样本数重建。Elasticity 测试集取原始数组末尾的案例；评估 `--ntest` 取
checkpoint 保存测试区间的前缀，不能直接向 loader 传更小的测试样本数，
否则会改为数组末尾的其他案例。`test_indices` 记录原始案例的左闭右开区间。
原版评估支持 `--regression-checkpoint`、`--inr-checkpoint`、`--output-root`、
`--data-dir`、`--output`、`--batch-size`、`--device`、`--ntest`。

Airfoil / Elasticity / Pipe 的原版 eval 默认从当前测试区间等距取 K=10 个案例，
包括首尾；不足 10 个时取全部，可用 `--visualization-k` 调整。重构图保存在对应
结果目录的 `visualization_error/`，每张 2 行 3 列：行是 INR-in/out，列是
GT、INR reconstruction、绝对误差。几何 x/y 通道分别保存一张图，标量 output
行在两个通道图中重复，所以 10 案例通常产生 20 张 PNG。GT/重构每行共用色标，
error 色标从 0 开始；Airfoil/Pipe 使用完整计算网格像素图，Elasticity 使用该
案例真值几何的散点图。显示值保留 loader 的归一化。INR-out 使用真实输出单独
适配的 latent 重构，属于诊断图；mapper 预测仍仅依赖测试输入。绘图复用指标
计算的最终重构，不增加适配步数。`manifest.json` 和 test metrics 记录测试局部
索引、原始案例 ID、每通道 relative L2 与文件路径，smoke 文件名加后缀。
实现共用 `run_adaptor/evaluate_airfoil.py` 与 `run_adaptor/visualize_error.py`。

## 七个数据集的入口与数据准备

两套目录均已提供全部 7 个数据集的 run/eval：`airfoil`（NACA-Euler）、
`elasticity`、`pipe`、`cylinder_flow`、`airfoil_flow`（MeshGraphNets IVP）、
`navier_stokes`、`shallow_water`。原版评估名为 `evaluate_<dataset>.py`，
`run_codelib` 评估名为 `eval_<dataset>.py`。完整入口表、配置参数、指标定义和下载策略见
`run_adaptor/DATASETS.md`，两套方法的配置分别在各自的 `config.yaml`。

数据路径的 `airfoil` 与 `airfoil_flow` 不可混用。Pipe 使用本机已有原始数据的
符号链接。SW 两个 HDF5 已完整下载；原始 16/2 trajectory 各有 160 帧，默认
切为 64/8 个 40 帧窗口，训练仅观察每窗口前 20 帧。SW loader 按所需窗口数读取
HDF5 trajectory 切片，避免 smoke 读入全量训练数据。

IVP Airfoil 已完整下载官方 train/valid/test 的 1000/100/100 条 trajectory，
每条 601 帧、5233 个节点。`datasets/airfoil_flow/download_manifest.json` 三个
split 均标记 complete，实际文件大小与记录数量已经核对，TFRecord 边界完整，
没有遗留 .part 文件。完整文件总计 60,607,322,400 bytes，约 60.6 GB / 56.4 GiB，
可使用完整样本配置。下载器保留补齐命令：
`python run_adaptor/download_missing_data.py --dataset airfoil-ivp --full-ivp`。
下载支持 .part 续传，成功后更新 manifest，已有完整 split 会跳过。
训练期间只取首末帧，不产生全量 `static_*.h5` 缓存。

原版 SW 分别训练 height/vorticity INR，再按原作者代码训练联合 ODE；
`run_codelib/extended.py` 的 SW/NS 使用共享联合场 INR 与原 Derivative latent
ODE。这个 SW 扩展与原版双 INR 架构不同。IVP 对变长 mesh 独立适配，optimizer
batch 内按案例平均 loss；原版 graph loop 按节点平均，两者权重不同。
所有动态评估仅拟合首帧，再 RK4 rollout，关闭 teacher forcing；网格必须与
checkpoint 重放一致。该扩展已不包含训练 code 检索。

Pipe loader 使用固定 1000/200 分区及完整训练 geometry 统计；smoke 不再被
旧代码强制加载 1000/200 案例。原版 IVP regression 已修复模型误存到 `inr/`
的问题，评估仍兼容旧路径。原版 W&B disabled 模式保留用户指定的 checkpoint
名称。静态原版 code normalization 编码 batch size 固定 4，时序固定 2。

检查命令：`~/miniconda3/envs/marble/bin/python -m unittest discover -s tests -v`。
`tests/test_dataset_entrypoints.py` 验证 split/时间窗口、全部配置与普通 ODE；
`tests/test_shared_pipeline.py` 验证原编码/解码/processor 等价性与旧检索配置拒绝；
`tests/test_anchormix.py` 验证梯度、读取、保存加载、GridMix 兼容及精确续训。
本次不运行完整收敛训练。

## 早期统一运行器索引

- `run/config.yaml`：从作者 Hydra 配置、bash 启动器和论文参数表整理出的统一配置。
- `run/coral_experiment.py`：读取配置、固定随机数、加载原始数据、训练与评估的公共入口。
- `run/run_airfoil.py`：NACA airfoil 静态几何设计任务。
- `run/run_elasticity.py`：elasticity 静态几何设计任务。
- `run/run_cylinder_flow.py`：MeshGraphNets cylinder-flow 首末帧 IVP 任务。
- `run/run_navier_stokes.py`：Navier–Stokes 时序动力学任务。
- `run/outputs/`：指标与 checkpoint，已在 `.gitignore` 中排除。

默认读取 `smoke` 模式：模型结构、学习率、SIREN 频率等仍采用作者配置，只把
INR 和 REG/ODE 阶段各缩短为 5 epoch，并确定性选取 8/4 个样本和每场最多
512 个空间点。`--mode author` 使用完整样本数、空间点和原始 epoch。

每次运行还会在 `run/outputs/<dataset>/visualization/` 保存一个测试样本的完整
空间场数据。可视化会重新读取同一原始样本的全部空间坐标，与训练时的空间采样
相互独立；它不运行 INR 或 ODE，也不绘制重构、预测或误差。静态/IVP 每个通道
生成一张 `inr_in_channel_<i>__inr_out_channel_<j>.png`，左列是数据集的
INR-in 数据，右列是对应的 INR-out 数据。输入和输出通道数不同时，标量一侧会
重复配对，以保留多通道一侧的每个通道。时序数据每个物理通道生成一张
`inr_in_out_channel_<i>.png`：前 20 帧与后 20 帧按相对时间配对，形成
`k` 行、2 列的完整场图片。`k` 由 `common.visualization_steps` 或 CLI
`--visualization-steps` 控制。Airfoil 的 `221×51` 规则网格和 Navier–Stokes
的 `64×64` 规则网格使用 `imshow` 像素图；elasticity 与 cylinder-flow 的
非结构网格使用坐标散点图。

## 原始数据与 split

本地运行器禁止读取用户生成的 `processed_*.h5`。规则网格数据直接走
`coral/utils/data/load_data.py`，cylinder-flow 直接解析原始 TFRecord；旧
`CylinderFlowDataset` 中生成 `static_*.h5` 的缓存分支不在本地运行路径中。

| 数据集 | 是否时序 | 原始文件 | 固定 split / 任务 |
| --- | --- | --- | --- |
| NACA airfoil | 否 | `datasets/airfoil/naca/NACA_Cylinder_{X,Y,Q}.npy` | train `0:1000`，test `1000:1200`；几何坐标到 Q 的第 5 个通道 |
| elasticity | 否 | `datasets/elasticity/Meshes/Random_UnitCell_{XY,sigma}_10.npy` | train 前 1000，test 全数据最后 200；单元网格到应力 |
| cylinder-flow | 原始数据是 600 帧，但本文任务只取首末帧 | `datasets/cylinder_flow/{train,valid,test}.tfrecord` | 文件级 split 为 1000/100/100；首帧 `(p,vx,vy)` 到末帧 `(p,vx,vy)` |
| Navier–Stokes | 是 | `datasets/dino/navier_1e-3_256_2_{train,test}.shelve` | DINo 原始生成 split 为 256/16；40 帧按前 20 帧 In-t、后 20 帧 Out-t |

cylinder-flow 是可变节点数的非结构网格。运行器使用 PyG 图 batch 与
`graph_outer_step`；author 模式保留每个图的全部节点，smoke 模式在每个图内做
固定索引采样，不把不同图裁成相同的最小网格。

Navier–Stokes 数据按照 CORAL README 指向的 `mkirchmeyer/DINo` 原始
`data_pdes.py` 生成：`nu=1e-3`、256/16 条轨迹、64×64、每条 40 帧。
`run/generate_navier_stokes_dino.py` 保留 DINo 的 GaussianRF 种子、30 时间单位
预热、伪谱/Crank–Nicolson 求解器、`dt=1e-3` 和 shelve 格式；生成参数与文件
SHA-256 记录在 `datasets/dino/manifest.json`。

## INR 与 REG/ODE 阶段

### 静态几何任务：airfoil、elasticity

1. 输入 INR 在公共查询坐标上重建几何场，每个样本通过 inner-loop 得到输入 code。
2. 输出 INR 重建物理输出场，并得到输出 code。
3. REG 阶段用 ResNet 学习 `输入 code -> 输出 code`。
4. 推理时先编码新几何，再做 code regression，最后由输出 INR 在任意查询坐标解码。

### 首末帧 IVP：cylinder-flow

流程同样是两个 INR 加一个 ResNet，但输入/输出分别是同一条 600 帧轨迹的首帧和
末帧。它利用时间端点构造监督对，不在中间 598 帧上训练 ODE，也不是逐块预测。

### 真正的时序建模：Navier–Stokes

1. 一个共享 INR 独立重建每一个物理时间帧；每帧得到一个 code。INR 没有把
   “前 20 帧整块”编码后直接输出“后 20 帧整块”。
2. INR 固定后，`Derivative` 网络学习潜变量微分方程 `dz/dt=f_theta(z)`。
3. 训练期在前 20 帧 code 上使用 scheduled sampling，并通过 RK4 积分。
4. 推理只给初始帧 code `z0`，Neural ODE 自回归积分出整条 40 帧 code 轨迹；
   同一个 INR 逐帧解码。前 20 帧报告 In-t，后 20 帧报告 Out-t 外推误差。

这一路径对应 `inr/inr.py`、`dynamics_modeling/train.py`、
`coral/utils/models/scheduling.py::ode_scheduling` 和论文第 3.2、4.2 节。

## 参数来源说明

`run/config.yaml` 优先保存仓库 bash 启动器覆盖 Hydra 默认值后实际生效的参数。
Navier–Stokes 存在可见冲突：代码启动器实际为 SIREN `w0=30`、INR batch 128、
ODE batch 64，而论文 Table 5 写的是 `w0=10`、batch 64/32；配置中同时记录了
这个差异，运行采用代码启动器值。cylinder-flow 也采用启动器明确覆盖的
`w0_in=20`、`w0_out=15`，而不是只抄论文表格。

所有入口固定 Python、NumPy、PyTorch 和 CUDA RNG，并启用确定性 CUDA 设置。
INR inner-loop 为 3 步、code 初始学习率为 0.01；具体每数据集的 latent/hidden、
epoch、batch、学习率和 scheduler 见 `run/config.yaml`。

## 核心代码索引

- `coral/siren.py`：SIREN 与 ModulatedSiren。
- `coral/metalearning.py`：规则网格 `outer_step` 与可变图 `graph_outer_step`。
- `coral/mlp.py`：MLP、ResNet、latent ODE 的 `Derivative`。
- `coral/losses.py`：MSE、相对 L2、PSNR 等损失。
- `coral/mfn.py`、`coral/fourier_features.py`：MFN/BACON 与 Fourier INR。
- `coral/utils/data/load_data.py`：算子与时序原始读取器、split 和坐标构造。
- `coral/utils/data/graph_dataset.py`：MeshGraphNets TFRecord 解析。
- `coral/utils/data/operator_dataset.py`：静态输入/输出场与 code 数据集。
- `coral/utils/data/dynamics_dataset.py`：时间帧场与 code 数据集。
- `coral/utils/data/load_modulations.py`：批量推断 modulation。
- `static/design_inr.py`、`static/design_regression.py`：静态几何原始训练入口。
- `static/static_inr.py`、`static/static_regression.py`：图/点集 IVP 原始入口。
- `inr/inr.py`、`dynamics_modeling/train.py`：时序 INR 与 latent ODE 原始入口。
- `baseline/`、`fno/`、`mppde/`：论文对比模型。

## 运行方法

从仓库根目录使用指定环境：

```bash
~/miniconda3/envs/marble/bin/python run/run_airfoil.py
~/miniconda3/envs/marble/bin/python run/run_elasticity.py
~/miniconda3/envs/marble/bin/python run/run_cylinder_flow.py
~/miniconda3/envs/marble/bin/python run/run_navier_stokes.py
```

调整时序可视化为 8 个时间步：

```bash
~/miniconda3/envs/marble/bin/python run/run_navier_stokes.py --visualization-steps 8
```

重新生成论文 NS 数据：

```bash
~/miniconda3/envs/marble/bin/python run/generate_navier_stokes_dino.py
```

完整作者规模示例：

```bash
~/miniconda3/envs/marble/bin/python run/run_airfoil.py --mode author
```

输出位于 `run/outputs/<dataset>/metrics.json` 和 `checkpoint.pt`。脚本可从任意
工作目录启动；不要修改 `datasets/` 指向的原始数据。
