# 七个数据集的入口、数据准备与验证

NACA-Euler（Geo-FNO Airfoil）与 MeshGraphNets Airfoil IVP 是不同任务，分别使用
`airfoil` 和 `airfoil_flow`。下面每个文件均可从项目根目录直接运行。

| 数据集 | run_adaptor 训练 / 评估 | run_codelib 训练 / 评估 | 数据目录 |
|---|---|---|---|
| NACA-Euler | run_airfoil.py / evaluate_airfoil.py | run_airfoil.py / eval_airfoil.py | datasets/airfoil |
| Elasticity | run_elasticity.py / evaluate_elasticity.py | run_elasticity.py / eval_elasticity.py | datasets/elasticity |
| Pipe | run_pipe.py / evaluate_pipe.py | run_pipe.py / eval_pipe.py | datasets/pipe |
| Cylinder IVP | run_cylinder_flow.py / evaluate_cylinder_flow.py | run_cylinder_flow.py / eval_cylinder_flow.py | datasets/cylinder_flow |
| Airfoil IVP | run_airfoil_flow.py / evaluate_airfoil_flow.py | run_airfoil_flow.py / eval_airfoil_flow.py | datasets/airfoil_flow |
| NS dynamics | run_navier_stokes.py / evaluate_navier_stokes.py | run_navier_stokes.py / eval_navier_stokes.py | datasets/dino |
| SW dynamics | run_shallow_water.py / evaluate_shallow_water.py | run_shallow_water.py / eval_shallow_water.py | datasets/dino |

配置分别集中在两个目录的 `config.yaml`。原版模式名为 `author`，`run_codelib` 为
`full`；两者的 `smoke` 默认 8/4 案例、5 epoch，保留配置中的网络和空间采样。
SW 的案例数指长度 40 的时间窗口：原始 train/test 为 16/2 条、每条 160 帧，
默认切为 64/8 个窗口；训练使用每窗口前 20 帧，评估报告 In-t、Out-t 及总误差。
NS 默认 256/16 条、每条 40 帧。IVP 为各官方 split 的前缀、每条首帧到末帧预测。

## 当前数据与下载顺序

2026-10-02 已验证的数据：

1. NACA-Euler、Elasticity、NS、Cylinder 已存在。Cylinder 原始数据约 16 GiB，
   无需重复下载；Pipe 原始数据也已存在于本机，补了 `datasets/pipe` 符号链接。
2. SW 两个官方 HDF5 已完整下载到 `datasets/dino`：test 167,774,208 bytes，
   train 1,342,179,328 bytes；先下载较小的 test，再下载 train。
3. IVP Airfoil 已下载官方 **完整数据集**：train/valid/test 各 1000/100/100 条
   trajectory，每条 601 帧、5233 个节点。`download_manifest.json` 三个 split
   均标记 complete；实际文件大小和 TFRecord 数量已核对，记录边界完整，
   没有遗留 .part 文件，可使用完整实验配置。
4. 完整 IVP Airfoil：train 50,506,102,000 bytes，valid/test 各 5,050,610,200
   bytes，合计 60,607,322,400 bytes（约 60.6 GB / 56.4 GiB）。下载按
   test → valid → train 完成；下载器会跳过已有完整 split。

来源：[MeshGraphNets 官方下载脚本](https://github.com/deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh)、
[DINo](https://github.com/mkirchmeyer/DINo)、
[SW 数据目录](https://drive.google.com/drive/folders/1RlH4E7-dnKOEA6Zur2EdK-HuEaW_PC8w)、
[Geo-FNO 数据目录](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8)。

```bash
PYTHON=~/miniconda3/envs/marble/bin/python

# 安装下载器依赖（已有数据时训练不需要 gdown）
$PYTHON -m pip install 'gdown>=5,<6' requests

# 默认下载完整 SW + IVP Airfoil 8/4/4 trajectory 前缀；已完成的文件会跳过
$PYTHON run_adaptor/download_missing_data.py

# 完整 IVP 下载，支持 .part 文件续传；成功后原子替换前缀文件并更新 manifest
$PYTHON run_adaptor/download_missing_data.py --dataset airfoil-ivp --full-ivp
```

前缀下载不会截断 trajectory 内的时间帧；TFRecord 每条记录完整保留。
原版 IVP 适配器只在内存中取首末帧，不创建 `static_*.h5`。SW 使用 HDF5 切片
读取所需时间窗口对应的原始 trajectory，不为 smoke 一次读入全部训练数据。

## 训练和评估示例

```bash
# 原版 Airfoil IVP：训练两个阶段，再独立评估
WANDB_MODE=offline $PYTHON run_adaptor/run_airfoil_flow.py --mode smoke
$PYTHON run_adaptor/evaluate_airfoil_flow.py --mode smoke

# Shared INR Airfoil IVP
$PYTHON run_codelib/run_airfoil_flow.py --mode smoke
$PYTHON run_codelib/eval_airfoil_flow.py --mode smoke

# SW dynamics
WANDB_MODE=offline $PYTHON run_adaptor/run_shallow_water.py --mode smoke
$PYTHON run_adaptor/evaluate_shallow_water.py --mode smoke
$PYTHON run_codelib/run_shallow_water.py --mode smoke
$PYTHON run_codelib/eval_shallow_water.py --mode smoke

# 已有 INR 后只训练下游：静态 / IVP 用 regression，dynamics 用 ode
$PYTHON run_codelib/run_navier_stokes.py --mode smoke --stage ode

# 默认普通网络；评估也可显式指定 checkpoint
$PYTHON run_codelib/run_cylinder_flow.py --mode smoke
$PYTHON run_codelib/eval_cylinder_flow.py --mode smoke --regression-checkpoint \
  run_codelib/outputs/cylinder-flow/model/cylinder-flow-shared-regression-smoke.pt
```

`run_codelib` 全部入口默认 `all`。Codelib 检索模型、库和开关已移除，目录名保留。
默认普通 SIREN + ResNet / Derivative；静态任务可选 AnchorMix，详见
[表示配置说明](../run_codelib/README.md)。旧检索 checkpoint 需重新训练。

原版训练要求 CUDA。原版 W&B 可用 `WANDB_MODE=offline` 或 `disabled`；适配器
会保留 disabled 模式中的指定 checkpoint 名称。运行完整实验时去掉 `--mode smoke`，
先确认 IVP Airfoil 已下载完整 split。未在本次工作中运行完整 epoch 训练。

## Shared INR 扩展的语义与比较边界

几何设计任务继续调用 `static/design_inr_shared.py` 和
`static/design_regression_shared.py`；IVP 和 dynamics 由 `run_codelib/extended.py`
实现，复用普通 `outer_step`、ResNet / Derivative 和 code 适配接口。

- IVP 保留变长 mesh，对每个 mesh 独立适配 code，再在 optimizer batch 内平均案例
  loss。原版 graph loop 在节点上平均 loss，两者在 mesh 点数不同时的加权不同。
  原版 Airfoil IVP 使用固定物理量统计与坐标除以 20，共享入口读取相同归一化字段。
  IVP mapper 输入和输出 code 均按训练统计标准化（Cylinder 按 latent 维度，
  Airfoil IVP 按全局 scalar），与作者 IVP regression 的方式一致。
- Shared INR dynamics 使用一个联合场 INR，每训练帧一个 code，普通 Derivative
  接收 normalized code。
  训练使用原配置的 RK4 和 teacher forcing；评估仅拟合首帧，关闭 teacher forcing。
  这是本地新扩展，原版 SW 使用 height/vorticity 两个 INR、合计两份 latent code，
  不能把 shared-INR SW 与原版双 INR SW 视为相同架构的严格消融。
- 下游阶段冻结 INR，重新提取 codes。Checkpoint 保存训练归一化统计、网格、
  关联 INR 路径与哈希，评估核对对应关系。
- 原版 dynamics 评估从训练 code 恢复统计；共享入口从 checkpoint 恢复统计。
  Dynamics 评估重放原采样并校验保存网格，再选择 `--ntest` 前缀，避免改变 RNG 网格。
  Elasticity 保留保存测试区间的前缀；Pipe 和 NACA-Euler 测试固定从案例 1000 开始。

## 验证

入口验证包括配置解析、split/采样约束与短程训练。仅 smoke 不代表收敛精度。

```bash
$PYTHON -m unittest discover -s tests -v
```

测试覆盖 Pipe 固定 split/统计、SW 时间窗口前缀、全部入口配置、普通 ODE 梯度、
原始编码/解码/processor 等价性、AnchorMix 与旧检索 checkpoint 拒绝。
