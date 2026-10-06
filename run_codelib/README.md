# Shared INR 训练与评估入口

Codelib 的检索模型、记忆库及专用参数已移除。`run_codelib/` 仅保留历史目录名，
继续使用原有阶段顺序和结果结构。默认模型为普通 SIREN、ResNet / Derivative；
AnchorMix 与旧二维 GridMix 的表示支持保留。

支持 NACA-Euler、Elasticity、Pipe、Cylinder IVP、Airfoil IVP、NS 和 SW。
数据准备与入口表见 [七个数据集说明](../run_adaptor/DATASETS.md)。全部训练入口
默认 `--stage all`，依次训练 INR、重新提取 latent，再训练 regression / ODE。

```bash
PYTHON=~/miniconda3/envs/marble/bin/python
$PYTHON run_codelib/run_airfoil.py --mode smoke
$PYTHON run_codelib/eval_airfoil.py --mode smoke
$PYTHON run_codelib/run_elasticity.py --mode smoke
$PYTHON run_codelib/eval_elasticity.py --mode smoke
$PYTHON run_codelib/run_cylinder_flow.py --mode smoke
$PYTHON run_codelib/eval_cylinder_flow.py --mode smoke
$PYTHON run_codelib/run_navier_stokes.py --mode smoke
$PYTHON run_codelib/eval_navier_stokes.py --mode smoke

# 已有匹配 INR 时只训练下游
$PYTHON run_codelib/run_airfoil.py --stage regression --inr-checkpoint path/to/inr.pt
$PYTHON run_codelib/run_navier_stokes.py --stage ode --inr-checkpoint path/to/inr.pt
```

配置来自 `config.yaml` 和各阶段基础 YAML。`smoke` 默认 8/4 案例、每阶段 5 epoch，
`full` 使用完整配置预算。可覆盖 `--epochs`、`--ntrain`、`--ntest`、`--batch-size`、
`--encode-batch-size`、`--device`、`--data-dir`、`--output-root`、`--wandb-mode`。
图 IVP 按 mesh 编码，时序按帧编码，这两类入口不使用 encode-batch-size。

静态训练复用清理后的 `static/design_inr_shared.py` 与 `static/design_regression_shared.py`；
图数据和时序由 `extended.py` 调用普通 `outer_step`、ResNet 和 Derivative。
保留原有数据划分、采样、损失、适配和训练预算。IVP 在 batch 内按案例平均 loss；
SW 仍是联合场共享 INR，与 `run_adaptor` 的双 INR 架构不同。

结果位于 `outputs/<dataset>/{inr,model,modulations,resolved_configs,visualization}`。
新实验名称为 `<dataset>-shared-inr` / `<dataset>-shared-regression`，smoke 加 `-smoke`。
Checkpoint 保存训练归一化统计、关联 INR 路径与 SHA-256，时序另存采样网格。
下游每次重新提取 codes，独立评估验证其 INR 来源。旧实验文件保持原样。

旧 Codelib 检索 checkpoint 会明确报错，需重新训练表示与配套 processor。
移除的 CLI 开关不再接受，旧 YAML 中的检索配置也会报错；兼容的普通 SIREN
与 GridMix checkpoint 仍可加载。新配置不再默认引用旧检索 INR。

静态评估保存 relative L2、MSE、逐案例指标和独立重构诊断，并打印 adaptation
步数与学习率。最终预测只拟合输入 latent；真实输出只用于指标和重构诊断。
时序评估只拟合首帧并使用 RK4 rollout，关闭 teacher forcing。

## AnchorMix：NACA-Euler / Elasticity

神经网络组件位于仓库根目录的 `anchormix/`，与 `coral/` 平级。SIREN、
latent MLP、processor、loss 和 latent adaptation 是原实现的独立副本，
该包不从 `coral` 导入代码。训练仍调用已有的
`static/design_inr_shared.py` → `static/design_regression_shared.py`；
数据 loader、split、采样、损失聚合、优化预算和 processor 算法沿用现有配置。
没有新增训练框架。不传 `--representation` 时使用普通 SIREN。

统一入口可直接运行，也可沿用 `run_airfoil.py` / `run_elasticity.py`：

```bash
PYTHON=~/miniconda3/envs/marble/bin/python

# 先训练两个 INR，再重新提取 latent，最后训练对应 processor。
# 默认只替换输出 INR；输入 INR 和 processor 保持原有配置。
$PYTHON run_codelib/pipeline.py --dataset airfoil --representation anchormix
$PYTHON run_codelib/pipeline.py --dataset elasticity --representation anchormix

# 小规模验证：只减案例和 epoch，不减模型宽度、深度或空间网格。
$PYTHON run_codelib/run_airfoil.py --stage all --mode smoke --epochs 1 \
  --ntrain 2 --ntest 1 --batch-size 1 --encode-batch-size 1 \
  --representation anchormix --wandb-mode disabled

# 统一评估入口；建议显式指定 processor checkpoint，结构完全来自 checkpoint。
$PYTHON run_codelib/eval_airfoil.py --dataset airfoil \
  --regression-checkpoint path/to/model.pt --device cuda --batch-size 4
$PYTHON run_codelib/eval_elasticity.py --mode smoke --representation anchormix

# 完整训练的 AnchorMix：评估也需选择同一实验。
$PYTHON run_codelib/eval_elasticity.py --mode full --representation anchormix

# 可学习位置 + gate + 全局/局部融合。
$PYTHON run_codelib/pipeline.py --dataset airfoil --representation anchormix \
  --learnable-pos true --pos-lr-scale 0.1 --pos-constraint bbox \
  --weight-mode distance_gate --fusion hybrid --global-alpha-init 0.1

# 从保存的最后一轮继续；epochs 表示总预算，不是额外轮数。
$PYTHON run_codelib/pipeline.py --dataset airfoil --stage inr \
  --resume path/to/inr.last.pt --epochs 5000

$PYTHON run_codelib/pipeline.py --help
$PYTHON run_codelib/eval_airfoil.py --help
```

评估不自动选择最新文件。不传表示参数时，当前默认配置选择普通 SIREN；训练时
通过 CLI 指定的 AnchorMix 不会反过来修改默认配置。也可用
`--regression-checkpoint` 明确选择 processor，此时网络和 anchors 从关联 checkpoint
恢复，不必重新填写 anchors 参数。运行前需激活 `marble` 环境，或使用上述 `$PYTHON`。

如果出现 INR SHA-256 不匹配，说明当前 INR 文件与该 processor 训练时使用的
快照不同，常见原因是在同一输出目录启动新训练而覆盖了 INR。需选择正确实验、
恢复匹配 INR，或重新训练配套 processor；不能跳过哈希校验混用两次实验。

`--representation {anchormix,siren,gridmix}` 与
`--representation-branch {in,out,both}` 控制替换范围，branch 默认 `out`。
未选中的分支保留 `inr_in` / `inr_out` 配置，默认是普通 SIREN；processor 使用原 ResNet。

训练优先级为显式 CLI > 配置文件 > 默认值。文件仍使用
`run_codelib/config.yaml`：公共 `anchormix` 段可由数据集的 `anchormix` 段覆盖，
也可在 `inr_overrides` 的 `inr_out.anchormix.*` /
`inr_in.anchormix.*` 指定各分支选项。数据集的 `representation` /
`representation_branch` 可替代 CLI 开关。M 读取各分支的 `grid_base`，
当前 CORAL 配置没有该字段时，采用已核对的原 GridMix 默认值 64；
CLI `--grid-base`（别名 `--num-bases`）可覆盖。
L、hidden channels、latent dimension、hypernet、w0 和 seed 来自原分支配置。
启用 scale 和 shift 时，C 包含每层的 scale/shift 两组通道，L 仍为 hidden
layer 数。系数继续使用原 `LatentToModulation` 的 Linear/SiLU 网络，无 softmax。

| 参数（CLI 使用 `--`，支持连字符或下划线） | 默认值 |
| --- | --- |
| num_anchors | 64 |
| anchor_init / anchor_path | fps / null |
| learnable_pos / pos_lr_scale | false / 0.1 |
| init_jitter / pos_constraint | 0.0 / none |
| read_mode / num_neighbors | knn / 8 |
| weight_mode / kernel_sigma | distance / 0.2 |
| attention_power | 1.0 |
| gate_hidden_dim | 64 |
| fusion | anchor |
| fusion_gate_hidden_dim / fusion_gate_init | 64 / 0.05 |
| global_alpha_init / learnable_global_alpha | 0.1 / true |
| query_chunk_size | 4096 |

这些值是工程起始值，并非实验最优值。布尔值接受 `true` / `false`
（大小写均可），不会把字符串 `False` 解析为真。

字典 `B[M,L,C,K]` 先与 `c(z)[batch,L,M]` 混合，得到各层 anchor values。
读取权重跨层共享：`softmax(-||x-p||²/(2*sigma²))`，可选 gate 为
跨 anchor 共享的小型 MLP，输入仅为 `(x, x-p)`，不依赖 z 或额外几何编码器。
`fusion=global` 完全复用独立副本中的原全局 modulation；`hybrid` 为
anchor + alpha × global。`read_mode=all` 仍然是随位置变化的 anchor 读取，
并不等价于 `fusion=global`。

`--attention-power`（也可写 `--attention_power`）控制读取权重锐化：
对原权重 `a` 使用 `a**power / sum(a**power)`，实现采用数学等价且数值稳定的
`softmax(power * scores)`。默认 1 保持原行为，>1 压低小权重、增强局部性，
0<值<1 使权重更均匀；必须为有限正数。它作用于 knn/all 的读取，不改变
字典混合系数 `c(z)`；`fusion=global` 时不参与计算。
`all` 保留全部 anchors 的平滑读取，不增加邻居硬切换。

纯 `distance` 模式下，锐化等价于 `sigma_eff = kernel_sigma / sqrt(power)`，
因此这是同一个距离带宽的另一种配置方式，并不新增表达能力。
`distance_gate` 下会同时放大距离项和 gate 分数；较大权重可能来自 gate，
不保证总是属于最近的 anchor。过大 power 会使权重接近单点选择，梯度和空间变化
可能变得很陡，不保证改善最终预测。

例如保持 32 anchors、sigma=0.2，使用 power=2（有效 sigma 约 0.1414）：

```bash
python run_codelib/run_elasticity.py --mode full --stage all \
  --representation anchormix --representation-branch out \
  --num-anchors 32 --read-mode all --weight-mode distance \
  --kernel-sigma 0.2 --attention-power 2 --fusion anchor \
  --output-root run_codelib/outputs_anchor_all_power2

python run_codelib/eval_elasticity.py --mode full --representation anchormix \
  --output-root run_codelib/outputs_anchor_all_power2
```

power 随配置/checkpoint 保存，evaluate/resume 恢复保存值；显式指定不一致的值会
报错。旧 checkpoint 缺少此字段时使用 1。改变 power 需要重新训练表示和配套
processor，实验使用独立输出目录；不能给旧 processor 在评估时直接换读取权重。

新增 `--fusion gated` 使用用户指定的互补门控融合，保持原 CORAL 全局 modulation
映射，并对每层拼接 global modulation 与 anchor modulation：

```
g_i(x,z) = sigmoid(MLP_i(concat(global_i(z), anchor_i(x,z))))
phi_i(x,z) = (1 - g_i(x,z)) * global_i(z) + g_i(x,z) * anchor_i(x,z)
```

默认 shift-only 时，gate 为 `[batch, points, layers, hidden_channels]`，每个位置、
每层、每通道都有独立权重。各层 MLP 参数独立，空间位置共享参数；采用批量矩阵
乘法同时计算各层 gate，并沿用查询分块。若显式启用 scale+shift，按现有通道
布局分别融合两组 modulation。

`fusion_gate_hidden_dim=64` 是融合 MLP 隐层宽度。`fusion_gate_init=0.05` 设置
初始 anchor 占比约 5%（global 约 95%），取值必须严格位于 `(0,1)`；末层使用
小非零权重和 `logit(0.05)` 偏置，保持各层梯度路径可用。
此 gate 依赖两分支的 modulation，因此依赖 z；它与 `weight_mode=distance_gate`
的坐标读取 gate 是两个不同模块。`attention_power` 仍仅作用于空间读取权重，
不作用于融合 gate 或字典系数。`global_alpha_init` / `learnable_global_alpha`
仅供旧 `hybrid` 模式使用，在 `gated` 模式下不参与计算。

下一版本保持 `all`、32 anchors、`attention_power=2`，只替换输出 INR：

```bash
python run_codelib/run_elasticity.py --mode full --stage all \
  --representation anchormix --representation-branch out \
  --num-anchors 32 --read-mode all --weight-mode distance \
  --kernel-sigma 0.2 --attention-power 2 --fusion gated \
  --fusion-gate-hidden-dim 64 --fusion-gate-init 0.05 \
  --output-root run_codelib/outputs_anchor_gated_power2

python run_codelib/eval_elasticity.py --mode full --representation anchormix \
  --output-root run_codelib/outputs_anchor_gated_power2
```

训练仍使用原 INR → latent 提取 → processor 阶段；gate 参数、完整配置和 anchors
随 checkpoint 保存，evaluate/resume 恢复同一结构。旧 anchor/global/hybrid
checkpoint 保持原行为。新版本使用独立目录并重新训练对应表示与 processor，
不复用前一版本的 latent 缓存。此命令从头训练，不隐式加载原版 SIREN 权重。

FPS 只使用传入训练案例查询坐标的并集，每次新训练只初始化一次。初始化先以
全训练坐标边界做空间桶降采样，最多保留 65536 个实际训练查询点，再执行 FPS；
相同公共网格不会被展开成全部案例的大数组。该边界只用于降采样和 bbox，
不会重新归一化模型坐标。NACA 查询坐标是原 loader 的 `[0,1]` 网格，
elasticity 是训练几何均值网格；`kernel_sigma` 和 `init_jitter` 都使用这些单位。
文件初始化使用 `--anchor-init file --anchor-path anchors.npy`，也支持仅包含
`[K,d]` tensor 的 `.pt` 文件。维度 d 从数据推断，不写死。

`anchor_init=zero` 严格保留零坐标；不添加默认扰动。重合 anchors 会使 KNN
出现距离并列，未选中的 tokens 可能收不到梯度；all 模式的初始空间权重也退化为
均匀权重。零位置实验推荐 `fusion=global`，或显式设置 jitter。
固定位置为 buffer，可学习位置为 Parameter，独立 optimizer group 的学习率为
`lr_inr * pos_lr_scale`。bbox 仅在外层更新后投影可学习位置；固定初始位置若在
bbox 外会报错，不会静默移动。`q=1` 的归一化权重恒为 1，仅靠距离/gate
不能为位置提供梯度；常规位置学习应使用多个邻居。

距离使用减法、平方、求和；仅离散 KNN 选择不求导，选中位置的距离会重新可导计算。
实现先混合 values，再按 query chunk 读取并解码；同一块所有层复用邻居/权重。
混合后的 values 在一次 forward 中只转换一次布局为 `[batch,K,L*C]`。
KNN 用 `scatter` 将 q 个权重写入 `[batch,chunk,K]`，未选中的权重严格为零，
再用一次 `bmm` 同时读取所有层和通道，已移除逐邻居 `gather` / 累加的 Python 循环。
该方式保留 KNN 的邻居选择语义，但矩阵乘法计算量为 `O(batch*N*K*L*C)`，
较大的 K 仍需实测；它不是只计算 q 个 values 的稀疏乘法。
不会构造 `[batch,L,N,K,C]` 或 `[batch,chunk,q,L*C]`。KNN 仍需在全部 K 个位置中搜索，搜索成本为
`O(batch*N*K*d)` 加 top-k，不能因 q 小就认为整体与 K 无关。
查询分块限制临时距离矩阵；二阶训练的 autograd 图仍有相应内存成本。
forward 不再通过 `positions_initialized.item()` 同步 GPU；初始化/加载时更新
Python 标志，保存的 buffer 和旧 checkpoint 格式保持兼容。

保留的查询分块循环用于限制峰值显存，SIREN 层循环有前后依赖；FPS 的逐 anchor
选择也依赖前一步，且只在初始化执行。可以通过 `--query-chunk-size` 调整分块。
GPU 性能对照脚本比较旧邻居循环与向量化读取，包含 forward、3 步 adaptation
及外循环 backward，报告预热后中位耗时和额外峰值显存，并检查 CUDA 二阶梯度：

```bash
$PYTHON tests/benchmark_anchormix.py --batch-size 8 --points 972 \
  --num-anchors 64 --num-neighbors 8
```

这是固定形状的合成数据基准，不等同于完整训练提速。计算顺序变化会产生浮点舍入
差异；数学定义、网络参数、损失和训练预算保持不变。

2026-10-04 的 RTX 5090 D / PyTorch 2.8.0 对照（沿用入口的确定性设置，
float32；batch=64、N=972、M=K=64、q=8、L=3、C=256，固定位置、distance/anchor）：
纯前向中位耗时 7.23 → 2.06 ms；3 步 adaptation + backward 为 1346 → 70.2 ms；
后者额外峰值显存 6.55 → 3.29 GiB。2 次预热、5 次交替测量，未包含数据读取或
optimizer step。更小的 batch 不保证前向提速；可学习位置 + gate + hybrid 的
分块样例中前向 2.45 → 2.73 ms，adaptation + backward 为 413 → 36.0 ms。

结果继续存入 `outputs/<dataset>/{inr,model,modulations,resolved_configs,...}`，
名字增加 `-anchormix-out`（或所选表示/分支）以区分原实验。不同超参数实验建议
指定不同的 `--output-root`。INR 的 `.pt` 保留原按训练输出损失选择最佳模型的规则；
`.last.pt` 保存最后一轮用于续训。state_dict 包含初始位置、训练后位置、训练边界、
坐标变换 offset/scale（identity）和初始化标志，checkpoint 另存配置、表示来源、
优化器和 RNG。resume/evaluate 直接恢复位置，既不重做 FPS，也不读取原 anchor 文件。
跨目录迁移 `.last.pt` 时也需保留同名最佳 `.pt`，以延续原来的最佳模型选择规则。
续训后需要重新训练对应 processor；原 processor 的 INR SHA-256 校验会拒绝不同权重。
processor 每次均从零重新提取 latent，缓存记录表示元数据和对应 INR 哈希，不读取旧缓存。

评估结构只来自 checkpoint。显式指定的表示/anchor 结构选项如不一致会报错；
数据目录、设备、batch size、query chunk 可覆盖。`--ntest` 仍只能取保存测试区间
的前缀，elasticity 会按保存的训练分区重建公共网格。
`--reconstruction-steps` / `--reconstruction-lr` 只改变重构诊断；评估会打印步数
和学习率。最终预测严格为输入场适配 → processor → 输出 INR，真实输出仅参与
指标和单独的重构诊断。指标定义与原入口一致。

兼容的旧二维 GridMix 使用 `anchormix/legacy_gridmix.py` 中的源代码快照恢复，
保留 `grid_bases` 和 modulation state_dict 名称，不转换为 AnchorMix；
该快照来自本机 `ICLR2025_GridMix/coral/siren.py`，运行不依赖相邻仓库。
旧 GridMix 训练快照可以导入并重新训练 processor，旧 processor 若缺少原评估所需
的归一化/哈希元数据，则仍需重新训练，不会猜测这些信息。

验证覆盖读取数学定义、1D/2D/3D、两种位置类型、全部读取/权重/融合组合、
原二阶 adaptation、分块输出与梯度一致性、文件删除后的 checkpoint 重放、
旧 GridMix 加载，以及连续训练和恢复训练的逐参数一致性。
已完成 NACA/elasticity 的真实数据 2/1 案例、每阶段 1 epoch smoke 和独立评估，
未运行完整收敛训练。
