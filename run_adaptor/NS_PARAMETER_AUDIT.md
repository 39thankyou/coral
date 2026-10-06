# CORAL Navier–Stokes 参数与数据协议核对

核对对象：本地 `2306.07266v2.pdf`、仓库配置/启动脚本/训练代码、历史 notebook，以及本地已完成实验的 resolved configs 和数据文件。

结论：没有一个 YAML 单独包含全部 NS 实验设置。论文报告值、仓库 bash 实际覆盖后的值、本地适配器实际值并不完全一致。尤其是本地数据的空间分辨率与再次下采样组合，未实现论文的 20% 采样协议。本记录不修改训练配置、数据或模型。

## 1. 应查看哪些来源

| 来源 | 内容与限制 |
| --- | --- |
| [论文](../2306.07266v2.pdf) 第 8 页，§4.2.2 | 数据规模、原始/训练分辨率、时间窗口、训练/测试网格协议 |
| 论文第 14 页，A.2 | NS 数据生成、时间起点、周期边界、强迫项 |
| 论文第 16–19 页，B.1 | INR、编码器、潜变量标准化、优化和调度 |
| 论文第 20 页，Table 5 | NS 与 SW 的主实验超参数 |
| 论文 C.2、Table 8 | 每条轨迹使用不同固定网格的额外实验，区别于主实验 |
| 论文第 26 页，C.8、Table 11 | w0、latent_dim、width 消融，不应把每组消融参数都当成主实验参数 |
| [inr/config/siren.yaml](../inr/config/siren.yaml) | INR 通用默认值，默认数据集写的是 SW，不是完整 NS 配置 |
| [dynamics_modeling/config/ode.yaml](../dynamics_modeling/config/ode.yaml) | ODE 通用默认值，默认也是 SW；单独看会读到 epochs=1000、sub_tr=0.0125 |
| [bash_dynamics/navier-stokes/inr.sh](../bash_dynamics/navier-stokes/inr.sh) | NS INR 的命令行覆盖，只有最后 Python 命令实际传入的变量才生效 |
| [bash_dynamics/navier-stokes/ode.sh](../bash_dynamics/navier-stokes/ode.sh) | NS ODE 的覆盖；声明 gamma_step=0.75，但没有传入命令 |
| [inr/inr.py](../inr/inr.py)、[dynamics_modeling/train.py](../dynamics_modeling/train.py) | 优化器、梯度裁剪、调度器参数、checkpoint 选择、潜变量标准化 |
| [load_data.py](../coral/utils/data/load_data.py) | 数据生成文件的读取、采样执行次数、训练/评估实际点数 |
| [scheduling.py](../coral/utils/models/scheduling.py) | RK4 默认值、scheduled sampling、概率阈值 |
| [expe/config/run_names.py](../expe/config/run_names.py) | 历史实验名称与采样设置的映射，不含完整超参数 |
| [expe/notebooks/codes.ipynb](../expe/notebooks/codes.ipynb) | 保存了 20% 实验实际每帧 819 点的历史输出；接口与当前 loader 已不相同 |
| [本地 INR 配置](outputs/navier-stokes-dino/resolved_configs/inr.yaml)、[本地 ODE 配置](outputs/navier-stokes-dino/resolved_configs/ode.yaml) | 本地这次实验的实际配置，不是作者当年发布结果的配置文件 |

`static/config/static/` 对应双 INR 的静态映射/设计任务和潜变量回归，不是本次单个共享 INR + Neural ODE 的 NS 时序任务配置。`bash_static/navier_stokes/` 中还存在 `dataset_name=navier-stokes`、BACON、`training/inr_meta_sgd.py` 等旧实验；不能与 `navier-stokes-dino` 的动态实验混用。

## 2. 论文数据协议与本地实验

论文第 8、14 页给出的主实验设置：

- 物理量：涡量，单通道；黏性系数 ν=1e-3；周期边界。
- 256 条训练轨迹、16 条测试轨迹，每条轨迹 40 帧，帧间隔 1。
- 论文描述在 256×256 网格上生成轨迹，再降采样到 64×64。
- 前 20 帧为 In-t；后 20 帧为 Out-t。预测从初始帧开始，测试时不做 teacher forcing。
- 主实验采用 100%、20%、5% 三种空间采样设置。
- 训练与测试使用不同的随机网格，但保持相同稀疏度；各自网格在轨迹间共享。固定网格本身符合论文主实验，并不要求每个 epoch 重新抽点。
- 论文 A.2 写保留物理时刻 t=20,…,59 的 40 帧。

以训练用 64×64 网格为分母，采用当前代码的向下取整方式：

| 设置 | 应有的每帧点数 |
| --- | ---: |
| 100% | 4096 |
| 20% | 819 |
| 5% | 204 |

这里的点数为根据论文协议计算的值；20% 的 819 点还得到历史 notebook 输出的直接支持。

本地只读检查 `datasets/dino/navier_1e-3_256_2_{train,test}.shelve`，得到 256/16 个记录，每个记录的 `data.shape=(1,40,64,64)`。因此**本地存储数据已经是 64×64**，不能根据文件名中的 `256` 判断空间分辨率。

当前本地加载链路：

| 用途 | 加载与规则下采样 | 再抽取 20% |
| --- | --- | ---: |
| 训练 | 64×64 → 32×32 → 16×16 | 51 |
| 训练轨迹评估 | 64×64 → 32×32 | 204 |
| 测试 | 64×64 → 32×32 | 204 |

51 点约为 64×64 的 1.25%，并非论文 20% 实验的 819 点。本地训练/测试的稀疏度也不同。

`load_data.py` 的重复训练下采样在仓库初始提交 `999fdc2` 中就存在；官方当前公开文件中也保留了它。但没有找到解释该重复操作设计意图的注释或对应完整实验配置。

不能仅把 `sub_from` 改为 4 就声称复现论文：当前训练数据会执行两遍该参数。若原始输入为 256×256，当前 sub_from=2 的训练分支会得到 64×64，但评估分支会得到 128×128，仍不对应论文主实验的相同稀疏度协议。应以实际张量形状和实验协议核对，而不是只对比参数名称。

本地 [manifest.json](../datasets/dino/manifest.json) 还记载 warmup_time=30；论文 A.2 描述从 t=20 开始保留轨迹。这一数据生成时间口径差异需要确认，不能只凭 DINo 文件名声称与论文数据完全一致。所查 DINo 当前公开代码默认 NS size=64，并有 T=30 的预热调用；直接采用 DINo 当前默认参数不等同于 CORAL 论文的数据生成设置。

## 3. 主实验网络与训练参数对照

“论文”来自 Table 5 和 B.1；“仓库实际”是 base YAML 加上 bash 命令真正传入的覆盖；“本地”是已完成运行保存的 resolved configs。

| 参数 | 论文 | 仓库 NS 启动脚本实际生效值 | 本地已完成实验 |
| --- | --- | --- | --- |
| INR 模型 | 调制 SIREN | SIREN | SIREN |
| latent_dim | 128 | 128 | 128 |
| INR depth | 4 | 4 | 4 |
| INR width | 128 | 128 | 128 |
| w0 | **10** | **30** | **30** |
| INR batch size | **64** | **128** | **128** |
| INR epochs | 10000 | 10000 | 10000 |
| INR 外循环学习率 | 5e-6 | 5e-6 | 5e-6 |
| 编码步数 K | 3 | 3 | 3 |
| 编码步长 α | 0.01 | 初始 0.01，参与学习 | 初始 0.01，参与学习 |
| meta-α 学习率 | Table 5 未列出 | 5e-6 | 5e-6 |
| ODE 网络 depth | 3 | 3 | 3 |
| ODE 网络 width | 512 | 512 | 512 |
| ODE 激活 | Swish | 项目自定义 Swish | 同左 |
| ODE solver | RK4 | RK4 | RK4 |
| ODE batch size | **32** | **64** | **64** |
| ODE epochs | 10000 | 10000 | 10000 |
| ODE 学习率 | 1e-3 | 1e-3 | 1e-3 |
| ODE scheduler factor | **0.75** | **0.9** | **0.75** |
| Scheduled sampling 初值 | 0.99 | 0.99 | 0.99 |
| Scheduled sampling 衰减 | 每 10 epoch 乘 0.99 | 每 10 epoch 乘 0.99 | 同左 |
| 优化器 | Adam | AdamW，weight_decay=0 | 同左 |
| seed | Table 5 未列出 | 123 | 123 |

关于 α：B.1.4 说明学习 α 时还会报告 meta-α 学习率，但 Table 5 未列这一行。因此不能断言论文 NS 主实验也以 5e-6 学习 α；当前代码确实学习 α。这是需要原实验 checkpoint/config 进一步确认的差异，不能把未列值直接补成“论文值”。

关于 ODE factor：`ode.sh` 声明的 `gamma_step=0.75` 没有进入最后的 Python 命令，所以原 YAML 的 0.9 生效。本地显式设置 0.75，**与论文一致，但与该 bash 的实际行为不同**。

Table 11 还研究过 w0=20/30、latent_dim=64/256、width=64/256；[ablation_w0.py](../expe/config/ablation_w0.py) 和 [ablation_code_dim.py](../expe/config/ablation_code_dim.py) 保存了相应历史 run 名称。它们表明 w0=30 属于作者研究过的设置，但不能由此证明默认启动脚本与论文主实验完全一致。

## 4. 不在 ode.yaml 里的关键实现细节

| 设置 | 当前实现 | 来源 |
| --- | --- | --- |
| INR 输入/输出 | 二维坐标 → 单通道涡量；所有帧共享一套 INR 权重 | inr/inr.py、load_inr.py |
| INR 调制 | shift=true，scale=false，use_latent=true | inr/config/siren.yaml |
| 超网络 | depth=1，为 latent→调制的单个线性层；配置 width=128 在这一层数下不构成隐藏层 | coral/siren.py::LatentToModulation |
| SIREN 频率 | 首层和后续层都取 cfg.inr.w0 | load_inr.py |
| 编码起点 | 每帧 code 从 0 开始，训练/测试都使用 3 步内循环 | metalearning.py、训练入口 |
| 二阶梯度 | INR 训练时对内循环保留高阶梯度 | metalearning.py::inner_loop_step |
| INR 梯度裁剪 | 参数梯度按值裁剪到 ±1 | inr/inr.py |
| INR scheduler | 虽构造 ReduceLROnPlateau，但仅 fourier_features 分支调用 step；当前 SIREN 不调用 | inr/inr.py |
| 第二阶段 code 提取 | INR 固定后，在训练前一次性提取；batch_size=2 条轨迹，时间帧摊入 batch | train.py、load_modulations.py |
| code 标准化 | 每个 latent 维度，统计所有训练轨迹的前 20 帧；同一统计量用于训练/测试/反归一化 | train.py |
| ODE 时间轴 | torch.arange(T)，间隔 1 | train.py |
| 自定义 Swish | beta 可学习，初始 0.5；实现为 x·sigmoid(x·softplus(beta))/1.1 | coral/mlp.py |
| ODE scheduler | ReduceLROnPlateau；监测 code_train_mse；patience=250、threshold=0.01、threshold_mode=rel、min_lr=1e-5、cooldown=0、eps=1e-8 | train.py |
| Scheduled sampling 小概率截断 | epsilon<1e-3 时设为 0；step=0 时就执行一次概率衰减 | scheduling.py、train.py |
| 测试 rollout | epsilon=0，仅从初始潜变量积分整段时间 | dynamics_modeling/eval.py |
| INR checkpoint | 按训练重建 MSE 最低保存 | inr/inr.py |
| ODE checkpoint（本例 20+20 帧） | 按训练轨迹完整 40 帧自由预测的物理场 MSE 最低保存 | train.py、eval.py |
| 测试频率 | 每 100 epoch 与最后一轮 | 训练入口 |

优化器未显式传入的 beta/eps 使用安装版本的 PyTorch 默认值，而非 YAML 字段。当前两阶段权重衰减都为零。

## 5. 历史实验记录能补充什么

[expe/config/run_names.py](../expe/config/run_names.py) 映射了 sub_from=4 下：

| 空间比例 | INR run_name | ODE run_name |
| --- | --- | --- |
| 100% | jolly-universe-3145 | valiant-river-3296 |
| 20% | legendary-sky-4204 | jedi-parsec-4274 |
| 5% | stilted-elevator-3317 | fragrant-silence-3321 |

`codes.ipynb` 保存的输出对应 20% 这组 run，训练与测试都是每帧 819 点，明确区别于当前 51/204 点。其代码从上述 `.pt` 文件里的 `cfg` 读取具体超参数；但 notebook 的旧 loader 参数/返回值与当前版本不同，不能当成可直接执行的脚本。

`dynamics_modeling/eval_coral.ipynb` 则保存过 3276 点的另一组评估输出，说明仓库混有多个实验设置，不能把任意 notebook 的一组值当成唯一官方配置。

本地未发现上述历史 run 对应的原始 checkpoint；公开网页也没有取得其完整 W&B 配置。因此可以整理出论文报告参数及当前仓库行为，但尚不能恢复这些历史 run 的每一个未公开细节。

## 6. 后续复现应如何区分目标

1. **按论文主实验协议复现**：优先满足原始数据生成和 64×64 基础网格、20%→819 点、训练/测试同等稀疏度、Table 5 的 w0=10 和 batch=64/32、scheduler factor=0.75；明确标注 α 是否学习等未完全公开项。
2. **按仓库某份脚本运行**：精确保留该脚本实际传入参数和当前 loader 行为，并记录实际数据形状，不能将结果自动标为论文主实验复现。
3. **在现有本地 64×64 数据上做对照**：令 sub_from=1、sub_tr=sub_te=0.2 可以避免进一步规则降采样，得到训练/测试各 819 点；这只是对当前代码的可推导对照设置，不证明本地数据等价于论文的 256×256 数值求解后降采样数据，也不解决全部历史配置差异。

本次仅完成资料核对，不自动选择其中一个协议或重新训练。

## 7. 外部一手资料

- [CORAL 作者仓库](https://github.com/LouisSerrano/coral)：README 将 NS/SW 数据生成指向 DINo。
- [论文 v2 在线版](https://arxiv.org/html/2306.07266v2)：本报告主要依据用户提供的本地同版本 PDF。
- [作者当前公开的数据加载器](https://github.com/LouisSerrano/coral/blob/main/coral/utils/data/load_data.py)：保留重复的训练 sub_from 调用。
- [DINo utils.py](https://github.com/mkirchmeyer/DINo/blob/main/utils.py)：当前 NS 默认 size=64。
- [DINo data_pdes.py](https://github.com/mkirchmeyer/DINo/blob/main/data_pdes.py)：NS 数值生成及 T=30 预热调用。
