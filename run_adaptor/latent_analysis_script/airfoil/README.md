# Airfoil latent 分析

从仓库根目录运行，使用已训练的 INR / mapper。默认从原始数据重新编码，
与 `viz_airfoil_task_ltt.py` 使用相同的零初始化和 inner-loop；也可通过
`--codes-file` 复用该脚本生成的 `codes.npz`，加载时会校验 INR SHA-256、
128 维 code 和原始案例顺序。

```bash
PYTHON=~/miniconda3/envs/marble/bin/python
CODES=run_adaptor/outputs/airfoil/visualization/modulations_tsne/codes.npz
$PYTHON run_adaptor/latent_analysis_script/airfoil/local_consistency.py --codes-file "$CODES"
$PYTHON run_adaptor/latent_analysis_script/airfoil/local_interpolation.py --codes-file "$CODES"
$PYTHON run_adaptor/latent_analysis_script/airfoil/ltt_replace.py --codes-file "$CODES" --ref 10
```

默认结果在 `run_adaptor/outputs/airfoil/latent_analysis/` 下，分别保存到
`local_consistency/`、`local_interpolation/`、`ltt_replace/`。
`--output-dir` 指定这三个子目录的**父目录**；`--mode smoke` 会在分析子目录名后
添加 `-smoke`，并在重新编码时选择 smoke checkpoint。

- **邻域一致性**：分别在原始 input/output 128 维空间，用欧氏距离为每个测试案例
  找训练邻居，默认 `--neighbors 5`。与每个查询同样数量的均匀随机训练配对比较，
  随机配对允许抽中邻居。保存逐配对 `pairs.csv`、汇总 `metadata.json`、对照图及
  `codes.npz`。场差异以公共计算网格索引对齐，报告 MSE 和以测试场为分母的相对 L2。
  output code 来自真实输出，其一致性是重构诊断，不代表预测能力。
- **局部插值**：只根据测试 input code 找训练集邻居，将对应训练 output code
  按归一化 `1/d` 加权，再用固定输出 INR 解码。距离为零时只给精确匹配项平均权重。
  与同一测试集上的 mapper 比较，并单列真实测试输出 code 的重构参考。
  mapper 严格使用完整训练 input code 的均值和样本标准差，因此不支持截短其训练集。
  保存逐案例指标、邻居及权重、预测 code、汇总和比较图。`--neighbors` 预先指定，
  脚本不会利用测试输出选择 k 或权重。
- **Latent 替换**：默认取测试案例 1000、1001、1002，可用 `--case-ids 1000 1005`
  指定，或用 `--num-cases 1` 缩小规模。在 output 空间选择最近训练案例，也可用
  `--neighbor-space in` 按 input code 选邻居。随机 donor 从除该邻居外的训练案例中抽取。
  对同一对案例的 **input 和 output code 分别插值**：
  `z(alpha)=(1-alpha)z_original+alpha*z_target`，固定两个 decoder 与公共查询网格。
  `--ref 10` 表示含端点共 **10 张 PNG**，α=0,1/9,…,1（ref 至少为 2）。
  每张图为 **两行五列**：上行 INR-in 几何，下行 INR-out 流场；五列分别为
  origin case、target case、decoded z、abs from org、abs from target。
  几何行显示二维坐标解码后的网格，误差为逆归一化后的物理坐标差的逐点模长，
  同时计入 x/y 分量；流场误差为逐点绝对差。所有差异按公共计算网格索引对齐。
  原案例和目标案例的真实流场各自显示在其物理网格上，解码流场与所有误差图显示在
  固定的原案例物理网格上。两行使用同一对案例与相同 α，每行的色标跨帧及两类 donor
  固定，两个误差列共享范围。重新生成时会清理当前序列多余的旧帧。
  默认放大翼型周围的物理窗口 `[-0.5, 1.5] × [-0.75, 0.75]`，
  可用 `--view-bounds xmin xmax ymin ymax` 修改，或 `--full-domain` 展示整个远场。
  解码始终使用完整网格。仅做可视化，不计算或输出标量误差指标。图片按案例、donor 类型和 ref
  分目录保存，`metadata.json` 记录帧序列与来源。

所有分析支持 `--device`、`--batch-size`、`--ntest`、`--seed`、`--data-dir`。
`--ntrain` 可用于一致性和替换的训练前缀；缓存仅支持官方 split 的前缀。
`--mapper-checkpoint` 可用于局部插值指定 mapper，必须匹配当前 INR 与编码步数。
分析不训练或更新网络参数。
