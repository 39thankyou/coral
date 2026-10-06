# Shared INR 潜在编码方式对比：科研规划

## 1. 研究主题

**建议题目：**

> 面向连续场表示的 Shared INR 潜在编码机制研究：Auto-decoding、规则网格编码与点集编码的系统比较

研究对象是一组由采样点表示的函数：

\[
\mathcal D_i=\{(x_{ij},y_{ij})\}_{j=1}^{N_i},
\]

目标是将其编码为潜在变量，并通过共享隐式神经表示（Shared INR）重建连续函数：

\[
z_i=E(\mathcal D_i),\qquad \hat y_i(x)=f_\theta(x;z_i).
\]

核心问题不是再提出一种新的 modulation，而是回答：

> 在固定 Shared INR、固定 latent 容量和固定训练预算下，不同 latent code 生成机制如何影响连续场重构、采样泛化、潜在空间结构、下游算子学习以及推理效率？

---

## 2. Research Gap

现有 Functa、CORAL 及其扩展通常采用 auto-decoding，通过测试时优化获得每个函数实例的 latent。其他工作则分别采用 CNN、Point/Set Encoder 或 Transformer 生成隐式表示参数，但这些方法往往同时改变 latent 结构、decoder、modulation 和训练目标，因此难以判断性能差异究竟来自编码器，还是来自整个架构变化。

目前缺少以下受控比较：

1. 固定同一 Shared INR 和同一 modulation，只改变 latent code 的生成方式；
2. 同时考察规则网格、不规则点集、稀疏观测和变几何场；
3. 区分“函数重构能力”和“latent 用于下游 operator learning 的能力”；
4. 同时报告重构误差、编码时间、采样分布变化鲁棒性和 latent 几何结构。

因此，本研究将问题定义为：

> 不同编码机制对共享连续函数表示的影响尚未得到系统、受控且跨采样形式的验证。

---

## 3. 主要研究问题

### RQ1：Auto-decoding 是否仍然具有明显的重构优势？

比较其与端到端 encoder 在相同 latent 维度下的重构上限及计算代价。

### RQ2：规则网格先验是否决定了 CNN encoder 的优势？

检验 CNN 在固定网格、随机稀疏采样、局部缺失和跨分辨率条件下的变化。

### RQ3：Point/Set/Transformer encoder 是否更适合不规则采样？

重点比较置换不变性、点数变化和采样位置变化下的稳定性。

### RQ4：重构最优的 latent 是否也最适合下游 operator learning？

检验 latent 的平滑性、插值性以及输入 latent 到输出 latent 的映射难度。

### RQ5：Encoder 初始化加少量 refinement 能否兼顾速度和精度？

研究：

\[
z^{(0)}=E_\phi(\mathcal D),
\]

\[
z^{(k+1)}=z^{(k)}-\eta\nabla_z\mathcal L,
\]

在 1、3、5、10 步 refinement 下是否能够接近完整 auto-decoding。

---

## 4. 研究假设

- **H1：** Auto-decoding 在同等 latent 容量下通常具有最低重构误差，但编码和测试成本最高。
- **H2：** CNN encoder 在规则、对齐网格上表现较强，但对随机采样、局部缺失和跨分辨率测试更敏感。
- **H3：** DeepSets、PointNet 或 Set Transformer 对不规则点集和点数变化更稳健。
- **H4：** 端到端 encoder 的重构误差可能略高，但 latent 流形更规整，更有利于下游 operator learning。
- **H5：** Encoder + 少量 latent refinement 能够显著缩小 amortization gap，并保留快速推理优势。
- **H6：** 单个 global latent 在局部高频、移动前沿和复杂几何情况下更容易形成瓶颈，多 token 或 spatial latent 可作为第二阶段扩展。

---

## 5. 总体实验框架

### 5.1 第一阶段：严格受控比较

固定：

- Shared INR backbone：SIREN 或带 Fourier Features 的 MLP；
- latent 维度：建议 \(d\in\{32,64,128\}\)；
- modulation：统一采用 FiLM；
- 隐藏层数、宽度和激活函数；
- 训练/测试划分；
- observed points 与 query points；
- loss、optimizer 和训练预算。

只替换：

\[
\mathcal D_i\rightarrow z_i
\]

的编码方式。

### 5.2 待比较方法

| 编号 | 方法 | 输入要求 | 主要特点 |
|---|---|---|---|
| B1 | Auto-decoding | 任意采样点 | 重构上限高，测试时需优化 |
| B2 | CNN/UNet Encoder | 规则网格或先栅格化 | 强局部与平移归纳偏置，仅用于规则场或栅格化输入 |
| B3 | DeepSets/PointNet | 无序点集 | 置换不变，计算较轻 |
| B4 | Set Transformer | 无序点集 | 建模点间全局关系 |
| B5 | Perceiver/Cross-Attention Encoder | 任意点数 | 将点集压缩为固定 latent/token |
| B6 | Encoder + Refinement | 任意采样 | 速度与重构精度折中 |

第一阶段所有方法统一输出单个 global latent：

\[
z_i\in\mathbb R^d.
\]

不设置 Flatten + MLP 基线。对于数千个无序非结构点，展平输入要求固定点数和固定排列，会人为引入不存在的序列结构；点的轻微重排也会使输入向量发生大幅变化，因此它不适合作为跨规则网格与非结构网格的公共编码器。若需要一个轻量端到端下限基线，使用 DeepSets 的逐点 MLP 加 mean/sum pooling 更合理。

### 5.3 第二阶段：表示结构上限比较

在完成严格对比后，再允许不同 latent 拓扑：

- global latent；
- spatial latent grid；
- multiple latent tokens；
- coordinate-dependent cross-attention latent；
- local feature map + INR decoder。

该阶段研究的是“编码结构与连续解码器的整体上限”，不能与第一阶段混为同一个公平性结论。

---

## 6. 合成函数簇设计

建议构造一套 **可控多机制合成函数簇（Controlled Function Family Suite）**。所有函数定义在：

\[
(x,y)\in[0,1]^2.
\]

每个函数由低维参数 \(\lambda\) 生成，可精确控制内在维数、频率、局部性、位移和非光滑程度。

### 6.1 子函数簇 A：可控频谱混合

定义：

\[
f_{\lambda}(x,y)=
\sum_{m=1}^{K}
a_m
\sin\left(2\pi(k_mx+l_my)+\phi_m\right).
\]

参数包括：

\[
\lambda=\{a_m,k_m,l_m,\phi_m\}_{m=1}^{K}.
\]

建议设置：

- \(K\in\{2,4,8,16\}\)；
- 低频：\(k_m,l_m\in[1,4]\)；
- 中频：\([5,12]\)；
- 高频：\([13,32]\)；
- 混合频谱：同时采样低频和高频。

用途：

- 检验频谱复杂度增加时不同 encoder 的退化；
- 检验 latent 是否能平滑表示频率变化；
- 对比 FiLM、GFM 或 Fourier feature 是否真正改善高频表示。

### 6.2 子函数簇 B：局部多尺度 Gabor 结构

定义：

\[
f_{\lambda}(x,y)=
\sum_{m=1}^{K}
a_m
\exp\left(-\frac{\|R_{\theta_m}([x,y]^T-c_m)\|^2}{2\sigma_m^2}\right)
\cos\left(2\pi\omega_m^TR_{\theta_m}([x,y]^T-c_m)+\phi_m\right).
\]

其中：

- \(c_m\) 控制局部结构位置；
- \(\sigma_m\) 控制局部尺度；
- \(\omega_m\) 控制局部频率；
- \(\theta_m\) 控制方向。

用途：

- 模拟尾迹、局部涡、热斑和局部高频结构；
- 检验 global latent 是否丢失局部信息；
- 比较 CNN、Point/Set Encoder 和多 token 表示。

### 6.3 子函数簇 C：输运与几何形变

从基础模板 \(f_0\) 出发：

\[
f_{\lambda}(x,y)=f_0(T_{\lambda}^{-1}(x,y)),
\]

其中 \(T_{\lambda}\) 包含：

- 平移；
- 旋转；
- 缩放；
- 剪切；
- 平滑非刚性形变。

可选基础模板：

\[
f_0(x,y)=\exp\left(-\frac{(x-0.5)^2+(y-0.5)^2}{2\sigma^2}\right),
\]

或多个涡旋/Gaussian 混合。

用途：

- 检验 encoder 对移动结构的编码能力；
- 检验 latent 插值是否对应物理上的连续形变；
- 检验 CNN 平移先验、Set Transformer 和 Point/Set Encoder 的差异。

### 6.4 子函数簇 D：移动前沿与近不连续结构

定义：

\[
f_{\lambda}(x,y)=
\tanh\left(\frac{d_{\lambda}(x,y)}{\varepsilon}\right),
\]

其中 \(d_{\lambda}(x,y)\) 是点到参数化曲线的有符号距离。例如：

\[
d_{\lambda}(x,y)=y-
\left[a\sin(2\pi kx+\phi)+b\right].
\]

通过调整 \(\varepsilon\) 控制前沿锐度。

用途：

- 模拟激波、界面、边界层或相变前沿；
- 研究高梯度区域误差；
- 验证全局频谱方法与局部空间编码的适用边界。

### 6.5 子函数簇 E：混合机制函数簇

最终可构造：

\[
f_{\lambda}
=
\alpha f_{\text{spectral}}
+\beta f_{\text{local}}
+\gamma f_{\text{transport}}
+\delta f_{\text{front}},
\]

并控制：

\[
\alpha,\beta,\gamma,\delta\ge 0.
\]

这样可以逐步增加函数簇复杂度，并观察不同方法何时失效。

### 6.6 合成数据划分

建议每个子函数簇：

- 训练：5,000–20,000 个函数实例；
- 验证：500–2,000；
- 测试：1,000–2,000；
- 每个函数原始生成 \(128\times128\) 高分辨率真值；
- 输入观测点比例：100%、50%、25%、10%、5%；
- query points 与 observed points 分离。

测试集同时包含：

1. 参数插值；
2. 参数边界；
3. 频率外推；
4. 不同采样率；
5. 不同采样模式；
6. 含噪观测。

---

## 7. CORAL 数据集建议

不建议一开始全部使用。推荐主实验采用：

### 7.1 Navier–Stokes

代表规则域、动态场和多时间状态，适合 CNN、Set Transformer/Perceiver 和 auto-decoding 的直接比较。

### 7.2 Cylinder-flow 或 Airfoil-flow

代表不规则网格/点集场，用于检验采样位置变化和点数变化下的编码鲁棒性。二者选一个作为主实验，另一个作为附录扩展。

### 7.3 NACA-Euler

代表变几何静态流场，更接近几何到流场预测，也可用于下游 latent operator 学习。

### 7.4 可选：Elasticity

用于说明结论不只适用于流体问题。

推荐最小组合：

\[
\boxed{
\text{合成函数簇}
+
\text{Navier–Stokes}
+
\text{Cylinder/Airfoil-flow}
+
\text{NACA-Euler}
}
\]

---

## 8. 采样条件设计

每个实际数据集至少构造以下输入模式：

1. 固定规则采样；
2. 每个实例随机采样；
3. 局部区域缺失；
4. 边界附近加密；
5. 训练与测试采用不同点数；
6. 训练低分辨率、测试高分辨率；
7. 添加 0.5%、1%、2%、5% 噪声。

这部分是区分 encoder 和 auto-decoding 的关键，不能只复现 CORAL 原始设置。

---

## 9. 评价指标

### 9.1 函数重构指标

- MSE；
- relative \(L_2\) error；
- PSNR（适合规则图像式场）；
- 未观测区域误差；
- 跨分辨率查询误差。

### 9.2 局部与频谱指标

高梯度区域误差：

\[
E_{\text{grad}}
=
\frac{1}{|\Omega_g|}
\int_{\Omega_g}|\hat u-u|^2dx,
\]

其中 \(\Omega_g\) 由真值梯度分位数定义。

频段误差：

\[
E_{\text{high}}
=
\sum_{\|\omega\|>\omega_c}
|\hat u(\omega)-u(\omega)|^2.
\]

### 9.3 编码与推理效率

- 单实例 latent 生成时间；
- refinement 时间；
- Shared INR 查询时间；
- 峰值显存；
- 训练总 GPU 时长；
- 达到固定误差阈值所需时间。

### 9.4 Latent 质量

- 参数邻域保持程度；
- latent kNN 与真实参数 kNN 的一致性；
- latent 线性/球面插值误差；
- latent 到物理参数的线性探测性能；
- 不同随机种子下 latent 对齐稳定性。

### 9.5 下游 operator learning

训练：

\[
G_\psi:z_{\mathrm{in}}\rightarrow z_{\mathrm{out}},
\]

并评价最终解码后的场误差。

需要分别报告：

- latent prediction error；
- decoded field error；
- rollout error；
- 几何外推/参数外推误差。

---

## 10. 关键消融实验

1. latent 维度：32、64、128、256；
2. Shared INR 宽度与深度；
3. FiLM、Shift、Scale 三种统一 modulation；
4. observed/query 点是否重叠；
5. encoder 训练时是否随机改变采样点；
6. auto-decoding 内循环步数；
7. encoder refinement 步数；
8. 相同训练 step 与相同 wall-clock 两种公平预算；
9. global latent 与 multiple tokens；
10. 是否加入 latent regularization 或 contrastive consistency。

---

## 11. 实验公平性原则

- 第一阶段所有方法输出相同维度的 global latent；
- Shared INR 与 FiLM 参数完全共享；
- 参数量尽量控制在 ±10%；
- 同时报告等训练步数和等计算时间两组结果；
- observed points 仅用于编码，query points 用于评估，防止重构泄漏；
- 至少使用 3 个随机种子，关键结果建议 5 个；
- 不用单个最优样本展示误差图，应给出中位样本和最差分位样本；
- 对每个 encoder 单独调参，但公开搜索范围和预算。

---

## 12. 预期可能得到的结论

本研究不应预设某一种 encoder 必然最好，而是可能形成条件化结论：

- 规则网格、密集观测：CNN encoder 性价比较高；
- 不规则采样、点数变化：Set/Transformer encoder 更稳健；
- 极稀疏或分布外采样：Auto-decoding 仍具有优势；
- 快速前向任务：端到端 encoder 更适合；
- 高精度重构：Encoder + refinement 可能是最佳折中；
- 局部结构复杂时：单 global latent 可能成为主要瓶颈；
- 重构误差最低的 latent 不一定最适合下游 operator learning。

这种结果比“提出一种新的 modulation 并优于几个基线”更有公共研究价值。

---

## 13. 分阶段科研计划

### 阶段 1：复现与统一框架（第 1–2 个月）

- 复现 CORAL/Functa 的 shared INR 与 auto-decoding；
- 建立统一数据接口：规则网格与不规则点集；
- 实现 CNN、DeepSets/PointNet、Set Transformer 和 Perceiver/Cross-Attention Encoder；
- 固定 global latent + FiLM decoder；
- 在一个小型合成频谱函数簇上调通流程。

**阶段结果：** 统一代码框架和第一版公平性基线。

### 阶段 2：合成函数簇机制实验（第 3–4 个月）

- 完成频谱、局部 Gabor、输运和前沿四类函数簇；
- 控制内在维度、频率和采样率；
- 分析 encoder 退化边界；
- 增加 encoder + refinement。

**阶段结果：** 形成主要机制图和 research gap 的实验证据。

### 阶段 3：CORAL 实际任务验证（第 5–7 个月）

- Navier–Stokes；
- Cylinder/Airfoil-flow；
- NACA-Euler；
- 构造稀疏、缺失、跨分辨率和含噪设置；
- 比较重构、效率和采样泛化。

**阶段结果：** 完成主实验表格和实际 PDE 结论。

### 阶段 4：下游 operator 与扩展架构（第 8–9 个月）

- 训练 latent-to-latent operator；
- 比较不同 latent 的下游可学习性；
- 增加 spatial latent 和 multiple tokens；
- 分析 global bottleneck 的边界。

**阶段结果：** 完成 operator 实验和扩展消融。

### 阶段 5：论文整理（第 10 个月）

- 整理 Intro 中的预实验观察；
- 将每个现象对应到完整实验；
- 明确结论边界，不声称普适最优；
- 完成代码、配置和数据生成脚本整理。

---

## 14. 论文结构建议

1. **Introduction**：公共任务、auto-decoding 成本、缺少受控 encoder 比较、预实验误差分布；
2. **Related Work**：Functa/CORAL、generalizable INR、set encoding、implicit decoder；
3. **Problem Formulation**：采样点到 latent、Shared INR 和公平比较协议；
4. **Methods under Comparison**：不同编码器与 refinement；
5. **Controlled Synthetic Suite**：函数簇与复杂度控制；
6. **Experiments**：重构、采样泛化、效率、latent 结构、operator；
7. **Analysis**：amortization gap、global bottleneck、不同采样先验；
8. **Limitations**：数据集范围、encoder 参数预算、3D 扩展和真实实验数据；
9. **Conclusion**：给出条件化选择建议，而非宣称单一方法全面占优。

---

## 15. 最核心的论文贡献表达

建议将贡献限定为：

1. 提出一套固定 Shared INR 与 latent 预算的公平比较协议；
2. 构造可控制频谱、局部性、输运和前沿复杂度的合成函数簇；
3. 系统比较 auto-decoding、像素编码、点集编码和 Transformer 编码；
4. 揭示重构性能、采样泛化、编码效率和 latent 下游可学习性之间的权衡；
5. 验证 encoder + 少量 refinement 作为快速连续场表示的折中方案。

最终研究结论应当回答：

> 对不同采样形式和函数簇结构，应如何选择 latent code 生成方式，而不是再提出一个只在特定数据分布上占优的新调制模块。