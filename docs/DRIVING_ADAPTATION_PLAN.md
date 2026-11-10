# 改造计划：驾驶场景的潜空间可达性安全过滤器（WOMD 向量化路线）

目标：实现"面向学习型驾驶策略、带统计保证的潜空间安全过滤器"。
方法来自 latent_cbf，世界模型用 CAT-K 的 SMART，闭环环境和失败数据用 CtRL-Sim，STL 监控参考 pSTL。

说明：本文件基于 latent_cbf 的代码阅读，以及另外三个仓库的 README 和文件结构写成。没有运行过任何脚本，也没有读完三个外部仓库的源码。工期均为粗略估计。

## 0. 总体架构

| 模块 | 来源 | 作用 |
|---|---|---|
| 世界模型 | CAT-K 的 SMART，冻结 | 在 GPU 上做多智能体 token 回放；自车的 agent 特征作为潜状态 z |
| 被过滤策略 | CAT-K 的 `ego_gmm`，或任意输出运动 token 的自车策略 | 提出名义动作 |
| 安全边界 l(z) | 新写，训练方式沿用 latent_cbf | 从 z 回归 STL 鲁棒度，带梯度惩罚 |
| 值函数 Q(z, a) | 新写，备份公式沿用 PyHJ | 在 SMART 想象中做离散动作的 HJ avoid 学习 |
| 过滤器 | 新写，逻辑沿用 latent_cbf | 在 top-K token 中屏蔽 Q 低于阈值 δ 的，选离名义动作最近的 |
| 阈值 δ | 新写 | 由 conformal 标定 |
| "真实"环境 | CtRL-Sim 加 Nocturne | 生成失败数据，提供标定真值，做最终闭环评估 |
| STL 监控器 | 新写，参考 pSTL 的 `stl_d_lib.py` | 在 Nocturne 特权状态上计算鲁棒度 ρ |

关键性质：世界模型（SMART）和评估环境（CtRL-Sim 加 Nocturne）是两个独立的模型。值函数在前者里学，真值在后者里取，conformal 的结论因此不是自己验证自己。

## 1. 设计决策

### 已定

- **数据集**：WOMD。三个模块都围绕它，不需要跨数据集迁移。
- **动作空间**：SMART 的离散运动 token，每个 token 对应 0.5 秒。过滤器以 2 Hz 运行。
- **HJ 求解器**：离散 Q 学习，沿用 `PytorchReachability/PyHJ/policy/modelfree/BasePolicy_Annealing_Avoid.py:387` 的折扣 avoid 备份。不再使用 DDPG。
- **多智能体语义先用方案 A**：他车由 SMART 采样，值函数是采样转移下的，安全性是概率意义的，由 conformal 给界。

### 待确认

1. **许可**。CAT-K 是 NVIDIA Source Code License，我只读了开头，用途限制需要你通读。WOMD 本身也有使用条款。pSTL 仓库没有许可证，只参考写法，STL 库自行实现。
2. **SMART 权重**。优先凭 Waymo 注册向作者索取 SMART-tiny 权重。拿不到则自训 `smart_nano_1M`（单张 A100 可训，作者报告其指标明显低于 tiny）。
3. **"端到端"的定位**。WOMD 上的策略以轨迹和地图为输入，属于 mid-to-mid。两个选择：把标题改成"学习型规划器"，或者在最后加一个传感器输入的实验（见阶段 6）。
4. **方案 B 是否纳入**。用 CtRL-Sim 的 tilt 充当扰动方，做博弈版 HJ。建议作为消融，主线稳定后再决定。

## 2. 分阶段计划

### 阶段 0：环境与基线（约 2 周）

- 在 Linux 或 WSL2 上复现 latent_cbf 的 Dubins 流程。目的是吃透方法和超参数（`gamma_pyhj`、`gp_weight`、`gradient_thresh`），不是为了复用代码。
- 搭好 CAT-K 环境，跑通 `scripts/cache_womd.sh` 和 `scripts/local_val.sh`。
- 确认现有的 CtRL-Sim 环境可用，跑通 `eval_planner.py`。
- **完成标准**：三个仓库各自的官方流程都能在同一台机器上跑通；许可问题有结论。

### 阶段 1：可行性验证（约 1 到 2 周，是否继续的关口）

要回答的问题：SMART 的自车特征里有没有足够的安全信息。

1. 在 WOMD 日志和少量带碰撞的回放上，提取每个时刻的自车 agent 特征 z。
2. 用最简单的标签（是否碰撞、到最近车辆的距离）训练一个小 MLP 头。
3. 检查留出集上的可分性和回归误差，并看特征随时间的变化是否平滑。

- **通过标准**：碰撞分类 AUROC 高，距离回归误差在可用范围内。具体阈值在看到第一批结果后定。
- **不通过的退路**：回到 Dreamer RSSM，自己写向量编码器和解码器，其余阶段不变。这条退路能最大限度复用 latent_cbf 的代码。

### 阶段 2：数据桥接、失败数据与 STL 标注（约 3 到 4 周）

1. **格式桥接**。Nocturne 用 json 场景，CAT-K 用自己缓存的 pickle。写一个转换器，把 Nocturne 闭环回放出的状态序列转成 SMART 的输入格式。这是本阶段最容易低估的工作。
2. **失败数据**。用 CtRL-Sim 的对抗策略（`cfgs/policy/ctrl_sim_adversary.yaml`）对自车策略做闭环回放，收集碰撞和险情。已知负向 veh-veh tilt 会饱和，所以还要补充：不同 tilt 强度、自车动作加噪、不同训练程度的自车策略。
3. **STL 监控器**。新增 `src/latent_cbf/stl/`，自行实现 Always、Eventually、Until 和原子命题的鲁棒度。
   - 首批规则：不碰撞、不出可行驶区域、最小 TTC 或安全距离。
   - 输入是 Nocturne 的特权状态，输出每个时刻的 ρ_t，`failure = (ρ_t < 0)`。
   - CtRL-Sim 离线数据里已有按时刻的碰撞和出界奖励分量，可用来交叉检查。
4. **数据集**。每条样本：SMART 输入、自车 token、ρ_t。

- **完成标准**：数据集中违规片段占比足够（目标先定 20% 以上），STL 标签与 CtRL-Sim 的碰撞标记一致。

### 阶段 3：潜空间 HJ（约 3 到 4 周）

1. **批量想象环境**。包一层 `src/smart/utils/rollout.py`：给定场景前缀和自车 token 序列，批量推进并返回每步的 z。替代 latent_cbf 里 batch 为 1 的 `dubins-wm-dp.py`。
2. **边界函数头 l(z)**。回归截断后的 ρ_t，梯度惩罚照搬 `dreamer_offline.py:250` 的 `_margin_gp_step`。同时训练一个二值标签版本作对比。
3. **Q 网络**。输入 z，输出每个候选 token 的 Q 值。token 词表有上千个，先只对策略的 top-K 或一个聚类后的小词表建模。
4. **训练**。离散 avoid 备份：目标为 γ·min(l(z), max_a' Q(z', a')) + (1−γ)·l(z)。沿用 latent_cbf 的做法，先令 γ 为 0 预热，再升到接近 1。想象起点一半来自数据，一半沿名义策略展开。
5. **有限时域**。WOMD 场景只有 9 秒，去掉 1 秒历史后最多 16 个 token。值函数的含义是"在场景剩余时间内保持安全"，论文中要写明。

- **完成标准**：在留出场景上，Q 的符号与 SMART 想象回放的实际违规一致率高；值函数随 token 步进的变化合理。

### 阶段 4：过滤器与 Nocturne 闭环（约 3 周）

1. **过滤器**。新增 `LatentSafetyFilter`：名义 token 的 Q 高于阈值则放行，否则在 Q 合格的 top-K token 中选离名义动作最近的。同时实现 least-restrictive 版本（直接切换到 argmax Q）。
2. **动作落地**。SMART token 是 0.5 秒的位移，Nocturne 的自车接口是加速度和转向，或直接设置位姿。先用直接设置位姿，再评估是否需要逆动力学跟踪。
3. **接入评估**。把"自车策略加过滤器"注册成 CtRL-Sim 的一个规划器，复用 `evaluators/planner_adversary_evaluator.py`。

- **完成标准**：在对抗和非对抗两种他车设置下，过滤后的 STL 违规率显著下降，干预率和进度损失有记录。此时阈值仍是手调的。

### 阶段 5：conformal 标定（约 2 到 3 周）

新增 `src/latent_cbf/conformal/calibrate.py`。

1. **真值**。对标定状态 s，在 Nocturne 中从 s 出发执行安全策略（argmax Q）直到场景结束，`y = 1` 当且仅当全程 ρ_t ≥ 0。
2. **标定集**。从带过滤器的闭环回放中抽取 N 个状态，场景与训练集不相交。
3. **界**。取 `y = 0` 的标定状态上 Q 值的 (1−α) 分位数作为 δ，则对新的不安全状态有 P(Q > δ) ≤ α，即"把不安全误判为安全"的概率有界。
4. **验证**。独立测试场景上报告经验误判率随 α 的变化，检查是否落在界内。

**已知风险**：conformal 要求标定分布与部署分布可交换，而部署分布取决于过滤器本身，δ 一变分布就变。应对是迭代标定一到两轮，并在论文中明确写出假设。这是最容易被审稿人质疑的点。

- **完成标准**：测试集上的经验误判率不超过 α（在有限样本误差范围内）。

### 阶段 6：实验与论文（约 4 周，部分与阶段 5 并行）

- **指标**：STL 违规率、碰撞率、干预率、干预幅度、进度或目标到达率、经验误判率。
- **基线**：无过滤；只用 l(z) 做一步前瞻（无 HJ）；阈值为 0 的未标定 HJ 过滤器；基于 TTC 的规则制动；STL guidance（参考 pSTL 的做法）。
- **消融**：二值标签对 STL 鲁棒度；有无梯度惩罚；CBF 式对 least-restrictive；不同 α 和标定集大小；方案 A 对方案 B。
- **可选的传感器实验**：若要保住"端到端"的说法，在 CARLA 或 NAVSIM 上补一个小规模实验。工期另算，不在下面的里程碑内。

## 3. latent_cbf 代码的去留

| 部分 | 去留 |
|---|---|
| `dreamerv3_torch/`、`diffusion4robotics/`、Dubins 环境 | 主线不用，只在阶段 0 复现和阶段 1 的退路中使用 |
| PyHJ 的 avoid 备份和 γ 预热 | 保留公式，改写成离散 Q 版本 |
| `_margin_gp_step` 的梯度惩罚训练 | 基本照搬 |
| `FilteredDiffusionController` 的 CBF 和 LR 逻辑 | 保留思路，按 token 动作重写 |
| PyHJ 的 Collector、Trainer 框架 | 不用，批量想象下自己写训练循环更简单 |

实际从 latent_cbf 直接复用的代码不多，它在这条路线里主要是方法和超参数的参考。

## 4. 风险清单

| 风险 | 影响 | 应对 |
|---|---|---|
| SMART 自车特征不含足够安全信息，或非马尔可夫性导致 Q 学不稳 | 主线不成立 | 阶段 1 的关口；退路是 Dreamer 加向量编码器 |
| Nocturne 与 SMART 的格式和频率不一致（10 Hz 对 2 Hz） | 阶段 2、4 延期 | 尽早写转换器并用日志回放做一致性测试 |
| 失败数据单一（tilt 饱和） | l 和 Q 在边界附近不准 | 多种来源混合；报告按失败类型分组的结果 |
| SMART 想象与 Nocturne 中的他车行为不一致 | Q 在部署时偏乐观 | conformal 的真值取自 Nocturne，偏差会体现在 δ 上；报告两者的差距 |
| conformal 的可交换性假设 | 统计保证站不住 | 迭代标定；明确写出假设 |
| 算力和权重 | 工期 | 先索取权重；否则用 nano 配置 |
| "第一个"的声明不成立 | 贡献被削弱 | 阶段 2 结束前系统查新，重点查潜空间安全过滤器加 conformal、以及驾驶中学习型 HJ 过滤器的已有工作 |

## 5. 里程碑

| 里程碑 | 累计时间 | 可交付物 |
|---|---|---|
| M0 环境与基线 | 第 2 周 | 三个仓库跑通，许可结论 |
| M1 可行性关口 | 第 4 周 | 特征可分性报告，继续或退回的决定 |
| M2 数据与 STL | 第 8 周 | 带 STL 标签和失败片段的数据集 |
| M3 潜空间 HJ | 第 12 周 | 训练好的 l(z) 和 Q |
| M4 闭环过滤 | 第 15 周 | Nocturne 中的过滤前后对比 |
| M5 conformal | 第 18 周 | 带统计界的过滤器 |
| M6 实验完成 | 第 22 周 | 全部基线和消融 |

## 6. 参考仓库

| 仓库 | 可用的部分 | 限制 |
|---|---|---|
| NVlabs/catk | `src/smart/utils/rollout.py`、`configs/model/ego_gmm.yaml`、`configs/model/smart_nano_1M.yaml` | NVIDIA Source Code License；SMART-tiny 需 8 张 A100 训练数天；权重需向作者索取 |
| montrealrobotics/ctrl-sim | `eval_planner.py`、`evaluators/planner_adversary_evaluator.py`、`cfgs/policy/ctrl_sim_adversary.yaml` | MIT 许可；依赖 Singularity 和 Nocturne 编译 |
| mengyuest/pSTL-diffusion-policy | `stl_d_lib.py` 的写法、驾驶 STL 规则模板、STL guidance 基线 | 无许可证，只参考不拷贝 |
