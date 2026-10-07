# Harness Generalization

## Proposal：面向跨环境迁移的 Agent Harness Adaptation

### 1. 研究背景

最近的研究（如 LIFE-HARNESS）表明，Agent 的失败并不完全来源于 LLM 本身，很多失败实际上发生在模型与环境的交互过程中，例如：

- 没有正确理解工具的使用方式；
- 没有维护当前环境的状态；
- Action 格式不正确；
- 发生错误后不断重复相同的操作；
- 无法判断任务是否真正完成。

因此，越来越多的工作开始将优化目标从模型本身转移到 Harness（推理时的运行框架）。Harness 通常负责 Prompt 构建、Observation 处理、Memory 管理、State Tracking、Tool 调用、Action Validation、Error Recovery 和 Completion Verification。

LIFE-HARNESS 已经证明：即使完全冻结 LLM，仅仅通过优化 Harness，也可以显著提高 Agent 的表现。

### 2. 问题

现有方法大多在单一环境内部进行 Harness 优化：

```text
Environment A → Harness Evolution → Harness A
```

但真实世界中的 Agent 往往需要面对多个不同环境，例如 WebShop、Terminal、Database 和 Customer Service。当 Agent 从一个环境进入另一个环境时，之前学到的 Harness 应该如何处理，仍缺乏研究。

一种方法是在每个新环境中从零开始学习新的 Harness；另一种方法是将已有 Harness 迁移到新环境并继续适应。我们研究后者是否可行，以及哪些 Harness 知识真正具有迁移能力。

### 3. Observation 1：Harness 的趋同进化

不同环境中的 Harness 可能会独立演化出相似策略。例如：

- WebShop：购买前检查商品是否满足所有条件；
- Terminal：修改文件前检查当前状态；
- Airline：修改订单前检查当前订单和规则。

这些规则背后可能遵循共同原则：执行重要操作之前，先验证当前状态。

另一个共同原则是：连续失败后不要无限重复相同操作，而应主动调整策略。这说明 Harness 中可能存在真正可迁移的通用知识。

### 4. Observation 2：Harness Overfitting

随着 Harness 不断进化，它可能逐渐积累越来越多与当前环境相关的规则。Harness evolution 可能存在两个阶段：

```text
Early Evolution → 学习通用策略 → ID ↑ / OOD ↑
Late Evolution  → 积累环境特定规则 → ID ↑ / OOD ↓
```

### 5. 核心假设

一个 Evolution 后的 Harness 可能同时包含两类知识：

- 可迁移知识：状态跟踪、失败恢复、约束验证、结果验证；
- 环境特定知识：商品属性解析、Traceback 解析、Airline Policy。

现有工作并没有明确区分这两种知识。

### 6. 研究问题

- **RQ1**：不同环境中的 Harness 是否会独立学习到相似策略？
- **RQ2**：Harness 中哪些知识具有跨环境的迁移能力？
- **RQ3**：如何利用可迁移知识，帮助 Harness 更快适应新的环境？

### 7. 方法

将 Harness 分解为：

```text
Harness = General Knowledge + Environment-specific Knowledge
```

基本流程为：

```text
WebShop → Harness Evolution → 识别可迁移知识 → Transfer → Terminal → 继续适应
```

目标不是直接迁移整个 Harness，而是提取其中真正具有泛化能力的部分，并利用这些知识帮助 Harness 更快适应新的环境。

### 8. 实验设计

#### Experiment 1：Convergent Evolution

分别在 WebShop、Terminal、Airline 和 Retail 中独立优化 Harness，观察不同环境是否会独立产生相似策略。

#### Experiment 2：Cross-Environment Transfer

比较：

- **Baseline**：Terminal 中从零开始优化 Harness；
- **Ours**：先在 WebShop 中优化 Harness，再迁移到 Terminal 并继续适应。

指标包括 Success Rate、Adaptation Speed、Evolution Rounds 和 Trajectory 数量。

#### Experiment 3：Harness Overfitting

保存 Evolution 过程中的多个 Harness：`H0 → H5 → H10 → H20 → H40`，比较 ID Performance、OOD Performance 和 Harness Complexity，分析 Harness 是否会随着 Evolution 逐渐失去跨环境泛化能力。

已有观察：

| Model | Method | HumanEval | AlfWorld |
|---|---|---:|---:|
| Qwen3-4B | meta-harness | 58.5% | 14.2% |
| Qwen3-4B | life-harness | 54.3% | 18.6% |
| Qwen3-8B | meta-harness | 82.5% | 22.4% |
| Qwen3-8B | life-harness | 80.0% | 32.8% |

## Generalization-aware Harness Evolution

核心思想：在 Harness evolution 时，不只看当前环境性能，还判断每次新增或修改的 Harness 机制是否具有通用性。

```text
原始 Harness
    ↓
在环境 A 中 Evolution
    ↓
产生新的 Harness 修改
    ↓
判断修改的通用性
    ├── 通用机制 → 保留
    └── 环境特定规则 → 降低优先级 / 抑制
    ↓
得到更通用的 Harness
```

### 8.1 Harness Evolution

沿用 LIFE-HARNESS：根据环境反馈定位失败原因并修改 Harness，例如新增状态检查、错误恢复和结果验证。

### 8.2 Modification Generality Evaluation

判断每次修改是否具有通用性。例如，“执行动作前检查当前状态”适用于 WebShop、Terminal 和 Airline；而“如果 WebShop 页面缺少价格则点击某个按钮”只适用于 WebShop，应降低其优先级。

### 8.3 Generalization-aware Selection

修改 Harness 时，不选择当前环境 reward 最高的更新，而选择：

```text
Score = 当前环境收益 + 通用性收益 − 环境依赖惩罚
```

最终目标是让 Harness evolution 学习通用 Agent 能力，而不是只记住当前环境技巧。

## 一句话总结

LIFE-HARNESS 证明了 Harness 可以在单一环境中不断优化，而我们进一步研究：一个 Harness 学到的知识，能否被迁移到新的环境，并帮助 Agent 更快地完成适应？

## 复现材料与轨迹

仓库包含复现实验所需的源代码、配置和数据 split。运行后产生的完整轨迹、模型权重、虚拟环境和日志保留在本地运行目录，不提交到 Git；各实验目录的 README/RESUME 文档记录了启动服务、重建环境和恢复运行的命令。

以 SWE-bench Lite Meta-Harness 为例，一次运行目录的结构是：

```text
runs/<run-name>/
├── evolution.jsonl                 # evolution 输入任务
├── heldout.jsonl                   # heldout 输入任务，只在最后评估
├── baseline/trajectories/<id>/<id>.traj.json
├── iteration-XX/
│   ├── candidate-XX.yaml            # proposer 生成的 candidate
│   ├── candidate_manifest.json
│   ├── proposal.txt
│   ├── proposer_trace.json
│   ├── trajectories/<id>/<id>.traj.json
│   ├── trajectories/preds.json
│   └── evaluation.json
└── final/
    ├── trajectories/<id>/<id>.traj.json
    ├── trajectories/preds.json
    └── summary.json
```

每个 `.traj.json` 是一个 task 的原始 agent 交互轨迹，`preds.json` 是 worker 的汇总预测，`evaluation.json` 保存 evaluator 的 reward/score 和候选结果。每轮 proposer 只访问上一轮 evolution 的反馈和被工具选中的 evolution trajectory；`heldout.jsonl` 及 `final/heldout` 在 proposer 完成全部 evolution 后才使用。因此恢复同一 run 时，必须保留这些轨迹、`preds.json`、`evaluation.json`、candidate 文件和 proposer trace。

`runs/`、旧版 `SWE-bench/trajectories/`、模型和日志由 `.gitignore` 排除，不会误被提交。要复现同一条运行的中断状态，需要另外保存并恢复对应 run 目录；只有 Git 仓库而没有运行产物时，按文档命令可以复现同样的 pipeline，但会从新的 run 状态开始。

本仓库另外通过 Git LFS 提供了两个完整的 SWE-bench Lite run bundle：
`repro_artifacts/qwen3-4b-swe-lite-metaharness-v2-20261005.tar.zst` 和
`repro_artifacts/qwen3-8b-swe-lite-metaharness-20261006.tar.zst`。安装 Git LFS
并执行对应 resume 文档中的解包命令后，可以恢复这些 run 的轨迹和中间状态。
