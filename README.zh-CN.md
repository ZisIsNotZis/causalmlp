<p align="center">
  <img src="assets/logo.svg" width="112" height="112" alt="causalmlp logo">
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

# causalmlp

### 一个 MLP，每个隐藏宽度都能给出预测。

**一次前向传播，既给出早期答案，也给出逐步变好的答案——不需要额外的输出头，也不需要第二个模型。**

`causalmlp` 是一个小而可复现的 PyTorch 实验，研究“因果隐藏单元”：把普通 MLP 改造成“前 `i` 个隐藏单元本身就是一个分类器”的形式，让每个前缀长度 `i` 都能独立给出预测。真正的问题不是它能不能训练，而是准确率会不会随着解锁更多隐藏单元自然上升，以及它相对普通 MLP 要付出什么代价。这个仓库保留了那次运行的原始结果和研究报告，也保留了失败与不完整的结论。

## 为什么做这个

常见的 early-exit 方案会另加分类器，或者同时训练多个网络。这个实验问的是一个更窄的问题：能否只用一个共享权重的 MLP，把所有中间宽度都变成可用的出口，用隐藏单元贡献的累积均值来支撑每个出口，让它在自己的宽度上就有意义？

实验给出的简短回答是：累积**均值**归一化很关键；均值和求和的差异使因果模型成为一个与普通 MLP 不同的函数；想追上基线，需要明显更多的隐藏宽度。完整过程，包括没有成功的部分，见 [report.md](report.md)。

## 一览

| | |
|---|---|
| **任务** | 64 维特征、10 类的合成分类任务 |
| **核心机制** | 对各隐藏单元的对数几率贡献取累积均值 |
| **输出** | `[batch, hidden_size, num_classes]`——每个前缀宽度一个出口 |
| **变体** | 共享输出权重，或每个出口一个独立输出头 |
| **基线** | 相同隐藏宽度的普通 MLP |
| **证据** | [results.json](results.json) 中的一次保存运行，以及 [report.md](report.md) 中的叙述 |
| **状态** | 研究归档——不是软件包，没有测试套件，也没有 CI |

> **指标提醒。** [results.json](results.json) 里，因果模型那一列是每个出口的 `exp(-交叉熵)` 均值，而 `regular_acc` 是 top-1 准确率。两者不是同一种指标，所以保存的 `0.755` 与 `0.889` **不能**当作可直接相减的准确率差距。待修复的问题记录在 [docs/plan/evaluation.md](docs/plan/evaluation.md)。

## 快速开始

需要 Python 和 PyTorch。主脚本可以在 CPU 上运行；`experiment.py` 里的精简对比是按 CUDA 写的。

```bash
git clone https://github.com/ZisIsNotZis/causalmlp.git
cd causalmlp
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install torch numpy matplotlib tqdm
python train.py --device cpu --hidden-size 128 --epochs 20
```

运行后会生成 `results.json`、`training_results.png` 和 `per_hidden_accuracy.png`。

精简的“参考模型对因果模型”对比（需要 CUDA）：

```bash
python experiment.py
```

快速检查导入和输出形状：

```bash
python causal_mlp.py
```

## 现在能做什么

- 一次前向传播，在每个有效隐藏宽度上都给出预测。
- 用累积均值归一化隐藏贡献，避免原始累积求和的尺度漂移。
- 训练时可以选择加入单调性惩罚。
- 在同一个合成任务上与普通 MLP 对比。
- 保存每个出口的指标和训练曲线，便于检查。
- 通过 `--model cumsum` 使用共享输出权重，或通过 `--model heads` 使用每个出口独立的输出头。

## 结构

```text
x ──► Linear ──► SiLU ──► h[0] … h[H-1]
                              │
                 每个单元的贡献  h[j] · W2[:, j]
                              │
                    对 j ≤ i 取累积均值
                              │
                    出口 i ──► 类别 logits
```

出口 `i` 只能使用隐藏单元 `0..i`。最后一个出口是所有单元贡献的均值，而不是总和；这正是因果 MLP 无法与同宽度普通 MLP 完全等价的架构原因。

| 路径 | 作用 |
|---|---|
| [`causal_mlp.py`](causal_mlp.py) | 模型变体、因果损失、逐出口指标 |
| [`train.py`](train.py) | 可直接运行的训练/评估流程和绘图 |
| [`experiment.py`](experiment.py) | 多种隐藏宽度下的精简 CUDA 实验 |
| [`mymlp.py`](mymlp.py)、[`mymlp2.py`](mymlp2.py) | 作者更早的参考实验 |
| [`report.md`](report.md) | 发现、对比和开放的研究问题 |
| [`results.json`](results.json) | 一次运行保存下来的逐出口结果 |
| [`docs/`](docs/) | 项目状态与评估修复记录 |

## 如何开发

这个仓库没有打包、测试套件或 CI，它刻意保持成一个小的研究草稿。修改后请保证模块仍可导入，合成实验仍能在仓库根目录运行。可用的手动检查：

```bash
python causal_mlp.py
python train.py --device cpu --epochs 1
```

如果改动会影响报告中的数字，请写清它是怎么产生的，并绑定到代码版本、数据、随机种子和指标定义。不要把 `exp(-交叉熵)` 和 top-1 准确率混在一起比较。

## 如何贡献

欢迎提 issue 和 pull request。简短流程与证据要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。核心只有几句：让结论可复现，让指标各自独立，把归档结果当作需要限定条件的证据，而不是需要维护的装饰。

## 路线图

### 已实现

- [x] 用累积均值聚合隐藏单元，为每个前缀宽度提供出口。
- [x] 共享权重和逐出口独立输出头两种变体。
- [x] 合成数据上的普通 MLP 基线。
- [x] 可选的单调性正则项。
- [x] 逐出口评估、绘图和结果保存。
- [x] 覆盖累积均值与累积求和、均值/求和的理论差异、以及观察到的单调性的研究报告。

### 计划与开放问题

- [ ] **修复评估结论：**追溯 `results.json` 的生产者与代码版本，再在同一批留出样本上用 top-1 准确率比较两个模型。
- [ ] 增加一个回归测试样例，让 top-1 准确率和平均分配概率故意不同。
- [ ] 在真实数据集上验证该机制，而不只是在合成特征上。
- [ ] 尝试权重约束（例如 `W ≥ 0`）能否消除剩余的单调性违反。
- [ ] 把思路扩展到单隐藏层之外。
- [ ] 评估早期出口能否在简单样本上真正节省计算。

## 未来设想

一个网络先给出粗略答案，再根据可用算力逐步细化：简单输入走早期出口，困难输入多用一些隐藏单元，而且不必为了拥有这个选项再训练第二个模型。这个仓库朝这个想法迈出了一小步，也把它的限制写在了明处。下一步最有价值的不是更大的结论，而是更可信的测量。

## 许可证

GNU AGPL-3.0-only，见 [LICENSE](LICENSE)。版权所有 © ZisIsNotZis。参考脚本（`mymlp.py`、`mymlp2.py`）是作者自己更早的代码，不是第三方代码。
