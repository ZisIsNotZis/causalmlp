<p align="center">
  <img src="assets/logo.svg" width="112" height="112" alt="causalmlp logo">
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a>
</p>

# causalmlp

### One MLP, a usable prediction at every hidden size.

**A single forward pass yields an early answer and progressively better ones — no extra output heads, no second model.**

`causalmlp` is a small, reproducible PyTorch study of *causal hidden units*. A standard MLP is reshaped so that the first `i` hidden units already form a classifier, and every prefix length `i` produces its own prediction. The interesting question is not whether it trains, but whether accuracy rises naturally with each unlocked unit — and what it costs compared with an ordinary MLP. The saved run and the research report keep the negative and partial findings, not just the wins.

## Why this exists

Early-exit models usually add separate classifiers or train multiple networks. This experiment asks a narrower question: can one weight-shared MLP expose all of its intermediate widths as exits, using a cumulative mean over hidden contributions, so that each exit is meaningful on its own?

The short answer from the experiments: cumulative **mean** normalization matters, the mean-vs-sum difference makes the causal model a genuinely different function from a standard MLP, and matching the baseline takes substantially more hidden width. The full account, including what failed, is in [report.md](report.md).

## At a glance

| | |
|---|---|
| **Task** | Synthetic 64-feature, 10-class classification |
| **Core mechanism** | Cumulative mean of per-hidden-unit logit contributions |
| **Output** | `[batch, hidden_size, num_classes]` — one exit per prefix width |
| **Variants** | Shared output weights, or a separate head per exit |
| **Baseline** | Standard MLP with the same hidden width |
| **Evidence** | One saved run in [results.json](results.json); narrative in [report.md](report.md) |
| **Status** | Research archive — no package, test suite, or CI |

> **Metric caveat.** In [results.json](results.json), the causal series is mean `exp(-cross-entropy)` per exit, while `regular_acc` is top-1 accuracy. They are different metrics, so the saved `0.755` versus `0.889` should **not** be read as an apples-to-apples accuracy gap. The open repair work is recorded in [docs/plan/evaluation.md](docs/plan/evaluation.md).

## Quick start

Requires Python and PyTorch. The main script runs on CPU; the compact comparison in `experiment.py` is written against CUDA.

```bash
git clone https://github.com/ZisIsNotZis/causalmlp.git
cd causalmlp
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install torch numpy matplotlib tqdm
python train.py --device cpu --hidden-size 128 --epochs 20
```

This writes `results.json`, `training_results.png`, and `per_hidden_accuracy.png`.

For the compact reference-versus-causal comparison (CUDA required):

```bash
python experiment.py
```

For a quick import-and-shape self-check:

```bash
python causal_mlp.py
```

## What it can do now

- Produce predictions at every effective hidden width from one forward pass.
- Normalize cumulative contributions with a mean, avoiding the scale drift of a raw cumulative sum.
- Optionally add a monotonicity penalty during training.
- Compare against a standard MLP on the same synthetic task.
- Save per-exit metrics and training plots for inspection.
- Choose shared output weights (`--model cumsum`) or one head per exit (`--model heads`).

## Architecture

```text
x ──► Linear ──► SiLU ──► h[0] … h[H-1]
                              │
                 per-unit contribution  h[j] · W2[:, j]
                              │
                    cumulative mean over j ≤ i
                              │
                    exit i ──► class logits
```

Exit `i` can only use hidden units `0..i`. The final exit is the mean of all per-unit contributions, not their sum; that is the architectural reason the causal MLP cannot become identical to an ordinary MLP of the same width.

| Path | Role |
|---|---|
| [`causal_mlp.py`](causal_mlp.py) | Model variants, causal loss, per-exit metrics |
| [`train.py`](train.py) | Runnable training/evaluation loop and plots |
| [`experiment.py`](experiment.py) | Compact CUDA experiment over several hidden widths |
| [`mymlp.py`](mymlp.py), [`mymlp2.py`](mymlp2.py) | The owner's earlier reference experiments |
| [`report.md`](report.md) | Findings, comparisons, and open research questions |
| [`results.json`](results.json) | Saved per-exit output from one run |
| [`docs/`](docs/) | Project status and the evaluation-repair note |

## How to develop

There is no packaging, test suite, or CI; this is a deliberately small research scratchpad. Keep changes importable and the synthetic experiment runnable from the repository root. A useful manual check after a change:

```bash
python causal_mlp.py
python train.py --device cpu --epochs 1
```

If you change a reported number, record how it was produced and bind it to the code revision, data, seed, and metric definition. Do not mix `exp(-cross-entropy)` with top-1 accuracy in a comparison.

## How to contribute

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the short workflow and the evidence expectations. In brief: keep claims reproducible, keep metrics distinct, and treat the archived results as evidence to qualify rather than decoration to preserve.

## Roadmap

### Implemented

- [x] Cumulative-mean hidden aggregation for per-prefix exits.
- [x] Shared-weight and per-exit-head variants.
- [x] Standard MLP baseline on synthetic data.
- [x] Optional monotonicity regularization.
- [x] Per-exit evaluation, plots, and a saved result file.
- [x] Research report covering cummean vs cumsum, the mean-vs-sum limit, and observed monotonicity.

### Planned and open

- [ ] **Repair the evaluation claim:** trace `results.json` to its producer and revision, then compare both models with top-1 accuracy on the same held-out examples.
- [ ] Add a regression fixture where top-1 accuracy and mean assigned probability deliberately differ.
- [ ] Validate the mechanism on a real dataset rather than synthetic features.
- [ ] Test whether weight constraints (for example `W ≥ 0`) can remove the remaining monotonicity violations.
- [ ] Extend the idea beyond a single hidden layer.
- [ ] Evaluate whether early exits can save computation on easy samples.

## Future vision

A single network that can answer early and refine as compute allows: route an easy input to an early exit, spend more hidden units on a hard one, and never train a second model to get the option. This repository is a small, honest step toward that idea, with its limits written down rather than hidden. The next useful move is not a bigger claim but a trustworthy measurement.

## License

GNU AGPL-3.0-only. See [LICENSE](LICENSE). The reference scripts (`mymlp.py`, `mymlp2.py`) are the owner's own earlier code, not third-party code.
