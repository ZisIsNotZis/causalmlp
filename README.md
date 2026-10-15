# causalmlp — Causal Hidden MLP (multi-exit prediction in one pass)

**One-line:** A tiny PyTorch experiment on training a single MLP whose accuracy
improves monotonically as more hidden units are "unlocked", so one forward pass
yields a prediction at every effective hidden size (early-exit / progressive
refinement without extra heads).

**Why it exists.** I wanted to know whether a standard MLP can be reshaped so
that its *prefix* of hidden units is itself a usable classifier, and whether
accuracy then increases naturally with prefix length. `report.md` records the
answer honestly, including the parts that did not work:

- Cumulative **mean** normalization is essential; raw cumulative **sum** drifts
  in scale and fails to be monotonic.
- The causal model computes a *mean* of hidden contributions while a standard
  MLP computes a *sum*, so the two are mathematically different functions and
  the causal MLP cannot reach 100% agreement no matter the width.
- It needs roughly **4× the hidden units** (128 vs 32) to match a plain MLP on
  the same task.
- ~90% of adjacent exits are strictly increasing; a handful of violations
  remain. `results.json` is a real run: `is_monotonic: false`, final max
  accuracy **0.755** vs regular MLP **0.889** (gap 0.134).

This is a negative/partial result on synthetic data, kept because the finding
(mean-vs-sum, monotonicity-without-regularization) is the interesting part.

**How to run.** Requires Python + PyTorch. CUDA is assumed by the small
one-file experiments; `train.py` has a CPU path.

```bash
python train.py --device cpu --hidden-size 128 --epochs 20
```

Outputs `results.json`, `training_results.png`, `per_hidden_accuracy.png`.
For the compact reference-vs-causal comparison:

```bash
python experiment.py        # requires CUDA (coded against .cuda())
```

**Status.** Current (PyTorch 2.x era, written Feb 2026). Uses `torch.compile`,
`silu`, AMP `GradScaler`, `set_float32_matmul_precision`. Only minor drift:
`torch.cuda.amp.GradScaler` / `torch.cuda.amp.autocast` should be spelled
`torch.amp.GradScaler('cuda')` / `torch.amp.autocast('cuda')` on newer PyTorch.
No packaging, no tests, no CI — a research scratchpad that is complete enough to
reproduce. Archived as-is.

**License.** GNU AGPL-3.0-only; see [LICENSE](LICENSE). The report's
"reference implementation" (`mymlp.py`, `mymlp2.py`) is the owner's own earlier
code in the same folder, not third-party code.
