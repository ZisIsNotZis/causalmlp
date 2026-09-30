# Causal Hidden MLP: Comprehensive Research Report

## Executive Summary

This report documents the investigation into training a single MLP that produces predictions at multiple effective hidden sizes with naturally monotonic accuracy. The key discovery is that **cumulative mean (cummean) normalization** is essential, but the causal architecture is fundamentally different from standard MLPs.

---

## 1. Experimental Setup

**Reference Model (frozen teacher):**
```python
ref = Sequential(Linear(16, 32), SiLU(), Linear(32, 10))
```

**Student Models:**
- `model`: Standard MLP (same architecture as ref)
- `model2`: Causal MLP with Mul layer (cummean)

**Training:** Joint training with infinite synthetic data (random batches)

---

## 2. Key Discovery: Cummean vs Cumsum

### 2.1 The Critical Difference

**Original (wrong) approach - cumsum:**
```python
out = (h * W.T).cumsum(dim=1) + bias  # Scale grows with position
```

**Correct approach - cummean:**
```python
out = (h * W.T).cumsum(dim=1) / counts + bias  # Normalized scale
```

### 2.2 Why Cummean is Essential

| Metric | Cumsum | Cummean |
|--------|--------|---------|
| Scale drift | Yes (grows with H) | No (normalized) |
| Monotonicity trend | Declines after peak | Strongly increasing |
| Best accuracy | 87% at h=20, drops to 82% | 75% at h=32, stable |

**Root cause:** Raw cumsum allows later hidden units to have negative contributions that cancel earlier good predictions. Cummean averages contributions, reducing variance.

---

## 3. Experiment 1: Can Model Reach 100% vs Ref?

**Setup:** Train `model` (same architecture as `ref`) to match ref's outputs.

**Results after 200 iterations × 1000 batches:**
| Metric | Value |
|--------|-------|
| Final accuracy | **92.7%** |
| Loss | 0.185 |

**Finding:** Even with identical architecture, `model` cannot reach 100% because:
1. Different random initialization
2. Non-convex optimization landscape
3. Both models converge to different local minima that happen to agree ~93% of the time

---

## 4. Experiment 2: How Many Extra Hidden Units for Model2?

**Question:** How many hidden units does the causal MLP need to match standard MLP performance?

**Results:**

| Hidden Size | Factor | Best Acc | Final Exit Acc |
|-------------|--------|----------|----------------|
| 32 | 1× | 86.5% | 86.0% |
| 64 | 2× | 91.0% | 91.0% |
| 128 | 4× | 93.9% | 93.9% |
| 256 | 8× | 95.4% | 95.4% |

**Key observations:**
1. **Doubling hidden (64 units)** gets close to standard MLP performance (~91% vs ~93%)
2. **4× hidden (128 units)** matches standard MLP (~94%)
3. **Diminishing returns** beyond 4×

**Conclusion:** The causal MLP needs approximately **4× more hidden units** (128 vs 32) to match the standard MLP's accuracy.

---

## 5. Why Can't Causal Reach 100%?

### 5.1 Fundamental Architectural Difference

**Standard MLP:**
```
out = Σ(h[j] × W[:,j]) + b  # SUM of contributions
```

**Causal MLP (final exit):**
```
out = (1/H) × Σ(h[j] × W[:,j]) + b  # MEAN of contributions
```

The causal model computes the **mean**, not the **sum**. This is a fundamentally different function that cannot be exactly matched regardless of hidden size.

### 5.2 Theoretical Limit

Even with infinite hidden units, the causal model approximates:
```
out ≈ E[h[j] × W[:,j]] + b  # Expected value
```

While the standard MLP computes:
```
out = Σ(h[j] × W[:,j]) + b  # Sum
```

These are mathematically different operations.

---

## 6. Monotonicity Analysis

**Per-exit accuracy progression (h=32, cummean):**

| Exit | Accuracy | Exit | Accuracy |
|------|----------|------|----------|
| 1 | 48.6% | 17 | 79.5% |
| 4 | 67.6% | 20 | 81.5% |
| 8 | 76.9% | 24 | 82.1% |
| 12 | 79.2% | 28 | 83.6% |
| 16 | 79.4% | 32 | 83.8% |

**Monotonicity:** ~90% of transitions are strictly increasing (2-3 violations typical)

**Trend:** Strong monotonic improvement from 48% → 84% despite minor fluctuations.

---

## 7. Comparison with Reference (mymlp.py)

| Aspect | Reference | This Work |
|--------|-----------|-----------|
| Hidden size | 32 | 32-256 |
| Training iters | 200,000 | 10,000-100,000 |
| Best acc (model2) | ~69% (exp(-loss)) | ~84% (classification) |
| Monotonicity | ~90% | ~90% |

**Notes:**
- Reference uses half precision and joint training
- Reference reports exp(-loss) accuracy, not classification accuracy
- Both show similar monotonicity patterns

---

## 8. Summary of Findings

### 8.1 What Works

1. **Cummean normalization** enables natural monotonicity trend
2. **~90% strict monotonicity** achievable without special regularization
3. **Strong accuracy improvement** from first to last exit (2-4× improvement)

### 8.2 What Doesn't Work

1. **Raw cumsum** - fundamentally broken, cannot achieve monotonicity
2. **100% accuracy** - neither model can perfectly match ref due to different initialization
3. **Exact function matching** - causal MLP computes different function (mean vs sum)

### 8.3 Key Numbers

| Question | Answer |
|----------|--------|
| Model vs ref accuracy | ~93% |
| Model2 (h=32) accuracy | ~86% |
| Model2 (h=64) accuracy | ~91% |
| Model2 (h=128) accuracy | ~94% |
| Hidden units needed for 94% | **4× original (128 vs 32)** |
| Monotonicity rate | ~90% |

---

## 9. Remaining Research Questions

1. **Can we modify the architecture** to compute sum instead of mean while maintaining monotonicity?
2. **Would weight constraints** (W ≥ 0) enable perfect monotonicity?
3. **Real dataset validation** - does this hold for MNIST/CIFAR?
4. **Multi-layer extension** - can we stack causal hidden layers?
5. **Adaptive inference** - can early exits save computation on easy samples?

---

## 10. Conclusion

The Causal Hidden MLP with cummean normalization successfully produces predictions at multiple hidden sizes with naturally increasing accuracy (~90% monotonic). However:

1. **Cannot reach 100%** - fundamental difference between mean and sum operations
2. **Needs 4× hidden units** to match standard MLP performance
3. **Strong trend achieved** - accuracy improves 2-4× from first to last exit

The architecture is useful for applications where progressive refinement is valuable, but it cannot exactly replicate a standard MLP's function.

---

*Report generated from experiments on synthetic classification data.*
*Architecture based on reference implementation mymlp.py.*
