# CS728 PA3 — Bonus: Alternative Head Selection Strategies

## How to run

```bash
cd CS728_PA3
python bonus/run3_bonus.py --model <path-to-Llama-3.2-1B-Instruct>
```

Optional args:
```
--k_values 10 20 30      # K values to evaluate (default: 10 20 30)
--train_samples 200      # training queries used for head selection (default: 200)
--seed 64
```

A single Phase 1 pass caches all attention signals, then all 6 strategies × 3 K values are evaluated sequentially.

---

## Strategies

**Group A — Score-function variants** (same independent per-head pipeline, different signal):

| Strategy | Description |
|---|---|
| `reciprocal_rank` | score += 1/(rank+1); partial credit for near-misses |
| `margin` | score += gold\_score − best\_non\_gold; rewards confident separation |
| `avg_attention` | score += mean attention on gold tool; rank-free direct signal |

**Group B — Structurally different selection methods:**

| Strategy | Description |
|---|---|
| `greedy` | Iteratively add the head that most improves combined train Recall@1; accounts for head interactions |
| `entropy` | Pick heads with lowest mean attention entropy over tool tokens (unsupervised, no gold labels) |
| `consistency` | Pick heads most consistently appearing in top-10 best heads across training queries |

---

## Results

### Effect of K — official strategy (from run3.py)

| K  | Recall@1 | Recall@5 |
|----|----------|----------|
| 10 | 0.2520   | 0.5826   |
| 20 | 0.1718   | 0.5702   |
| 30 | 0.1334   | 0.5214   |

---

### Group A: Score-function variants

| Strategy | R@1 K=10 | R@5 K=10 | R@1 K=20 | R@5 K=20 | R@1 K=30 | R@5 K=30 |
|---|---|---|---|---|---|---|
| reciprocal_rank | 0.2430 | 0.5938 | 0.1632 | 0.5112 | 0.0684 | 0.4904 |
| margin          | 0.2542 | 0.4470 | 0.2730 | 0.4964 | 0.2426 | 0.4556 |
| avg_attention   | 0.0506 | 0.5148 | 0.0598 | 0.4528 | 0.0562 | 0.3500 |

### Group B: Structurally different

| Strategy | R@1 K=10 | R@5 K=10 | R@1 K=20 | R@5 K=20 | R@1 K=30 | R@5 K=30 |
|---|---|---|---|---|---|---|
| greedy      | 0.2736 | 0.4900 | 0.2628 | 0.5054 | 0.2468 | 0.4786 |
| entropy     | 0.0112 | 0.0500 | 0.0112 | 0.0532 | 0.0108 | 0.0642 |
| consistency | 0.2444 | 0.5962 | 0.0620 | 0.5668 | 0.0614 | 0.5284 |
