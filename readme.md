# Action-Aware Markov Chain for Sequential Recommendation

This repository contains the reference implementation used in our study on **lightweight sequential recommendation with action-aware Markov chains**, developed for an academic submission (ACM Hypertext).

The goal is to evaluate how different history weighting strategies affect recommendation quality under a fixed Markov transition model.

---

## Method

User behavior is modeled as a sequence of **action–item states**  
(e.g. `listen_123`, `like_456`).  
A global first-order Markov chain is estimated from interaction logs:

$$
P(s' \mid s) = \frac{C(s, s')}{\sum_x C(s, x)}
$$

At recommendation time, candidate items are aggregated from multiple recent
states instead of using only the last interaction.

---

## History Weighting (Decay)

Given the last `DEPTH` states ({(s_k, t_k)}), scores are aggregated as

$$
\text{score}(i) = \sum_k w_k \, P(i \mid s_k)
$$

Three weighting schemes are supported:

### Time decay (DecayPop)
Exponential decay based on timestamps:

$$
w_k \propto \exp\left(-\ln(2)\cdot \frac{t_1 - t_k}{h}\right)
$$

with half-life `h = 21.6` hours (as used in the paper).

### Order-based exponential decay
Time-agnostic positional decay:

$$
w_k \propto \gamma^{k-1}
$$

### Order-based linear decay
Linearly decreasing weights with depth:

$$
w_k \propto 1 - (k-1)\frac{1-\lambda}{K-1}
$$

---

## Evaluation

- Dataset: **Yambda-50M**
- Temporal split following the official Yambda protocol
- Metrics:
  - **Recall@N**
  - **NDCG@N**
- Typical settings: `N ∈ {10, 100}`, `DEPTH ∈ [5, 100]`

All results reported in the paper are produced with this code.

---

## Project Structure

```
.
├── download_dataset.py      # Dataset loading
├── markov_chain_polars.py   # Markov chain construction (Polars)
├── state_weights.py         # History weighting schemes
├── run_experiment.py        # Main experiment script
└── README.md
```

---

## Running Experiments

Time-based decay (DecayPop, as in the paper):

```bash
python run_experiment.py
```

Order-based exponential decay:

```python
DECAY_MODE = "order_exp"
GAMMA = 0.85
```

Order-based linear decay:

```python
DECAY_MODE = "order_linear"
MIN_W = 0.05
```

---

## Notes

- The method is **training-free** and requires no parameter learning.
- All experiments use the same Markov transition matrix; only history weighting changes.
- The implementation mirrors the logic used in the experimental notebook.
- Results are fully reproducible under the Yambda evaluation protocol.

---

## License

Released for research.
