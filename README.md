# IDinsight / Evidential Bandits Proof

Small standalone proof for C4GT DMP 2026 Issue #766.

It demonstrates a modular bandit policy interface that could map into Evidential's allocation layer:

- `BanditPolicy` interface
- Thompson Sampling for Beta-Bernoulli rewards
- UCB1 for deterministic upper-confidence allocation
- replayable simulator
- unit tests for selection, updates, posterior summaries, and regret-ish behavior

## Run

```bash
uv run python -m unittest discover -s tests
uv run python simulate.py
```

## Why this shape

For Evidential, the important design point is not only the algorithm. It is having a stable allocation contract:

```python
arm = policy.select_arm(context=None)
policy.update(arm, reward)
summary = policy.posterior_summary()
```

That contract can later support:

- Thompson Sampling
- UCB
- contextual policies using user-level features
- audit logging for assignment decisions
- dashboard summaries for nonprofits
