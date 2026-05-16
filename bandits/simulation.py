from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict

from bandits import BanditPolicy


@dataclass
class SimulationResult:
    pulls: Dict[str, int]
    rewards: Dict[str, float]
    cumulative_reward: float
    best_arm_pull_rate: float


def run_simulation(policy: BanditPolicy, true_rates: Dict[str, float], rounds: int = 1000, seed: int = 7) -> SimulationResult:
    if rounds <= 0:
        raise ValueError("rounds must be greater than 0")
    if not true_rates:
        raise ValueError("at least one true rate is required")

    missing_rates = set(policy.arms) - set(true_rates)
    extra_rates = set(true_rates) - set(policy.arms)
    if missing_rates or extra_rates:
        raise ValueError("policy arms must exactly match true_rates keys")

    invalid_rates = {
        arm: rate
        for arm, rate in true_rates.items()
        if rate < 0 or rate > 1
    }
    if invalid_rates:
        raise ValueError("true_rates values must be probabilities in [0, 1]")

    rng = random.Random(seed)
    pulls = {arm: 0 for arm in true_rates}
    rewards = {arm: 0.0 for arm in true_rates}
    best_arm = max(true_rates, key=true_rates.get)

    for _ in range(rounds):
        arm = policy.select_arm()
        reward = 1 if rng.random() < true_rates[arm] else 0
        policy.update(arm, reward)
        pulls[arm] += 1
        rewards[arm] += reward

    return SimulationResult(
        pulls=pulls,
        rewards=rewards,
        cumulative_reward=sum(rewards.values()),
        best_arm_pull_rate=pulls[best_arm] / rounds,
    )
