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
