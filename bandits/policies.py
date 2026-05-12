from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    pulls: int
    successes: int
    failures: int
    mean_reward: float
    score: float


class BanditPolicy(ABC):
    """Allocation contract for experiment bandit policies."""

    def __init__(self, arms: Iterable[str], seed: Optional[int] = None):
        self.arms = list(arms)
        if not self.arms:
            raise ValueError("at least one arm is required")
        self.rng = random.Random(seed)

    @abstractmethod
    def select_arm(self, context: Optional[dict] = None) -> str:
        """Choose which variant a user should receive."""

    @abstractmethod
    def update(self, arm: str, reward: int | float) -> None:
        """Update policy state after observing an outcome."""

    @abstractmethod
    def posterior_summary(self) -> Dict[str, ArmSummary]:
        """Return dashboard/audit-friendly state."""


class ThompsonSamplingPolicy(BanditPolicy):
    """Beta-Bernoulli Thompson Sampling for binary rewards."""

    def __init__(self, arms: Iterable[str], alpha: float = 1.0, beta: float = 1.0, seed: Optional[int] = None):
        super().__init__(arms, seed=seed)
        self.alpha = {arm: float(alpha) for arm in self.arms}
        self.beta = {arm: float(beta) for arm in self.arms}
        self.pulls = {arm: 0 for arm in self.arms}
        self.successes = {arm: 0 for arm in self.arms}

    def select_arm(self, context: Optional[dict] = None) -> str:
        samples = {
            arm: self.rng.betavariate(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: int | float) -> None:
        self._validate_arm(arm)
        if reward not in (0, 1):
            raise ValueError("ThompsonSamplingPolicy expects binary reward 0 or 1")
        self.pulls[arm] += 1
        self.successes[arm] += int(reward)
        self.alpha[arm] += int(reward)
        self.beta[arm] += 1 - int(reward)

    def posterior_summary(self) -> Dict[str, ArmSummary]:
        out = {}
        for arm in self.arms:
            pulls = self.pulls[arm]
            successes = self.successes[arm]
            failures = pulls - successes
            mean = self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
            out[arm] = ArmSummary(arm, pulls, successes, failures, mean, mean)
        return out

    def _validate_arm(self, arm: str) -> None:
        if arm not in self.alpha:
            raise KeyError(f"unknown arm: {arm}")


class UCB1Policy(BanditPolicy):
    """UCB1 for rewards in [0, 1]."""

    def __init__(self, arms: Iterable[str], exploration: float = 2.0, seed: Optional[int] = None):
        super().__init__(arms, seed=seed)
        self.exploration = exploration
        self.pulls = {arm: 0 for arm in self.arms}
        self.reward_sum = {arm: 0.0 for arm in self.arms}
        self.total_pulls = 0

    def select_arm(self, context: Optional[dict] = None) -> str:
        for arm in self.arms:
            if self.pulls[arm] == 0:
                return arm
        scores = {}
        for arm in self.arms:
            avg = self.reward_sum[arm] / self.pulls[arm]
            bonus = math.sqrt(self.exploration * math.log(self.total_pulls) / self.pulls[arm])
            scores[arm] = avg + bonus
        return max(scores, key=scores.get)

    def update(self, arm: str, reward: int | float) -> None:
        if arm not in self.pulls:
            raise KeyError(f"unknown arm: {arm}")
        if reward < 0 or reward > 1:
            raise ValueError("UCB1Policy expects reward in [0, 1]")
        self.pulls[arm] += 1
        self.reward_sum[arm] += float(reward)
        self.total_pulls += 1

    def posterior_summary(self) -> Dict[str, ArmSummary]:
        out = {}
        for arm in self.arms:
            pulls = self.pulls[arm]
            successes = int(round(self.reward_sum[arm]))
            failures = pulls - successes
            mean = self.reward_sum[arm] / pulls if pulls else 0.0
            if pulls and self.total_pulls > 1:
                score = mean + math.sqrt(self.exploration * math.log(self.total_pulls) / pulls)
            else:
                score = float("inf")
            out[arm] = ArmSummary(arm, pulls, successes, failures, mean, score)
        return out
