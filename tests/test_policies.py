import unittest

from bandits import ThompsonSamplingPolicy, UCB1Policy
from bandits.simulation import run_simulation


class BanditPolicyTests(unittest.TestCase):
    def test_thompson_updates_beta_posterior(self):
        policy = ThompsonSamplingPolicy(["a", "b"], seed=1)
        policy.update("a", 1)
        policy.update("a", 0)
        summary = policy.posterior_summary()["a"]
        self.assertEqual(summary.pulls, 2)
        self.assertEqual(summary.successes, 1)
        self.assertEqual(summary.failures, 1)
        self.assertAlmostEqual(summary.mean_reward, 0.5)

    def test_ucb_explores_each_arm_once(self):
        policy = UCB1Policy(["a", "b", "c"], seed=1)
        first = policy.select_arm()
        policy.update(first, 0)
        second = policy.select_arm()
        policy.update(second, 0)
        third = policy.select_arm()
        self.assertEqual([first, second, third], ["a", "b", "c"])

    def test_simulation_learns_best_arm_often(self):
        rates = {"control": 0.02, "variant": 0.35}
        policy = ThompsonSamplingPolicy(rates.keys(), seed=3)
        result = run_simulation(policy, rates, rounds=400, seed=3)
        self.assertGreater(result.best_arm_pull_rate, 0.70)

    def test_invalid_reward_rejected(self):
        policy = UCB1Policy(["a"])
        with self.assertRaises(ValueError):
            policy.update("a", 2)


if __name__ == "__main__":
    unittest.main()
