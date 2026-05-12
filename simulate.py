from bandits import ThompsonSamplingPolicy, UCB1Policy
from bandits.simulation import run_simulation

true_rates = {"control": 0.08, "sms_reminder": 0.12, "ai_followup": 0.18}

for policy in [
    ThompsonSamplingPolicy(true_rates.keys(), seed=42),
    UCB1Policy(true_rates.keys(), seed=42),
]:
    result = run_simulation(policy, true_rates, rounds=500, seed=11)
    print(policy.__class__.__name__)
    print("pulls", result.pulls)
    print("cumulative_reward", result.cumulative_reward)
    print("best_arm_pull_rate", round(result.best_arm_pull_rate, 3))
    print("summary", policy.posterior_summary())
    print()
