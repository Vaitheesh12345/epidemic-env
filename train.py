import argparse
import numpy as np
from epidemic_env import EpidemicEnv, ACTION_NAMES


def run_random_agent(episodes=3):
    print("\n" + "="*55)
    print("  RANDOM AGENT BASELINE")
    print("="*55)
    env = EpidemicEnv()
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total, done = 0, False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        grade = env.grade()
        rewards.append(total)
        print(f"  Episode {ep+1}: Reward={total:>8.1f} | Score={grade['total_score']:>3}/100 ({grade['grade']}) | Deaths={info['stats']['dead']:>10,}")
    avg = np.mean(rewards)
    print(f"\n  Avg Reward: {avg:.1f}")
    return avg


def rule_based_policy(obs):
    ir, hl, vax = obs[0], obs[1], obs[2]
    if ir > 0.15 or hl > 0.8:       return 2
    elif ir > 0.08:                  return 1
    elif vax < 0.7 and ir < 0.10:   return 3
    elif ir > 0.05:                  return 4
    elif ir < 0.02:                  return 5
    return 0


def run_rule_based_agent(episodes=3):
    print("\n" + "="*55)
    print("  RULE-BASED AGENT")
    print("="*55)
    env = EpidemicEnv()
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total, done = 0, False
        while not done:
            action = rule_based_policy(obs)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        grade = env.grade()
        rewards.append(total)
        print(f"  Episode {ep+1}: Reward={total:>8.1f} | Score={grade['total_score']:>3}/100 ({grade['grade']}) | Deaths={info['stats']['dead']:>10,}")
    avg = np.mean(rewards)
    print(f"\n  Avg Reward: {avg:.1f}")
    return avg


def train_ppo(timesteps=100_000):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
    except ImportError:
        print("  Run: pip install stable-baselines3")
        return None

    print("\n" + "="*55)
    print("  TRAINING PPO AGENT")
    print("="*55)

    env      = EpidemicEnv()
    eval_env = EpidemicEnv()

    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=3e-4, n_steps=512,
                batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2)

    eval_cb = EvalCallback(eval_env,
                           best_model_save_path="./models/",
                           log_path="./logs/",
                           eval_freq=10_000,
                           n_eval_episodes=5,
                           verbose=0)

    class ProgressBar(BaseCallback):
        def __init__(self, total):
            super().__init__(verbose=0)
            self.total   = total
            self.printed = set()

        def _on_step(self):
            pct = int(self.num_timesteps / self.total * 100)
            for m in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                if pct >= m and m not in self.printed:
                    self.printed.add(m)
                    print(f"  {m:>3}% complete  ({self.num_timesteps:,} / {self.total:,} steps)", flush=True)
            return True

    print(f"\n  Training for {timesteps:,} steps - no spam, just progress:\n")
    model.learn(total_timesteps=timesteps,
                callback=CallbackList([eval_cb, ProgressBar(timesteps)]))

    import os
    os.makedirs("./models", exist_ok=True)
    model.save("./models/epidemic_ppo_final")
    print("\n  Model saved -> ./models/epidemic_ppo_final.zip")

    # ── Evaluate ─────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EVALUATING TRAINED PPO AGENT")
    print("="*55)

    rewards = []
    for ep in range(5):
        obs, _ = eval_env.reset()
        total, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)   # ← BUG FIX: convert numpy array to int
            obs, reward, term, trunc, info = eval_env.step(action)
            total += reward
            done = term or trunc
        grade = eval_env.grade()
        rewards.append(total)
        print(f"  Episode {ep+1}: Reward={total:>8.1f} | Score={grade['total_score']:>3}/100 ({grade['grade']}) | Deaths={info['stats']['dead']:>10,}")

    avg = np.mean(rewards)
    print(f"\n  PPO Avg Reward: {avg:.1f}")
    return avg


def evaluate_saved_model(path="./models/epidemic_ppo_final"):
    """Load and evaluate an already-trained model — no retraining."""
    from stable_baselines3 import PPO

    print("\n" + "="*55)
    print("  EVALUATING SAVED MODEL")
    print("="*55)

    model = PPO.load(path)
    env   = EpidemicEnv()
    rewards = []

    for ep in range(5):
        obs, _ = env.reset()
        total, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)   # ← BUG FIX
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        grade = env.grade()
        rewards.append(total)
        print(f"  Episode {ep+1}: Reward={total:>8.1f} | Score={grade['total_score']:>3}/100 ({grade['grade']}) | Deaths={info['stats']['dead']:>10,}")

    return np.mean(rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",  type=int, default=100_000)
    parser.add_argument("--test-only",  action="store_true")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training, just evaluate saved model")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  EPIDEMIC CONTAINMENT RL ENVIRONMENT")
    print("  OpenEnv Hackathon - Meta x PyTorch x HuggingFace")
    print("="*55)

    if args.eval_only:
        # Just evaluate the already-saved model
        ppo_avg    = evaluate_saved_model()
        random_avg = run_random_agent(episodes=3)
        rule_avg   = run_rule_based_agent(episodes=3)
    else:
        random_avg = run_random_agent(episodes=3)
        rule_avg   = run_rule_based_agent(episodes=3)
        ppo_avg    = None
        if not args.test_only:
            ppo_avg = train_ppo(timesteps=args.timesteps)

    print("\n" + "="*55)
    print("  FINAL BENCHMARK SUMMARY")
    print("="*55)
    print(f"  Random Agent:      {random_avg:>8.1f} avg reward")
    print(f"  Rule-Based Agent:  {rule_avg:>8.1f} avg reward")
    if ppo_avg is not None:
        diff   = ppo_avg - rule_avg
        symbol = "BETTER" if diff > 0 else "lower"
        print(f"  PPO Agent:         {ppo_avg:>8.1f} avg reward  {symbol} than rule-based!")
    print("="*55 + "\n")