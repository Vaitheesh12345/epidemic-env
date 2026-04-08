from epidemic_env.env import EpidemicEnv
from stable_baselines3 import PPO
import numpy as np

env = None
obs = None

def reset():
    global env, obs
    env = EpidemicEnv()
    obs, _ = env.reset()
    return {"status": "reset", "observation": obs.tolist()}

def step(action):
    global env, obs
    obs, reward, terminated, truncated, info = env.step(action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated)
    }

def predict(obs_input):
    model = PPO.load("models/epidemic_ppo_final.zip")
    obs_array = np.array(obs_input)
    action, _ = model.predict(obs_array, deterministic=True)
    return {"action": int(action)}