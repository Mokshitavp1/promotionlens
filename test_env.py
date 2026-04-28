# add to test_env.py, replace everything with this
import os
import sys

# Keep env checks reproducible and independent from external LLM quotas.
os.environ.setdefault("LLM_BACKEND", "mock")

sys.path.append("src")
import numpy as np
from bias_env import BiasEnv
from gymnasium.utils.env_checker import check_env

base_profile = {
    "name": "Rahul Verma",
    "role": "Senior Engineer",
    "review_text": "Shows potential but inconsistent delivery. Has good ideas but struggles to drive them to completion independently.",
    "college": "JNTU Hyderabad",
    "score": 6.8
}

env = BiasEnv(base_profile)

# ENV CHECKER
print("=== RUNNING ENV CHECKER ===")
check_env(env)
print("✅ Env checker passed!")

# MANUAL STEP TEST - all 8 actions
print("\n=== MANUAL STEP TEST ===")
state, _ = env.reset()
print(f"Initial state: {state}")

for action in range(8):
    env2 = BiasEnv(base_profile)
    env2.reset()
    new_state, reward, done, _, info = env2.step(action)
    print(f"Action {action}: reward={round(reward,3)} bias={info['total_bias']} bias_reduction={info['bias_reduction']}")