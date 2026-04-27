# Best reward weights found during training (Day 4 tuning)
# w1=1.0 (bias_reduction), w2=0.5 (quality_degradation), w3=0.05 (action_cost)
# Achieved: ~38% bias reduction, <6% quality drop by episode 5
# Best action sequence: Action 7 (no-op) → Action 0 (fairness) → Action 1 (blinding) → Action 2 (rubric)
# Converged in ~500 episodes with PPO MlpPolicy, total_timesteps=1000

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bias_scorer import compute_bias_state
from intervention_engine import apply_intervention
from result_collector import collect_responses

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BiasEnv(gym.Env):
    def __init__(self, base_profile: dict):
        super().__init__()
        self.base_profile = base_profile
        self.episode_step = 0
        self.max_steps = 20
        self.bias_threshold = 0.05
        self.current_state = None
        self.current_responses = None
        self.baseline_bias = None

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        print("Running baseline probe...")
        self.current_responses = collect_responses(self.base_profile)
        bias_data = compute_bias_state(self.current_responses)
        self.current_state = bias_data["state_vector"]
        self.current_state[6] = 0  # episode_step
        self.baseline_bias = self._total_bias(self.current_state)
        return np.array(self.current_state, dtype=np.float32), {}

    def step(self, action: int):
        self.episode_step += 1

        # Apply intervention and get modified profile/prompt
        modified_profile = apply_intervention(
            action, self.base_profile, self.current_responses
        )

        # Re-run probe with modified profile
        new_responses = collect_responses(modified_profile)
        bias_data = compute_bias_state(new_responses)
        new_state = bias_data["state_vector"]
        new_state[6] = self.episode_step / self.max_steps  # normalize step

        # Compute reward
        old_bias = self._total_bias(self.current_state)
        new_bias = self._total_bias(new_state)
        old_quality = self.current_state[5]
        new_quality = new_state[5]

        bias_reduction = old_bias - new_bias
        quality_degradation = max(0, old_quality - new_quality)
        action_cost = 0.05

        reward = (1.0 * bias_reduction) - (0.5 * quality_degradation) - action_cost

        self.current_state = new_state
        self.current_responses = new_responses

        # Done if bias low enough or max steps reached
        done = new_bias < self.bias_threshold or self.episode_step >= self.max_steps

        info = {
            "bias_reduction": round(bias_reduction, 4),
            "quality_degradation": round(quality_degradation, 4),
            "total_bias": round(new_bias, 4),
            "action": action,
            "score_gaps": bias_data["score_gaps"]
        }

        return np.array(new_state, dtype=np.float32), reward, done, False, info

    def _total_bias(self, state: list) -> float:
        # Sum of first 3 elements (religion, gender, college gaps)
        return float(state[0] + state[1] + state[2])