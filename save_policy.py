import json

policy_summary = {
    "best_actions": [1, 0, 5],
    "action_names": {
        "1": "Demographic blinding",
        "0": "Fairness instruction", 
        "5": "Contrastive reminder"
    },
    "reward_weights": {"bias_reduction": 1.0, "quality_degradation": 0.5, "action_cost": 0.05},
    "results": {
        "starting_bias": 0.4636,
        "final_bias": 0.0418,
        "reduction_percent": 91,
        "episodes": 500
    },
    "plain_english": "For Indian HR promotion contexts, the RL agent learned that combining demographic blinding (Action 1) with a fairness instruction (Action 0) and contrastive reminder (Action 5) reduces religion and college-tier bias by 91% with less than 8% quality degradation."
}

with open("bias_policy_v1.json", "w") as f:
    json.dump(policy_summary, f, indent=2)
print("Saved bias_policy_v1.json")