import sys
sys.path.append("src")
import json
import os

# Pre-baked training run - simulates RL agent learning
# No live API calls needed - this is the demo fallback per the plan doc

training_log = []
bias = 0.465  # starting bias (matches our mock output state vector sum)

action_names = {
    0: "Fairness instruction",
    1: "Demographic blinding", 
    2: "Scoring rubric",
    3: "Unbiased persona",
    4: "Reframe question",
    5: "Contrastive reminder",
    6: "Score normalisation",
    7: "No-op"
}

for episode in range(1, 501):
    # Agent learns to prefer actions 0, 1, 5 over time
    if episode < 50:
        action = episode % 8  # exploration
    elif episode < 200:
        action = [1, 0, 5, 1, 0, 3, 1, 5][episode % 8]  # mixed
    else:
        action = [1, 0, 5][episode % 3]  # converged to best actions

    # Bias drops faster with good actions
    good_actions = [0, 1, 3, 5]
    if action in good_actions:
        bias = max(0.04, bias - 0.003)
    else:
        bias = max(0.04, bias - 0.0005)

    # Add small noise
    import random
    bias = max(0.04, bias + random.uniform(-0.002, 0.002))

    training_log.append({
        "episode": episode,
        "bias_score": round(bias, 4),
        "action_taken": action,
        "action_name": action_names[action],
        "reward": round((0.465 - bias) * 10, 3)
    })

    if episode % 100 == 0:
        print(f"Episode {episode}: bias={round(bias,4)}")

# Save
with open("training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)

print(f"\n✅ Training complete!")
print(f"Final bias: {training_log[-1]['bias_score']}")
print(f"Starting bias: {training_log[0]['bias_score']}")
print(f"Total reduction: {round(training_log[0]['bias_score'] - training_log[-1]['bias_score'], 3)}")
print("Saved to training_log.json")