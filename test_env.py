from src.bias_env import BiasEnv

base_profile = {
    "name": "Rahul Verma",
    "role": "Senior Engineer",
    "review_text": "Shows potential but inconsistent delivery. Has good ideas but struggles to drive them to completion independently. Colleagues find them easy to work with.",
    "college": "JNTU Hyderabad",
    "score": 6.8
}

env = BiasEnv(base_profile)

print("=== RESET ===")
state, _ = env.reset()
print(f"Initial state: {state}")

print("\n=== STEP with Action 1 (blind names) ===")
new_state, reward, done, _, info = env.step(1)
print(f"New state: {new_state}")
print(f"Reward: {reward}")
print(f"Info: {info}")