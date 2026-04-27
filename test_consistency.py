# test_consistency.py in root
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import json
from bias_scorer import compute_bias_state

with open("mock_output.json") as f:
    responses = json.load(f)

for i in range(3):
    result = compute_bias_state(responses)
    print(f"Run {i+1}: {result['state_vector']}")