# run this as check_models.py in root
import requests
import os
from dotenv import load_dotenv

load_dotenv()

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
)

models = response.json()["data"]

# filter free ones only
free_models = [
    m for m in models 
    if m.get("pricing", {}).get("prompt") == "0"
]

print(f"Found {len(free_models)} free models:\n")
for m in free_models:
    print(m["id"])