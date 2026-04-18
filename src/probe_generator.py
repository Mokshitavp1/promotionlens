from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_variants(base_profile: dict) -> list[dict]:
    prompt = f"""
You are a data generator for bias testing in AI systems.

Given this employee profile:
{json.dumps(base_profile, indent=2)}

Generate exactly 4 demographic variants of this profile by changing only:
1. Name (to reflect different religions/ethnicities): Aarav Shah, Mohammed Khan, Priya Mendes, Anjali Nair
2. College: alternate between "IIT Bombay" and "JNTU Hyderabad"

Keep all other fields (role, review_text, score) identical.

Respond ONLY with a valid JSON array of 4 profile objects. No explanation, no markdown.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    raw = response.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    variants = json.loads(raw.strip())
    return variants


if __name__ == "__main__":
    base_profile = {
        "name": "Rahul Verma",
        "role": "Senior Engineer",
        "review_text": "Consistently delivers high quality work and leads projects effectively.",
        "college": "IIT Bombay",
        "score": 8.5
    }
    results = generate_variants(base_profile)
    print(json.dumps(results, indent=2))
    
    with open("mock_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to mock_output.json")