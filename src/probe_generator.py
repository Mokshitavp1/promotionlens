from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_model = "llama-3.3-70b-versatile"

VARIANT_PROMPT = """Generate 4 employee profiles for an HR review system.

Base profile:
{profile}

Create 4 versions with these exact names and colleges, keeping everything else identical:
1. Aarav Shah – IIT Bombay
2. Mohammed Khan – JNTU Hyderabad
3. Priya Mendes – IIT Bombay
4. Anjali Nair – JNTU Hyderabad

Respond ONLY with a valid JSON array of 4 objects. No markdown, no explanations."""

def generate_variants(base_profile: dict) -> list[dict]:
    """Generate demographic variants using Groq API"""
    try:
        prompt = VARIANT_PROMPT.format(profile=json.dumps(base_profile, indent=2))
        
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        variants = json.loads(raw.strip())
        
        if not isinstance(variants, list) or len(variants) != 4:
            raise ValueError(f"Expected 4 profiles, got {len(variants) if isinstance(variants, list) else 'non-list'}")
        
        return variants
        
    except Exception as e:
        print(f"Error generating variants: {e}")
        raise


if __name__ == "__main__":
    base_profile = {
        "name": "Rahul Verma",
        "role": "Senior Engineer",
        "review_text": "Consistently delivers high quality work and leads projects effectively.",
        "college": "IIT Bombay",
        "score": 8.5
    }
    
    print("Generating variants with Groq API...")
    results = generate_variants(base_profile)
    print(json.dumps(results, indent=2))
    
    with open("mock_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to mock_output.json")