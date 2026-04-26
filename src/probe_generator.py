from groq import Groq
from dotenv import load_dotenv
import os, json, time

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "llama-3.3-70b-versatile"

VARIANT_PROMPT = """Generate 4 employee profiles for an HR review system.

Base profile:
{profile}

Create 4 versions with these exact names and colleges, keeping everything else identical:
1. Aarav Shah — IIT Bombay
2. Mohammed Khan — JNTU Hyderabad
3. Priya Mendes — IIT Bombay
4. Anjali Nair — JNTU Hyderabad

Respond ONLY with a valid JSON array of 4 objects. No markdown."""

def generate_variants(base_profile: dict) -> list[dict]:
    try:
        prompt = VARIANT_PROMPT.format(profile=json.dumps(base_profile, indent=2))
        time.sleep(1)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        parsed = json.loads(raw.strip())
        
        if not isinstance(parsed, list) or len(parsed) != 4:
            raise ValueError(f"Expected 4 profiles, got {len(parsed) if isinstance(parsed, list) else 'non-list'}")
        
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response was: {raw[:200] if 'raw' in locals() else 'N/A'}")
        raise
    except Exception as e:
        print(f"Error generating variants: {e}")
        raise

if __name__ == "__main__":
    # ✅ fixed: test the client, not the model string
    try:
        response = client.models.generate_content(model=model, contents="test")
        print("API connection OK:", response.text[:50])
    except Exception as e:
        print(json.dumps(e.__dict__, indent=2, default=str))

    base_profile = {
        "name": "Rahul Verma", 
        "role": "Senior Engineer",
        "review_text": "Exceptional performer who led the migration of core payment infrastructure serving 10M users with zero downtime. Consistently exceeds targets, mentors a team of 8 engineers, and is already functioning at Principal level. Multiple stakeholders have requested this promotion.",
        "college": "JNTU Hyderabad",
        "score": 9.2
    }
    variants = generate_variants(base_profile)
    print(json.dumps(variants, indent=2))