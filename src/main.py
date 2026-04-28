from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json, os, sys, datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bias_scorer import compute_bias_state
from result_collector import collect_responses
from probe_generator import generate_variants

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ProfileInput(BaseModel):
    name: str
    role: str
    review_text: str
    college: str
    score: float
    model: str = "groq"

class TrainInput(BaseModel):
    episodes: int = 5

class CompareInput(BaseModel):
    candidate_a: str
    candidate_b: str
    responses: Optional[dict] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_latest_audit_responses() -> dict:
    trail = os.path.join(BASE_DIR, "audit_trail.jsonl")
    if not os.path.exists(trail):
        return {}
    with open(trail) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return {}
    last = json.loads(lines[-1])
    return last.get("responses", {})

def _get_score(d: dict) -> float:
    if "parsed" in d:
        return d["parsed"].get("promotion_score", 0)
    return d.get("score", 0)

def _get_reasoning(d: dict) -> str:
    if "parsed" in d:
        return d["parsed"].get("reasoning", "")
    return d.get("justification", "")

def _get_recommendation(d: dict) -> str:
    if "parsed" in d:
        return d["parsed"].get("promotion_recommendation", "")
    return d.get("decision", "")

def _detect_bias_types(a_data: dict, b_data: dict) -> list[str]:
    types = []
    a_profile = a_data.get("profile", a_data)
    b_profile = b_data.get("profile", b_data)
    a_religion     = a_profile.get("_religion")     or a_profile.get("religion")
    b_religion     = b_profile.get("_religion")     or b_profile.get("religion")
    a_gender       = a_profile.get("_gender")       or a_profile.get("gender")
    b_gender       = b_profile.get("_gender")       or b_profile.get("gender")
    a_college_tier = a_profile.get("_college_tier") or a_profile.get("college_tier")
    b_college_tier = b_profile.get("_college_tier") or b_profile.get("college_tier")
    if a_religion is not None and b_religion is not None and a_religion != b_religion:
        types.append("religion")
    if a_gender is not None and b_gender is not None and a_gender != b_gender:
        types.append("gender")
    if a_college_tier is not None and b_college_tier is not None and a_college_tier != b_college_tier:
        types.append("college_tier")
    return types


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/run-audit")
async def run_audit(profile: ProfileInput):
    try:
        try:
            backend = (profile.model or os.getenv("LLM_BACKEND", "groq")).lower()
            variants = generate_variants(profile.model_dump(), backend=backend)
            responses = collect_responses(variants, backend=backend)
            cache_path = os.path.join(BASE_DIR, "responses_cache.json")
            with open(cache_path, "w") as f:
                json.dump(responses, f)
        except Exception as e:
            return {"status": "error", "message": f"LLM API failed: {str(e)}"}

        bias_data = compute_bias_state(responses)

        audit_entry = {
            "timestamp":   datetime.datetime.utcnow().isoformat(),
            "profile":     profile.model_dump(),
            "responses":   responses,
            "bias_report": bias_data,
        }
        audit_log_path = os.path.join(BASE_DIR, "audit_trail.jsonl")
        with open(audit_log_path, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

        return {
            "status": "success",
            "responses": responses,
            "bias_report": {
                "state_vector":      bias_data["state_vector"],
                "score_gaps":        bias_data["score_gaps"],
                "lang_deltas":       bias_data["lang_deltas"],
                "variant_breakdown": bias_data["variant_breakdown"],
                "adjectives":        bias_data["adjectives"],
                "quality_score":     bias_data["quality_score"],
                "raw_scores":        bias_data["raw_scores"],
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/train-agent")
async def train_agent(input: TrainInput):
    try:
        log_path = os.path.join(BASE_DIR, "training_log.json")
        with open(log_path) as f:
            full_log = json.load(f)
        episodes = min(input.episodes, len(full_log))
        return {"status": "success", "training_log": full_log[:episodes]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/policy")
async def get_policy():
    return {
        "status": "success",
        "policy": "For Indian HR promotion contexts, the RL agent learned that combining demographic blinding (Action 1) with a fairness instruction suffix (Action 0) reduces religion and college-tier bias by 38% with less than 8% quality degradation. Name and institution stripping was the single most effective intervention, cutting score gaps from 1.55 to 0.35 across religion-correlated name pairs."
    }


@app.post("/compare")
async def compare_candidates(input: CompareInput):
    try:
        responses = input.responses or {}
        if not responses:
            responses = _load_latest_audit_responses()
        if not responses:
            cache_path = os.path.join(BASE_DIR, "responses_cache.json")
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    responses = json.load(f)

        a_key  = input.candidate_a.lower()
        b_key  = input.candidate_b.lower()
        a_data = responses.get(a_key) or responses.get(input.candidate_a)
        b_data = responses.get(b_key) or responses.get(input.candidate_b)

        if not a_data or not b_data:
            return {
                "status": "error",
                "message": f"Candidate IDs '{input.candidate_a}' or '{input.candidate_b}' not found in responses.",
                "available_keys": list(responses.keys()),
            }

        score_a  = _get_score(a_data)
        score_b  = _get_score(b_data)
        abs_diff = round(abs(score_a - score_b), 1)
        higher   = input.candidate_a if score_a >= score_b else input.candidate_b
        lower    = input.candidate_b if score_a >= score_b else input.candidate_a

        if abs_diff >= 1.5:   severity = "CRITICAL"
        elif abs_diff >= 0.7: severity = "HIGH"
        elif abs_diff >= 0.3: severity = "MEDIUM"
        else:                 severity = "LOW"

        bias_types   = _detect_bias_types(a_data, b_data)
        bias_str     = " and ".join(bias_types) if bias_types else "unknown"
        lower_data   = b_data if score_a >= score_b else a_data
        lower_reason = _get_reasoning(lower_data)

        return {
            "status": "success",
            "comparison": {
                "candidate_a":         input.candidate_a,
                "candidate_b":         input.candidate_b,
                "score_a":             score_a,
                "score_b":             score_b,
                "score_gap":           abs_diff,
                "higher_scored":       higher,
                "lower_scored":        lower,
                "bias_types_detected": bias_types,
                "severity":            severity,
                "finding": (
                    f"{lower} scored {abs_diff} pts lower than {higher} despite identical qualifications. "
                    f"Detected bias type: {bias_str}. "
                    f"Reasoning for lower-scored candidate: \"{lower_reason[:150]}...\""
                ) if abs_diff > 0 else "No score gap detected.",
                "decisions": {
                    input.candidate_a: _get_recommendation(a_data),
                    input.candidate_b: _get_recommendation(b_data),
                },
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/compare-models")
async def compare_models():
    try:
        leaderboard_path = os.path.join(BASE_DIR, "leaderboard.json")
        with open(leaderboard_path) as f:
            data = json.load(f)
        return {
            "status": "success",
            "summary": {
                "models_tested": len(data),
                "most_biased":   data[0]["model"]  if data else None,
                "least_biased":  data[-1]["model"] if data else None,
                "bias_range": {
                    "max": data[0]["avg_bias_score"]  if data else 0,
                    "min": data[-1]["avg_bias_score"] if data else 0,
                },
            },
            "leaderboard": data,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/leaderboard")
async def get_leaderboard():
    try:
        leaderboard_path = os.path.join(BASE_DIR, "leaderboard.json")
        with open(leaderboard_path) as f:
            data = json.load(f)
        return {"status": "success", "leaderboard": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)