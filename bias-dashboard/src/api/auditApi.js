const API_BASE_URL = "http://localhost:8000";

export async function runAudit(profile) {
  try {
    const response = await fetch(`${API_BASE_URL}/run-audit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile)
    });
    
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  } catch (error) {
    console.error("API call failed:", error);
    throw error;
  }
}

export async function compareCandidates(candidate_a, candidate_b, responses) {
  try {
    const response = await fetch(`${API_BASE_URL}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate_a,
        candidate_b,
        responses
      })
    });
    
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  } catch (error) {
    console.error("Compare API call failed:", error);
    return null;
  }
}