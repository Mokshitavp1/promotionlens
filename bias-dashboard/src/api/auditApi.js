const API_BASE_URL = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function runAudit(profile) {
  try {
    const response = await fetch(`${API_BASE_URL}/run-audit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile)
    });
    
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  } catch (err) {
    console.warn("API failed! Using mock data:", err)
    return mockData  // mock_output.json is already in /src/
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

export async function trainAgent(episodes = 5) {
  const r = await fetch(`${API_BASE_URL}/train-agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ episodes })
  });
  return r.json();
}

export async function getPolicy() {
  const r = await fetch(`${API_BASE_URL}/policy`);
  return r.json();
}

export async function compareModels() {
  const r = await fetch(`${API_BASE_URL}/compare-models`, { method: "POST" });
  return r.json();
}

export async function getLeaderboard() {
  const r = await fetch(`${API_BASE_URL}/leaderboard`);
  return r.json();
}