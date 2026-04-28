const API_BASE_URL = import.meta.env.VITE_API_BASE || "http://localhost:8000"

export async function runAudit(profile) {
  const r = await fetch(`${API_BASE_URL}/run-audit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  })
  if (!r.ok) throw new Error(`API error: ${r.status}`)
  return r.json()
}

export async function compareCandidates(candidate_a, candidate_b, responses) {
  const r = await fetch(`${API_BASE_URL}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate_a, candidate_b, responses }),
  })
  if (!r.ok) throw new Error(`API error: ${r.status}`)
  return r.json()
}

export async function trainAgent(episodes = 5) {
  const r = await fetch(`${API_BASE_URL}/train-agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ episodes }),
  })
  if (!r.ok) throw new Error(`API error: ${r.status}`)
  return r.json()
}

export async function getPolicy() {
  const r = await fetch(`${API_BASE_URL}/policy`)
  if (!r.ok) throw new Error(`API error: ${r.status}`)
  return r.json()
}

export async function compareModels() {
  const r = await fetch(`${API_BASE_URL}/compare-models`, { method: "POST" })
  if (!r.ok) throw new Error(`API error: ${r.status}`)
  return r.json()
}

export async function getLeaderboard() {
  const r = await fetch(`${API_BASE_URL}/leaderboard`)
  if (!r.ok) throw new Error(`API error: ${r.status}`)
  return r.json()
}