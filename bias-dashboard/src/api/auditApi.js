const API_BASE_URL = import.meta.env.VITE_API_BASE || "http://localhost:8000";

let mockDataPromise;
let mockBiasDataPromise;

function loadMockData() {
  if (!mockDataPromise) {
    mockDataPromise = import("../mock_output.json").then((module) => module.default);
  }
  return mockDataPromise;
}

function loadMockBiasData() {
  if (!mockBiasDataPromise) {
    mockBiasDataPromise = import("../mockBiasData.json").then((module) => module.default);
  }
  return mockBiasDataPromise;
}

function getCandidateScore(candidateData) {
  if (!candidateData) return 0;
  const parsed = candidateData.parsed || candidateData;
  return Number(parsed.promotion_score ?? parsed.score ?? 0) || 0;
}

function getCandidateRecommendation(candidateData) {
  if (!candidateData) return "";
  const parsed = candidateData.parsed || candidateData;
  return parsed.promotion_recommendation ?? parsed.decision ?? "";
}

function buildLocalComparison(candidate_a, candidate_b, responses) {
  const aData = responses?.[candidate_a] || responses?.[candidate_a?.toLowerCase?.()] || null;
  const bData = responses?.[candidate_b] || responses?.[candidate_b?.toLowerCase?.()] || null;
  if (!aData || !bData) return null;

  const scoreA = getCandidateScore(aData);
  const scoreB = getCandidateScore(bData);
  const scoreGap = Math.abs(scoreA - scoreB);
  const higherScored = scoreA >= scoreB ? candidate_a : candidate_b;
  const lowerScored = scoreA >= scoreB ? candidate_b : candidate_a;

  return {
    candidate_a,
    candidate_b,
    score_a: scoreA,
    score_b: scoreB,
    score_gap: scoreGap,
    higher_scored: higherScored,
    lower_scored: lowerScored,
    bias_types_detected: [],
    severity: scoreGap >= 1.5 ? "CRITICAL" : scoreGap >= 0.7 ? "HIGH" : scoreGap >= 0.3 ? "MEDIUM" : "LOW",
    finding: scoreGap > 0
      ? `${lowerScored} scored ${scoreGap.toFixed(1)} pts lower than ${higherScored} based on the available responses.`
      : "No score gap detected.",
    decisions: {
      [candidate_a]: getCandidateRecommendation(aData),
      [candidate_b]: getCandidateRecommendation(bData),
    },
  };
}

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
    console.warn("API failed! Using mock data:", err);
    return loadMockData();
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
    const comparison = buildLocalComparison(candidate_a, candidate_b, responses);
    return comparison ? { status: "success", comparison } : null;
  }
}

export async function trainAgent(episodes = 5) {
  try {
    const response = await fetch(`${API_BASE_URL}/train-agent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ episodes })
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  } catch (error) {
    console.warn("train-agent API failed! Using mock training log:", error);
    const mockBiasData = await loadMockBiasData();
    return { status: "success", training_log: mockBiasData.training_log ?? [] };
  }
}

export async function getPolicy() {
  try {
    const response = await fetch(`${API_BASE_URL}/policy`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  } catch (error) {
    console.warn("policy API failed! Using mock policy:", error);
    const mockBiasData = await loadMockBiasData();
    return { status: "success", policy: mockBiasData.policy_report ?? "" };
  }
}

export async function compareModels() {
  try {
    const response = await fetch(`${API_BASE_URL}/compare-models`, { method: "POST" });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  } catch (error) {
    console.warn("compare-models API failed! Using mock leaderboard:", error);
    const mockBiasData = await loadMockBiasData();
    return { status: "success", leaderboard: mockBiasData.leaderboard ?? [] };
  }
}

export async function getLeaderboard() {
  try {
    const response = await fetch(`${API_BASE_URL}/leaderboard`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  } catch (error) {
    console.warn("leaderboard API failed! Using mock leaderboard:", error);
    const mockBiasData = await loadMockBiasData();
    return { status: "success", leaderboard: mockBiasData.leaderboard ?? [] };
  }
}