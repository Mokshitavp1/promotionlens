import { useState } from "react"
import { runAudit, compareCandidates, getLeaderboard, getPolicy, trainAgent } from "./api/auditApi"
import BiasHero from "./components/BiasHero"
import BiasScoreGauge from "./components/BiasScoreGauge"
import ProbeResultCard from "./components/ProbeResultCard"
import AdjectiveBreakdown from "./components/AdjectiveBreakdown"
import PolicyReport from "./components/PolicyReport"
import LLMLeaderboard from "./components/LLMLeaderboard"
import TrainingCurve from "./components/TrainingCurve"
import ExportPanel from "./components/ExportPanel"
import ProfileForm from "./components/ProfileForm"

function computeBiasScore(biasReport) {
  const gaps = biasReport?.score_gaps
  if (!gaps) return 0
  const values = Object.values(gaps)
  if (!values.length) return 0
  const avg = values.reduce((a, b) => a + b, 0) / values.length
  return Math.min(avg / 10, 1)
}

async function fetchComparison(responses) {
  try {
    const r = await compareCandidates("mohammed_jntu", "aarav_iit", responses)
    if (!r?.comparison) return null
    const comp = r.comparison
    comp.candidate_a_name = responses?.mohammed_jntu?.profile?.name || "Mohammed Khan"
    comp.candidate_b_name = responses?.aarav_iit?.profile?.name || "Aarav Shah"
    return comp
  } catch {
    return null
  }
}

async function fetchSupportingData() {
  const [lbResult, policyResult, trainResult] = await Promise.allSettled([
    getLeaderboard(),
    getPolicy(),
    trainAgent(20),
  ])
  return {
    leaderboard: lbResult.status === "fulfilled" ? lbResult.value?.leaderboard ?? [] : [],
    policy: policyResult.status === "fulfilled" ? policyResult.value?.policy ?? "" : "",
    trainingLog: trainResult.status === "fulfilled" ? trainResult.value?.training_log ?? [] : [],
  }
}

const makeBlankForm = () => ({
  name: "", role: "Senior Engineer",
  review_text: "", college: "", score: 8.0, model: "groq"
})

export default function App() {
  const [phase, setPhase] = useState("idle")
  const [auditError, setAuditError] = useState(null)
  const [profiles, setProfiles] = useState([makeBlankForm()])

  const [batchResults, setBatchResults] = useState([])   // array of {profile, auditData, biasScore, comparison}
  const [leaderboard, setLeaderboard] = useState([])
  const [policy, setPolicy] = useState("")
  const [trainingLog, setTrainingLog] = useState(null)

  function updateProfile(idx, updated) {
    setProfiles(prev => prev.map((p, i) => i === idx ? updated : p))
  }

  function addProfile() {
    if (profiles.length >= 5) return
    setProfiles(prev => [...prev, makeBlankForm()])
  }

  function removeProfile(idx) {
    if (profiles.length === 1) return
    setProfiles(prev => prev.filter((_, i) => i !== idx))
  }

  async function handleAudit() {
    const valid = profiles.filter(p => p.name.trim() && p.review_text.trim())
    if (!valid.length) return
    setPhase("loading")
    setAuditError(null)

    try {
      // Fire all audits in parallel
      const auditResults = await Promise.allSettled(valid.map(p => runAudit(p)))

      const succeeded = []
      const errors = []

      for (let i = 0; i < auditResults.length; i++) {
        const r = auditResults[i]
        if (r.status === "rejected" || r.value?.status === "error") {
          errors.push(`${valid[i].name || `Profile ${i + 1}`}: ${r.reason?.message || r.value?.message || "failed"}`)
        } else {
          succeeded.push({ profile: valid[i], res: r.value })
        }
      }

      if (!succeeded.length) {
        setAuditError(errors.join(" · "))
        setPhase("error")
        return
      }

      // Build batch results with comparison per profile
      const batchPromise = Promise.all(succeeded.map(async ({ profile, res }) => {
        const comp = await fetchComparison(res.responses || {})
        return {
          profile,
          auditData: res,
          biasScore: computeBiasScore(res.bias_report),
          comparison: comp,
        }
      }))
      const supportingPromise = fetchSupportingData()

      const [batch, supporting] = await Promise.all([batchPromise, supportingPromise])

      setBatchResults(batch)
      setLeaderboard(supporting.leaderboard)
      setPolicy(supporting.policy)
      setTrainingLog(supporting.trainingLog)

      if (errors.length) {
        setAuditError(`${errors.length} profile(s) failed: ${errors.join(" · ")}`)
      }

      setPhase("results")
    } catch (err) {
      setAuditError(err.message ?? "Unexpected error — check the console.")
      setPhase("error")
    }
  }

  function handleReset() {
    setPhase("idle")
    setAuditError(null)
    setBatchResults([])
    // keep profiles intact so user can tweak and re-run
  }

  const canSubmit = profiles.some(p => p.name.trim() && p.review_text.trim())

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", paddingBottom: 60 }}>
      <header style={{
        borderBottom: "1px solid var(--border)",
        padding: "24px 48px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <div>
          <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: 28, fontWeight: 800, color: "var(--text)", margin: 0, letterSpacing: "-0.5px" }}>
            Promotion<span style={{ color: "var(--accent)" }}>Lens</span>
          </h1>
          <p style={{ color: "var(--muted)", fontSize: 12, fontFamily: "'DM Mono', monospace", margin: "4px 0 0 0" }}>
            RL-powered bias auditing for LLM promotion decisions
          </p>
        </div>
        {phase === "results" && (
          <ExportPanel
            auditData={batchResults[0]?.auditData}
            leaderboard={leaderboard}
            biasScore={batchResults[0]?.biasScore}
          />
        )}
      </header>

      <main style={{ maxWidth: 1400, margin: "0 auto", padding: "40px 48px" }}>

        {(phase === "idle" || phase === "error") && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Profile slots */}
            {profiles.map((form, idx) => (
              <div key={idx} style={{ position: "relative" }}>
                {profiles.length > 1 && (
                  <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    marginBottom: 8,
                  }}>
                    <span style={{ fontSize: 11, fontFamily: "'DM Mono', monospace", color: "var(--muted)", letterSpacing: 2, textTransform: "uppercase" }}>
                      Profile {idx + 1}
                    </span>
                    <button
                      onClick={() => removeProfile(idx)}
                      style={{ background: "transparent", border: "none", color: "var(--danger)", fontSize: 11, fontFamily: "'DM Mono', monospace", cursor: "pointer", padding: "2px 8px" }}
                    >
                      ✕ remove
                    </button>
                  </div>
                )}
                <ProfileForm
                  form={form}
                  onChange={updated => updateProfile(idx, updated)}
                  loading={false}
                />
              </div>
            ))}

            {/* Add profile + run */}
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              {profiles.length < 5 && (
                <button
                  onClick={addProfile}
                  style={{ background: "transparent", border: "1px dashed var(--border)", color: "var(--muted)", borderRadius: 8, padding: "10px 20px", fontSize: 12, fontFamily: "'DM Mono', monospace", cursor: "pointer", letterSpacing: "0.05em" }}
                >
                  + add another profile
                </button>
              )}
              <button
                onClick={handleAudit}
                disabled={!canSubmit}
                style={{
                  background: canSubmit ? "var(--accent)" : "var(--border)",
                  color: canSubmit ? "#fff" : "var(--muted)",
                  border: "none", borderRadius: 8, padding: "10px 28px",
                  fontFamily: "'DM Mono', monospace", fontSize: 13,
                  cursor: canSubmit ? "pointer" : "not-allowed",
                  transition: "background 0.15s",
                }}
              >
                → run audit{profiles.length > 1 ? ` (${profiles.filter(p => p.name.trim() && p.review_text.trim()).length} profiles)` : ""}
              </button>
            </div>

            {phase === "error" && auditError && (
              <div style={{
                background: "rgba(255,77,109,0.08)", border: "1px solid var(--danger)",
                borderRadius: 8, padding: "14px 20px", display: "flex",
                justifyContent: "space-between", alignItems: "center",
                fontFamily: "'DM Mono', monospace", fontSize: 13,
              }}>
                <span style={{ color: "var(--danger)" }}>✗ {auditError}</span>
                <button
                  onClick={() => setAuditError(null)}
                  style={{ background: "transparent", border: "1px solid var(--danger)", color: "var(--danger)", borderRadius: 6, padding: "4px 12px", fontSize: 12, cursor: "pointer", fontFamily: "'DM Mono', monospace" }}
                >
                  clear
                </button>
              </div>
            )}
          </div>
        )}

        {phase === "loading" && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 400, gap: 16 }}>
            <div style={{ width: 48, height: 48, border: "3px solid var(--border)", borderTop: "3px solid var(--accent)", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
            <p style={{ color: "var(--muted)", fontFamily: "'DM Mono', monospace", fontSize: 14 }}>
              running {profiles.filter(p => p.name.trim() && p.review_text.trim()).length} audit{profiles.length > 1 ? "s" : ""}...
            </p>
            <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
          </div>
        )}

        {phase === "results" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 48 }}>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <p style={{ color: "var(--muted)", fontSize: 12, fontFamily: "'DM Mono', monospace", margin: 0 }}>
                {batchResults.length} profile{batchResults.length > 1 ? "s" : ""} audited
              </p>
              <button
                onClick={handleReset}
                style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--muted)", borderRadius: 6, padding: "6px 16px", fontSize: 12, cursor: "pointer", fontFamily: "'DM Mono', monospace" }}
              >
                ← new audit
              </button>
            </div>

            {/* Partial error banner */}
            {auditError && (
              <div style={{ background: "rgba(255,183,3,0.08)", border: "1px solid var(--warn)", borderRadius: 8, padding: "12px 20px", fontFamily: "'DM Mono', monospace", fontSize: 12, color: "var(--warn)" }}>
                ⚠ {auditError}
              </div>
            )}

            {/* One section per profile */}
            {batchResults.map(({ profile, auditData, biasScore, comparison }, idx) => (
              <div key={idx} style={{ display: "flex", flexDirection: "column", gap: 24 }}>

                {/* Profile header */}
                <div style={{ borderBottom: "1px solid var(--border)", paddingBottom: 16 }}>
                  <p style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, fontWeight: 700, color: "var(--text)", margin: 0 }}>
                    {profile.name}
                    <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: "var(--muted)", fontWeight: 400, marginLeft: 16 }}>
                      {profile.role} · {profile.college} · {profile.score}/10
                    </span>
                  </p>
                </div>

                {comparison && <BiasHero comparison={comparison} biasReport={auditData?.bias_report} />}

                <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 24 }}>
                  <BiasScoreGauge score={biasScore} />
                  {policy && idx === 0 && <PolicyReport report={policy} />}
                </div>

                {idx === 0 && (
                  <ProbeResultCard
                    results={auditData?.responses || {}}
                    baselineName={profile.name}
                  />
                )}

                {auditData?.bias_report?.adjectives && (
                  <AdjectiveBreakdown adjectives={auditData.bias_report.adjectives} />
                )}

              </div>
            ))}

            {/* Shared components — shown once at the bottom */}
            {trainingLog && <TrainingCurve trainingLog={trainingLog} />}
            {leaderboard.length > 0 && <LLMLeaderboard leaderboard={leaderboard} />}

          </div>
        )}

      </main>
    </div>
  )
}