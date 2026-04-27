import { useState, useEffect } from "react"
import { runAudit, compareCandidates, getLeaderboard, getPolicy, trainAgent } from "./api/auditApi"
import BiasHero from "./components/BiasHero"
import BiasScoreGauge from "./components/BiasScoreGauge"
import ProbeResultCard from "./components/ProbeResultCard"
import AdjectiveBreakdown from "./components/AdjectiveBreakdown"
import PolicyReport from "./components/PolicyReport"
import LLMLeaderboard from "./components/LLMLeaderboard"
import TrainingCurve from "./components/TrainingCurve"

const BASE_PROFILE = {
  name: "Rahul Verma",
  role: "Senior Engineer",
  review_text: "Consistently delivers high quality work and leads projects effectively.",
  college: "IIT Bombay",
  score: 8.5
}

export default function App() {
  const [auditData, setAuditData]     = useState(null)
  const [comparison, setComparison]   = useState(null)
  const [leaderboard, setLeaderboard] = useState([])
  const [policy, setPolicy]           = useState("")
  const [trainingLog, setTrainingLog] = useState(null)
  const [biasScore, setBiasScore]     = useState(0)
  const [loading, setLoading]         = useState(true)
  const [error, setError]             = useState(null)

  useEffect(() => {
    // Fire all requests in parallel — don't let one block another
    getLeaderboard().then(r => r?.leaderboard && setLeaderboard(r.leaderboard)).catch(() => {})
    getPolicy().then(r => r?.policy && setPolicy(r.policy)).catch(() => {})
    trainAgent(20).then(r => r?.training_log && setTrainingLog(r.training_log)).catch(() => {})

    runAudit(BASE_PROFILE)
      .then(res => {
        setAuditData(res)

        // Compute overall bias score from score_gaps
        const gaps = res.bias_report?.score_gaps
        if (gaps) {
          const avg = Object.values(gaps).reduce((a, b) => a + b, 0) / Object.keys(gaps).length
          setBiasScore(Math.min(avg / 10, 1))
        }

        // Compare two specific variants
        compareCandidates("aarav_iit", "mohammed_jntu", res.responses || {})
          .then(r => r?.comparison && setComparison(r.comparison))
          .catch(() => {})
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="p-8 text-center text-lg">Running bias audit...</div>
  if (error)   return <div className="p-8 text-red-600">Error: {error}</div>

  const responses  = auditData?.responses || {}
  const adjectives = auditData?.bias_report?.adjectives || {}

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <h1 className="text-4xl font-bold text-blue-700 mb-8">PromotionLens — Bias Dashboard</h1>

      {comparison && <div className="mb-6"><BiasHero comparison={comparison} /></div>}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <BiasScoreGauge score={biasScore} />
        <PolicyReport report={policy} />
      </div>

      <div className="mb-6">
        <ProbeResultCard results={responses} />
      </div>

      <div className="mb-6">
        <AdjectiveBreakdown adjectives={adjectives} />
      </div>

      {trainingLog && (
        <div className="mb-6">
          <TrainingCurve trainingLog={trainingLog} />
        </div>
      )}

      <div className="mb-6">
        <LLMLeaderboard leaderboard={leaderboard} />
      </div>
    </div>
  )
}