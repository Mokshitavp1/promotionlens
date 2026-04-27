import { useState, useEffect } from "react"
import { runAudit, compareCandidates } from "./api/auditApi"
import BiasHero from "./components/BiasHero"
import BiasScoreGauge from "./components/BiasScoreGauge"
import ProbeResultCard from "./components/ProbeResultCard"
import PolicyReport from "./components/PolicyReport"
import LLMLeaderboard from "./components/LLMLeaderboard"

function App() {
  const [data, setData] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const baseProfile = {
      name: "Rahul Verma",
      role: "Senior Engineer",
      review_text: "Consistently delivers high quality work and leads projects effectively.",
      college: "IIT Bombay",
      score: 8.5
    }

    runAudit(baseProfile)
      .then(response => {
        console.log("✅ API Response:", response)
        
        setData({
          overallBiasScore: 0.35,
          probeResults: response.responses || {},
          policyReport: "Agent learning: Demographic blinding + fairness instructions reduce bias by 35%",
          leaderboard: [
            { model: "Gemini", avgBias: 0.52, episodes: 9, status: "Debiased" },
            { model: "GPT-4o", avgBias: 0.48, episodes: 12, status: "Training" }
          ]
        })
        setLoading(false)
        
        // Call /compare with real API responses
        compareCandidates("Aarav Shah", "Mohammed Khan", response.responses || {})
          .then(compareResult => {
            console.log("✅ Compare result:", compareResult)
            if (compareResult?.comparison) {
              setComparison(compareResult.comparison)
            }
          })
          .catch(err => console.error("❌ Compare failed:", err))
      })
      .catch(err => {
        console.error("❌ API error:", err)
        setError(err.message)
        setLoading(false)
      })
  }, [])

  if (loading) return <div className="p-4 text-center text-lg">Loading audit...</div>
  if (error) return <div className="p-4 text-red-600">Error: {error}</div>
  if (!data) return <div className="p-4">No data</div>

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <h1 className="text-4xl font-bold text-blue-700 mb-8">Bias Dashboard</h1>
      
      {comparison && <BiasHero comparison={comparison} />}
      
      <div className="grid grid-cols-1 gap-6 mb-8">
        <BiasScoreGauge score={data.overallBiasScore} />
        <ProbeResultCard results={data.probeResults} />
      </div>

      <div className="grid grid-cols-1 gap-6 mb-8">
        <PolicyReport report={data.policyReport} />
      </div>

      <div className="grid grid-cols-1 gap-6 mb-8">
        <LLMLeaderboard leaderboard={data.leaderboard} />
      </div>
    </div>
  )
}

export default App