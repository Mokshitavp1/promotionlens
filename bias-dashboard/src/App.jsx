import { useState, useEffect } from "react"
import biasData from "./mockBiasData.json"
import BiasScoreGauge from "./components/BiasScoreGauge"
import ProbeResultCard from "./components/ProbeResultCard"
import AdjectiveBreakdown from "./components/AdjectiveBreakdown"
import TrainingCurve from "./components/TrainingCurve"
import PolicyReport from "./components/PolicyReport"
import LLMLeaderboard from "./components/LLMLeaderboard"

function App() {
  const [data, setData] = useState(null)

  useEffect(() => {
    setData(biasData)
  }, [])

  if (!data) return <div className="p-4">Loading...</div>

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <h1 className="text-4xl font-bold text-blue-700 mb-8">Bias Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <BiasScoreGauge score={data.overallBiasScore} />
        <div className="md:col-span-2">
          <ProbeResultCard results={data.probeResults} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 mb-8">
        <TrainingCurve episodes={data.episodes} />
      </div>

      <div className="grid grid-cols-1 gap-6 mb-8">
        <PolicyReport report={data.policyReport} />
      </div>

      <div className="grid grid-cols-1 gap-6 mb-8">
        <LLMLeaderboard leaderboard={data.leaderboard} />
      </div>

      <div className="grid grid-cols-1 gap-6">
        <AdjectiveBreakdown adjectives={data.adjectives} />
      </div>
    </div>
  )
}

export default App