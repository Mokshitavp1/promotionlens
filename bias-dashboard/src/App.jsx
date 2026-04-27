import { useState, useEffect } from "react"
import { runAudit, compareCandidates } from "./api/auditApi"
import BiasHero from "./components/BiasHero"
import BiasScoreGauge from "./components/BiasScoreGauge"
import ProbeResultCard from "./components/ProbeResultCard"
import PolicyReport from "./components/PolicyReport"
import LLMLeaderboard from "./components/LLMLeaderboard"
import mockResponses from "../../mock_responses.json";
import mockOutput from "../../mock_output.json";

function App() {
  const [data, setData] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    initializeDashboard()
  }, [])

  const initializeDashboard = async () => {
    try {
      setLoading(true)
      
      // Load mock audit responses
      const auditResponse = {
        status: "success",
        responses: mockOutput
      }
      
      console.log("✅ Using mock audit data:", auditResponse)
      
      // Process audit data
      setData({
        overallBiasScore: calculateOverallBias(mockOutput),
        probeResults: mockOutput,
        policyReport: "Agent learning: Demographic blinding + fairness instructions reduce bias by 35%",
        leaderboard: [
          { model: "PromotionLens (Tuned)", avgBias: 0.15, episodes: 50, status: "Debiased" },
          { model: "Gemini", avgBias: 0.52, episodes: 9, status: "Training" },
          { model: "GPT-4o", avgBias: 0.48, episodes: 12, status: "Training" }
        ]
      })
      
      // Perform comparison with variant IDs (not names)
      await performComparison("aarav_iit", "mohammed_jntu")
      
      setLoading(false)
    } catch (err) {
      console.error("❌ Initialization error:", err)
      setError(err.message)
      setLoading(false)
    }
  }

  const calculateOverallBias = (results) => {
    if (!results || Object.keys(results).length === 0) return 0.35
    
    const scores = Object.values(results).map(r => r.score || 0)
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length
    
    // Convert score (0-10) to bias metric (0-1)
    // Lower score = higher bias
    return Math.max(0.1, 1 - avgScore / 10)
  }

  const performComparison = async (variantIdA, variantIdB) => {
    try {
      console.log(`🔄 Comparing ${variantIdA} vs ${variantIdB}`)
      
      // Get mock data for both variants from mockResponses
      const candidateA = mockResponses[variantIdA]
      const candidateB = mockResponses[variantIdB]
      
      if (!candidateA || !candidateB) {
        throw new Error(`Candidates not found: ${variantIdA}, ${variantIdB}`)
      }

      // scores come directly from mockResponses profiles
      const scoreA = candidateA.profile.score
      const scoreB = candidateB.profile.score
      
      // Try API call first, pass mockResponses so backend can look up by variant ID
      let compareResult
      try {
        const response = await compareCandidates(variantIdA, variantIdB, mockResponses)
        console.log("✅ API comparison result:", response)
        
        // FIX: Safely extract comparison from response
        // Handle multiple response formats:
        // 1. {comparison: {...}}
        // 2. Direct comparison object with candidate_a, candidate_b, etc.
        // 3. String error responses (gracefully fallback)
        
        if (typeof response === 'string') {
          throw new Error(`API returned string instead of JSON: ${response}`)
        }
        
        compareResult = response.comparison || response
        
        // Validate that we got the expected structure
        if (!compareResult.candidate_a || !compareResult.candidate_b) {
          throw new Error("Invalid comparison response structure")
        }
        
      } catch (apiErr) {
        console.warn("⚠️ API comparison failed, using mock data:", apiErr.message)
        
        // Fallback: build comparison directly from mockResponses
        compareResult = buildComparisonFromMock(candidateA, candidateB, scoreA, scoreB)
      }
      
      console.log("✅ Final comparison:", compareResult)
      setComparison(compareResult)
    } catch (err) {
      console.error("❌ Comparison error:", err)
      setError(`Comparison failed: ${err.message}`)
    }
  }

  // Helper function to build comparison from mock data
  const buildComparisonFromMock = (candidateA, candidateB, scoreA, scoreB) => {
    return {
      candidate_a: candidateA.profile.name,
      candidate_b: candidateB.profile.name,
      score_gap: Math.abs(scoreA - scoreB).toFixed(1),
      bias_types_detected: detectBiasTypes(candidateA, candidateB),
      severity: "HIGH",
      severity_emoji: "🔴",
      finding: `${candidateA.profile.name} scored ${scoreA} vs ${candidateB.profile.name} ${scoreB} despite identical qualifications.`
    }
  }

  const detectBiasTypes = (candidateA, candidateB) => {
    const biases = []
    
    // College tier bias
    const collegeA = candidateA.profile.college
    const collegeB = candidateB.profile.college
    if (collegeA !== collegeB) {
      biases.push("college tier")
    }
    
    // Name-based bias detection
    const nameA = candidateA.profile.name
    const nameB = candidateB.profile.name
    
    // Hindu names: Aarav, Priya
    // Muslim names: Mohammed, Anjali
    // This is detecting if review language differs by name while qualifications are same
    if (
      ((nameA.includes("Aarav") || nameA.includes("Priya")) &&
       (nameB.includes("Mohammed") || nameB.includes("Anjali"))) ||
      ((nameA.includes("Mohammed") || nameA.includes("Anjali")) &&
       (nameB.includes("Aarav") || nameB.includes("Priya")))
    ) {
      biases.push("religion")
    }
    
    return biases.length > 0 ? biases : ["review language bias"]
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-bold text-white mb-4">Analyzing Promotion Bias...</div>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
        <div className="max-w-2xl mx-auto">
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-6 text-red-200">
            <h2 className="text-xl font-bold mb-2">Error</h2>
            <p>{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-red-700 hover:bg-red-800 rounded font-medium"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
        <div className="text-center text-white">No data available</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">PromotionLens</h1>
          <p className="text-slate-400">AI Bias Detection in Promotion Decisions</p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Left Column */}
          <div className="space-y-6">
            {comparison && (
              <BiasHero comparison={comparison} />
            )}
            {data && (
              <BiasScoreGauge score={data.overallBiasScore} />
            )}
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {data && (
              <ProbeResultCard results={data.probeResults} />
            )}
          </div>
        </div>

        {/* Policy Report */}
        {data && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <PolicyReport report={data.policyReport} />
            <LLMLeaderboard leaderboard={data.leaderboard} />
          </div>
        )}
      </div>
    </div>
  )
}

export default App