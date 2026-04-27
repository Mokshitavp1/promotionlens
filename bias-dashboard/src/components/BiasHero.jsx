export default function BiasHero({ comparison }) {
  if (!comparison) {
    return <div className="bg-white p-6 rounded-lg shadow">Loading bias analysis...</div>
  }

  const { 
    lower_scored, 
    higher_scored, 
    score_gap, 
    bias_types_detected, 
    severity_emoji,
    finding
  } = comparison

  return (
    <div className={`border-2 rounded-lg p-8 mb-8 ${
      severity_emoji === "🔴" ? "bg-red-50 border-red-300" : 
      severity_emoji === "🟡" ? "bg-yellow-50 border-yellow-300" :
      "bg-green-50 border-green-300"
    }`}>
      <div className="flex items-center gap-3 mb-6">
        <span className="text-4xl">{severity_emoji}</span>
        <h2 className={`text-3xl font-bold ${
          severity_emoji === "🔴" ? "text-red-700" :
          severity_emoji === "🟡" ? "text-yellow-700" :
          "text-green-700"
        }`}>BIAS DETECTED</h2>
      </div>

      <p className="text-lg font-semibold text-gray-800 mb-6">
        {lower_scored} scored <span className="text-red-600 font-bold">{score_gap} points lower</span> than {higher_scored}
        <br />
        for the <span className="italic">IDENTICAL employee profile.</span>
      </p>

      <div className="space-y-3 mb-8 bg-white p-4 rounded border border-gray-200">
        {bias_types_detected && bias_types_detected.includes("religion") && (
          <div className="flex justify-between items-center">
            <span className="font-semibold text-gray-700">Religion bias:</span>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-red-600">{score_gap.toFixed(2)} pts</span>
              <span className="text-2xl">🔴</span>
            </div>
          </div>
        )}
        {bias_types_detected && bias_types_detected.includes("college tier") && (
          <div className="flex justify-between items-center">
            <span className="font-semibold text-gray-700">College bias:</span>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-red-600">{score_gap.toFixed(2)} pts</span>
              <span className="text-2xl">🔴</span>
            </div>
          </div>
        )}
        {bias_types_detected && bias_types_detected.includes("gender") && (
          <div className="flex justify-between items-center">
            <span className="font-semibold text-gray-700">Gender bias:</span>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-red-600">{score_gap.toFixed(2)} pts</span>
              <span className="text-2xl">🔴</span>
            </div>
          </div>
        )}
        {(!bias_types_detected || bias_types_detected.length === 0) && (
          <div className="flex justify-between items-center">
            <span className="font-semibold text-gray-700">Gender bias:</span>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold text-green-600">0.00 pts</span>
              <span className="text-2xl">🟢</span>
            </div>
          </div>
        )}
      </div>

      <p className="text-sm text-gray-700 mb-6 bg-gray-50 p-3 rounded italic">
        "{finding}"
      </p>

      <div className="flex gap-4">
        <button className="px-6 py-3 bg-red-600 text-white font-semibold rounded-lg hover:bg-red-700 transition">
          [ Run Audit ]
        </button>
        <button className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition">
          [ Train Agent ]
        </button>
      </div>
    </div>
  )
}