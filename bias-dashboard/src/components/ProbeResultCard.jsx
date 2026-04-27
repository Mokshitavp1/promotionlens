export default function ProbeResultCard({ results }) {
  if (!results || typeof results !== 'object') {
    return <div className="bg-white p-6 rounded-lg shadow">No results available</div>
  }

  // Convert object to array
  const resultArray = Array.isArray(results)
    ? results
    : Object.entries(results).map(([name, data]) => {
        // Handle both old format (decision/justification) and new format (reasoning)
        const score = typeof data === 'object' ? data.score : 0
        const decision = typeof data === 'object' ? data.decision : undefined
        const justification = typeof data === 'object' ? data.justification : undefined
        const reasoning = typeof data === 'object' ? data.reasoning : ''
        
        return {
          name,
          decision: decision || (score >= 8 ? "Recommend" : "Consider"),
          score: typeof score === 'number' ? score : 0,
          justification: justification || reasoning || "No details available"
        }
      })

  if (resultArray.length === 0) {
    return <div className="bg-white p-6 rounded-lg shadow">No results</div>
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Promotion Decisions</h2>
      <div className="space-y-4">
        {resultArray.map((result, idx) => (
          <div
            key={idx}
            className={`border-l-4 pl-4 py-3 rounded transition-shadow hover:shadow-md ${
              result.decision === "Recommend" || result.decision === "Promoted"
                ? "border-green-500 bg-green-50"
                : "border-yellow-500 bg-yellow-50"
            }`}
          >
            <div className="flex justify-between items-start gap-4">
              <div className="flex-1">
                <p className="font-semibold text-gray-800 text-base">{result.name}</p>
                <p
                  className={`text-sm font-bold mt-1 ${
                    result.decision === "Recommend" || result.decision === "Promoted"
                      ? "text-green-600"
                      : "text-yellow-600"
                  }`}
                >
                  {result.decision}
                </p>
              </div>
              <div className="text-right">
                <span className="text-2xl font-bold text-blue-600">
                  {typeof result.score === 'number' ? result.score.toFixed(1) : '0.0'}
                </span>
                <p className="text-xs text-gray-500 font-medium">Score</p>
              </div>
            </div>
            <p className="text-sm text-gray-600 mt-3 leading-relaxed">{result.justification}</p>
          </div>
        ))}
      </div>
      
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <p className="text-xs text-blue-800 font-medium">
          <span className="font-semibold">💡 Note:</span> Variations in scores and decisions may indicate demographic bias in the AI model's evaluation process.
        </p>
      </div>
    </div>
  )
}