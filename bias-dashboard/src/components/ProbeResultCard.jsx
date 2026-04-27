export default function ProbeResultCard({ results }) {
  if (!results || typeof results !== 'object') {
    return <div className="bg-white p-6 rounded-lg shadow">No results available</div>
  }

  // Convert object to array
  const resultArray = Array.isArray(results)
    ? results
    : Object.entries(results).map(([name, data]) => ({
        name,
        decision: data.decision,
        score: data.score,
        justification: data.justification
      }))

  if (resultArray.length === 0) {
    return <div className="bg-white p-6 rounded-lg shadow">No results</div>
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Promotion Decisions</h2>
      <div className="space-y-4">
        {resultArray.map((result, idx) => (
          <div
            key={idx}
            className={`border-l-4 pl-4 py-3 rounded ${
              result.decision === "Recommend"
                ? "border-green-500 bg-green-50"
                : "border-red-500 bg-red-50"
            }`}
          >
            <div className="flex justify-between items-start">
              <div>
                <p className="font-semibold text-gray-800">{result.name}</p>
                <p
                  className={`text-sm font-bold mt-1 ${
                    result.decision === "Recommend"
                      ? "text-green-600"
                      : "text-red-600"
                  }`}
                >
                  {result.decision}
                </p>
              </div>
              <span className="text-2xl font-bold text-blue-600">{result.score.toFixed(1)}</span>
            </div>
            <p className="text-sm text-gray-600 mt-3">{result.justification}</p>
          </div>
        ))}
      </div>
    </div>
  )
}