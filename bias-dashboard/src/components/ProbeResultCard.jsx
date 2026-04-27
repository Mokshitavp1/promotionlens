export default function ProbeResultCard({ results }) {
  if (!results || typeof results !== 'object' || Object.keys(results).length === 0)
    return <div className="bg-white p-6 rounded-lg shadow">No results available</div>

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Promotion Decisions</h2>
      <div className="space-y-4">
        {Object.entries(results).map(([vid, data]) => {
          const p = data?.parsed || {}
          const rec = p.promotion_recommendation || data?.decision || "—"
          const score = p.promotion_score ?? data?.score ?? 0
          const reason = p.reasoning || data?.justification || ""
          const name = data?.profile?.name || vid
          const isYes = rec.includes("yes") || rec === "Recommend"

          return (
            <div key={vid} className={`border-l-4 pl-4 py-3 rounded ${isYes ? "border-green-500 bg-green-50" : "border-red-500 bg-red-50"}`}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-semibold text-gray-800">{name}</p>
                  <p className={`text-sm font-bold mt-1 ${isYes ? "text-green-600" : "text-red-600"}`}>{rec}</p>
                </div>
                <span className="text-2xl font-bold text-blue-600">{Number(score).toFixed(1)}</span>
              </div>
              <p className="text-sm text-gray-600 mt-3">{reason}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}