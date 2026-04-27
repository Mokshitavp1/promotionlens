export default function LLMLeaderboard({ leaderboard }) {
  if (!leaderboard || leaderboard.length === 0)
    return <div className="bg-white p-6 rounded-lg shadow">No leaderboard data</div>

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">🏆 LLM Leaderboard</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="bg-blue-100 border-b-2 border-blue-500">
            <tr>
              <th className="px-4 py-3 font-bold text-gray-800">Rank</th>
              <th className="px-4 py-3 font-bold text-gray-800">Model</th>
              <th className="px-4 py-3 font-bold text-gray-800">Avg Bias Score</th>
              <th className="px-4 py-3 font-bold text-gray-800">Episodes to Debias</th>
              <th className="px-4 py-3 font-bold text-gray-800">Religion Gap</th>
              <th className="px-4 py-3 font-bold text-gray-800">College Gap</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((entry, idx) => {
              const bias = entry.avg_bias_score ?? 0
              const scoreColor = bias < 0.2 ? "text-green-600" : bias < 0.5 ? "text-yellow-600" : "text-red-600"
              return (
                <tr key={idx} className={idx % 2 === 0 ? "bg-gray-50" : "bg-white"}>
                  <td className="px-4 py-3 font-bold">#{idx + 1}</td>
                  <td className="px-4 py-3 font-semibold text-gray-800">{entry.model}</td>
                  <td className={`px-4 py-3 font-bold ${scoreColor}`}>{bias.toFixed(3)}</td>
                  <td className="px-4 py-3 text-gray-700">{entry.episodes_to_debias ?? "—"}</td>
                  <td className="px-4 py-3 text-gray-700">{entry.score_gap_religion?.toFixed(2) ?? "—"}</td>
                  <td className="px-4 py-3 text-gray-700">{entry.score_gap_college?.toFixed(2) ?? "—"}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-sm text-gray-500">Lower bias score = better. Sorted most → least biased.</p>
    </div>
  )
}