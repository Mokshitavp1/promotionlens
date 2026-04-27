export default function LLMLeaderboard({ leaderboard }) {
  if (!leaderboard || leaderboard.length === 0) {
    return <div className="bg-white p-6 rounded-lg shadow">No leaderboard data</div>
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">🏆 LLM Leaderboard</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="bg-blue-100 border-b-2 border-blue-500">
            <tr>
              <th className="px-4 py-3 font-bold text-gray-800">Rank</th>
              <th className="px-4 py-3 font-bold text-gray-800">Model</th>
              <th className="px-4 py-3 font-bold text-gray-800">Bias Score</th>
              <th className="px-4 py-3 font-bold text-gray-800">Episodes to Debias</th>
              <th className="px-4 py-3 font-bold text-gray-800">Status</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((entry, idx) => {
              // Use correct field names: avgBias and episodes (from App.jsx)
              const biasScore = entry.avgBias || entry.biasScore || 0
              const episodes = entry.episodes || entry.episodesToDebias || 0
              
              const scoreColor = biasScore < 0.4 ? "text-green-600" : biasScore < 0.7 ? "text-yellow-600" : "text-red-600"
              const statusIcon = entry.status || (biasScore < 0.4 ? "✓" : biasScore < 0.7 ? "⚠" : "✗")
              
              return (
                <tr key={idx} className={idx % 2 === 0 ? "bg-gray-50" : "bg-white"}>
                  <td className="px-4 py-3 font-bold text-lg">#{idx + 1}</td>
                  <td className="px-4 py-3 font-semibold text-gray-800">{entry.model}</td>
                  <td className={`px-4 py-3 font-bold text-lg ${scoreColor}`}>{biasScore.toFixed(2)}</td>
                  <td className="px-4 py-3 text-gray-700">{episodes}</td>
                  <td className="px-4 py-3 text-2xl">{statusIcon}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-4 p-4 bg-gray-50 rounded text-sm text-gray-600">
        <p><strong>Lower bias score = Better performance</strong></p>
      </div>
    </div>
  )
}