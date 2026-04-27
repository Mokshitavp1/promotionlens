export default function TrainingCurve({ episodes }) {
  if (!episodes || episodes.length === 0) {
    return <div className="bg-white p-6 rounded-lg shadow">No training data</div>
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Training Progress</h2>
      <div className="space-y-4">
        {episodes.map((ep, idx) => (
          <div key={idx} className="flex justify-between items-center">
            <span className="font-semibold">Episode {ep.episode}</span>
            <div className="flex-1 mx-4 bg-gray-200 rounded h-2">
              <div
                className="bg-blue-500 h-2 rounded"
                style={{ width: `${Math.max(0, Math.min(100, (1 - ep.bias) * 100))}%` }}
              />
            </div>
            <span className="text-sm font-bold">{(ep.bias || 0).toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}