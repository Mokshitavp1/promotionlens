export default function TrainingCurve({ episodes }) {
  const maxScore = 100
  const width = 700
  const height = 350
  const padding = 50

  const points = episodes.map((ep, idx) => ({
    x: padding + (idx / (episodes.length - 1)) * (width - 2 * padding),
    y: height - padding - (ep.score / maxScore) * (height - 2 * padding),
    score: ep.score,
    episode: ep.episode,
  }))

  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ")

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">📈 Training Curve - Bias Score Over Episodes</h2>
      
      <svg width={width} height={height} className="border border-gray-200 rounded bg-white">
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((val) => (
          <g key={val}>
            <line
              x1={padding}
              y1={height - padding - (val / maxScore) * (height - 2 * padding)}
              x2={width - padding}
              y2={height - padding - (val / maxScore) * (height - 2 * padding)}
              stroke="#e5e7eb"
              strokeDasharray="4"
            />
            <text x={padding - 35} y={height - padding - (val / maxScore) * (height - 2 * padding) + 4} fontSize="12" fill="#666">
              {val}
            </text>
          </g>
        ))}

        {/* Y-axis label */}
        <text x="10" y="20" fontSize="12" fontWeight="bold" fill="#333">Bias Score</text>

        {/* Axes */}
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#000" strokeWidth="2" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#000" strokeWidth="2" />

        {/* X-axis label */}
        <text x={width - 100} y={height - 10} fontSize="12" fontWeight="bold" fill="#333">Episode</text>

        {/* Line */}
        <path d={pathD} stroke="#ef4444" strokeWidth="3" fill="none" />

        {/* Points */}
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="6" fill="#ef4444" stroke="#fff" strokeWidth="2" />
            <text x={p.x} y={height - padding + 25} fontSize="12" textAnchor="middle" fill="#666" fontWeight="bold">
              {p.episode}
            </text>
            <text x={p.x} y={p.y - 15} fontSize="11" textAnchor="middle" fill="#ef4444" fontWeight="bold">
              {p.score}
            </text>
          </g>
        ))}
      </svg>

      <div className="mt-4 p-4 bg-green-50 rounded">
        <p className="text-sm text-gray-700">
          <strong>Improvement:</strong> Bias score dropped from {episodes[0].score} to {episodes[episodes.length - 1].score} 
          ({Math.round(((episodes[0].score - episodes[episodes.length - 1].score) / episodes[0].score) * 100)}% reduction)
        </p>
      </div>
    </div>
  )
}