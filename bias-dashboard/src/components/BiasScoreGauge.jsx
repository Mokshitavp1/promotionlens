export default function BiasScoreGauge({ score }) {
  // Ensure score is a number and round it
  const numericScore = typeof score === 'number' ? Math.round(score * 100) : 0
  
  const radius = 45
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (numericScore / 100) * circumference

  const color = numericScore < 30 ? "#10b981" : numericScore < 60 ? "#f59e0b" : "#ef4444"

  return (
    <div className="bg-white p-6 rounded-lg shadow text-center">
      <h2 className="text-xl font-bold mb-4">Overall Bias Score</h2>
      <svg width="120" height="120" className="mx-auto">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="#e5e7eb" strokeWidth="8" />
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          style={{ transform: "rotate(-90deg)", transformOrigin: "60px 60px" }}
        />
      </svg>
      <p className="text-4xl font-bold mt-4" style={{ color }}>
        {numericScore}
      </p>
      <p className="text-gray-600 text-sm mt-2">
        {numericScore < 30 ? "Low Bias ✓" : numericScore < 60 ? "Moderate Bias ⚠" : "High Bias ✗"}
      </p>
    </div>
  )
}