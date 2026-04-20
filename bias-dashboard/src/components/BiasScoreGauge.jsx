export default function BiasScoreGauge({ score }) {
  const radius = 45
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (score / 100) * circumference

  const color = score < 30 ? "#10b981" : score < 60 ? "#f59e0b" : "#ef4444"

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
        {score}
      </p>
      <p className="text-gray-600 text-sm mt-2">
        {score < 30 ? "Low Bias ✓" : score < 60 ? "Moderate Bias ⚠" : "High Bias ✗"}
      </p>
    </div>
  )
}