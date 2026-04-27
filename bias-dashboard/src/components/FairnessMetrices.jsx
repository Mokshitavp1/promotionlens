import React from "react"

export default function FairnessMetrics({ data }) {
  const metrics = [
    {
      name: "Demographic Parity",
      value: 0.92,
      target: 0.95,
      description: "Proportion of positive outcomes across demographic groups",
      status: "good"
    },
    {
      name: "Equalized Odds",
      value: 0.88,
      target: 0.90,
      description: "Equal TPR and FPR across protected attributes",
      status: "good"
    },
    {
      name: "Calibration Difference",
      value: 0.15,
      target: 0.10,
      description: "Predicted vs actual positive outcome rates",
      status: "warning"
    },
    {
      name: "Predictive Parity",
      value: 0.91,
      target: 0.93,
      description: "Positive predictive value consistency",
      status: "good"
    },
    {
      name: "False Positive Rate Gap",
      value: 0.08,
      target: 0.05,
      description: "Difference in false positive rates across groups",
      status: "warning"
    },
    {
      name: "Statistical Parity Difference",
      value: 0.06,
      target: 0.10,
      description: "Maximum difference in selection rates",
      status: "good"
    }
  ]

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Fairness Metrics</h2>
        <p className="text-gray-600">Comprehensive fairness assessment across multiple dimensions</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric, idx) => {
          const progress = (metric.value / metric.target) * 100
          const isGood = metric.status === "good"
          
          return (
            <div key={idx} className="bg-white rounded-lg shadow p-6 border-t-4 border-gray-200 hover:shadow-lg transition">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="font-bold text-gray-800">{metric.name}</h3>
                  <p className="text-xs text-gray-500 mt-1">{metric.description}</p>
                </div>
                <span className={`text-2xl ${isGood ? "text-green-500" : "text-yellow-500"}`}>
                  {isGood ? "✓" : "⚠"}
                </span>
              </div>

              <div className="mb-4">
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-semibold text-gray-700">{metric.value.toFixed(2)}</span>
                  <span className="text-xs text-gray-500">Target: {metric.target.toFixed(2)}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      isGood ? "bg-green-500" : "bg-yellow-500"
                    }`}
                    style={{ width: `${Math.min(100, progress)}%` }}
                  />
                </div>
              </div>

              <div className="text-xs text-gray-600">
                {isGood ? (
                  <span className="text-green-600 font-medium">✓ Within acceptable range</span>
                ) : (
                  <span className="text-yellow-600 font-medium">⚠ Needs improvement</span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { label: "Metrics Meeting Target", value: "4/6", color: "bg-green-100 text-green-800" },
          { label: "Average Fairness Score", value: "0.84", color: "bg-blue-100 text-blue-800" },
          { label: "Risk Areas", value: "2", color: "bg-yellow-100 text-yellow-800" },
          { label: "Last Audit", value: "Today", color: "bg-purple-100 text-purple-800" }
        ].map((stat, idx) => (
          <div key={idx} className={`p-4 rounded-lg ${stat.color}`}>
            <p className="text-xs font-semibold uppercase">{stat.label}</p>
            <p className="text-2xl font-bold mt-2">{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Recommendations */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-6">
        <h3 className="font-bold text-amber-900 mb-4">Recommendations</h3>
        <ul className="space-y-3">
          <li className="flex gap-3">
            <span className="text-amber-600 font-bold">1.</span>
            <span className="text-gray-700">
              Address calibration difference - recalibrate decision thresholds to reduce gap from 0.15 to below 0.10
            </span>
          </li>
          <li className="flex gap-3">
            <span className="text-amber-600 font-bold">2.</span>
            <span className="text-gray-700">
              Monitor false positive rate gap - implement stratified monitoring across protected attributes
            </span>
          </li>
          <li className="flex gap-3">
            <span className="text-amber-600 font-bold">3.</span>
            <span className="text-gray-700">
              Continue demographic blinding in feature engineering to maintain parity metrics
            </span>
          </li>
        </ul>
      </div>
    </div>
  )
}