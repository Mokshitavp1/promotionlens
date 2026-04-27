import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

export default function TrainingCurve({ trainingLog }) {
  if (!trainingLog || trainingLog.length === 0)
    return <div className="bg-white p-6 rounded-lg shadow">No training data</div>

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">📉 Agent Training — Bias Reduction Over Episodes</h2>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={trainingLog}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="episode" label={{ value: "Episode", position: "insideBottom", offset: -2 }} />
          <YAxis domain={[0, 1]} label={{ value: "Bias Score", angle: -90, position: "insideLeft" }} />
          <Tooltip formatter={(v) => v.toFixed(3)} />
          <Line type="monotone" dataKey="bias_score" stroke="#3b82f6" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}