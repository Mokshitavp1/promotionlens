export default function TrainingCurve({ trainingLog }) {
  if (!trainingLog || !Array.isArray(trainingLog) || trainingLog.length === 0)
    return null

  const safeNumber = (value, fallback = 0) => {
    const num = Number(value)
    return Number.isFinite(num) ? num : fallback
  }

  const getBias = (step) => safeNumber(step?.overall_bias_score ?? step?.bias_score ?? step?.score)

  const initialBias = getBias(trainingLog[0])
  const finalBias = getBias(trainingLog[trainingLog.length - 1])
  const reduction = initialBias > 0 ? (((initialBias - finalBias) / initialBias) * 100).toFixed(1) : "0.0"
  const maxBias = Math.max(...trainingLog.map(getBias), 1)

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 28 }}>
      <p style={{ color: "var(--muted)", fontSize: 11, fontFamily: "'DM Mono', monospace", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>Training Progress</p>
      <p style={{ color: "var(--muted)", fontSize: 13, marginBottom: 24 }}>
        Bias score reduced from <span style={{ color: "var(--danger)" }}>{initialBias.toFixed(3)}</span> to <span style={{ color: "var(--accent2)" }}>{finalBias.toFixed(3)}</span> ({reduction}% improvement)
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 28 }}>
        {[
          { label: "Starting Score", value: initialBias.toFixed(3), color: "var(--danger)" },
          { label: "Reduction", value: `${reduction}%`, color: "var(--accent)" },
          { label: "Final Score", value: finalBias.toFixed(3), color: "var(--accent2)" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ background: "var(--surface2)", borderRadius: 10, padding: 16, border: "1px solid var(--border)" }}>
            <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 24, fontWeight: 800, color }}>{value}</div>
            <div style={{ fontSize: 12, color: "var(--muted)", fontFamily: "'DM Mono', monospace", marginTop: 4 }}>{label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {trainingLog.map((step, index) => {
          const score = getBias(step)
          return (
            <div key={index} style={{ background: "rgba(255,255,255,0.02)", borderRadius: 10, padding: 16, border: "1px solid var(--border)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: "var(--muted)" }}>
                  Iteration {step?.iteration ?? index + 1}
                </span>
                <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: "var(--text)" }}>
                  {score.toFixed(3)}
                </span>
              </div>
              <div style={{ width: "100%", height: 6, background: "var(--border)", borderRadius: 4, overflow: "hidden" }}>
                <div style={{ height: 6, borderRadius: 4, background: "linear-gradient(90deg, var(--danger), var(--warn), var(--accent2))", width: `${(score / maxBias) * 100}%`, transition: "width 0.5s ease" }} />
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)", fontFamily: "'DM Mono', monospace", marginTop: 6 }}>
                {step?.note || step?.summary || "Bias optimization step completed"}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}