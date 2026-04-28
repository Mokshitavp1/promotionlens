export default function BiasHero({ comparison }) {
  if (!comparison || !comparison.candidate_a) return null

  const { candidate_a_name, candidate_b_name, candidate_a, candidate_b, score_gap, bias_types_detected, finding, severity } = comparison
  const gap = parseFloat(score_gap) || 0
  const nameA = candidate_a_name || candidate_a
  const nameB = candidate_b_name || candidate_b

  // No gap — render clean green state, don't say "bias detected"
  if (gap === 0 || severity === "LOW") {
    return (
      <div style={{ background: "rgba(0,230,118,0.05)", border: "1px solid rgba(0,230,118,0.3)", borderRadius: 12, padding: 32 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#00e676", boxShadow: "0 0 12px #00e676" }} />
          <span style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 20, color: "#00e676", letterSpacing: 2, textTransform: "uppercase" }}>
            No Bias Detected
          </span>
        </div>
        <p style={{ color: "var(--muted)", fontSize: 13, fontFamily: "'DM Mono', monospace", lineHeight: 1.7, margin: 0 }}>
          {finding || `${nameA} and ${nameB} scored within acceptable range on the identical profile.`}
        </p>
      </div>
    )
  }

  const isCritical = severity === "CRITICAL" || severity === "HIGH"
  const color = isCritical ? "var(--danger)" : "var(--warn)"
  const bg    = isCritical ? "rgba(255,77,109,0.06)" : "rgba(255,183,3,0.06)"
  const glow  = isCritical ? "rgba(255,77,109,0.08)" : "rgba(255,183,3,0.08)"

  return (
    <div style={{ background: bg, border: `1px solid ${color}`, borderRadius: 12, padding: 32, position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 0, right: 0, width: 300, height: 300, background: `radial-gradient(circle, ${glow} 0%, transparent 70%)`, pointerEvents: "none" }} />
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
        <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, boxShadow: `0 0 12px ${color}` }} />
        <span style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 20, color, letterSpacing: 2, textTransform: "uppercase" }}>
          Bias Detected — {severity}
        </span>
      </div>
      <p style={{ color: "var(--text)", fontSize: 16, marginBottom: 24, lineHeight: 1.6 }}>
        <code style={{ background: "var(--surface2)", padding: "2px 8px", borderRadius: 4, fontFamily: "'DM Mono', monospace", fontSize: 14 }}>{nameA}</code>
        {" "}scored{" "}
        <span style={{ color, fontWeight: 700 }}>{gap.toFixed(1)} pts lower</span>
        {" "}than{" "}
        <code style={{ background: "var(--surface2)", padding: "2px 8px", borderRadius: 4, fontFamily: "'DM Mono', monospace", fontSize: 14 }}>{nameB}</code>
        {" "}on the <em>identical</em> profile.
      </p>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        {(bias_types_detected || []).map(t => (
          <span key={t} style={{ background: "rgba(255,77,109,0.15)", border: "1px solid rgba(255,77,109,0.3)", borderRadius: 20, padding: "4px 14px", fontSize: 12, fontFamily: "'DM Mono', monospace", color: "var(--danger)", textTransform: "uppercase", letterSpacing: 1 }}>{t}</span>
        ))}
      </div>
      <p style={{ color: "var(--muted)", fontSize: 13, fontFamily: "'DM Mono', monospace", lineHeight: 1.7, borderTop: "1px solid var(--border)", paddingTop: 16 }}>{finding}</p>
    </div>
  )
}