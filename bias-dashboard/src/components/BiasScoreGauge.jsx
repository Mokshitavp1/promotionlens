export default function BiasScoreGauge({ score }) {
  const pct = Math.round((score || 0) * 100)
  const radius = 52
  const circ = 2 * Math.PI * radius
  const offset = circ - (pct / 100) * circ
  const color = pct < 30 ? "var(--accent2)" : pct < 60 ? "var(--warn)" : "var(--danger)"
  const label = pct < 30 ? "Low" : pct < 60 ? "Moderate" : "High"

  return (
    <div style={{ background:"var(--surface)", border:"1px solid var(--border)", borderRadius:12, padding:28, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center" }}>
      <p style={{ color:"var(--muted)", fontSize:11, fontFamily:"'DM Mono', monospace", letterSpacing:2, textTransform:"uppercase", marginBottom:20 }}>Bias Score</p>
      <div style={{ position:"relative" }}>
        <svg width={140} height={140}>
          <circle cx={70} cy={70} r={radius} fill="none" stroke="var(--border)" strokeWidth={8} />
          <circle cx={70} cy={70} r={radius} fill="none" stroke={color} strokeWidth={8}
            strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
            style={{ transform:"rotate(-90deg)", transformOrigin:"70px 70px", transition:"stroke-dashoffset 1s ease" }} />
        </svg>
        <div style={{ position:"absolute", inset:0, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center" }}>
          <span style={{ fontFamily:"'Syne', sans-serif", fontSize:32, fontWeight:800, color, lineHeight:1 }}>{pct}</span>
          <span style={{ fontSize:10, color:"var(--muted)", fontFamily:"'DM Mono', monospace", marginTop:4 }}>/100</span>
        </div>
      </div>
      <p style={{ color, fontSize:13, fontFamily:"'DM Mono', monospace", marginTop:16, letterSpacing:1 }}>{label} Bias</p>
    </div>
  )
}