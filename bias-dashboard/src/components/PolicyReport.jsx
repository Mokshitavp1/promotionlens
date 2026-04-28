export default function PolicyReport({ report }) {
  return (
    <div style={{ background:"var(--surface)", border:"1px solid var(--border)", borderRadius:12, padding:28 }}>
      <p style={{ color:"var(--muted)", fontSize:11, fontFamily:"'DM Mono', monospace", letterSpacing:2, textTransform:"uppercase", marginBottom:16 }}>Agent Policy — What the RL Agent Learned</p>
      <p style={{ color:"var(--text)", lineHeight:1.8, fontSize:15, borderLeft:"3px solid var(--accent)", paddingLeft:20 }}>{report}</p>
    </div>
  )
}