export default function LLMLeaderboard({ leaderboard }) {
  if (!leaderboard || leaderboard.length === 0) return null

  return (
    <div style={{ background:"var(--surface)", border:"1px solid var(--border)", borderRadius:12, padding:28 }}>
      <p style={{ color:"var(--muted)", fontSize:11, fontFamily:"'DM Mono', monospace", letterSpacing:2, textTransform:"uppercase", marginBottom:20 }}>LLM Leaderboard</p>
      <table style={{ width:"100%", borderCollapse:"collapse" }}>
        <thead>
          <tr style={{ borderBottom:"1px solid var(--border)" }}>
            {["Rank","Model","Avg Bias","Episodes","Religion Gap","College Gap"].map(h => (
              <th key={h} style={{ padding:"10px 16px", fontSize:11, fontFamily:"'DM Mono', monospace", color:"var(--muted)", letterSpacing:1, textTransform:"uppercase", fontWeight:500, textAlign:"left" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {leaderboard.map((entry, idx) => {
            const bias = entry.avg_bias_score ?? 0
            const color = bias < 0.3 ? "var(--accent2)" : bias < 0.8 ? "var(--warn)" : "var(--danger)"
            return (
              <tr key={idx} style={{ borderBottom:"1px solid var(--border)", background: idx % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                <td style={{ padding:"12px 16px", fontSize:13, color:"var(--muted)", fontFamily:"'DM Mono', monospace" }}>#{idx+1}</td>
                <td style={{ padding:"12px 16px", fontSize:13, color:"var(--text)", fontWeight:600 }}>{entry.model}</td>
                <td style={{ padding:"12px 16px", fontSize:13, color, fontFamily:"'DM Mono', monospace", fontWeight:700 }}>{bias.toFixed(3)}</td>
                <td style={{ padding:"12px 16px", fontSize:13, color:"var(--muted)", fontFamily:"'DM Mono', monospace" }}>{entry.episodes_to_debias ?? "—"}</td>
                <td style={{ padding:"12px 16px", fontSize:13, color:"var(--muted)", fontFamily:"'DM Mono', monospace" }}>{entry.score_gap_religion?.toFixed(2) ?? "—"}</td>
                <td style={{ padding:"12px 16px", fontSize:13, color:"var(--muted)", fontFamily:"'DM Mono', monospace" }}>{entry.score_gap_college?.toFixed(2) ?? "—"}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p style={{ marginTop:16, fontSize:12, color:"var(--muted)", fontFamily:"'DM Mono', monospace" }}>Lower bias score = better. Sorted most → least biased.</p>
    </div>
  )
}