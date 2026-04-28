export default function ProbeResultCard({ results, baselineName }) {
  if (!results || typeof results !== "object" || Object.keys(results).length === 0)
    return null

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 28 }}>
      <div style={{ marginBottom: 20 }}>
        <p style={{ color: "var(--muted)", fontSize: 11, fontFamily: "'DM Mono', monospace", letterSpacing: 2, textTransform: "uppercase", margin: 0 }}>
          Demographic Probes
        </p>
        {baselineName && (
          <p style={{ color: "var(--muted)", fontSize: 11, fontFamily: "'DM Mono', monospace", margin: "6px 0 0 0", opacity: 0.6 }}>
            5 demographic variants run against {baselineName}'s performance baseline
          </p>
        )}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {Object.entries(results).map(([vid, data]) => {
          const p = data?.parsed || {}
          const rec = p.promotion_recommendation || data?.decision || "—"
          const score = p.promotion_score ?? data?.score ?? 0
          const reason = p.reasoning || data?.justification || ""
          const name = data?.profile?.name || vid
          const isYes = (typeof rec === "string") && (rec.toLowerCase().includes("yes"))

          return (
            <div key={vid} style={{
              borderLeft: `3px solid ${isYes ? "var(--accent2)" : "var(--danger)"}`,
              paddingLeft: 16, paddingTop: 12, paddingBottom: 12,
              borderRadius: 4,
              background: isYes ? "rgba(0,212,170,0.04)" : "rgba(255,77,109,0.04)",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                  <p style={{ fontWeight: 600, color: "var(--text)", margin: 0 }}>{name}</p>
                  <p style={{ fontSize: 12, fontFamily: "'DM Mono', monospace", color: isYes ? "var(--accent2)" : "var(--danger)", marginTop: 4, marginBottom: 0 }}>{rec}</p>
                </div>
                <span style={{ fontFamily: "'Syne', sans-serif", fontSize: 28, fontWeight: 800, color: "var(--accent)" }}>
                  {Number(score).toFixed(1)}
                </span>
              </div>
              <p style={{ fontSize: 13, color: "var(--muted)", marginTop: 10, lineHeight: 1.7, marginBottom: 0 }}>{reason}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}