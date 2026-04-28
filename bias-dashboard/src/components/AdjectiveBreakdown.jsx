export default function AdjectiveBreakdown({ adjectives }) {
  const normalizedAdjectives = (() => {
    if (!adjectives || typeof adjectives !== 'object') return null

    if (Array.isArray(adjectives)) {
      return adjectives.reduce((acc, item, idx) => {
        if (item && typeof item === 'object') {
          const label = item.name || item.title || `Adjectives ${idx + 1}`
          acc[label] = item
        }
        return acc
      }, {})
    }

    // If it's an object with variant IDs as keys, return as-is
    // Each value should have { agentic: [...], communal: [...] }
    return adjectives
  })()

  if (!normalizedAdjectives || Object.keys(normalizedAdjectives).length === 0)
    return null

  return (
    <div style={{ background:"var(--surface)", border:"1px solid var(--border)", borderRadius:12, padding:28 }}>
      <p style={{ color:"var(--muted)", fontSize:11, fontFamily:"'DM Mono', monospace", letterSpacing:2, textTransform:"uppercase", marginBottom:20 }}>Adjective Breakdown</p>
      {Object.keys(normalizedAdjectives).map(name => {
        const data = normalizedAdjectives[name] || {}
        const agentic = Array.isArray(data.agentic) ? data.agentic : []
        const communal = Array.isArray(data.communal) ? data.communal : []
        return (
          <div key={name} style={{ borderBottom:"1px solid var(--border)", paddingBottom:20, marginBottom:20 }}>
            <p style={{ fontFamily:"'DM Mono', monospace", fontSize:12, color:"var(--muted)", letterSpacing:1, textTransform:"uppercase", marginBottom:12 }}>{name}</p>
            <div style={{ marginBottom:12 }}>
              <p style={{ fontSize:11, fontFamily:"'DM Mono', monospace", color:"var(--accent)", letterSpacing:1, textTransform:"uppercase", marginBottom:8 }}>Agentic</p>
              <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
                {agentic.length > 0 ? agentic.map((w, i) => (
                  <span key={i} style={{ background:"rgba(108,99,255,0.15)", border:"1px solid rgba(108,99,255,0.3)", borderRadius:20, padding:"3px 12px", fontSize:12, color:"var(--accent)", fontFamily:"'DM Mono', monospace" }}>{w}</span>
                )) : <span style={{ fontSize:12, color:"var(--muted)", fontFamily:"'DM Mono', monospace" }}>none</span>}
              </div>
            </div>
            <div>
              <p style={{ fontSize:11, fontFamily:"'DM Mono', monospace", color:"#ff6b9d", letterSpacing:1, textTransform:"uppercase", marginBottom:8 }}>Communal</p>
              <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
                {communal.length > 0 ? communal.map((w, i) => (
                  <span key={i} style={{ background:"rgba(255,107,157,0.15)", border:"1px solid rgba(255,107,157,0.3)", borderRadius:20, padding:"3px 12px", fontSize:12, color:"#ff6b9d", fontFamily:"'DM Mono', monospace" }}>{w}</span>
                )) : <span style={{ fontSize:12, color:"var(--muted)", fontFamily:"'DM Mono', monospace" }}>none</span>}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}