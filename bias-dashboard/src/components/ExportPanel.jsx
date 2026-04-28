import { jsPDF } from "jspdf"
import autoTable from "jspdf-autotable"

export default function ExportPanel({ auditData, leaderboard, biasScore }) {
  function exportCSV() {
    const rows = [["Candidate", "College", "Score", "Recommendation", "Readiness", "Agentic Adjectives", "Communal Adjectives"]]
    const breakdown = auditData?.bias_report?.variant_breakdown || []
    breakdown.forEach(v => {
      rows.push([v.name, v.college, v.score, v.recommendation, v.readiness,
        (v.adjectives_agentic || []).join("; "), (v.adjectives_communal || []).join("; ")])
    })
    const gaps = auditData?.bias_report?.score_gaps || {}
    rows.push([])
    rows.push(["Bias Gaps", "", "", "", "", "", ""])
    rows.push(["Religion Gap", gaps.religion ?? "—", "", "", "", "", ""])
    rows.push(["Gender Gap", gaps.gender ?? "—", "", "", "", "", ""])
    rows.push(["College Gap", gaps.college ?? "—", "", "", "", "", ""])

    const csv = rows.map(r => r.map(c => `"${c}"`).join(",")).join("\n")
    const blob = new Blob([csv], { type: "text/csv" })
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob)
    a.download = "promotionlens_audit.csv"; a.click()
  }

  function exportPDF() {
    const doc = new jsPDF()
    doc.setFont("helvetica", "bold")
    doc.setFontSize(20)
    doc.text("PromotionLens — Bias Audit Report", 14, 20)
    doc.setFont("helvetica", "normal")
    doc.setFontSize(10)
    doc.setTextColor(120)
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 28)
    doc.text(`Overall Bias Score: ${Math.round((biasScore || 0) * 100)}/100`, 14, 34)

    const gaps = auditData?.bias_report?.score_gaps || {}
    doc.setTextColor(0)
    doc.setFontSize(13)
    doc.setFont("helvetica", "bold")
    doc.text("Score Gaps by Bias Dimension", 14, 46)
    autoTable(doc, {
      startY: 50,
      head: [["Dimension", "Gap (pts)"]],
      body: Object.entries(gaps).map(([k, v]) => [k.charAt(0).toUpperCase() + k.slice(1), v.toFixed(3)]),
      styles: { fontSize: 10 }, headStyles: { fillColor: [108, 99, 255] }
    })

    const breakdown = auditData?.bias_report?.variant_breakdown || []
    doc.setFontSize(13)
    doc.setFont("helvetica", "bold")
    doc.text("Candidate Breakdown", 14, doc.lastAutoTable.finalY + 14)
    autoTable(doc, {
      startY: doc.lastAutoTable.finalY + 18,
      head: [["Name", "College", "Score", "Recommendation", "Readiness"]],
      body: breakdown.map(v => [v.name, v.college, v.score, v.recommendation, v.readiness]),
      styles: { fontSize: 9 }, headStyles: { fillColor: [108, 99, 255] }
    })

    if (leaderboard?.length) {
      doc.addPage()
      doc.setFontSize(13)
      doc.setFont("helvetica", "bold")
      doc.text("LLM Leaderboard", 14, 20)
      autoTable(doc, {
        startY: 24,
        head: [["Model", "Avg Bias", "Religion Gap", "College Gap", "Episodes to Debias"]],
        body: leaderboard.map(e => [e.model, e.avg_bias_score?.toFixed(3), e.score_gap_religion?.toFixed(2), e.score_gap_college?.toFixed(2), e.episodes_to_debias]),
        styles: { fontSize: 9 }, headStyles: { fillColor: [108, 99, 255] }
      })
    }

    doc.save("promotionlens_audit.pdf")
  }

  const btnStyle = {
    padding:"8px 18px", borderRadius:8, fontSize:12, fontFamily:"'DM Mono', monospace",
    cursor:"pointer", border:"1px solid var(--border)", fontWeight:500, letterSpacing:0.5
  }

  return (
    <div style={{ display:"flex", gap:10 }}>
      <button onClick={exportCSV} style={{ ...btnStyle, background:"var(--surface2)", color:"var(--accent2)" }}>
        ↓ Export CSV
      </button>
      <button onClick={exportPDF} style={{ ...btnStyle, background:"var(--accent)", color:"#fff", border:"none" }}>
        ↓ Export PDF
      </button>
    </div>
  )
}