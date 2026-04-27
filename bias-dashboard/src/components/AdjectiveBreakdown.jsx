export default function AdjectiveBreakdown({ adjectives }) {
  console.log("adjectives prop:", adjectives)
  if (!adjectives || typeof adjectives !== 'object' || Object.keys(adjectives).length === 0) {
    return <div className="bg-white p-6 rounded-lg shadow">No adjective data</div>
  }

  const names = Object.keys(adjectives)

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Adjective Breakdown</h2>
      <div className="space-y-6">
        {names.map(name => {
          const data = adjectives[name] || {}
          const agentic = Array.isArray(data.agentic) ? data.agentic : []
          const communal = Array.isArray(data.communal) ? data.communal : []
          
          return (
            <div key={name} className="border-b pb-4">
              <p className="font-semibold text-gray-800 mb-3">{name}</p>
              
              <div className="mb-3">
                <p className="text-sm font-semibold text-blue-600 mb-2">Agentic Words:</p>
                <div className="flex flex-wrap gap-2">
                  {agentic.length > 0 ? (
                    agentic.map((word, idx) => (
                      <span key={idx} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                        {word}
                      </span>
                    ))
                  ) : (
                    <span className="text-gray-500 text-sm">None</span>
                  )}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold text-pink-600 mb-2">Communal Words:</p>
                <div className="flex flex-wrap gap-2">
                  {communal.length > 0 ? (
                    communal.map((word, idx) => (
                      <span key={idx} className="px-3 py-1 bg-pink-100 text-pink-800 rounded-full text-sm">
                        {word}
                      </span>
                    ))
                  ) : (
                    <span className="text-gray-500 text-sm">None</span>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}