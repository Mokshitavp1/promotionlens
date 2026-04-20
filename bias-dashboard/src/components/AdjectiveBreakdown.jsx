export default function AdjectiveBreakdown({ adjectives }) {
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Adjectives Used in Reviews</h2>
      
      <div className="space-y-3 mb-6">
        {adjectives.map((adj, idx) => (
          <div key={idx} className="flex items-center justify-between">
            <span className={`font-semibold ${adj.biased ? "text-red-600" : "text-green-600"}`}>
              {adj.word}
            </span>
            <div className="flex items-center gap-3">
              <div className="w-32 bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${adj.biased ? "bg-red-500" : "bg-green-500"}`}
                  style={{ width: `${(adj.count / 6) * 100}%` }}
                />
              </div>
              <span className="text-sm text-gray-600 w-8">{adj.count}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div className="bg-red-50 p-3 rounded">
          <p className="font-semibold text-red-700">Biased Words</p>
          <p className="text-gray-700">
            {adjectives
              .filter((a) => a.biased)
              .map((a) => a.word)
              .join(", ")}
          </p>
        </div>
        <div className="bg-green-50 p-3 rounded">
          <p className="font-semibold text-green-700">Neutral Words</p>
          <p className="text-gray-700">
            {adjectives
              .filter((a) => !a.biased)
              .map((a) => a.word)
              .join(", ")}
          </p>
        </div>
      </div>
    </div>
  )
}