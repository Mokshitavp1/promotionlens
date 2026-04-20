export default function ProbeResultCard({ results }) {
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">Probe Results</h2>
      <div className="space-y-4">
        {results.map((result, idx) => (
          <div key={idx} className="border-l-4 border-blue-500 pl-4">
            <p className="font-semibold text-gray-800">{result.label}</p>
            <div className="flex justify-between items-center mt-2">
              <span className="text-lg font-bold text-blue-600">{result.value}</span>
              <span className="text-sm text-gray-500">
                Confidence: {(result.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded h-2 mt-2">
              <div
                className="bg-blue-500 h-2 rounded"
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}