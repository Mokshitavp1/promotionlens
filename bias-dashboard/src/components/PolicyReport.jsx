export default function PolicyReport({ report }) {
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">📋 Policy Report</h2>
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
        <p className="text-gray-800 leading-relaxed">
          {report}
        </p>
      </div>
      <div className="mt-4 p-4 bg-gray-50 rounded text-sm text-gray-600">
        <p><strong>Summary:</strong> Agent learned to reduce bias through interventions and policy adjustments across multiple episodes.</p>
      </div>
    </div>
  )
}