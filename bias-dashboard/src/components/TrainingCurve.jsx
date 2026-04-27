export default function TrainingCurve({ trainingLog }) {
  if (!trainingLog || !Array.isArray(trainingLog) || trainingLog.length === 0) {
    return (
      <div className="bg-slate-900 text-white rounded-2xl p-6 border border-slate-800 shadow-lg">
        <h3 className="text-xl font-semibold mb-2">Training Progress</h3>
        <p className="text-slate-400 text-sm">No training data available.</p>
      </div>
    );
  }

  const safeNumber = (value, fallback = 0) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
  };

  const initialBias = safeNumber(
    trainingLog[0]?.overall_bias_score ??
      trainingLog[0]?.bias_score ??
      trainingLog[0]?.score
  );

  const finalBias = safeNumber(
    trainingLog[trainingLog.length - 1]?.overall_bias_score ??
      trainingLog[trainingLog.length - 1]?.bias_score ??
      trainingLog[trainingLog.length - 1]?.score
  );

  const reduction =
    initialBias > 0
      ? (((initialBias - finalBias) / initialBias) * 100).toFixed(1)
      : "0.0";

  const maxBias = Math.max(
    ...trainingLog.map((step) =>
      safeNumber(step?.overall_bias_score ?? step?.bias_score ?? step?.score)
    ),
    1
  );

  return (
    <div className="bg-slate-900 text-white rounded-2xl p-6 border border-slate-800 shadow-lg">
      <div className="mb-6">
        <h3 className="text-2xl font-bold mb-2">Training Progress</h3>
        <p className="text-slate-400 text-sm">
          Bias score reduced from {initialBias.toFixed(2)} to {finalBias.toFixed(2)} ({reduction}% improvement)
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="text-2xl font-bold text-red-400">
            {initialBias.toFixed(2)}
          </div>
          <div className="text-sm text-slate-400 mt-1">Starting Score</div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="text-2xl font-bold text-cyan-400">{reduction}%</div>
          <div className="text-sm text-slate-400 mt-1">Reduction</div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="text-2xl font-bold text-green-400">
            {finalBias.toFixed(2)}
          </div>
          <div className="text-sm text-slate-400 mt-1">Final Score</div>
        </div>
      </div>

      <div className="space-y-4">
        {trainingLog.map((step, index) => {
          const score = safeNumber(
            step?.overall_bias_score ?? step?.bias_score ?? step?.score
          );

          const barWidth = `${(score / maxBias) * 100}%`;

          return (
            <div key={index} className="bg-slate-800/60 rounded-xl p-4 border border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-slate-200">
                  Iteration {step?.iteration ?? index + 1}
                </span>
                <span className="text-sm text-slate-300">
                  {score.toFixed(2)}
                </span>
              </div>

              <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden mb-2">
                <div
                  className="h-full bg-gradient-to-r from-red-500 via-yellow-400 to-green-500 rounded-full transition-all duration-500"
                  style={{ width: barWidth }}
                />
              </div>

              <div className="text-xs text-slate-400">
                {step?.note || step?.summary || "Bias optimization step completed"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}