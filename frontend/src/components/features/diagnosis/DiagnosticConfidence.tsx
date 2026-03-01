export function DiagnosticConfidence({ percentage }: { percentage: number }) {
  return (
    <div className="flex-none w-full md:w-auto flex flex-col items-center justify-center bg-white/5 rounded-2xl p-6 border border-white/10 backdrop-blur-sm">
      <div
        className="relative w-36 h-36 rounded-full mb-4"
        style={{ background: `conic-gradient(#3b82f6 0% ${percentage}%, rgba(255,255,255,0.05) ${percentage}% 100%)` }}
      >
        <div className="absolute inset-[12px] bg-[#0D1B2A] rounded-full flex flex-col items-center justify-center shadow-inner">
          <span className="text-4xl font-bold text-white tracking-tighter">{percentage}%</span>
        </div>
      </div>
      <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-blue-200 text-center">
        Diagnosztikai<br />Konfidencia
      </span>
    </div>
  );
}

export default DiagnosticConfidence;
