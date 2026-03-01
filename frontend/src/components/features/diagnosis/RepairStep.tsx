import { MaterialIcon } from '../../ui/MaterialIcon';

export function RepairStep({
  number,
  title,
  description,
  tools,
  expertTip,
}: {
  number: number;
  title: string;
  description: string;
  tools: { icon: string; name: string }[];
  expertTip: string;
}) {
  return (
    <div className="relative group">
      <div className="absolute left-0 md:left-0 top-0 w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center font-bold z-10 border-4 border-slate-50 shadow-sm group-hover:scale-110 transition-transform">
        {number}
      </div>
      <div className="ml-12 md:ml-16 bg-white rounded-2xl p-6 md:p-8 shadow-sm border border-slate-100 hover:shadow-md transition-all duration-300">
        <h4 className="text-xl font-bold text-slate-900 mb-3">{title}</h4>
        <p className="text-slate-600 text-sm leading-relaxed mb-6">{description}</p>

        {/* Szükséges szerszámok */}
        <div className="mb-6">
          <h5 className="text-[10px] font-bold uppercase text-slate-400 mb-3 tracking-widest">Szükséges szerszámok</h5>
          <div className="flex flex-wrap gap-2">
            {tools.map((tool, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-slate-600 text-xs font-medium hover:bg-white transition-colors"
              >
                <MaterialIcon name={tool.icon} className="text-sm font-light" />
                {tool.name}
              </div>
            ))}
          </div>
        </div>

        {/* Szakértői tipp */}
        <div className="mt-4 border-l-4 border-amber-400 bg-amber-50 p-5 rounded-r-xl">
          <div className="flex items-start gap-4">
            <div className="p-1.5 bg-amber-100 rounded-full text-amber-600">
              <MaterialIcon name="lightbulb" className="text-lg" />
            </div>
            <div>
              <span className="block text-xs font-black text-amber-700 uppercase tracking-wide mb-1.5">Szakértői Tipp</span>
              <p className="text-sm text-slate-800 font-medium leading-relaxed">{expertTip}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default RepairStep;
