/**
 * ResultPage - Részletes AI Diagnosztikai Jelentés
 * Design: Navy theme (#0D1B2A), Space Grotesk font, Material Symbols icons
 * Sticky header, floating bottom bar, responsive design
 */

import { useParams, useNavigate } from 'react-router-dom';
import { useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { useToast } from '../contexts/ToastContext';
import { useDiagnosisDetail } from '../services/hooks';
import { DiagnosisResponse, PartWithPrice } from '../services/api';

// Material Symbol ikon komponens
function MaterialIcon({ name, className = '' }: { name: string; className?: string }) {
  return (
    <span className={`material-symbols-outlined ${className}`} style={{ fontVariationSettings: "'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24" }}>
      {name}
    </span>
  );
}

// Autó kép URL - Unsplash vagy fallback
function getVehicleImageUrl(make: string, model: string): string {
  const searchQuery = encodeURIComponent(`${make} ${model} car professional photo`);
  return `https://source.unsplash.com/800x600/?${searchQuery}`;
}

const fallbackImages: Record<string, string> = {
  'Toyota': 'https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?w=800&h=600&fit=crop',
  'Honda': 'https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=800&h=600&fit=crop',
  'Ford': 'https://images.unsplash.com/photo-1551830820-330a71b99659?w=800&h=600&fit=crop',
  'BMW': 'https://images.unsplash.com/photo-1555215695-3004980ad54e?w=800&h=600&fit=crop',
  'Mercedes-Benz': 'https://images.unsplash.com/photo-1618843479619-f3d0d81e4d10?w=800&h=600&fit=crop',
  'Mercedes': 'https://images.unsplash.com/photo-1618843479619-f3d0d81e4d10?w=800&h=600&fit=crop',
  'Audi': 'https://images.unsplash.com/photo-1606664515524-ed2f786a0bd6?w=800&h=600&fit=crop',
  'Volkswagen': 'https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=800&h=600&fit=crop',
  'Tesla': 'https://images.unsplash.com/photo-1560958089-b8a1929cea89?w=800&h=600&fit=crop',
  'Skoda': 'https://images.unsplash.com/photo-1609521263047-f8f205293f24?w=800&h=600&fit=crop',
  'Opel': 'https://images.unsplash.com/photo-1612825173281-9a193378527e?w=800&h=600&fit=crop',
  'default': 'https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800&h=600&fit=crop',
};

// Conic gradient circular progress
function DiagnosticConfidence({ percentage }: { percentage: number }) {
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
        Diagnosztikai<br/>Konfidencia
      </span>
    </div>
  );
}

// Javítási lépés komponens
function RepairStep({
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

function DiagnosisResultContent({ result }: { result: DiagnosisResponse }) {
  const toast = useToast();
  const [imageError, setImageError] = useState(false);

  // Autó kép URL generálása
  const vehicleImage = useMemo(() => {
    const make = result.vehicle_make || 'default';
    if (fallbackImages[make]) {
      return fallbackImages[make];
    }
    return getVehicleImageUrl(result.vehicle_make, result.vehicle_model);
  }, [result.vehicle_make, result.vehicle_model]);

  const licensePlate = 'N/A';
  const vin = 'N/A';
  const mileage = 'N/A';
  const engineInfo = 'N/A';

  // Elsődleges DTC kód
  const primaryDTC = result.dtc_codes[0] || 'P0303';
  const dtcDescription = result.probable_causes[0]?.title || 'Henger 3 Égéskimaradás';

  // Ügyfél panasza
  const customerComplaint = result.symptoms || 'Reggelente rángat a motor hidegindításnál. A check engine lámpa villog, amikor autópályára hajtok fel.';

  // AI elemzés szöveg
  const aiAnalysisDetails = result.root_cause_analysis
    || result.probable_causes[0]?.description
    || 'Az AI elemzés nem tartalmaz részletes leírást ehhez a hibakódhoz.';

  // Konfidencia százalék
  const confidencePercentage = Math.round(result.confidence_score * 100);

  // Javítási lépések
  const repairSteps = result.recommended_repairs.length > 0
    ? result.recommended_repairs.slice(0, 5).map((repair, index) => ({
        number: index + 1,
        title: repair.title,
        description: repair.description,
        tools: (repair.tools_needed?.length > 0)
          ? repair.tools_needed.map(t => ({ icon: t.icon_hint || 'handyman', name: t.name }))
          : repair.parts_needed.slice(0, 3).map(part => ({ icon: 'handyman', name: part })),
        expertTip: repair.expert_tips?.[0] || repair.root_cause_explanation || 'Ellenőrizze a kapcsolódó alkatrészeket is a javítás során.',
      }))
    : [];

  // PDF mentés
  const handleSavePDF = () => {
    toast.info('PDF mentés elindítva...');
    window.print();
  };

  // Munkalap nyomtatás
  const handlePrintWorksheet = () => {
    toast.info('Munkalap nyomtatása...');
    window.print();
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col relative font-['Noto_Sans',sans-serif]">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white/80 backdrop-blur-md">
        <div className="px-6 md:px-10 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-[#0D1B2A] text-white shadow-lg shadow-[#0D1B2A]/20">
              <MaterialIcon name="build_circle" className="text-2xl" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 font-['Space_Grotesk',sans-serif]">MechanicAI</h1>
          </div>

          <nav className="hidden md:flex items-center gap-8">
            <a className="text-sm font-medium text-slate-500 hover:text-[#0D1B2A] transition-colors" href="/diagnosis">Vezérlőpult</a>
            <a className="text-sm font-medium text-slate-500 hover:text-[#0D1B2A] transition-colors" href="/history">Előzmények</a>
            <a className="text-sm font-medium text-slate-500 hover:text-[#0D1B2A] transition-colors" href="#">Beállítások</a>
          </nav>

          <div className="flex items-center gap-4">
            <button className="hidden md:flex items-center justify-center h-9 px-4 rounded-lg bg-slate-100 text-slate-600 text-xs font-bold hover:bg-slate-200 transition-colors">
              Kijelentkezés
            </button>
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 border border-slate-200 shadow-sm"></div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 w-full max-w-7xl mx-auto p-4 md:p-8 lg:p-10">
        {/* Page Title */}
        <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-200 text-slate-700 text-xs font-bold uppercase tracking-wider mb-4">
              <MaterialIcon name="assignment" className="text-sm" />
              Diagnosztikai Folyamat #{result.id?.slice(-4) || '4829'}
            </div>
            <h2 className="text-4xl font-bold tracking-tight text-slate-900 font-['Space_Grotesk',sans-serif]">Javítási javaslat</h2>
          </div>
          <div className="flex items-center gap-4 bg-white px-4 py-2 rounded-full border border-slate-200 shadow-sm">
            <div className="flex h-3 w-3 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </div>
            <span className="text-sm font-bold text-slate-700">Rendszer státusz: ONLINE</span>
          </div>
        </div>

        {/* Wizard Steps */}
        <div className="mb-12">
          <div className="flex items-center justify-between w-full max-w-2xl relative">
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-0.5 bg-slate-200 -z-10"></div>

            {/* Step 1 - Completed */}
            <div className="flex items-center gap-3 bg-slate-50 pr-6">
              <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center shadow-lg shadow-green-500/20">
                <MaterialIcon name="check" className="text-base" />
              </div>
              <span className="text-xs font-bold uppercase text-slate-500">Adatfelvétel</span>
            </div>

            {/* Step 2 - Completed */}
            <div className="flex items-center gap-3 bg-slate-50 px-6">
              <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center shadow-lg shadow-green-500/20">
                <MaterialIcon name="check" className="text-base" />
              </div>
              <span className="text-xs font-bold uppercase text-slate-500">Elemzés</span>
            </div>

            {/* Step 3 - Active */}
            <div className="flex items-center gap-3 bg-slate-50 pl-6">
              <div className="w-8 h-8 rounded-full bg-[#0D1B2A] text-white flex items-center justify-center shadow-lg shadow-[#0D1B2A]/30 ring-4 ring-slate-50">
                <span className="text-xs font-bold">3</span>
              </div>
              <span className="text-xs font-bold uppercase text-[#0D1B2A]">Jelentés</span>
            </div>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12">
          {/* Left Column - Vehicle Info */}
          <div className="lg:col-span-4 space-y-6">
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden sticky top-24">
              {/* Vehicle Image */}
              <div className="relative h-56 bg-slate-100 group">
                <img
                  src={imageError ? fallbackImages['default'] : vehicleImage}
                  alt={`${result.vehicle_make} ${result.vehicle_model}`}
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                  onError={() => setImageError(true)}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent to-transparent"></div>
                <div className="absolute bottom-5 left-6 text-white">
                  <h3 className="text-2xl font-bold font-['Space_Grotesk',sans-serif] tracking-tight">
                    {result.vehicle_make} {result.vehicle_model}
                  </h3>
                  <p className="text-sm font-medium text-slate-300">{engineInfo} • {result.vehicle_year}</p>
                </div>
              </div>

              {/* Vehicle Details */}
              <div className="p-6 pt-8">
                <div className="space-y-4 text-sm mb-8">
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-slate-500 font-medium">Rendszám</span>
                    <span className="font-bold text-slate-900 font-mono bg-slate-100 px-2 py-0.5 rounded">{licensePlate}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-slate-500 font-medium">Alvázszám</span>
                    <span className="font-bold text-slate-900 font-mono text-xs text-right">{vin}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-slate-500 font-medium">Futásteljesítmény</span>
                    <span className="font-bold text-slate-900">{mileage}</span>
                  </div>
                </div>

                {/* DTC Code Box */}
                <div className="bg-red-50 rounded-xl p-5 border border-red-100 mb-8 relative overflow-hidden">
                  <div className="absolute -right-4 -top-4 text-red-100">
                    <MaterialIcon name="warning" className="text-[100px]" />
                  </div>
                  <div className="relative z-10">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] font-bold uppercase text-red-600 tracking-wider">Elsődleges Hibakód</span>
                    </div>
                    <div className="text-4xl font-black text-slate-900 font-['Space_Grotesk',sans-serif] mb-1">{primaryDTC}</div>
                    <div className="text-sm font-bold text-red-800">{dtcDescription}</div>
                  </div>
                </div>

                {/* Customer Complaint */}
                <div>
                  <h4 className="text-xs font-bold uppercase text-slate-400 mb-3 flex items-center gap-2 tracking-wide">
                    <MaterialIcon name="person" className="text-base" />
                    Ügyfél panasz
                  </h4>
                  <div className="relative">
                    <span className="absolute -left-1 -top-2 text-3xl text-slate-200 font-serif">"</span>
                    <p className="text-sm italic text-slate-600 leading-relaxed pl-4">
                      {customerComplaint}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - AI Analysis & Repair Steps */}
          <div className="lg:col-span-8 space-y-10">
            {/* AI Analysis Section */}
            <section className="bg-[#0D1B2A] rounded-3xl p-8 lg:p-10 shadow-xl shadow-[#0D1B2A]/10 relative overflow-hidden text-white group">
              <div className="absolute top-0 right-0 w-80 h-80 bg-blue-600/20 rounded-full blur-[80px] -translate-y-1/2 translate-x-1/3 group-hover:bg-blue-600/30 transition-colors duration-700"></div>
              <div className="absolute bottom-0 left-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-[60px] translate-y-1/2 -translate-x-1/3"></div>

              <div className="flex flex-col md:flex-row gap-8 items-start relative z-10">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-white/10 backdrop-blur-md border border-white/10">
                      <MaterialIcon name="psychology" className="text-blue-200" />
                    </div>
                    <h3 className="text-xl font-bold tracking-tight">AI Diagnosztikai Elemzés</h3>
                  </div>
                  <p className="text-white/90 leading-relaxed mb-6 text-lg font-light">
                    A rendszer <span className="text-white font-bold decoration-blue-400 underline underline-offset-4 decoration-2">{primaryDTC} hibakódot</span> detektált, amely közvetlen összefüggésben áll a motorvezérlő (ECU) által észlelt főtengely-szöggyorsulás ingadozással.
                  </p>
                  <div className="text-sm text-slate-300 leading-relaxed space-y-4 border-t border-white/10 pt-4">
                    <p>{aiAnalysisDetails}</p>
                  </div>
                </div>

                <DiagnosticConfidence percentage={confidencePercentage} />
              </div>
            </section>

            {/* Repair Steps Section */}
            <section>
              <div className="flex items-center gap-4 mb-10">
                <h3 className="text-2xl font-bold text-slate-900 font-['Space_Grotesk',sans-serif]">Priorizált javítási terv</h3>
                <div className="h-px flex-1 bg-slate-200"></div>
                <span className="text-xs font-bold uppercase text-slate-500 tracking-wider">{repairSteps.length} lépés</span>
              </div>

              {repairSteps.length > 0 ? (
                <div className="relative pl-4 md:pl-6 space-y-12">
                  {/* Vertical line */}
                  <div className="absolute left-[28px] md:left-[36px] top-6 bottom-6 w-0.5 bg-slate-200 rounded-full"></div>

                  {repairSteps.map((step) => (
                    <RepairStep
                      key={step.number}
                      number={step.number}
                      title={step.title}
                      description={step.description}
                      tools={step.tools}
                      expertTip={step.expertTip}
                    />
                  ))}
                </div>
              ) : (
                <div className="bg-slate-100 rounded-2xl p-8 text-center">
                  <MaterialIcon name="info" className="text-4xl text-slate-400 mb-3" />
                  <p className="text-slate-500 font-medium">Nincs elérhető javítási javaslat ehhez a diagnosztikához.</p>
                </div>
              )}
            </section>

            {/* Parts & Prices Section */}
            {result.parts_with_prices && result.parts_with_prices.length > 0 && (
              <section>
                <div className="flex items-center gap-4 mb-10">
                  <h3 className="text-2xl font-bold text-slate-900 font-['Space_Grotesk',sans-serif]">Alkatrészek és Árak</h3>
                  <div className="h-px flex-1 bg-slate-200"></div>
                  <span className="text-xs font-bold uppercase text-slate-500 tracking-wider">{result.parts_with_prices.length} alkatrész</span>
                </div>

                {/* Parts Table */}
                <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden mb-8">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-200">
                          <th className="text-left px-6 py-4 text-[10px] font-bold uppercase text-slate-500 tracking-wider">Alkatrész</th>
                          <th className="text-left px-6 py-4 text-[10px] font-bold uppercase text-slate-500 tracking-wider">Kategória</th>
                          <th className="text-right px-6 py-4 text-[10px] font-bold uppercase text-slate-500 tracking-wider">Ár (HUF)</th>
                          <th className="text-right px-6 py-4 text-[10px] font-bold uppercase text-slate-500 tracking-wider">Munkaidő</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {result.parts_with_prices.map((part: PartWithPrice, idx: number) => (
                          <tr key={part.id || idx} className="hover:bg-slate-50 transition-colors">
                            <td className="px-6 py-4">
                              <div className="font-bold text-slate-900 text-sm">{part.name}</div>
                              {part.name_en && <div className="text-xs text-slate-400 mt-0.5">{part.name_en}</div>}
                            </td>
                            <td className="px-6 py-4">
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-blue-50 text-blue-700 border border-blue-100">
                                {part.category}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-right">
                              <span className="font-bold text-slate-900 text-sm">
                                {part.price_range_min?.toLocaleString('hu-HU')} - {part.price_range_max?.toLocaleString('hu-HU')}
                              </span>
                              <span className="text-xs text-slate-400 ml-1">Ft</span>
                            </td>
                            <td className="px-6 py-4 text-right text-sm text-slate-600">
                              {part.labor_hours} óra
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Total Cost Estimate Card */}
                {result.total_cost_estimate && (
                  <div className="bg-gradient-to-br from-[#0D1B2A] to-[#1B2838] rounded-2xl p-8 text-white shadow-xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-blue-600/10 rounded-full blur-[60px] -translate-y-1/2 translate-x-1/3"></div>
                    <div className="relative z-10">
                      <div className="flex items-center gap-3 mb-6">
                        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-white/10 border border-white/10">
                          <MaterialIcon name="payments" className="text-green-300" />
                        </div>
                        <h4 className="text-lg font-bold">Becsült Javítási Költség</h4>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-6">
                        <div>
                          <div className="text-[10px] font-bold uppercase text-slate-400 tracking-wider mb-1">Alkatrészek</div>
                          <div className="text-lg font-bold">
                            {result.total_cost_estimate.parts_min?.toLocaleString('hu-HU')} - {result.total_cost_estimate.parts_max?.toLocaleString('hu-HU')} Ft
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] font-bold uppercase text-slate-400 tracking-wider mb-1">Munkadíj</div>
                          <div className="text-lg font-bold">
                            {result.total_cost_estimate.labor_min?.toLocaleString('hu-HU')} - {result.total_cost_estimate.labor_max?.toLocaleString('hu-HU')} Ft
                          </div>
                        </div>
                        <div>
                          <div className="text-[10px] font-bold uppercase text-slate-400 tracking-wider mb-1">Becsült idő</div>
                          <div className="text-lg font-bold">{result.total_cost_estimate.estimated_hours} óra</div>
                        </div>
                      </div>

                      <div className="border-t border-white/10 pt-6">
                        <div className="flex items-end justify-between">
                          <div>
                            <div className="text-[10px] font-bold uppercase text-slate-400 tracking-wider mb-1">Összesen</div>
                            <div className="text-3xl font-black font-['Space_Grotesk',sans-serif]">
                              {result.total_cost_estimate.total_min?.toLocaleString('hu-HU')} - {result.total_cost_estimate.total_max?.toLocaleString('hu-HU')} Ft
                            </div>
                          </div>
                          <div className="text-right">
                            <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                              result.total_cost_estimate.difficulty === 'easy' ? 'bg-green-500/20 text-green-300' :
                              result.total_cost_estimate.difficulty === 'medium' ? 'bg-yellow-500/20 text-yellow-300' :
                              result.total_cost_estimate.difficulty === 'hard' ? 'bg-orange-500/20 text-orange-300' :
                              'bg-red-500/20 text-red-300'
                            }`}>
                              {result.total_cost_estimate.difficulty === 'easy' ? 'Könnyű' :
                               result.total_cost_estimate.difficulty === 'medium' ? 'Közepes' :
                               result.total_cost_estimate.difficulty === 'hard' ? 'Nehéz' :
                               'Szakértő'}
                            </span>
                          </div>
                        </div>
                      </div>

                      {result.total_cost_estimate.disclaimer && (
                        <p className="mt-4 text-xs text-slate-400 italic">{result.total_cost_estimate.disclaimer}</p>
                      )}
                    </div>
                  </div>
                )}
              </section>
            )}
          </div>
        </div>
      </main>

      {/* Fixed Bottom Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-50 bg-slate-900 text-white shadow-[0_-4px_20px_-5px_rgba(0,0,0,0.3)] p-4 border-t border-slate-800 backdrop-blur-md bg-opacity-95 print:hidden">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-end gap-4">
          <button
            onClick={handleSavePDF}
            className="w-full sm:w-auto px-6 py-3.5 rounded-lg border border-slate-600 text-slate-200 font-bold hover:bg-slate-800 transition-all flex items-center justify-center gap-2 shadow-sm"
          >
            <MaterialIcon name="picture_as_pdf" className="text-xl" />
            Jelentés mentése PDF-ként
          </button>
          <button
            onClick={handlePrintWorksheet}
            className="w-full sm:w-auto px-6 py-3.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-bold shadow-lg shadow-blue-900/30 transition-all flex items-center justify-center gap-2"
          >
            <MaterialIcon name="print" className="text-xl" />
            Munkalap nyomtatása
          </button>
        </div>
      </div>

      {/* Bottom spacer for fixed bar */}
      <div className="h-40 md:h-28"></div>
    </div>
  );
}

export default function ResultPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data: result, isLoading, error, refetch } = useDiagnosisDetail(id);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-[#0D1B2A] mx-auto mb-4" />
          <p className="text-slate-600 font-medium">Diagnózis betöltése...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center font-['Noto_Sans',sans-serif]">
        <div className="text-center max-w-md mx-auto px-4">
          <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <span className="material-symbols-outlined text-4xl text-slate-400">error_outline</span>
          </div>
          <h2 className="text-xl font-bold text-slate-900 mb-2">
            A diagnózis nem található
          </h2>
          <p className="text-slate-600 mb-6">
            Lehetséges, hogy a diagnózis törölve lett vagy nem létezik.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              onClick={() => refetch()}
              className="px-6 py-3 border-2 border-slate-300 text-slate-700 font-bold rounded-xl hover:bg-slate-50 transition-colors"
            >
              Újrapróbálás
            </button>
            <button
              onClick={() => navigate('/diagnosis')}
              className="px-6 py-3 bg-[#0D1B2A] text-white font-bold rounded-xl hover:bg-[#1a2d42] transition-colors"
            >
              Új diagnózis készítése
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  return <DiagnosisResultContent result={result} />;
}
