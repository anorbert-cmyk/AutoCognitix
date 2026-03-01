/**
 * DemoResultPage - Bemutató AI Diagnosztikai Jelentés
 *
 * Pre-filled demo page with realistic P0300 misfire simulation.
 * Shows what a paid user would receive after running a full diagnosis.
 * Includes real Hungarian auto parts prices from Bárdi Autó, Uni Autó, AUTODOC.
 *
 * Design: Navy theme (#0D1B2A), Space Grotesk font, Material Symbols icons
 */

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  demoDiagnosisResponse,
  demoParts,
  demoVehicleImage,
  demoVehicleDetails,
} from '../data/demoData';
import { MaterialIcon } from '../components/ui/MaterialIcon';
import { DiagnosticConfidence } from '../components/features/diagnosis/DiagnosticConfidence';
import { RepairStep } from '../components/features/diagnosis/RepairStep';
import { PartStoreCardGrid } from '../components/features/diagnosis/PartStoreCard';

function DemoBanner({ onStartDiagnosis }: { onStartDiagnosis: () => void }) {
  return (
    <div className="bg-gradient-to-r from-amber-500 via-amber-400 to-yellow-400 text-slate-900 px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-3 shadow-lg print:hidden">
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-white/30 backdrop-blur-sm">
          <MaterialIcon name="visibility" className="text-2xl text-slate-900" />
        </div>
        <div>
          <h2 className="text-base font-bold font-['Space_Grotesk',sans-serif]">
            Bemutató mód – Így néz ki egy teljes diagnosztikai jelentés
          </h2>
          <p className="text-sm text-slate-700">
            Szimulált P0300 hibakód (égéskimaradás) · VW Golf VII 1.4 TSI · Valós alkatrész árak
          </p>
        </div>
      </div>
      <button
        onClick={onStartDiagnosis}
        className="flex-shrink-0 px-6 py-3 bg-[#0D1B2A] text-white font-bold rounded-xl hover:bg-[#1a2d42] transition-colors shadow-lg flex items-center gap-2"
      >
        <MaterialIcon name="add_circle" className="text-xl" />
        Saját diagnózis indítása
      </button>
    </div>
  );
}

// =============================================================================
// Main DemoResultPage Component
// =============================================================================

export default function DemoResultPage() {
  const navigate = useNavigate();
  const [imageError, setImageError] = useState(false);
  const result = demoDiagnosisResponse;
  const vehicle = demoVehicleDetails;

  const primaryDTC = result.dtc_codes[0];
  const dtcDescription = result.probable_causes[0]?.title || 'Több hengeres égéskimaradás';
  const confidencePercentage = Math.round(result.confidence_score * 100);

  // Map repair steps
  const repairSteps = result.recommended_repairs.map((repair, index) => ({
    number: index + 1,
    title: repair.title,
    description: repair.description,
    tools: repair.tools_needed.length > 0
      ? repair.tools_needed.map((t) => ({ icon: t.icon_hint || 'handyman', name: t.name }))
      : repair.parts_needed.slice(0, 3).map((part) => ({ icon: 'handyman', name: part })),
    expertTip: repair.expert_tips?.[0] || 'Kérjen szakemberi segítséget, ha bizonytalan.',
  }));

  const handleStartDiagnosis = () => navigate('/diagnosis');

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col relative font-['Noto_Sans',sans-serif]">
      {/* Demo Banner */}
      <DemoBanner onStartDiagnosis={handleStartDiagnosis} />

      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white/80 backdrop-blur-md">
        <div className="px-6 md:px-10 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-[#0D1B2A] text-white shadow-lg shadow-[#0D1B2A]/20">
              <MaterialIcon name="build_circle" className="text-2xl" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 font-['Space_Grotesk',sans-serif]">MechanicAI</h1>
            <span className="ml-2 px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 text-[10px] font-bold uppercase tracking-wider border border-amber-200">
              Demo
            </span>
          </div>

          <nav className="hidden md:flex items-center gap-8">
            <Link className="text-sm font-medium text-slate-500 hover:text-[#0D1B2A] transition-colors" to="/">Főoldal</Link>
            <Link className="text-sm font-medium text-slate-500 hover:text-[#0D1B2A] transition-colors" to="/diagnosis">Új diagnózis</Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 w-full max-w-7xl mx-auto p-4 md:p-8 lg:p-10">
        {/* Page Title */}
        <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-100 text-amber-800 text-xs font-bold uppercase tracking-wider mb-4">
              <MaterialIcon name="play_circle" className="text-sm" />
              Bemutató Diagnosztikai Folyamat
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
            <div className="flex items-center gap-3 bg-slate-50 pr-6">
              <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center shadow-lg shadow-green-500/20">
                <MaterialIcon name="check" className="text-base" />
              </div>
              <span className="text-xs font-bold uppercase text-slate-500">Adatfelvétel</span>
            </div>
            <div className="flex items-center gap-3 bg-slate-50 px-6">
              <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center shadow-lg shadow-green-500/20">
                <MaterialIcon name="check" className="text-base" />
              </div>
              <span className="text-xs font-bold uppercase text-slate-500">Elemzés</span>
            </div>
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
                  src={imageError ? 'https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800&h=600&fit=crop' : demoVehicleImage}
                  alt={`${result.vehicle_make} ${result.vehicle_model}`}
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                  loading="lazy"
                  decoding="async"
                  onError={() => setImageError(true)}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent to-transparent"></div>
                <div className="absolute bottom-5 left-6 text-white">
                  <h3 className="text-2xl font-bold font-['Space_Grotesk',sans-serif] tracking-tight">
                    {result.vehicle_make} {result.vehicle_model}
                  </h3>
                  <p className="text-sm font-medium text-slate-300">{vehicle.engineInfo} · {result.vehicle_year}</p>
                </div>
              </div>

              {/* Vehicle Details */}
              <div className="p-6 pt-8">
                <div className="space-y-4 text-sm mb-8">
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-slate-500 font-medium">Rendszám</span>
                    <span className="font-bold text-slate-900 font-mono bg-slate-100 px-2 py-0.5 rounded">{vehicle.licensePlate}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-slate-500 font-medium">Alvázszám</span>
                    <span className="font-bold text-slate-900 font-mono text-xs text-right">{vehicle.vin}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-slate-500 font-medium">Futásteljesítmény</span>
                    <span className="font-bold text-slate-900">{vehicle.mileage}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-slate-500 font-medium">Üzemanyag</span>
                    <span className="font-bold text-slate-900">{vehicle.fuelType}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-100">
                    <span className="text-slate-500 font-medium">Váltó</span>
                    <span className="font-bold text-slate-900">{vehicle.transmission}</span>
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
                    <div className="flex gap-2 mt-3">
                      {result.dtc_codes.slice(1).map((code) => (
                        <span
                          key={code}
                          className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold text-red-700 bg-red-100 border border-red-200"
                        >
                          {code}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Customer Complaint */}
                <div>
                  <h4 className="text-xs font-bold uppercase text-slate-400 mb-3 flex items-center gap-2 tracking-wide">
                    <MaterialIcon name="person" className="text-base" />
                    Ügyfél panasz + szerelői megjegyzés
                  </h4>
                  <div className="relative">
                    <span className="absolute -left-1 -top-2 text-3xl text-slate-200 font-serif">&ldquo;</span>
                    <p className="text-sm italic text-slate-600 leading-relaxed pl-4">
                      {result.symptoms}
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
                    A rendszer <span className="text-white font-bold decoration-blue-400 underline underline-offset-4 decoration-2">{primaryDTC} hibakódot</span> és {result.dtc_codes.length - 1} kapcsolódó kódot detektált,
                    amelyek a motorvezérlő (ECU) által észlelt főtengely-szöggyorsulás ingadozásra utalnak.
                  </p>
                  <div className="text-sm text-slate-300 leading-relaxed space-y-4 border-t border-white/10 pt-4 whitespace-pre-line">
                    <p>{result.root_cause_analysis}</p>
                  </div>
                </div>

                <DiagnosticConfidence percentage={confidencePercentage} />
              </div>
            </section>

            {/* Probable Causes */}
            <section>
              <div className="flex items-center gap-4 mb-6">
                <h3 className="text-2xl font-bold text-slate-900 font-['Space_Grotesk',sans-serif]">Lehetséges okok</h3>
                <div className="h-px flex-1 bg-slate-200"></div>
              </div>
              <div className="space-y-4">
                {result.probable_causes.map((cause, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm hover:shadow-md transition-all"
                  >
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <h4 className="text-base font-bold text-slate-900">{cause.title}</h4>
                      <span
                        className={`flex-shrink-0 inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold ${
                          cause.confidence >= 0.8
                            ? 'bg-green-100 text-green-700'
                            : cause.confidence >= 0.5
                            ? 'bg-amber-100 text-amber-700'
                            : 'bg-slate-100 text-slate-600'
                        }`}
                      >
                        {Math.round(cause.confidence * 100)}%
                      </span>
                    </div>
                    <p className="text-sm text-slate-600 leading-relaxed">{cause.description}</p>
                    <div className="flex items-center gap-2 mt-3">
                      {cause.related_dtc_codes.map((code) => (
                        <span key={code} className="px-2 py-0.5 rounded bg-slate-100 text-slate-600 text-[10px] font-mono font-bold">
                          {code}
                        </span>
                      ))}
                      {cause.components.map((comp) => (
                        <span key={comp} className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 text-[10px] font-medium">
                          {comp}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Repair Steps Section */}
            <section>
              <div className="flex items-center gap-4 mb-10">
                <h3 className="text-2xl font-bold text-slate-900 font-['Space_Grotesk',sans-serif]">Priorizált javítási terv</h3>
                <div className="h-px flex-1 bg-slate-200"></div>
                <span className="text-xs font-bold uppercase text-slate-500 tracking-wider">{repairSteps.length} lépés</span>
              </div>

              <div className="relative pl-4 md:pl-6 space-y-12">
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
            </section>

            {/* Parts Store Cards Section */}
            <PartStoreCardGrid parts={demoParts} />

            {/* Total Cost Estimate Card */}
            {result.total_cost_estimate && (
              <section>
                <div className="bg-gradient-to-br from-[#0D1B2A] to-[#1B2838] rounded-2xl p-8 text-white shadow-xl relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-64 h-64 bg-blue-600/10 rounded-full blur-[60px] -translate-y-1/2 translate-x-1/3"></div>
                  <div className="relative z-10">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-white/10 border border-white/10">
                        <MaterialIcon name="payments" className="text-green-300" />
                      </div>
                      <h4 className="text-lg font-bold">Becsült Javítási Összköltség</h4>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-6">
                      <div>
                        <div className="text-[10px] font-bold uppercase text-slate-400 tracking-wider mb-1">Alkatrészek</div>
                        <div className="text-lg font-bold">
                          {result.total_cost_estimate.parts_min?.toLocaleString('hu-HU')} – {result.total_cost_estimate.parts_max?.toLocaleString('hu-HU')} Ft
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] font-bold uppercase text-slate-400 tracking-wider mb-1">Munkadíj</div>
                        <div className="text-lg font-bold">
                          {result.total_cost_estimate.labor_min?.toLocaleString('hu-HU')} – {result.total_cost_estimate.labor_max?.toLocaleString('hu-HU')} Ft
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
                            {result.total_cost_estimate.total_min?.toLocaleString('hu-HU')} – {result.total_cost_estimate.total_max?.toLocaleString('hu-HU')} Ft
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
              </section>
            )}

            {/* Sources Section */}
            <section>
              <div className="flex items-center gap-4 mb-6">
                <h3 className="text-2xl font-bold text-slate-900 font-['Space_Grotesk',sans-serif]">Felhasznált források</h3>
                <div className="h-px flex-1 bg-slate-200"></div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {result.sources.map((source, idx) => (
                  <div
                    key={idx}
                    className="bg-white rounded-xl p-4 border border-slate-200 shadow-sm flex items-start gap-3"
                  >
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center">
                      <MaterialIcon
                        name={
                          source.type === 'database' ? 'storage' :
                          source.type === 'tsb' ? 'description' :
                          source.type === 'forum' ? 'forum' :
                          source.type === 'manual' ? 'menu_book' : 'source'
                        }
                        className="text-base text-slate-500"
                      />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-slate-900 truncate">{source.title}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">{source.type}</span>
                        <span className="text-[10px] text-slate-400">·</span>
                        <span className={`text-[10px] font-bold ${
                          source.relevance_score >= 0.8 ? 'text-green-600' :
                          source.relevance_score >= 0.5 ? 'text-amber-600' : 'text-slate-500'
                        }`}>
                          {Math.round(source.relevance_score * 100)}% relevancia
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </div>
      </main>

      {/* Fixed Bottom Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-50 bg-slate-900 text-white shadow-[0_-4px_20px_-5px_rgba(0,0,0,0.3)] p-4 border-t border-slate-800 backdrop-blur-md bg-opacity-95 print:hidden">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <MaterialIcon name="info" className="text-amber-400" />
            <span className="text-sm text-slate-300">
              Ez egy <strong className="text-amber-400">bemutató jelentés</strong> szimulált adatokkal. A valós diagnózishoz indítson saját elemzést.
            </span>
          </div>
          <button
            onClick={handleStartDiagnosis}
            className="flex-shrink-0 w-full sm:w-auto px-6 py-3.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-bold shadow-lg shadow-blue-900/30 transition-all flex items-center justify-center gap-2"
          >
            <MaterialIcon name="add_circle" className="text-xl" />
            Saját diagnózis készítése
          </button>
        </div>
      </div>

      {/* Bottom spacer for fixed bar */}
      <div className="h-40 md:h-28"></div>
    </div>
  );
}
