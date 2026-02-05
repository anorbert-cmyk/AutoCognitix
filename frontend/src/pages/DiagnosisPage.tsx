/**
 * DiagnosisPage - Magyar nyelvű diagnosztikai oldal
 * Teljes wizard flow: Adatbevitel → Elemzés → Jelentés
 */

import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  Car,
  Mic,
  QrCode,
  Clock,
  Sparkles,
  User,
  Wrench,
  HelpCircle,
  ChevronDown,
} from 'lucide-react';
import { useAnalyzeDiagnosis, useDiagnosisHistory } from '../services/hooks';
import { cn } from '@/lib/utils';
import { ApiError } from '../services/api';
import { useToast } from '../contexts/ToastContext';
import { AnalysisProgress } from '../components/features/diagnosis/AnalysisProgress';

const currentYear = new Date().getFullYear();

// Gyártó opciók a legördülő listához
const manufacturers = [
  'Toyota',
  'Honda',
  'Ford',
  'Chevrolet',
  'Volkswagen',
  'BMW',
  'Mercedes-Benz',
  'Audi',
  'Nissan',
  'Hyundai',
  'Kia',
  'Mazda',
  'Subaru',
  'Lexus',
  'Jeep',
  'Tesla',
  'Volvo',
  'Porsche',
  'Land Rover',
  'Jaguar',
  'Fiat',
  'Alfa Romeo',
  'Peugeot',
  'Renault',
  'Citroën',
  'Škoda',
  'Seat',
  'Opel',
  'Mini',
  'Suzuki',
  'Mitsubishi',
  'Dodge',
  'Ram',
  'GMC',
  'Buick',
  'Cadillac',
  'Lincoln',
  'Acura',
  'Infiniti',
];

type WizardStep = 'input' | 'analysis' | 'report';

export default function DiagnosisPage() {
  const navigate = useNavigate();
  const toast = useToast();

  // Wizard állapot
  const [currentStep, setCurrentStep] = useState<WizardStep>('input');
  const [diagnosisId, setDiagnosisId] = useState<string | null>(null);

  // Űrlap állapot
  const [dtcCode, setDtcCode] = useState('');
  const [vehicleMake, setVehicleMake] = useState('');
  const [vehicleModel, setVehicleModel] = useState('');
  const [vehicleYear, setVehicleYear] = useState('');
  const [ownerComplaints, setOwnerComplaints] = useState('');
  const [mechanicNotes, setMechanicNotes] = useState('');

  // API hookok
  const analyzeDiagnosis = useAnalyzeDiagnosis();
  const { data: historyData } = useDiagnosisHistory({ limit: 2 });

  // Diktálás kezelő
  const handleDictation = useCallback(() => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      toast.error('A böngésző nem támogatja a beszédfelismerést');
      return;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SpeechRecognitionConstructor = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognitionConstructor) return;
    const recognition = new SpeechRecognitionConstructor();
    recognition.lang = 'hu-HU';
    recognition.continuous = false;
    recognition.interimResults = false;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setOwnerComplaints((prev) => prev + (prev ? ' ' : '') + transcript);
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      toast.error('Hiba történt a beszédfelismerés során');
    };
    recognition.start();
    toast.info('Beszéljen most...');
  }, [toast]);

  // Űrlap beküldés - elemzés indítása
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!dtcCode.trim()) {
      toast.error('Hibakód megadása kötelező');
      return;
    }

    // Elemzés lépésre váltás
    setCurrentStep('analysis');

    try {
      const result = await analyzeDiagnosis.mutateAsync({
        vehicleMake: vehicleMake.trim() || 'Ismeretlen',
        vehicleModel: vehicleModel.trim() || 'Ismeretlen',
        vehicleYear: vehicleYear ? parseInt(vehicleYear) : currentYear,
        dtcCodes: [dtcCode.trim().toUpperCase()],
        symptoms: ownerComplaints.trim() || 'Nincs megadva',
        additionalContext: mechanicNotes.trim() || undefined,
      });

      setDiagnosisId(result.id);
      // Az AnalysisProgress komponens kezeli a navigációt az eredményhez
    } catch (err) {
      setCurrentStep('input');
      if (err instanceof ApiError) {
        toast.error(err.detail);
      } else {
        toast.error('Ismeretlen hiba történt');
      }
    }
  }, [analyzeDiagnosis, vehicleMake, vehicleModel, vehicleYear, dtcCode, ownerComplaints, mechanicNotes, toast]);

  const handleCancelAnalysis = useCallback(() => {
    setCurrentStep('input');
    setDiagnosisId(null);
  }, []);

  const handleAnalysisComplete = useCallback(() => {
    if (diagnosisId) {
      navigate(`/diagnosis/${diagnosisId}`);
    }
  }, [diagnosisId, navigate]);

  const recentItems = historyData?.items || [];

  // Elemzés folyamatban lépés megjelenítése
  if (currentStep === 'analysis') {
    return (
      <AnalysisProgress
        diagnosisId={diagnosisId || undefined}
        onCancel={handleCancelAnalysis}
        onComplete={handleAnalysisComplete}
        vehicleInfo={
          vehicleMake
            ? {
                make: vehicleMake,
                model: vehicleModel || 'Ismeretlen',
                year: vehicleYear ? parseInt(vehicleYear) : currentYear,
                dtcCode: dtcCode.toUpperCase(),
              }
            : undefined
        }
      />
    );
  }

  // Adatbeviteli űrlap lépés megjelenítése
  return (
    <div className="min-h-screen bg-[#f8fafc] flex flex-col">
      {/* Fő tartalom */}
      <main className="flex-1 w-full max-w-5xl mx-auto p-4 md:p-8 lg:p-12">
        {/* Fejléc szekció */}
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-blue-200 bg-blue-50 text-[#2563eb] text-xs font-bold uppercase tracking-wider mb-4">
            <AlertTriangle className="h-3.5 w-3.5" />
            AI Diagnosztikai Eszköz
          </div>
          <h2 className="text-4xl font-black tracking-tight text-slate-900 mb-3">
            Új diagnosztikai folyamat
          </h2>
          <p className="text-lg text-slate-600 max-w-3xl">
            Adja meg a jármű adatait és az OBD hibakódokat. Az AI modellünk elemzi az adatokat és javaslatokat tesz a lehetséges okokra és javítási eljárásokra.
          </p>
        </div>

        {/* Fő kártya */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
          {/* Wizard lépések */}
          <div className="flex border-b border-slate-200 bg-slate-50 px-6 py-4 gap-8 overflow-x-auto">
            <div className="flex items-center gap-2 text-[#2563eb] font-bold text-sm whitespace-nowrap">
              <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[#2563eb] text-white text-xs font-bold">
                1
              </span>
              Adatbevitel
            </div>
            <div className="flex items-center gap-2 text-slate-400 font-medium text-sm whitespace-nowrap">
              <span className="flex items-center justify-center w-6 h-6 rounded-full border-2 border-slate-300 text-xs">
                2
              </span>
              Elemzés
            </div>
            <div className="flex items-center gap-2 text-slate-400 font-medium text-sm whitespace-nowrap">
              <span className="flex items-center justify-center w-6 h-6 rounded-full border-2 border-slate-300 text-xs">
                3
              </span>
              Jelentés
            </div>
          </div>

          {/* Űrlap */}
          <form onSubmit={handleSubmit} className="p-6 md:p-8 lg:p-10 space-y-10">
            {/* DTC szekció */}
            <section>
              <label className="block text-sm font-bold uppercase tracking-wider text-slate-500 mb-3">
                Elsődleges hibakód (DTC)
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-5 flex items-center pointer-events-none">
                  <AlertTriangle className="h-6 w-6 text-slate-300 group-focus-within:text-[#2563eb] transition-colors" />
                </div>
                <input
                  type="text"
                  value={dtcCode}
                  onChange={(e) => setDtcCode(e.target.value.toUpperCase())}
                  placeholder="pl. P0300"
                  className="block w-full pl-16 pr-14 py-5 bg-slate-50 border-2 border-slate-200 rounded-xl text-3xl font-bold text-slate-900 placeholder:text-slate-300 focus:ring-0 focus:border-[#2563eb] focus:bg-white transition-all uppercase tracking-wide"
                />
                <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                  <button
                    type="button"
                    className="p-2 text-slate-400 hover:text-[#2563eb] transition-colors"
                    title="Szkennelés kamerával"
                  >
                    <QrCode className="h-6 w-6" />
                  </button>
                </div>
              </div>
              <p className="mt-2 text-sm text-slate-500">
                Adja meg az OBD-II olvasóból származó fő hibakódot.
              </p>
            </section>

            <hr className="border-slate-200" />

            {/* Jármű szekció */}
            <section>
              <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                <Car className="h-5 w-5 text-[#2563eb]" />
                Jármű azonosítás
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Gyártó - Legördülő */}
                <div className="space-y-2">
                  <label className="block text-sm font-bold text-slate-600">Gyártó</label>
                  <div className="relative">
                    <select
                      value={vehicleMake}
                      onChange={(e) => setVehicleMake(e.target.value)}
                      className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-14 px-4 font-medium appearance-none cursor-pointer"
                    >
                      <option value="">Válasszon gyártót</option>
                      {manufacturers.map((make) => (
                        <option key={make} value={make}>
                          {make}
                        </option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 pr-4 flex items-center pointer-events-none">
                      <ChevronDown className="h-5 w-5 text-slate-400" />
                    </div>
                  </div>
                </div>

                {/* Modell */}
                <div className="space-y-2">
                  <label className="block text-sm font-bold text-slate-600">Modell</label>
                  <input
                    type="text"
                    value={vehicleModel}
                    onChange={(e) => setVehicleModel(e.target.value)}
                    placeholder="Camry"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-14 px-4 font-medium"
                  />
                </div>

                {/* Évjárat */}
                <div className="space-y-2">
                  <label className="block text-sm font-bold text-slate-600">Évjárat</label>
                  <input
                    type="number"
                    value={vehicleYear}
                    onChange={(e) => setVehicleYear(e.target.value)}
                    placeholder="2019"
                    min="1990"
                    max={currentYear + 1}
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-14 px-4 font-medium"
                  />
                </div>
              </div>
            </section>

            {/* Szöveges mezők szekció */}
            <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Tulajdonos panaszai */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
                    <User className="h-4 w-4 text-slate-400" />
                    Tulajdonos panaszai
                  </label>
                  <button
                    type="button"
                    onClick={handleDictation}
                    className="text-xs text-[#2563eb] font-bold hover:underline flex items-center gap-1"
                  >
                    <Mic className="h-3.5 w-3.5" />
                    Diktálás
                  </button>
                </div>
                <textarea
                  value={ownerComplaints}
                  onChange={(e) => setOwnerComplaints(e.target.value)}
                  placeholder="pl. Az ügyfél reggel egyenetlen alapjáratról számol be, a motorhiba lámpa villog, különösen autópályán gyorsításkor..."
                  rows={5}
                  className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:bg-white focus:border-[#2563eb] focus:ring-0 resize-none p-4 font-medium"
                />
              </div>

              {/* Szerelői jegyzetek */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
                    <Wrench className="h-4 w-4 text-slate-400" />
                    Szerelői jegyzetek / Tesztút
                  </label>
                  <span className="text-xs text-slate-400 font-medium">
                    Privát jegyzetek
                  </span>
                </div>
                <textarea
                  value={mechanicNotes}
                  onChange={(e) => setMechanicNotes(e.target.value)}
                  placeholder="pl. Megerősített gyújtáskimaradás a 3. hengerben terheléses teszt során. A gyújtógyertyák kopottak, de a tekercsek jól reagálnak..."
                  rows={5}
                  className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:bg-white focus:border-[#2563eb] focus:ring-0 resize-none p-4 font-medium"
                />
              </div>
            </section>

            {/* Művelet sáv */}
            <div className="pt-8 flex flex-col md:flex-row items-center justify-between gap-6 border-t border-slate-200">
              <div className="flex items-center gap-2 text-slate-400 text-sm font-medium">
                <Clock className="h-4 w-4" />
                <span>Automatikusan mentve 2 perce</span>
              </div>
              <div className="flex w-full md:w-auto gap-4">
                <button
                  type="button"
                  className="flex-1 md:flex-none h-14 px-8 rounded-xl border-2 border-slate-300 text-slate-700 font-bold hover:bg-slate-50 transition-colors"
                >
                  Piszkozat mentése
                </button>
                <button
                  type="submit"
                  disabled={analyzeDiagnosis.isPending}
                  className={cn(
                    "flex-1 md:flex-none h-14 px-8 rounded-xl bg-[#2563eb] hover:bg-[#1d4ed8] text-white font-bold transition-colors flex items-center justify-center gap-2 shadow-lg shadow-blue-200",
                    analyzeDiagnosis.isPending && "opacity-70 cursor-not-allowed"
                  )}
                >
                  <Sparkles className="h-4 w-4" />
                  AI Megoldás Generálása
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* Alsó kártyák */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Legutóbbi 1 */}
          <div
            className="p-5 rounded-xl bg-white border border-slate-200 flex items-start gap-4 shadow-sm cursor-pointer hover:border-[#2563eb] hover:shadow-md transition-all"
            onClick={() => recentItems[0] && navigate(`/diagnosis/${recentItems[0].id}`)}
          >
            <div className="bg-slate-100 text-slate-500 p-3 rounded-lg">
              <Clock className="h-5 w-5" />
            </div>
            <div>
              <h4 className="font-bold text-slate-900">
                Legutóbbi: {recentItems[0]?.vehicle_make || 'Toyota'} {recentItems[0]?.vehicle_model || 'Camry'}
              </h4>
              <p className="text-sm text-slate-500 mt-1">
                <span className="font-mono text-[#2563eb] font-semibold">
                  {recentItems[0]?.dtc_codes?.[0] || 'P0420'}
                </span>{' '}
                - Katalizátor rendszer hatásfoka küszöb alatt
              </p>
              <button className="text-xs text-[#2563eb] font-bold mt-2 hover:underline">
                Jelentés megtekintése
              </button>
            </div>
          </div>

          {/* Legutóbbi 2 */}
          <div
            className="p-5 rounded-xl bg-white border border-slate-200 flex items-start gap-4 shadow-sm cursor-pointer hover:border-[#2563eb] hover:shadow-md transition-all"
            onClick={() => recentItems[1] && navigate(`/diagnosis/${recentItems[1].id}`)}
          >
            <div className="bg-slate-100 text-slate-500 p-3 rounded-lg">
              <Clock className="h-5 w-5" />
            </div>
            <div>
              <h4 className="font-bold text-slate-900">
                Legutóbbi: {recentItems[1]?.vehicle_make || 'Ford'} {recentItems[1]?.vehicle_model || 'F-150'}
              </h4>
              <p className="text-sm text-slate-500 mt-1">
                <span className="font-mono text-[#2563eb] font-semibold">
                  {recentItems[1]?.dtc_codes?.[0] || 'P0303'}
                </span>{' '}
                - 3. henger gyújtáskimaradás észlelve
              </p>
              <button className="text-xs text-[#2563eb] font-bold mt-2 hover:underline">
                Jelentés megtekintése
              </button>
            </div>
          </div>

          {/* Segítség kártya */}
          <div className="p-5 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center">
            <div className="text-center">
              <HelpCircle className="h-8 w-8 text-[#2563eb] mx-auto mb-2" />
              <p className="text-sm font-bold text-slate-700">
                Segítségre van szüksége egy kóddal?
              </p>
              <button className="text-xs text-[#2563eb] font-bold mt-1 hover:underline">
                DTC adatbázis böngészése
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Lábléc */}
      <footer className="mt-auto py-8 text-center text-slate-500 text-sm font-medium border-t border-slate-200 bg-white">
        <p>© {currentYear} MechanicAI. Fejlett diagnosztikai algoritmusokkal működik.</p>
      </footer>
    </div>
  );
}
