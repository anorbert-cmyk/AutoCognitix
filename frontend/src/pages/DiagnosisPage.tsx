/**
 * DiagnosisPage - Exact match to provided HTML design
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
} from 'lucide-react';
import { useAnalyzeDiagnosis, useDiagnosisHistory } from '../services/hooks';
import { cn } from '@/lib/utils';
import { ApiError } from '../services/api';
import { useToast } from '../contexts/ToastContext';

const currentYear = new Date().getFullYear();

export default function DiagnosisPage() {
  const navigate = useNavigate();
  const toast = useToast();

  // Form state
  const [dtcCode, setDtcCode] = useState('');
  const [vehicleMake, setVehicleMake] = useState('');
  const [vehicleModel, setVehicleModel] = useState('');
  const [vehicleYear, setVehicleYear] = useState('');
  const [ownerComplaints, setOwnerComplaints] = useState('');
  const [mechanicNotes, setMechanicNotes] = useState('');

  // API hooks
  const analyzeDiagnosis = useAnalyzeDiagnosis();
  const { data: historyData } = useDiagnosisHistory({ limit: 2 });

  // Dictation handler
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

  // Form submission
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!dtcCode.trim()) {
      toast.error('Hibakód megadása kötelező');
      return;
    }
    try {
      const result = await analyzeDiagnosis.mutateAsync({
        vehicleMake: vehicleMake.trim() || 'Ismeretlen',
        vehicleModel: vehicleModel.trim() || 'Ismeretlen',
        vehicleYear: vehicleYear ? parseInt(vehicleYear) : currentYear,
        dtcCodes: [dtcCode.trim().toUpperCase()],
        symptoms: ownerComplaints.trim() || 'Nincs megadva',
        additionalContext: mechanicNotes.trim() || undefined,
      });
      toast.success('Diagnózis sikeresen elkészült!');
      navigate(`/diagnosis/${result.id}`);
    } catch (err) {
      if (err instanceof ApiError) {
        toast.error(err.detail);
      } else {
        toast.error('Ismeretlen hiba történt');
      }
    }
  }, [analyzeDiagnosis, vehicleMake, vehicleModel, vehicleYear, dtcCode, ownerComplaints, mechanicNotes, toast, navigate]);

  const recentItems = historyData?.items || [];

  return (
    <div className="min-h-screen bg-[#F2F4F7] flex flex-col">
      {/* Main Content */}
      <main className="flex-1 w-full max-w-5xl mx-auto p-4 md:p-8 lg:p-12">
        {/* Header Section */}
        <div className="mb-10 text-center md:text-left">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded border border-gray-300 bg-white text-gray-700 text-xs font-bold uppercase tracking-wider mb-4">
            <AlertTriangle className="h-3.5 w-3.5" />
            AI Diagnosztikai Eszköz
          </div>
          <h2 className="text-4xl font-black tracking-tight text-[#1A1A1A] mb-3">
            Új diagnosztikai folyamat
          </h2>
          <p className="text-lg font-medium text-gray-600 max-w-3xl">
            Adja meg a jármű adatait és az OBD kódokat. Az AI modell elemzi az adatokat a lehetséges okok és javítási eljárások javaslatához.
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded shadow-sm border border-gray-300 overflow-hidden">
          {/* Wizard Steps */}
          <div className="flex border-b border-gray-200 bg-gray-50 px-6 py-4 gap-8 overflow-x-auto">
            <div className="flex items-center gap-2 text-[#0052CC] font-bold text-sm whitespace-nowrap">
              <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[#0052CC] text-white text-xs">
                1
              </span>
              Adatbevitel
            </div>
            <div className="flex items-center gap-2 text-gray-400 font-semibold text-sm whitespace-nowrap">
              <span className="flex items-center justify-center w-6 h-6 rounded-full border-2 border-gray-300 text-xs">
                2
              </span>
              Elemzés
            </div>
            <div className="flex items-center gap-2 text-gray-400 font-semibold text-sm whitespace-nowrap">
              <span className="flex items-center justify-center w-6 h-6 rounded-full border-2 border-gray-300 text-xs">
                3
              </span>
              Jelentés
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="p-6 md:p-8 lg:p-10 space-y-10">
            {/* DTC Section */}
            <section>
              <label className="block text-sm font-bold uppercase tracking-wider text-gray-700 mb-3">
                Elsődleges hibakód (DTC)
              </label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <AlertTriangle className="h-6 w-6 text-gray-400 group-focus-within:text-[#0052CC] transition-colors" />
                </div>
                <input
                  type="text"
                  value={dtcCode}
                  onChange={(e) => setDtcCode(e.target.value.toUpperCase())}
                  placeholder="pl. P0300"
                  className="block w-full pl-14 pr-14 py-4 bg-white border-2 border-gray-300 rounded text-3xl font-bold text-[#1A1A1A] placeholder:text-gray-300 focus:ring-0 focus:border-[#0052CC] transition-colors uppercase tracking-wide"
                />
                <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                  <button
                    type="button"
                    className="p-2 text-gray-400 hover:text-[#0052CC] transition-colors"
                    title="Szkennelés kamerával"
                  >
                    <QrCode className="h-6 w-6" />
                  </button>
                </div>
              </div>
              <p className="mt-2 text-sm font-medium text-gray-500">
                Adja meg a fő kódot az OBD-II olvasóból.
              </p>
            </section>

            <hr className="border-gray-200" />

            {/* Vehicle Section */}
            <section>
              <h3 className="text-xl font-bold text-[#1A1A1A] mb-6 flex items-center gap-2">
                <Car className="h-5 w-5 text-[#0052CC]" />
                Jármű azonosítás
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <label className="block text-sm font-bold text-gray-700">Gyártó</label>
                  <input
                    type="text"
                    value={vehicleMake}
                    onChange={(e) => setVehicleMake(e.target.value)}
                    placeholder="Toyota"
                    className="block w-full rounded border-gray-300 bg-white text-[#1A1A1A] focus:border-[#0052CC] focus:ring-1 focus:ring-[#0052CC] h-12 font-medium"
                  />
                </div>
                <div className="space-y-2">
                  <label className="block text-sm font-bold text-gray-700">Modell</label>
                  <input
                    type="text"
                    value={vehicleModel}
                    onChange={(e) => setVehicleModel(e.target.value)}
                    placeholder="Camry"
                    className="block w-full rounded border-gray-300 bg-white text-[#1A1A1A] focus:border-[#0052CC] focus:ring-1 focus:ring-[#0052CC] h-12 font-medium"
                  />
                </div>
                <div className="space-y-2">
                  <label className="block text-sm font-bold text-gray-700">Évjárat</label>
                  <input
                    type="number"
                    value={vehicleYear}
                    onChange={(e) => setVehicleYear(e.target.value)}
                    placeholder="2019"
                    min="1990"
                    max="2025"
                    className="block w-full rounded border-gray-300 bg-white text-[#1A1A1A] focus:border-[#0052CC] focus:ring-1 focus:ring-[#0052CC] h-12 font-medium"
                  />
                </div>
              </div>
            </section>

            {/* Textareas Section */}
            <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Owner Complaints */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-bold text-[#1A1A1A] flex items-center gap-2">
                    <User className="h-4 w-4 text-gray-500" />
                    Tulajdonos panaszai
                  </label>
                  <button
                    type="button"
                    onClick={handleDictation}
                    className="text-xs text-[#0052CC] font-bold hover:underline flex items-center gap-1 uppercase tracking-wide"
                  >
                    <Mic className="h-3.5 w-3.5" />
                    Diktálás
                  </button>
                </div>
                <textarea
                  value={ownerComplaints}
                  onChange={(e) => setOwnerComplaints(e.target.value)}
                  placeholder="pl. Az ügyfél egyenetlen alapjáratot panaszol reggelente, a motorhiba-jelző lámpa villog autópályán történő gyorsításkor..."
                  rows={5}
                  className="block w-full rounded border-gray-300 bg-gray-50 text-[#1A1A1A] placeholder:text-gray-400 focus:bg-white focus:border-[#0052CC] focus:ring-1 focus:ring-[#0052CC] resize-none p-4 font-medium"
                />
              </div>

              {/* Mechanic Notes */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-bold text-[#1A1A1A] flex items-center gap-2">
                    <Wrench className="h-4 w-4 text-gray-500" />
                    Szerelői jegyzetek / Tesztút
                  </label>
                  <span className="text-xs font-bold text-gray-400 uppercase tracking-wide">
                    Privát jegyzetek
                  </span>
                </div>
                <textarea
                  value={mechanicNotes}
                  onChange={(e) => setMechanicNotes(e.target.value)}
                  placeholder="pl. Terheléses teszt során a 3. henger gyújtáskimaradása megerősítve. A gyújtógyertyák kopottak, de a tekercsek jónak tűnnek..."
                  rows={5}
                  className="block w-full rounded border-gray-300 bg-gray-50 text-[#1A1A1A] placeholder:text-gray-400 focus:bg-white focus:border-[#0052CC] focus:ring-1 focus:ring-[#0052CC] resize-none p-4 font-medium"
                />
              </div>
            </section>

            {/* Action Bar */}
            <div className="pt-8 flex flex-col md:flex-row items-center justify-between gap-6 border-t border-gray-200">
              <div className="flex items-center gap-2 text-gray-500 text-sm font-medium">
                <Clock className="h-4 w-4" />
                <span>Automatikusan mentve 2 perce</span>
              </div>
              <div className="flex w-full md:w-auto gap-4">
                <button
                  type="button"
                  className="flex-1 md:flex-none h-14 px-8 rounded border-2 border-gray-300 text-gray-700 font-bold hover:bg-gray-50 transition-colors uppercase tracking-wide text-sm"
                >
                  Mentés
                </button>
                <button
                  type="submit"
                  disabled={analyzeDiagnosis.isPending}
                  className={cn(
                    "flex-1 md:flex-none h-14 px-8 rounded bg-[#0052CC] hover:bg-[#0041a3] text-white font-bold transition-colors flex items-center justify-center gap-2 shadow-sm uppercase tracking-wide text-sm",
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

        {/* Bottom Cards */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Recent 1 */}
          <div
            className="p-4 rounded bg-white border border-gray-300 flex items-start gap-3 shadow-sm cursor-pointer hover:border-[#0052CC] transition-colors"
            onClick={() => recentItems[0] && navigate(`/diagnosis/${recentItems[0].id}`)}
          >
            <div className="mt-1 bg-gray-100 text-gray-600 p-2 rounded">
              <Clock className="h-5 w-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-[#1A1A1A]">
                Legutóbbi: {recentItems[0]?.vehicle_make || 'Toyota'} {recentItems[0]?.vehicle_model || 'Camry'}
              </h4>
              <p className="text-xs font-medium text-gray-500 mt-1">
                {recentItems[0]?.dtc_codes?.[0] || 'P0420'} - Katalizátor rendszer hatékonysága határérték alatt
              </p>
              <button className="text-xs text-[#0052CC] font-bold mt-2 inline-block hover:underline">
                Jelentés megtekintése
              </button>
            </div>
          </div>

          {/* Recent 2 */}
          <div
            className="p-4 rounded bg-white border border-gray-300 flex items-start gap-3 shadow-sm cursor-pointer hover:border-[#0052CC] transition-colors"
            onClick={() => recentItems[1] && navigate(`/diagnosis/${recentItems[1].id}`)}
          >
            <div className="mt-1 bg-gray-100 text-gray-600 p-2 rounded">
              <Clock className="h-5 w-5" />
            </div>
            <div>
              <h4 className="font-bold text-sm text-[#1A1A1A]">
                Legutóbbi: {recentItems[1]?.vehicle_make || 'Ford'} {recentItems[1]?.vehicle_model || 'F-150'}
              </h4>
              <p className="text-xs font-medium text-gray-500 mt-1">
                {recentItems[1]?.dtc_codes?.[0] || 'P0303'} - 3. henger gyújtáskimaradás érzékelve
              </p>
              <button className="text-xs text-[#0052CC] font-bold mt-2 inline-block hover:underline">
                Jelentés megtekintése
              </button>
            </div>
          </div>

          {/* Help Card */}
          <div className="p-4 rounded bg-[#0052CC]/5 border border-[#0052CC]/20 flex items-center justify-center gap-3">
            <div className="text-center">
              <HelpCircle className="h-8 w-8 text-[#0052CC] mx-auto mb-1" />
              <p className="text-sm font-bold text-gray-700">
                Segítségre van szüksége egy kóddal?
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-auto py-8 text-center text-gray-500 text-sm font-medium border-t border-gray-200 bg-white">
        <p>© 2023 MechanicAI. Fejlett diagnosztikai algoritmusokkal támogatva.</p>
      </footer>
    </div>
  );
}
