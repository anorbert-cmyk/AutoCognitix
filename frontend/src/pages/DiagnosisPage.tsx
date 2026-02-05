/**
 * DiagnosisPage - Exact match to UI mockup
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

// Generate year options
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
  const [lastSaved] = useState<Date>(new Date(Date.now() - 2 * 60 * 1000)); // 2 minutes ago

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
  const handleSubmit = useCallback(async () => {
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

  // Recent diagnoses
  const recentItems = historyData?.items || [];

  return (
    <div className="min-h-screen bg-[#f8fafc]">
      <div className="max-w-[900px] mx-auto px-6 py-10">
        {/* Header Badge */}
        <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-[#eff6ff] text-[#1d4ed8] rounded-md text-xs font-medium tracking-wide mb-5">
          <AlertTriangle className="h-3.5 w-3.5" />
          AI DIAGNOSZTIKAI ESZKÖZ
        </div>

        {/* Page Title */}
        <h1 className="text-[32px] font-bold text-[#0f172a] mb-2">
          Új diagnosztikai folyamat
        </h1>
        <p className="text-[#64748b] text-base mb-8 max-w-[600px]">
          Adja meg a jármű adatait és az OBD kódokat. Az AI modell elemzi az adatokat a
          lehetséges okok és javítási eljárások javaslatához.
        </p>

        {/* Main Card */}
        <div className="bg-white rounded-xl border border-[#e2e8f0] shadow-sm">
          {/* Wizard Steps */}
          <div className="px-8 py-5 border-b border-[#e2e8f0]">
            <div className="flex items-center gap-10">
              {/* Step 1 - Active */}
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-full bg-[#2563eb] text-white text-sm font-medium flex items-center justify-center">
                  1
                </div>
                <span className="text-sm font-medium text-[#2563eb]">Adatbevitel</span>
              </div>
              {/* Step 2 - Inactive */}
              <div className="flex items-center gap-2.5 opacity-40">
                <div className="w-7 h-7 rounded-full border-2 border-[#94a3b8] text-[#94a3b8] text-sm font-medium flex items-center justify-center">
                  2
                </div>
                <span className="text-sm font-medium text-[#94a3b8]">Elemzés</span>
              </div>
              {/* Step 3 - Inactive */}
              <div className="flex items-center gap-2.5 opacity-40">
                <div className="w-7 h-7 rounded-full border-2 border-[#94a3b8] text-[#94a3b8] text-sm font-medium flex items-center justify-center">
                  3
                </div>
                <span className="text-sm font-medium text-[#94a3b8]">Jelentés</span>
              </div>
            </div>
          </div>

          {/* Form Content */}
          <div className="px-8 py-6">
            {/* DTC Section */}
            <div className="mb-8">
              <label className="block text-xs font-semibold text-[#0f172a] uppercase tracking-wider mb-3">
                ELSŐDLEGES HIBAKÓD (DTC)
              </label>
              <div className="relative">
                <div className="flex items-center border border-[#e2e8f0] rounded-lg px-4 h-[60px] bg-white focus-within:ring-2 focus-within:ring-[#2563eb] focus-within:border-[#2563eb]">
                  <AlertTriangle className="h-5 w-5 text-[#f59e0b] mr-3 flex-shrink-0" />
                  <input
                    type="text"
                    value={dtcCode}
                    onChange={(e) => setDtcCode(e.target.value.toUpperCase())}
                    placeholder="PL. P0300"
                    className="flex-1 text-2xl font-light text-[#f59e0b] placeholder:text-[#fcd34d] bg-transparent outline-none tracking-wide"
                  />
                  <button
                    type="button"
                    className="p-2 text-[#94a3b8] hover:text-[#64748b] transition-colors"
                  >
                    <QrCode className="h-6 w-6" />
                  </button>
                </div>
              </div>
              <p className="mt-2.5 text-sm text-[#94a3b8]">
                Adja meg a fő kódot az OBD-II olvasóból.
              </p>
            </div>

            {/* Divider */}
            <hr className="border-[#e2e8f0] mb-8" />

            {/* Vehicle Section */}
            <div className="mb-8">
              <h3 className="flex items-center gap-2 text-base font-semibold text-[#0f172a] mb-5">
                <Car className="h-5 w-5 text-[#2563eb]" />
                Jármű azonosítás
              </h3>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm text-[#64748b] mb-2">Gyártó</label>
                  <input
                    type="text"
                    value={vehicleMake}
                    onChange={(e) => setVehicleMake(e.target.value)}
                    placeholder="Toyota"
                    className="w-full h-11 px-4 text-sm text-[#0f172a] placeholder:text-[#94a3b8] bg-white border border-[#e2e8f0] rounded-lg outline-none focus:ring-2 focus:ring-[#2563eb] focus:border-[#2563eb] transition-all"
                  />
                </div>
                <div>
                  <label className="block text-sm text-[#64748b] mb-2">Modell</label>
                  <input
                    type="text"
                    value={vehicleModel}
                    onChange={(e) => setVehicleModel(e.target.value)}
                    placeholder="Camry"
                    className="w-full h-11 px-4 text-sm text-[#0f172a] placeholder:text-[#94a3b8] bg-white border border-[#e2e8f0] rounded-lg outline-none focus:ring-2 focus:ring-[#2563eb] focus:border-[#2563eb] transition-all"
                  />
                </div>
                <div>
                  <label className="block text-sm text-[#64748b] mb-2">Évjárat</label>
                  <input
                    type="text"
                    value={vehicleYear}
                    onChange={(e) => setVehicleYear(e.target.value)}
                    placeholder="2019"
                    className="w-full h-11 px-4 text-sm text-[#0f172a] placeholder:text-[#94a3b8] bg-white border border-[#e2e8f0] rounded-lg outline-none focus:ring-2 focus:ring-[#2563eb] focus:border-[#2563eb] transition-all"
                  />
                </div>
              </div>
            </div>

            {/* Complaints and Notes - Side by Side */}
            <div className="grid grid-cols-2 gap-6 mb-8">
              {/* Owner Complaints */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="flex items-center gap-2 text-sm font-medium text-[#0f172a]">
                    <User className="h-4 w-4 text-[#64748b]" />
                    Tulajdonos panaszai
                  </label>
                  <button
                    type="button"
                    onClick={handleDictation}
                    className="flex items-center gap-1.5 text-xs font-semibold text-[#2563eb] hover:text-[#1d4ed8] transition-colors"
                  >
                    <Mic className="h-3.5 w-3.5" />
                    DIKTÁLÁS
                  </button>
                </div>
                <textarea
                  value={ownerComplaints}
                  onChange={(e) => setOwnerComplaints(e.target.value)}
                  placeholder="pl. Az ügyfél egyenetlen alapjáratot panaszol reggelente, a motorhiba-jelző lámpa villog autópályán történő gyorsításkor..."
                  rows={5}
                  className="w-full px-4 py-3 text-sm text-[#0f172a] placeholder:text-[#94a3b8] bg-white border border-[#e2e8f0] rounded-lg outline-none focus:ring-2 focus:ring-[#2563eb] focus:border-[#2563eb] resize-none transition-all"
                />
              </div>

              {/* Mechanic Notes */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="flex items-center gap-2 text-sm font-medium text-[#0f172a]">
                    <Wrench className="h-4 w-4 text-[#64748b]" />
                    Szerelői jegyzetek / Tesztút
                  </label>
                  <span className="text-[10px] font-medium text-[#94a3b8] uppercase tracking-wider">
                    PRIVÁT JEGYZETEK
                  </span>
                </div>
                <textarea
                  value={mechanicNotes}
                  onChange={(e) => setMechanicNotes(e.target.value)}
                  placeholder="pl. Terheléses teszt során a 3. henger gyújtáskimaradása megerősítve. A gyújtógyertyák kopottak, de a tekercsek jónak tűnnek..."
                  rows={5}
                  className="w-full px-4 py-3 text-sm text-[#0f172a] placeholder:text-[#94a3b8] bg-white border border-[#e2e8f0] rounded-lg outline-none focus:ring-2 focus:ring-[#2563eb] focus:border-[#2563eb] resize-none transition-all"
                />
              </div>
            </div>

            {/* Action Bar */}
            <div className="flex items-center justify-between pt-6 border-t border-[#e2e8f0]">
              <div className="flex items-center gap-2 text-sm text-[#94a3b8]">
                <Clock className="h-4 w-4" />
                <span>Automatikusan mentve {Math.floor((Date.now() - lastSaved.getTime()) / 60000)} perce</span>
              </div>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  className="h-11 px-6 text-sm font-medium text-[#0f172a] bg-white border border-[#e2e8f0] rounded-lg hover:bg-[#f8fafc] transition-colors"
                >
                  MENTÉS
                </button>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={analyzeDiagnosis.isPending}
                  className={cn(
                    "h-11 px-6 text-sm font-medium text-white bg-[#2563eb] rounded-lg hover:bg-[#1d4ed8] transition-colors flex items-center gap-2",
                    analyzeDiagnosis.isPending && "opacity-70 cursor-not-allowed"
                  )}
                >
                  <Sparkles className="h-4 w-4" />
                  AI MEGOLDÁS GENERÁLÁSA
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Cards */}
        <div className="grid grid-cols-3 gap-4 mt-6">
          {/* Recent 1 */}
          {recentItems[0] ? (
            <div
              className="bg-white border border-[#e2e8f0] rounded-lg p-5 cursor-pointer hover:border-[#2563eb] transition-colors"
              onClick={() => navigate(`/diagnosis/${recentItems[0].id}`)}
            >
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-[#f1f5f9] rounded-lg flex items-center justify-center flex-shrink-0">
                  <Clock className="h-5 w-5 text-[#64748b]" />
                </div>
                <div className="min-w-0">
                  <p className="font-medium text-[#0f172a] text-sm">
                    Legutóbbi: {recentItems[0].vehicle_make} {recentItems[0].vehicle_model}
                  </p>
                  <p className="text-xs text-[#64748b] mt-0.5">
                    {recentItems[0].dtc_codes[0]} - Katalizátor rendszer hatékonysága határérték alatt
                  </p>
                  <button className="mt-2.5 text-xs font-medium text-[#2563eb] hover:text-[#1d4ed8]">
                    Jelentés megtekintése
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white border border-[#e2e8f0] rounded-lg p-5">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-[#f1f5f9] rounded-lg flex items-center justify-center flex-shrink-0">
                  <Clock className="h-5 w-5 text-[#64748b]" />
                </div>
                <div className="min-w-0">
                  <p className="font-medium text-[#0f172a] text-sm">Legutóbbi: Toyota Camry</p>
                  <p className="text-xs text-[#64748b] mt-0.5">P0420 - Katalizátor rendszer hatékonysága határérték alatt</p>
                  <button className="mt-2.5 text-xs font-medium text-[#2563eb] hover:text-[#1d4ed8]">
                    Jelentés megtekintése
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Recent 2 */}
          {recentItems[1] ? (
            <div
              className="bg-white border border-[#e2e8f0] rounded-lg p-5 cursor-pointer hover:border-[#2563eb] transition-colors"
              onClick={() => navigate(`/diagnosis/${recentItems[1].id}`)}
            >
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-[#f1f5f9] rounded-lg flex items-center justify-center flex-shrink-0">
                  <Clock className="h-5 w-5 text-[#64748b]" />
                </div>
                <div className="min-w-0">
                  <p className="font-medium text-[#0f172a] text-sm">
                    Legutóbbi: {recentItems[1].vehicle_make} {recentItems[1].vehicle_model}
                  </p>
                  <p className="text-xs text-[#64748b] mt-0.5">
                    {recentItems[1].dtc_codes[0]} - 3. henger gyújtáskimaradás érzékelve
                  </p>
                  <button className="mt-2.5 text-xs font-medium text-[#2563eb] hover:text-[#1d4ed8]">
                    Jelentés megtekintése
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white border border-[#e2e8f0] rounded-lg p-5">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-[#f1f5f9] rounded-lg flex items-center justify-center flex-shrink-0">
                  <Clock className="h-5 w-5 text-[#64748b]" />
                </div>
                <div className="min-w-0">
                  <p className="font-medium text-[#0f172a] text-sm">Legutóbbi: Ford F-150</p>
                  <p className="text-xs text-[#64748b] mt-0.5">P0303 - 3. henger gyújtáskimaradás érzékelve</p>
                  <button className="mt-2.5 text-xs font-medium text-[#2563eb] hover:text-[#1d4ed8]">
                    Jelentés megtekintése
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Help Card */}
          <div className="bg-white border border-[#e2e8f0] rounded-lg p-5 flex flex-col items-center justify-center text-center">
            <div className="w-12 h-12 bg-[#eff6ff] rounded-full flex items-center justify-center mb-3">
              <HelpCircle className="h-6 w-6 text-[#2563eb]" />
            </div>
            <p className="text-sm text-[#64748b]">
              Segítségre van szüksége egy kóddal?
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-10 text-center text-sm text-[#94a3b8]">
          © 2023 MechanicAI. Fejlett diagnosztikai algoritmusokkal támogatva.
        </div>
      </div>
    </div>
  );
}
