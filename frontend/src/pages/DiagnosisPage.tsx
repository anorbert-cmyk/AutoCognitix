/**
 * DiagnosisPage - New diagnosis form page
 * Redesigned based on UI mockup
 */

import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  Car,
  Mic,
  QrCode,
  Clock,
  Save,
  Sparkles,
  User,
  Wrench,
  HelpCircle,
  ExternalLink,
} from 'lucide-react';
import { useAnalyzeDiagnosis, useDiagnosisHistory } from '../services/hooks';
import { Button, Input, Textarea } from '@/components/lib';
import { cn } from '@/lib/utils';
import { ApiError } from '../services/api';
import { useToast } from '../contexts/ToastContext';

// Generate year options (current year down to 1990)
const currentYear = new Date().getFullYear();
const yearOptions = [
  { value: '', label: 'Évjárat' },
  ...Array.from({ length: currentYear - 1990 + 1 }, (_, i) => ({
    value: String(currentYear - i),
    label: String(currentYear - i),
  })),
];

interface RecentDiagnosis {
  id: string;
  vehicleInfo: string;
  dtcCode: string;
  description: string;
}

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
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [lastSaved, setLastSaved] = useState<Date | null>(null);

  // API hooks
  const analyzeDiagnosis = useAnalyzeDiagnosis();
  const { data: historyData } = useDiagnosisHistory({ limit: 3 });

  // Transform history data to recent diagnoses
  const recentDiagnoses: RecentDiagnosis[] = historyData?.items?.map((item) => ({
    id: item.id,
    vehicleInfo: `${item.vehicle_make} ${item.vehicle_model}`,
    dtcCode: item.dtc_codes[0] || 'N/A',
    description: item.dtc_codes.length > 1
      ? `${item.dtc_codes[0]} - Több hibakód`
      : `${item.dtc_codes[0]} - Diagnosztika`,
  })) || [];

  // Auto-save simulation
  const handleAutoSave = useCallback(() => {
    setLastSaved(new Date());
  }, []);

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

  // Form validation
  const validateForm = useCallback((): boolean => {
    const newErrors: Record<string, string> = {};

    // Validate DTC code
    const dtcPattern = /^[PBCU][0-9A-F]{4}$/;
    const code = dtcCode.trim().toUpperCase();

    if (!code) {
      newErrors.dtcCode = 'Hibakód megadása kötelező';
    } else if (!dtcPattern.test(code)) {
      newErrors.dtcCode = 'Érvénytelen hibakód formátum (pl. P0300)';
    }

    // Validate vehicle info (at least make is required)
    if (!vehicleMake.trim()) {
      newErrors.vehicleMake = 'Gyártó megadása kötelező';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [dtcCode, vehicleMake]);

  // Form submission
  const handleSubmit = useCallback(async () => {
    if (!validateForm()) return;

    try {
      const result = await analyzeDiagnosis.mutateAsync({
        vehicleMake: vehicleMake.trim(),
        vehicleModel: vehicleModel.trim() || vehicleMake.trim(),
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
  }, [validateForm, analyzeDiagnosis, vehicleMake, vehicleModel, vehicleYear, dtcCode, ownerComplaints, mechanicNotes, toast, navigate]);

  // Save draft
  const handleSave = useCallback(() => {
    handleAutoSave();
    toast.success('Piszkozat mentve');
  }, [handleAutoSave, toast]);

  // Format time ago
  const formatTimeAgo = (date: Date | null): string => {
    if (!date) return '';
    const minutes = Math.floor((Date.now() - date.getTime()) / 60000);
    if (minutes < 1) return 'most';
    if (minutes === 1) return '1 perce';
    return `${minutes} perce`;
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <div className="mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-primary-50 text-primary-700 rounded-full text-sm font-medium mb-4">
            <AlertTriangle className="h-4 w-4" />
            AI DIAGNOSZTIKAI ESZKÖZ
          </div>
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Új diagnosztikai folyamat
          </h1>
          <p className="text-muted-foreground">
            Adja meg a jármű adatait és az OBD kódokat. Az AI modell elemzi az adatokat a
            lehetséges okok és javítási eljárások javaslatához.
          </p>
        </div>

        {/* Main Form Card */}
        <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden">
          {/* Wizard Steps */}
          <div className="border-b border-border px-6 py-4">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center w-7 h-7 rounded-full bg-primary-600 text-white text-sm font-medium">
                  1
                </div>
                <span className="text-sm font-medium text-primary-600">Adatbevitel</span>
              </div>
              <div className="flex items-center gap-2 opacity-50">
                <div className="flex items-center justify-center w-7 h-7 rounded-full border-2 border-muted-foreground text-muted-foreground text-sm font-medium">
                  2
                </div>
                <span className="text-sm font-medium text-muted-foreground">Elemzés</span>
              </div>
              <div className="flex items-center gap-2 opacity-50">
                <div className="flex items-center justify-center w-7 h-7 rounded-full border-2 border-muted-foreground text-muted-foreground text-sm font-medium">
                  3
                </div>
                <span className="text-sm font-medium text-muted-foreground">Jelentés</span>
              </div>
            </div>
          </div>

          <div className="p-6 space-y-8">
            {/* DTC Code Section */}
            <div>
              <label className="block text-sm font-semibold text-foreground uppercase tracking-wide mb-3">
                Elsődleges hibakód (DTC)
              </label>
              <div className="relative">
                <Input
                  value={dtcCode}
                  onChange={(e) => setDtcCode(e.target.value.toUpperCase())}
                  placeholder="PL. P0300"
                  error={errors.dtcCode}
                  className="text-2xl font-light tracking-wide text-primary-500 placeholder:text-primary-300 h-16 pr-12"
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-muted-foreground hover:text-foreground transition-colors"
                  title="QR kód beolvasása"
                >
                  <QrCode className="h-6 w-6" />
                </button>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Adja meg a fő kódot az OBD-II olvasóból.
              </p>
            </div>

            <hr className="border-border" />

            {/* Vehicle Identification */}
            <div>
              <h3 className="flex items-center gap-2 text-base font-semibold text-foreground mb-4">
                <Car className="h-5 w-5 text-primary-600" />
                Jármű azonosítás
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1.5">
                    Gyártó
                  </label>
                  <Input
                    value={vehicleMake}
                    onChange={(e) => setVehicleMake(e.target.value)}
                    placeholder="Toyota"
                    error={errors.vehicleMake}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1.5">
                    Modell
                  </label>
                  <Input
                    value={vehicleModel}
                    onChange={(e) => setVehicleModel(e.target.value)}
                    placeholder="Camry"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-1.5">
                    Évjárat
                  </label>
                  <select
                    value={vehicleYear}
                    onChange={(e) => setVehicleYear(e.target.value)}
                    className={cn(
                      'w-full h-10 px-3 text-sm bg-background border border-input rounded-lg',
                      'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
                      'transition-colors'
                    )}
                  >
                    {yearOptions.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Complaints and Notes */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Owner Complaints */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <User className="h-4 w-4 text-muted-foreground" />
                    Tulajdonos panaszai
                  </label>
                  <button
                    type="button"
                    onClick={handleDictation}
                    className="flex items-center gap-1.5 text-sm font-medium text-primary-600 hover:text-primary-700 transition-colors"
                  >
                    <Mic className="h-4 w-4" />
                    DIKTÁLÁS
                  </button>
                </div>
                <Textarea
                  value={ownerComplaints}
                  onChange={(e) => setOwnerComplaints(e.target.value)}
                  placeholder="pl. Az ügyfél egyenetlen alapjáratot panaszol reggelente, a motorhiba-jelző lámpa villog autópályán történő gyorsításkor..."
                  rows={5}
                  className="resize-none"
                />
              </div>

              {/* Mechanic Notes */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <Wrench className="h-4 w-4 text-muted-foreground" />
                    Szerelői jegyzetek / Tesztút
                  </label>
                  <span className="text-xs text-muted-foreground uppercase tracking-wide">
                    Privát jegyzetek
                  </span>
                </div>
                <Textarea
                  value={mechanicNotes}
                  onChange={(e) => setMechanicNotes(e.target.value)}
                  placeholder="pl. Terheléses teszt során a 3. henger gyújtáskimaradása megerősítve. A gyújtógyertyák kopottak, de a tekercsek jónak tűnnek..."
                  rows={5}
                  className="resize-none"
                />
              </div>
            </div>

            {/* Action Bar */}
            <div className="flex items-center justify-between pt-4 border-t border-border">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Clock className="h-4 w-4" />
                {lastSaved
                  ? `Automatikusan mentve ${formatTimeAgo(lastSaved)}`
                  : 'Automatikus mentés aktív'
                }
              </div>
              <div className="flex items-center gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleSave}
                  leftIcon={<Save className="h-4 w-4" />}
                >
                  MENTÉS
                </Button>
                <Button
                  type="button"
                  variant="primary"
                  onClick={handleSubmit}
                  isLoading={analyzeDiagnosis.isPending}
                  leftIcon={<Sparkles className="h-4 w-4" />}
                  className="min-w-[200px]"
                >
                  AI MEGOLDÁS GENERÁLÁSA
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Diagnoses Cards */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          {recentDiagnoses.slice(0, 2).map((diagnosis) => (
            <div
              key={diagnosis.id}
              className="bg-card border border-border rounded-lg p-4 hover:border-primary-300 transition-colors cursor-pointer"
              onClick={() => navigate(`/diagnosis/${diagnosis.id}`)}
            >
              <div className="flex items-start gap-3">
                <div className="p-2 bg-muted rounded-lg">
                  <Clock className="h-5 w-5 text-muted-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-foreground truncate">
                    Legutóbbi: {diagnosis.vehicleInfo}
                  </p>
                  <p className="text-sm text-muted-foreground truncate">
                    {diagnosis.description}
                  </p>
                  <button className="mt-2 text-sm font-medium text-primary-600 hover:text-primary-700 flex items-center gap-1">
                    Jelentés megtekintése
                    <ExternalLink className="h-3 w-3" />
                  </button>
                </div>
              </div>
            </div>
          ))}

          {/* Help Card */}
          <div className="bg-card border border-border rounded-lg p-4 flex flex-col items-center justify-center text-center">
            <div className="p-3 bg-primary-50 rounded-full mb-3">
              <HelpCircle className="h-6 w-6 text-primary-600" />
            </div>
            <p className="text-sm text-muted-foreground">
              Segítségre van szüksége egy kóddal?
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-muted-foreground">
          © {new Date().getFullYear()} MechanicAI. Fejlett diagnosztikai algoritmusokkal támogatva.
        </div>
      </div>
    </div>
  );
}
