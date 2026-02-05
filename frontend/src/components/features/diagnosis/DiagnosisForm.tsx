import { useState } from 'react';
import { AlertCircle, Car, Mic } from 'lucide-react';
import { Button, Input, Select, Textarea, type SelectOption } from '@/components/lib';
import { cn } from '@/lib/utils';

export interface DiagnosisFormData {
  dtcCodes: string[];
  vehicleMake: string;
  vehicleModel: string;
  vehicleYear: string;
  ownerComplaints: string;
  mechanicNotes: string;
}

export interface DiagnosisFormProps {
  /** Callback when form is submitted */
  onSubmit: (data: DiagnosisFormData) => void;
  /** Whether form is submitting */
  isSubmitting?: boolean;
  /** Initial values */
  initialValues?: Partial<DiagnosisFormData>;
  /** Manufacturer options */
  manufacturers?: SelectOption[];
  /** Additional className */
  className?: string;
}

const defaultManufacturers: SelectOption[] = [
  { value: '', label: 'Válasszon gyártót' },
  { value: 'audi', label: 'Audi' },
  { value: 'bmw', label: 'BMW' },
  { value: 'ford', label: 'Ford' },
  { value: 'honda', label: 'Honda' },
  { value: 'hyundai', label: 'Hyundai' },
  { value: 'kia', label: 'Kia' },
  { value: 'mazda', label: 'Mazda' },
  { value: 'mercedes', label: 'Mercedes-Benz' },
  { value: 'nissan', label: 'Nissan' },
  { value: 'opel', label: 'Opel' },
  { value: 'peugeot', label: 'Peugeot' },
  { value: 'renault', label: 'Renault' },
  { value: 'seat', label: 'Seat' },
  { value: 'skoda', label: 'Škoda' },
  { value: 'toyota', label: 'Toyota' },
  { value: 'volkswagen', label: 'Volkswagen' },
  { value: 'volvo', label: 'Volvo' },
];

// Generate year options (current year down to 1990)
const currentYear = new Date().getFullYear();
const yearOptions: SelectOption[] = [
  { value: '', label: 'Évjárat' },
  ...Array.from({ length: currentYear - 1990 + 1 }, (_, i) => ({
    value: String(currentYear - i),
    label: String(currentYear - i),
  })),
];

/**
 * Diagnosis form component (Step 1 of the wizard).
 *
 * @example
 * ```tsx
 * <DiagnosisForm
 *   onSubmit={(data) => startAnalysis(data)}
 *   isSubmitting={isAnalyzing}
 * />
 * ```
 */
export function DiagnosisForm({
  onSubmit,
  isSubmitting = false,
  initialValues,
  manufacturers = defaultManufacturers,
  className,
}: DiagnosisFormProps) {
  const [dtcInput, setDtcInput] = useState(
    initialValues?.dtcCodes?.join(', ') || ''
  );
  const [vehicleMake, setVehicleMake] = useState(
    initialValues?.vehicleMake || ''
  );
  const [vehicleModel, setVehicleModel] = useState(
    initialValues?.vehicleModel || ''
  );
  const [vehicleYear, setVehicleYear] = useState(
    initialValues?.vehicleYear || ''
  );
  const [ownerComplaints, setOwnerComplaints] = useState(
    initialValues?.ownerComplaints || ''
  );
  const [mechanicNotes, setMechanicNotes] = useState(
    initialValues?.mechanicNotes || ''
  );

  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    // Validate DTC codes
    const dtcCodes = dtcInput
      .split(/[,\s]+/)
      .map((code) => code.trim().toUpperCase())
      .filter((code) => code.length > 0);

    if (dtcCodes.length === 0) {
      newErrors.dtcCodes = 'Legalább egy hibakód megadása kötelező';
    } else {
      const dtcPattern = /^[PBCU][0-9A-F]{4}$/;
      const invalidCodes = dtcCodes.filter((code) => !dtcPattern.test(code));
      if (invalidCodes.length > 0) {
        newErrors.dtcCodes = `Érvénytelen hibakód(ok): ${invalidCodes.join(', ')}`;
      }
    }

    // Validate vehicle info
    if (!vehicleMake) {
      newErrors.vehicleMake = 'Gyártó megadása kötelező';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) return;

    const dtcCodes = dtcInput
      .split(/[,\s]+/)
      .map((code) => code.trim().toUpperCase())
      .filter((code) => code.length > 0);

    onSubmit({
      dtcCodes,
      vehicleMake,
      vehicleModel,
      vehicleYear,
      ownerComplaints,
      mechanicNotes,
    });
  };

  const handleDictation = () => {
    // Check for Web Speech API support
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      alert('A böngésző nem támogatja a beszédfelismerést');
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
      alert('Hiba történt a beszédfelismerés során');
    };

    recognition.start();
  };

  return (
    <form onSubmit={handleSubmit} className={cn('space-y-6', className)}>
      {/* Top Section: DTC Input + Vehicle Info */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: DTC Code Input */}
        <div className="bg-card border border-border rounded-lg p-4 md:p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-primary-600" />
            Hibakód megadása
          </h3>
          <Input
            label="DTC kód(ok)"
            value={dtcInput}
            onChange={(e) => setDtcInput(e.target.value.toUpperCase())}
            placeholder="Pl. P0171, P0300"
            error={errors.dtcCodes}
            hint="Több kódot vesszővel vagy szóközzel válasszon el"
            leftIcon={<AlertCircle className="h-4 w-4" />}
          />
        </div>

        {/* Right: Vehicle Identification */}
        <div className="bg-card border border-border rounded-lg p-4 md:p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Car className="h-5 w-5 text-primary-600" />
            Jármű azonosítás
          </h3>
          <div className="space-y-4">
            <Select
              label="Gyártó"
              options={manufacturers}
              value={vehicleMake}
              onChange={setVehicleMake}
              placeholder="Válasszon gyártót"
              error={errors.vehicleMake}
            />
            <Input
              label="Modell"
              value={vehicleModel}
              onChange={(e) => setVehicleModel(e.target.value)}
              placeholder="Pl. Golf, 320d, A4"
            />
            <Select
              label="Évjárat"
              options={yearOptions}
              value={vehicleYear}
              onChange={setVehicleYear}
              placeholder="Évjárat"
            />
          </div>
        </div>
      </div>

      {/* Bottom Section: Complaints + Notes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: Owner Complaints */}
        <div className="bg-card border border-border rounded-lg p-4 md:p-6">
          <Textarea
            label="Tulajdonos panaszai"
            value={ownerComplaints}
            onChange={(e) => setOwnerComplaints(e.target.value)}
            placeholder="Írja le a tulajdonos által tapasztalt problémákat..."
            showCharacterCount
            maxLength={1000}
            rightAction={
              <button
                type="button"
                onClick={handleDictation}
                className="p-1 rounded hover:bg-muted transition-colors"
                title="Diktálás"
              >
                <Mic className="h-5 w-5" />
              </button>
            }
          />
        </div>

        {/* Right: Mechanic Notes */}
        <div className="bg-card border border-border rounded-lg p-4 md:p-6">
          <Textarea
            label="Szerelői jegyzetek"
            value={mechanicNotes}
            onChange={(e) => setMechanicNotes(e.target.value)}
            placeholder="Saját megfigyelések, mérési eredmények..."
            showCharacterCount
            maxLength={1000}
          />
        </div>
      </div>

      {/* Submit Button */}
      <div className="flex justify-center pt-4">
        <Button
          type="submit"
          variant="primary"
          size="lg"
          isLoading={isSubmitting}
          className="min-w-[280px]"
        >
          AI MEGOLDÁS GENERÁLÁSA
        </Button>
      </div>
    </form>
  );
}

export default DiagnosisForm;
