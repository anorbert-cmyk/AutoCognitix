/**
 * CalculatorPage — "Megeri megjavitani?" (Worth Repairing?) Calculator
 *
 * Full calculator page with form, recommendation, value comparison,
 * cost breakdown, factors, and alternative scenarios.
 */

import { useState, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Calculator,
  Car,
  Wrench,
  ChevronRight,
  Plus,
  Minus,
  Shield,
  Fuel,
} from 'lucide-react';

import { PageContainer } from '@/components/layouts';
import { Button, Input, Badge } from '@/components/lib';
import { RecommendationCard } from '@/components/features/calculator/RecommendationCard';
import { ValueComparison } from '@/components/features/calculator/ValueComparison';
import { useEvaluateCalculator } from '@/services/hooks/useCalculator';
import type { CalculatorRequest, CalculatorResponse } from '@/services/calculatorService';

// =============================================================================
// Constants
// =============================================================================

const manufacturers = [
  { value: '', label: 'Valasszon gyartot' },
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
  { value: 'skoda', label: 'Skoda' },
  { value: 'toyota', label: 'Toyota' },
  { value: 'volkswagen', label: 'Volkswagen' },
  { value: 'volvo', label: 'Volvo' },
];

const conditionOptions = [
  { value: 'excellent', label: 'Kivalo', description: 'Szinte uj allapot, minimalis kopas' },
  { value: 'good', label: 'Jo', description: 'Normalis hasznalati nyomok, megbizhato' },
  { value: 'fair', label: 'Elfogadhato', description: 'Latszanak a hasznalat jelei, mukodokepesen' },
  { value: 'poor', label: 'Rossz', description: 'Jelentos kopas, tobb javitas szukseges' },
] as const;

const fuelTypeOptions = [
  { value: '', label: 'Valasszon uzemanyagot' },
  { value: 'petrol', label: 'Benzin' },
  { value: 'diesel', label: 'Dizel' },
  { value: 'hybrid', label: 'Hibrid' },
  { value: 'electric', label: 'Elektromos' },
  { value: 'lpg', label: 'LPG' },
];

// =============================================================================
// HUF Formatter
// =============================================================================

function formatHUF(amount: number, currency: string = 'HUF'): string {
  return new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(amount);
}

// =============================================================================
// Sub-Components
// =============================================================================

function VehicleValueCard({ result }: { result: CalculatorResponse }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-sm">
      <div className="flex items-center gap-2 mb-4">
        <Car className="h-5 w-5 text-blue-500" />
        <h3 className="text-base font-bold text-slate-900">Jarmu becsult erteke</h3>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center">
          <p className="text-xs text-slate-500 mb-1">Minimum</p>
          <p className="text-lg font-bold text-slate-700 tabular-nums">
            {formatHUF(result.vehicle_value_min, result.currency)}
          </p>
        </div>
        <div className="text-center border-x border-slate-100">
          <p className="text-xs text-slate-500 mb-1">Atlagos</p>
          <p className="text-xl font-black text-blue-600 tabular-nums">
            {formatHUF(result.vehicle_value_avg, result.currency)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-slate-500 mb-1">Maximum</p>
          <p className="text-lg font-bold text-slate-700 tabular-nums">
            {formatHUF(result.vehicle_value_max, result.currency)}
          </p>
        </div>
      </div>
    </div>
  );
}

function RepairCostCard({ result }: { result: CalculatorResponse }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-sm">
      <div className="flex items-center gap-2 mb-4">
        <Wrench className="h-5 w-5 text-orange-500" />
        <h3 className="text-base font-bold text-slate-900">Becsult javitasi koltseg</h3>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <p className="text-xs text-slate-500 mb-1">Minimum</p>
          <p className="text-lg font-bold text-orange-600 tabular-nums">
            {formatHUF(result.repair_cost_min, result.currency)}
          </p>
        </div>
        <div className="text-center border-l border-slate-100">
          <p className="text-xs text-slate-500 mb-1">Maximum</p>
          <p className="text-lg font-bold text-orange-600 tabular-nums">
            {formatHUF(result.repair_cost_max, result.currency)}
          </p>
        </div>
      </div>
    </div>
  );
}

function FactorsList({ factors }: { factors: CalculatorResponse['factors'] }) {
  if (!factors || factors.length === 0) return null;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-sm">
      <h3 className="text-base font-bold text-slate-900 mb-4">Ertekelesben figyelembe vett tenyezok</h3>
      <div className="space-y-3">
        {factors.map((factor, index) => (
          <div key={index} className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              {factor.impact === 'positive' ? (
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-green-100">
                  <Plus className="h-3.5 w-3.5 text-green-600" />
                </div>
              ) : (
                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-red-100">
                  <Minus className="h-3.5 w-3.5 text-red-600" />
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-slate-800">{factor.name}</p>
              <p className="text-xs text-slate-500 mt-0.5">{factor.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AlternativeScenarios({ scenarios }: { scenarios: CalculatorResponse['alternative_scenarios'] }) {
  if (!scenarios || scenarios.length === 0) return null;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-sm">
      <h3 className="text-base font-bold text-slate-900 mb-4">Alternativ lehetosegek</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {scenarios.map((scenario, index) => (
          <div
            key={index}
            className="rounded-xl border border-slate-100 bg-slate-50/50 p-4 hover:border-slate-200 transition-colors"
          >
            <h4 className="text-sm font-bold text-slate-800 mb-1">{scenario.scenario}</h4>
            <p className="text-xs text-slate-500 mb-3 leading-relaxed">{scenario.description}</p>
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-slate-900 tabular-nums">
                {formatHUF(scenario.estimated_value)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ConfidenceBadge({ score }: { score: number }) {
  const percent = Math.round(score * 100);
  const colorClass =
    percent >= 80
      ? 'bg-green-100 text-green-700 border-green-200'
      : percent >= 60
      ? 'bg-yellow-100 text-yellow-700 border-yellow-200'
      : 'bg-red-100 text-red-700 border-red-200';

  return (
    <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-bold ${colorClass}`}>
      <Shield className="h-3.5 w-3.5" />
      Megbizhatosag: {percent}%
    </div>
  );
}

// =============================================================================
// Main Page Component
// =============================================================================

export default function CalculatorPage() {
  const [searchParams] = useSearchParams();
  const diagnosisId = searchParams.get('diagnosis_id');

  // Form state
  const [vehicleMake, setVehicleMake] = useState('');
  const [vehicleModel, setVehicleModel] = useState('');
  const [vehicleYear, setVehicleYear] = useState('');
  const [mileageKm, setMileageKm] = useState('');
  const [condition, setCondition] = useState<CalculatorRequest['condition'] | ''>('');
  const [fuelType, setFuelType] = useState('');
  const [repairCostHuf, setRepairCostHuf] = useState('');

  // API mutation
  const evaluateMutation = useEvaluateCalculator();
  const result = evaluateMutation.data;

  // Form validation
  const isFormValid = useMemo(() => {
    return (
      vehicleMake.length > 0 &&
      vehicleModel.trim().length > 0 &&
      parseInt(vehicleYear) >= 1990 &&
      parseInt(vehicleYear) <= 2030 &&
      parseInt(mileageKm) >= 0 &&
      parseInt(mileageKm) <= 999999 &&
      condition !== ''
    );
  }, [vehicleMake, vehicleModel, vehicleYear, mileageKm, condition]);

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid || !condition) return;

    const request: CalculatorRequest = {
      vehicle_make: vehicleMake,
      vehicle_model: vehicleModel.trim(),
      vehicle_year: parseInt(vehicleYear),
      mileage_km: parseInt(mileageKm),
      condition,
      repair_cost_huf: repairCostHuf ? parseInt(repairCostHuf) : undefined,
      diagnosis_id: diagnosisId || undefined,
      fuel_type: fuelType ? (fuelType as CalculatorRequest['fuel_type']) : undefined,
    };

    evaluateMutation.mutate(request);
  };

  return (
    <PageContainer maxWidth="xl" padding="md">
      {/* Page Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Calculator className="h-8 w-8 text-primary-600" />
          <h1 className="text-2xl font-bold text-foreground">
            Megeri megjavitani?
          </h1>
          {diagnosisId && (
            <Badge variant="info">Diagnozis alapjan</Badge>
          )}
        </div>
        <p className="text-muted-foreground">
          Szamolja ki, hogy megeri-e megjavitani a jarmuvet, vagy jobb eladni / bontasra adni.
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="mb-10">
        <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-sm">
          <h2 className="text-lg font-bold text-slate-900 mb-5">Jarmu adatok</h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {/* Vehicle Make */}
            <div>
              <label htmlFor="calc-make" className="block text-sm font-medium text-slate-700 mb-1.5">
                Gyarto *
              </label>
              <select
                id="calc-make"
                value={vehicleMake}
                onChange={(e) => setVehicleMake(e.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 shadow-sm focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-colors"
              >
                {manufacturers.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Vehicle Model */}
            <div>
              <label htmlFor="calc-model" className="block text-sm font-medium text-slate-700 mb-1.5">
                Modell *
              </label>
              <Input
                id="calc-model"
                type="text"
                placeholder="pl. Golf, Octavia, 3-as"
                value={vehicleModel}
                onChange={(e) => setVehicleModel(e.target.value)}
              />
            </div>

            {/* Vehicle Year */}
            <div>
              <label htmlFor="calc-year" className="block text-sm font-medium text-slate-700 mb-1.5">
                Evjarat *
              </label>
              <Input
                id="calc-year"
                type="number"
                placeholder="pl. 2018"
                min={1990}
                max={2030}
                value={vehicleYear}
                onChange={(e) => setVehicleYear(e.target.value)}
              />
            </div>

            {/* Mileage */}
            <div>
              <label htmlFor="calc-mileage" className="block text-sm font-medium text-slate-700 mb-1.5">
                Kilometerora allas (km) *
              </label>
              <Input
                id="calc-mileage"
                type="number"
                placeholder="pl. 150000"
                min={0}
                max={999999}
                value={mileageKm}
                onChange={(e) => setMileageKm(e.target.value)}
              />
            </div>

            {/* Fuel Type */}
            <div>
              <label htmlFor="calc-fuel" className="block text-sm font-medium text-slate-700 mb-1.5">
                <span className="flex items-center gap-1.5">
                  <Fuel className="h-3.5 w-3.5 text-slate-400" />
                  Uzemanyag
                </span>
              </label>
              <select
                id="calc-fuel"
                value={fuelType}
                onChange={(e) => setFuelType(e.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 shadow-sm focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-colors"
              >
                {fuelTypeOptions.map((f) => (
                  <option key={f.value} value={f.value}>
                    {f.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Repair Cost (optional) */}
            <div>
              <label htmlFor="calc-repair-cost" className="block text-sm font-medium text-slate-700 mb-1.5">
                Javitasi koltseg (Ft)
              </label>
              <Input
                id="calc-repair-cost"
                type="number"
                placeholder="Ha ismeri a javitasi koltseget"
                min={0}
                value={repairCostHuf}
                onChange={(e) => setRepairCostHuf(e.target.value)}
              />
              <p className="text-[11px] text-slate-400 mt-1">Nem kotelezo - ha ures, becsuljuk</p>
            </div>
          </div>

          {/* Condition Radio Buttons */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-700 mb-3">
              Altalanos allapot *
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {conditionOptions.map((opt) => (
                <label
                  key={opt.value}
                  className={`relative flex flex-col cursor-pointer rounded-xl border-2 p-4 transition-all ${
                    condition === opt.value
                      ? 'border-primary-500 bg-primary-50 shadow-sm'
                      : 'border-slate-200 bg-white hover:border-slate-300'
                  }`}
                >
                  <input
                    type="radio"
                    name="condition"
                    value={opt.value}
                    checked={condition === opt.value}
                    onChange={() => setCondition(opt.value)}
                    className="sr-only"
                  />
                  <span
                    className={`text-sm font-bold ${
                      condition === opt.value ? 'text-primary-700' : 'text-slate-800'
                    }`}
                  >
                    {opt.label}
                  </span>
                  <span className="text-[11px] text-slate-500 mt-0.5">{opt.description}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Submit */}
          <div className="flex items-center gap-4">
            <Button
              type="submit"
              disabled={!isFormValid || evaluateMutation.isPending}
              className="gap-2"
            >
              {evaluateMutation.isPending ? (
                <>
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Szamitas...
                </>
              ) : (
                <>
                  Kalkuacio inditasa
                  <ChevronRight className="h-4 w-4" />
                </>
              )}
            </Button>

            {diagnosisId && (
              <p className="text-xs text-slate-500">
                A diagnozis eredmenyeit is figyelembe vesszuk az ertekelesben.
              </p>
            )}
          </div>

          {/* Error display */}
          {evaluateMutation.isError && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3">
              <p className="text-sm text-red-700">
                {evaluateMutation.error?.message || 'Hiba tortent a szamitas soran.'}
              </p>
            </div>
          )}
        </div>
      </form>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Recommendation */}
          <RecommendationCard
            recommendation={result.recommendation}
            text={result.recommendation_text}
            ratio={result.ratio}
          />

          {/* Confidence Badge */}
          <div className="flex justify-end">
            <ConfidenceBadge score={result.confidence_score} />
          </div>

          {/* Value Comparison */}
          <ValueComparison
            vehicleValue={result.vehicle_value_avg}
            repairCost={(result.repair_cost_min + result.repair_cost_max) / 2}
            currency={result.currency}
          />

          {/* Value and Cost Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <VehicleValueCard result={result} />
            <RepairCostCard result={result} />
          </div>

          {/* Factors */}
          <FactorsList factors={result.factors} />

          {/* Alternative Scenarios */}
          <AlternativeScenarios scenarios={result.alternative_scenarios} />
        </div>
      )}
    </PageContainer>
  );
}
