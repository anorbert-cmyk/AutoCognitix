import { useParams, useNavigate } from 'react-router-dom'
import {
  AlertTriangle,
  Check,
  Clock,
  User,
  Wrench,
  FileText,
  Printer,
  Lightbulb,
  Loader2,
  AlertCircle,
  Car,
  Gauge,
} from 'lucide-react'
import { useToast } from '../contexts/ToastContext'
import { useDiagnosisDetail } from '../services/hooks'
import { DiagnosisResponse } from '../services/api'

// Wizard step component
function WizardStep({
  number,
  label,
  isCompleted,
  isActive,
}: {
  number: number
  label: string
  isCompleted: boolean
  isActive: boolean
}) {
  return (
    <div className="flex items-center">
      <div
        className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium ${
          isCompleted
            ? 'bg-green-500 text-white'
            : isActive
              ? 'bg-[#2563eb] text-white'
              : 'bg-gray-200 text-gray-600'
        }`}
      >
        {isCompleted ? <Check className="w-5 h-5" /> : number}
      </div>
      <span
        className={`ml-2 text-sm font-medium ${
          isActive ? 'text-gray-900' : 'text-gray-500'
        }`}
      >
        {label}
      </span>
    </div>
  )
}

// Circular progress indicator component
function CircularProgress({
  percentage,
  size = 120,
}: {
  percentage: number
  size?: number
}) {
  const strokeWidth = 8
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (percentage / 100) * circumference

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255,255,255,0.2)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#22c55e"
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-3xl font-bold text-white">{percentage}%</span>
      </div>
    </div>
  )
}

// Repair step component
function RepairStep({
  number,
  title,
  description,
  tools,
  expertTip,
}: {
  number: number
  title: string
  description: string
  tools: string[]
  expertTip: string
}) {
  return (
    <div className="bg-white rounded-xl p-6 shadow-sm">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-10 h-10 bg-[#2563eb] text-white rounded-full flex items-center justify-center font-bold text-lg">
          {number}
        </div>
        <div className="flex-1">
          <h4 className="text-lg font-semibold text-gray-900 mb-2">{title}</h4>
          <p className="text-gray-600 mb-4">{description}</p>

          {/* Tools needed */}
          <div className="flex items-center gap-2 mb-4">
            <Wrench className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-500">Szukseges szerszamok:</span>
            <div className="flex flex-wrap gap-2">
              {tools.map((tool, idx) => (
                <span
                  key={idx}
                  className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                >
                  {tool}
                </span>
              ))}
            </div>
          </div>

          {/* Expert tip */}
          <div className="border-l-4 border-amber-400 bg-amber-50 p-4 rounded-r-lg">
            <div className="flex items-start gap-2">
              <Lightbulb className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-medium text-amber-800">Szakertoi tipp:</span>
                <p className="text-amber-700 text-sm mt-1">{expertTip}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function DiagnosisResultContent({ result }: { result: DiagnosisResponse }) {
  const toast = useToast()

  // Generate mock data for display (in real app, this would come from API)
  const vehicleImage = 'https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?w=400&h=300&fit=crop'
  const licensePlate = 'ABC-123'
  const vin = result.dtc_codes.length > 0 ? 'JTDKN3DU5A0123456' : 'N/A'
  const mileage = '125,000 km'

  // Get the primary DTC code
  const primaryDTC = result.dtc_codes[0] || 'P0000'

  // Get customer complaint (symptoms)
  const customerComplaint = result.symptoms || 'A jarmu egyenetlenul jar, kulonosen gyorsitaskor.'

  // Generate repair steps from recommended_repairs or use defaults
  const repairSteps = result.recommended_repairs.length > 0
    ? result.recommended_repairs.slice(0, 3).map((repair, index) => ({
        number: index + 1,
        title: repair.title,
        description: repair.description,
        tools: repair.parts_needed.length > 0 ? repair.parts_needed.slice(0, 3) : ['Altalanos szerszamkeszlet'],
        expertTip: 'Ellenorizze a kapcsolodo alkatreszeket is a javitas soran.',
      }))
    : [
        {
          number: 1,
          title: 'Gyujtógyertya csere',
          description: 'Csereljuk ki a 3. hengerhez tartozo gyujtogyertyat. A regi gyertya kopott vagy serult elektrodaval rendelkezhet.',
          tools: ['Gyertyakulcs (16mm)', 'Nyomatekkulcs', 'Dieletromos zsir'],
          expertTip: 'Hideg motoron vegezze a cserét. Ellenorizze a gyertya resét (0.8-0.9mm Toyota motoroknal).',
        },
        {
          number: 2,
          title: 'Gyujtókábel ellenorzés',
          description: 'Ellenorizzuk a gyujtokabel allapotat es csatlakozasat. Keressunk repedéseket, kopasos nyomokat.',
          tools: ['Multiméter', 'Diagnosztikai lampa'],
          expertTip: 'A kabelek ellenallasa altalaban 10-15 kOhm/meter korul legyen. Ettol valo jelentos elteres cserét igenyel.',
        },
        {
          number: 3,
          title: 'Kompresszió meres',
          description: 'Merje meg a 3. henger kompressziojat. Az ertek 10%-nal nagyobb elteres a tobbi hengertol problemat jelez.',
          tools: ['Kompresszio mero', 'Gyertyakulcs'],
          expertTip: 'Meleg motoron vegezze a merest. Normal ertek: 12-14 bar Toyota 2.5L motoroknal.',
        },
      ]

  // Calculate confidence percentage
  const confidencePercentage = Math.round(result.confidence_score * 100)

  // Handlers
  const handleSavePDF = () => {
    toast.info('PDF mentes elinditva...')
    window.print()
  }

  const handlePrintWorksheet = () => {
    toast.info('Munkalap nyomtatasa...')
    window.print()
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header with wizard steps */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-[#0D1B2A] rounded-lg flex items-center justify-center">
                <Car className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-xl text-[#0D1B2A]">AutoCognitix</span>
            </div>

            {/* Wizard Steps */}
            <div className="hidden md:flex items-center gap-8">
              <WizardStep number={1} label="Adatfelvetel" isCompleted={true} isActive={false} />
              <div className="w-12 h-0.5 bg-green-500" />
              <WizardStep number={2} label="Elemzes" isCompleted={true} isActive={false} />
              <div className="w-12 h-0.5 bg-green-500" />
              <WizardStep number={3} label="Jelentes" isCompleted={true} isActive={true} />
            </div>

            {/* Placeholder for right side */}
            <div className="w-24" />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pb-32">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left sidebar */}
          <div className="lg:col-span-4 space-y-6">
            {/* Vehicle card */}
            <div className="bg-white rounded-xl shadow-sm overflow-hidden">
              <img
                src={vehicleImage}
                alt={`${result.vehicle_year} ${result.vehicle_make} ${result.vehicle_model}`}
                className="w-full h-48 object-cover"
              />
              <div className="p-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">
                  {result.vehicle_make} {result.vehicle_model} {result.vehicle_year}
                </h2>

                <div className="space-y-3">
                  <div className="flex items-center gap-3 text-gray-600">
                    <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                      <FileText className="w-4 h-4 text-gray-500" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Rendszam</p>
                      <p className="font-medium">{licensePlate}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 text-gray-600">
                    <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                      <Car className="w-4 h-4 text-gray-500" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">VIN</p>
                      <p className="font-medium text-sm">{vin}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 text-gray-600">
                    <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                      <Gauge className="w-4 h-4 text-gray-500" />
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Kilometerora allas</p>
                      <p className="font-medium">{mileage}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* DTC Code card */}
            <div className="bg-red-50 border border-red-200 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                <span className="text-sm font-medium text-red-600">Hibakod</span>
              </div>
              <div className="text-4xl font-bold text-red-600 mb-2">{primaryDTC}</div>
              <p className="text-red-700 text-sm">
                {result.probable_causes[0]?.title || '3. henger gyujtaskimaradas'}
              </p>
              {result.dtc_codes.length > 1 && (
                <div className="mt-3 pt-3 border-t border-red-200">
                  <p className="text-xs text-red-600 mb-2">Tovabbi hibakodok:</p>
                  <div className="flex flex-wrap gap-2">
                    {result.dtc_codes.slice(1).map((code) => (
                      <span
                        key={code}
                        className="px-2 py-1 bg-red-100 text-red-700 text-xs rounded-full font-medium"
                      >
                        {code}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Customer complaint */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <div className="flex items-center gap-2 mb-4">
                <User className="w-5 h-5 text-gray-400" />
                <span className="text-sm font-medium text-gray-700">Ugyfel panasza</span>
              </div>
              <blockquote className="text-gray-600 italic border-l-4 border-[#2563eb] pl-4">
                "{customerComplaint}"
              </blockquote>
            </div>
          </div>

          {/* Right content area */}
          <div className="lg:col-span-8 space-y-6">
            {/* AI Analysis header */}
            <div className="bg-[#0D1B2A] rounded-xl p-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold mb-1">AI Elemzes</h3>
                  <p className="text-gray-400 text-sm">
                    Mesterseges intelligencia altal generalt diagnozis
                  </p>
                </div>
                <CircularProgress percentage={confidencePercentage} />
              </div>

              <div className="mt-6 pt-6 border-t border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-400">Elemzes ideje: 2.3 masodperc</span>
                </div>
                <p className="text-gray-300 text-sm">
                  Az elemzes {result.sources?.length || 3} kulonbozo adatforras es {result.probable_causes.length} lehetseges ok alapjan keszult.
                </p>
              </div>
            </div>

            {/* Prioritized repair plan */}
            <div>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-[#2563eb] rounded-xl flex items-center justify-center">
                  <Wrench className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900">Priorizalt javitasi terv</h3>
                  <p className="text-gray-500 text-sm">A legvaloszinubb oktol a legkevesbe valoszinuig</p>
                </div>
              </div>

              <div className="space-y-4">
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
            </div>
          </div>
        </div>
      </main>

      {/* Fixed bottom bar */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 shadow-lg print:hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="text-sm text-gray-500">
              Diagnozis ID: <span className="font-mono text-gray-700">{result.id}</span>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={handleSavePDF}
                className="inline-flex items-center gap-2 px-6 py-3 bg-[#0D1B2A] text-white font-medium rounded-xl hover:bg-[#1a2d42] transition-colors"
              >
                <FileText className="w-5 h-5" />
                Jelentes mentese PDF-kent
              </button>
              <button
                onClick={handlePrintWorksheet}
                className="inline-flex items-center gap-2 px-6 py-3 border-2 border-[#0D1B2A] text-[#0D1B2A] font-medium rounded-xl hover:bg-gray-50 transition-colors"
              >
                <Printer className="w-5 h-5" />
                Munkalap nyomtatasa
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function ResultPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const { data: result, isLoading, error, refetch } = useDiagnosisDetail(id)

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-[#2563eb] mx-auto mb-4" />
          <p className="text-gray-600">Diagnozis betoltese...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto px-4">
          <AlertCircle className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            A diagnozis nem talalhato
          </h2>
          <p className="text-gray-600 mb-6">
            Lehetseges, hogy a diagnozis torolve lett vagy nem letezik.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              onClick={() => refetch()}
              className="px-6 py-3 border border-gray-300 text-gray-700 font-medium rounded-xl hover:bg-gray-50 transition-colors"
            >
              Ujraprobalas
            </button>
            <button
              onClick={() => navigate('/diagnosis')}
              className="px-6 py-3 bg-[#2563eb] text-white font-medium rounded-xl hover:bg-[#1d4ed8] transition-colors"
            >
              Uj diagnozis keszitese
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (!result) {
    return null
  }

  return <DiagnosisResultContent result={result} />
}
