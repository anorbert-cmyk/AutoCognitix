import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, AlertCircle, Wrench, Car, BookOpen } from 'lucide-react'

// Placeholder data - will be replaced with API call
const placeholderDTC = {
  code: 'P0101',
  descriptionEn: 'Mass Air Flow Circuit Range/Performance',
  descriptionHu: 'Levegőtömeg-mérő áramkör tartomány/teljesítmény hiba',
  category: 'Powertrain',
  severity: 'medium',
  system: 'Fuel and Air Metering',
  symptoms: [
    'Motor teljesítményvesztése',
    'Egyenetlen alapjárat',
    'Nehéz indítás',
    'Megnövekedett üzemanyag-fogyasztás',
    'Fekete füst a kipufogóból',
  ],
  possibleCauses: [
    'Szennyezett MAF szenzor',
    'Levegőszűrő eltömődése',
    'Vákuumszivárgás a szívórendszerben',
    'MAF szenzor meghibásodása',
    'Vezeték vagy csatlakozó probléma',
  ],
  diagnosticSteps: [
    'Vizuálisan ellenőrizze a MAF szenzort és a levegőszűrőt',
    'Ellenőrizze a MAF szenzor csatlakozóját és vezetékeit',
    'Tisztítsa meg a MAF szenzort speciális tisztítóval',
    'Ellenőrizze a szívórendszert vákuumszivárgás szempontjából',
    'Tesztelje a MAF szenzor jelét oszcilloszkóppal vagy multiméterrel',
  ],
  relatedCodes: ['P0100', 'P0102', 'P0103', 'P0171', 'P0174'],
  commonVehicles: [
    'Volkswagen Golf (2010-2020)',
    'Audi A3 (2012-2020)',
    'Ford Focus (2011-2018)',
    'Toyota Corolla (2014-2019)',
  ],
}

function getSeverityColor(severity: string) {
  const colors: Record<string, string> = {
    low: 'text-green-600 bg-green-100',
    medium: 'text-yellow-600 bg-yellow-100',
    high: 'text-orange-600 bg-orange-100',
    critical: 'text-red-600 bg-red-100',
  }
  return colors[severity] || 'text-gray-600 bg-gray-100'
}

function getSeverityLabel(severity: string) {
  const labels: Record<string, string> = {
    low: 'Alacsony',
    medium: 'Közepes',
    high: 'Magas',
    critical: 'Kritikus',
  }
  return labels[severity] || severity
}

export default function DTCDetailPage() {
  const { code } = useParams()
  const dtc = placeholderDTC // TODO: Fetch from API based on code

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
        {/* Back button */}
        <Link
          to="/diagnosis"
          className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-6"
        >
          <ArrowLeft className="h-4 w-4" />
          Vissza
        </Link>

        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-3xl font-mono font-bold text-primary-600">
                  {code}
                </span>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(
                    dtc.severity
                  )}`}
                >
                  {getSeverityLabel(dtc.severity)}
                </span>
              </div>
              <h1 className="text-xl font-medium text-gray-900">
                {dtc.descriptionHu}
              </h1>
              <p className="text-gray-500">{dtc.descriptionEn}</p>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="inline-flex items-center px-3 py-1 rounded bg-gray-100 text-gray-700 text-sm">
              {dtc.category}
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded bg-gray-100 text-gray-700 text-sm">
              {dtc.system}
            </span>
          </div>
        </div>

        {/* Content grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Symptoms */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-yellow-500" />
              Tünetek
            </h2>
            <ul className="space-y-2">
              {dtc.symptoms.map((symptom, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-yellow-500 mt-1">•</span>
                  <span className="text-gray-700">{symptom}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Possible Causes */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Wrench className="h-5 w-5 text-primary-500" />
              Lehetséges okok
            </h2>
            <ul className="space-y-2">
              {dtc.possibleCauses.map((cause, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-primary-500 mt-1">•</span>
                  <span className="text-gray-700">{cause}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Diagnostic Steps */}
          <div className="card p-6 md:col-span-2">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-green-500" />
              Diagnosztikai lépések
            </h2>
            <ol className="space-y-3">
              {dtc.diagnosticSteps.map((step, index) => (
                <li key={index} className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-100 text-primary-700 text-sm font-medium flex items-center justify-center">
                    {index + 1}
                  </span>
                  <span className="text-gray-700">{step}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Related Codes */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Kapcsolódó hibakódok
            </h2>
            <div className="flex flex-wrap gap-2">
              {dtc.relatedCodes.map((relatedCode) => (
                <Link
                  key={relatedCode}
                  to={`/dtc/${relatedCode}`}
                  className="inline-flex items-center px-3 py-2 rounded bg-primary-50 text-primary-700 font-mono font-medium hover:bg-primary-100 transition-colors"
                >
                  {relatedCode}
                </Link>
              ))}
            </div>
          </div>

          {/* Common Vehicles */}
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Car className="h-5 w-5 text-gray-500" />
              Gyakran érintett járművek
            </h2>
            <ul className="space-y-2">
              {dtc.commonVehicles.map((vehicle, index) => (
                <li key={index} className="text-gray-700">
                  {vehicle}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* CTA */}
        <div className="mt-8 text-center">
          <Link to="/diagnosis" className="btn-primary">
            Diagnózis indítása ezzel a kóddal
          </Link>
        </div>
      </div>
    </div>
  )
}
