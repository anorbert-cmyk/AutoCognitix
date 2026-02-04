import { Link, useParams } from 'react-router-dom'
import { AlertTriangle, CheckCircle, Wrench, DollarSign, ArrowLeft } from 'lucide-react'

// Placeholder data - will be replaced with API call
const placeholderResult = {
  id: 'placeholder-id',
  vehicle: {
    make: 'Volkswagen',
    model: 'Golf',
    year: 2018,
  },
  dtcCodes: ['P0101', 'P0171'],
  symptoms: 'A motor nehezen indul hidegben, egyenetlenül jár alapjáraton.',
  confidenceScore: 0.78,
  probableCauses: [
    {
      title: 'Levegőtömeg-mérő (MAF) szenzor hiba',
      description:
        'A P0101 hibakód a MAF szenzor jelproblémáját jelzi. A szenzor szennyeződése vagy meghibásodása okozhatja a tüneteket.',
      confidence: 0.85,
      relatedCodes: ['P0101'],
      components: ['Levegőtömeg-mérő szenzor', 'Levegőszűrő'],
    },
    {
      title: 'Vákuumszivárgás a szívórendszerben',
      description:
        'A P0171 hibakód sovány keverékre utal, amit vákuumszivárgás okozhat.',
      confidence: 0.72,
      relatedCodes: ['P0171'],
      components: ['Szívócső', 'Tömítések', 'Vákuumcsövek'],
    },
  ],
  recommendedRepairs: [
    {
      title: 'MAF szenzor tisztítása vagy cseréje',
      description:
        'Először próbálja meg speciális MAF tisztítóval megtisztítani a szenzort. Ha nem segít, cserélje ki.',
      estimatedCostMin: 5000,
      estimatedCostMax: 45000,
      difficulty: 'beginner',
      partsNeeded: ['MAF szenzor tisztító spray', 'Új MAF szenzor (ha szükséges)'],
    },
    {
      title: 'Vákuumrendszer ellenőrzése',
      description:
        'Ellenőrizze az összes vákuumcsövet és tömítést szivárgás szempontjából.',
      estimatedCostMin: 0,
      estimatedCostMax: 15000,
      difficulty: 'intermediate',
      partsNeeded: ['Vákuumcsövek', 'Szívócső tömítés'],
    },
  ],
}

function getConfidenceColor(confidence: number) {
  if (confidence >= 0.8) return 'text-green-600 bg-green-100'
  if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100'
  return 'text-red-600 bg-red-100'
}

function getDifficultyLabel(difficulty: string) {
  const labels: Record<string, string> = {
    beginner: 'Kezdő',
    intermediate: 'Közepes',
    advanced: 'Haladó',
    professional: 'Szakember',
  }
  return labels[difficulty] || difficulty
}

function getDifficultyColor(difficulty: string) {
  const colors: Record<string, string> = {
    beginner: 'text-green-600 bg-green-100',
    intermediate: 'text-yellow-600 bg-yellow-100',
    advanced: 'text-orange-600 bg-orange-100',
    professional: 'text-red-600 bg-red-100',
  }
  return colors[difficulty] || 'text-gray-600 bg-gray-100'
}

export default function ResultPage() {
  const { id } = useParams()
  // TODO: Fetch from API based on id
  console.log('Loading diagnosis result:', id)
  const result = placeholderResult

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
        {/* Back button */}
        <Link
          to="/diagnosis"
          className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-6"
        >
          <ArrowLeft className="h-4 w-4" />
          Új diagnózis
        </Link>

        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                {result.vehicle.year} {result.vehicle.make} {result.vehicle.model}
              </h1>
              <p className="text-gray-600">
                Hibakódok: {result.dtcCodes.join(', ')}
              </p>
            </div>
            <div
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-full font-medium ${getConfidenceColor(
                result.confidenceScore
              )}`}
            >
              <CheckCircle className="h-5 w-5" />
              {Math.round(result.confidenceScore * 100)}% megbízhatóság
            </div>
          </div>
        </div>

        {/* Probable Causes */}
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-500" />
            Valószínű okok
          </h2>
          <div className="space-y-4">
            {result.probableCauses.map((cause, index) => (
              <div key={index} className="card p-6">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      {cause.title}
                    </h3>
                    <p className="text-gray-600 mb-4">{cause.description}</p>
                    <div className="flex flex-wrap gap-2">
                      {cause.relatedCodes.map((code) => (
                        <Link
                          key={code}
                          to={`/dtc/${code}`}
                          className="inline-flex items-center px-2 py-1 rounded bg-primary-100 text-primary-700 text-sm font-medium hover:bg-primary-200"
                        >
                          {code}
                        </Link>
                      ))}
                      {cause.components.map((component) => (
                        <span
                          key={component}
                          className="inline-flex items-center px-2 py-1 rounded bg-gray-100 text-gray-700 text-sm"
                        >
                          {component}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(
                      cause.confidence
                    )}`}
                  >
                    {Math.round(cause.confidence * 100)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recommended Repairs */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Wrench className="h-5 w-5 text-primary-500" />
            Javasolt javítások
          </h2>
          <div className="space-y-4">
            {result.recommendedRepairs.map((repair, index) => (
              <div key={index} className="card p-6">
                <div className="flex items-start justify-between gap-4 mb-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    {repair.title}
                  </h3>
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${getDifficultyColor(
                      repair.difficulty
                    )}`}
                  >
                    {getDifficultyLabel(repair.difficulty)}
                  </span>
                </div>
                <p className="text-gray-600 mb-4">{repair.description}</p>

                <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                  {/* Cost estimate */}
                  <div className="flex items-center gap-2 text-gray-700">
                    <DollarSign className="h-5 w-5 text-gray-400" />
                    <span className="font-medium">
                      {repair.estimatedCostMin.toLocaleString()} -{' '}
                      {repair.estimatedCostMax.toLocaleString()} Ft
                    </span>
                  </div>

                  {/* Parts needed */}
                  {repair.partsNeeded.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {repair.partsNeeded.map((part) => (
                        <span
                          key={part}
                          className="inline-flex items-center px-2 py-1 rounded bg-gray-100 text-gray-700 text-sm"
                        >
                          {part}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="mt-8 flex justify-center gap-4">
          <Link to="/diagnosis" className="btn-primary">
            Új diagnózis indítása
          </Link>
        </div>
      </div>
    </div>
  )
}
