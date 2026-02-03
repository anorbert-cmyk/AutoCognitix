import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Car, AlertCircle, FileText, Loader2 } from 'lucide-react'

interface DiagnosisForm {
  vehicleMake: string
  vehicleModel: string
  vehicleYear: string
  dtcCodes: string
  symptoms: string
}

export default function DiagnosisPage() {
  const navigate = useNavigate()
  const [isLoading, setIsLoading] = useState(false)
  const [form, setForm] = useState<DiagnosisForm>({
    vehicleMake: '',
    vehicleModel: '',
    vehicleYear: '',
    dtcCodes: '',
    symptoms: '',
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      // TODO: Implement API call
      // const response = await api.post('/diagnosis/analyze', form)
      // navigate(`/diagnosis/${response.data.id}`)

      // Placeholder: simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000))
      navigate('/diagnosis/placeholder-id')
    } catch (error) {
      console.error('Diagnosis error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  // Sample data for dropdowns
  const makes = [
    'Audi', 'BMW', 'Ford', 'Honda', 'Hyundai', 'Kia', 'Mazda',
    'Mercedes-Benz', 'Nissan', 'Opel', 'Peugeot', 'Renault',
    'SEAT', 'Škoda', 'Suzuki', 'Toyota', 'Volkswagen', 'Volvo'
  ]

  const currentYear = new Date().getFullYear()
  const years = Array.from({ length: 35 }, (_, i) => currentYear - i)

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Új diagnosztika
          </h1>
          <p className="text-gray-600">
            Töltse ki az alábbi mezőket a pontos diagnózishoz.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Vehicle Information */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <Car className="h-5 w-5 text-primary-600" />
                <h2 className="card-title text-lg">Jármű adatok</h2>
              </div>
            </div>
            <div className="card-content space-y-4">
              <div className="grid md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Gyártó *
                  </label>
                  <select
                    name="vehicleMake"
                    value={form.vehicleMake}
                    onChange={handleChange}
                    required
                    className="input"
                  >
                    <option value="">Válasszon...</option>
                    {makes.map((make) => (
                      <option key={make} value={make}>
                        {make}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Modell *
                  </label>
                  <input
                    type="text"
                    name="vehicleModel"
                    value={form.vehicleModel}
                    onChange={handleChange}
                    required
                    placeholder="pl. Golf, Focus, Corolla"
                    className="input"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Évjárat *
                  </label>
                  <select
                    name="vehicleYear"
                    value={form.vehicleYear}
                    onChange={handleChange}
                    required
                    className="input"
                  >
                    <option value="">Válasszon...</option>
                    {years.map((year) => (
                      <option key={year} value={year}>
                        {year}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* DTC Codes */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-primary-600" />
                <h2 className="card-title text-lg">Hibakódok</h2>
              </div>
            </div>
            <div className="card-content">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                DTC kódok *
              </label>
              <input
                type="text"
                name="dtcCodes"
                value={form.dtcCodes}
                onChange={handleChange}
                required
                placeholder="pl. P0101, P0171, P0300"
                className="input"
              />
              <p className="mt-1 text-sm text-gray-500">
                Vesszővel elválasztva adja meg a hibakódokat.
              </p>
            </div>
          </div>

          {/* Symptoms */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary-600" />
                <h2 className="card-title text-lg">Tünetek leírása</h2>
              </div>
            </div>
            <div className="card-content">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tapasztalt tünetek *
              </label>
              <textarea
                name="symptoms"
                value={form.symptoms}
                onChange={handleChange}
                required
                rows={5}
                placeholder="Írja le részletesen a tapasztalt problémákat. Például: A motor nehezen indul hidegben, egyenetlenül jár alapjáraton, megnövekedett a fogyasztás..."
                className="textarea"
              />
              <p className="mt-1 text-sm text-gray-500">
                Minél részletesebb leírást ad, annál pontosabb diagnózist kaphat.
              </p>
            </div>
          </div>

          {/* Submit */}
          <div className="flex justify-end gap-4">
            <button
              type="button"
              onClick={() => navigate('/')}
              className="btn-outline"
            >
              Mégse
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Elemzés folyamatban...
                </>
              ) : (
                'Diagnózis indítása'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
