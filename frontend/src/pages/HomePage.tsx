import { Link } from 'react-router-dom'
import { Wrench, Zap, Shield, ArrowRight } from 'lucide-react'

export default function HomePage() {
  return (
    <div>
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary-600 to-primary-800 text-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              AI-alapú Gépjármű Diagnosztika
            </h1>
            <p className="text-xl md:text-2xl text-primary-100 mb-8 max-w-3xl mx-auto">
              Gyors és pontos hibakód-elemzés mesterséges intelligenciával.
              Hardver nélkül, bárhonnan elérhető.
            </p>
            <Link
              to="/diagnosis"
              className="inline-flex items-center gap-2 bg-white text-primary-700 hover:bg-primary-50 px-8 py-4 rounded-lg font-semibold text-lg transition-colors"
            >
              Diagnózis indítása
              <ArrowRight className="h-5 w-5" />
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Hogyan működik?
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Egyszerűen adja meg a hibakódokat és a tüneteket, az AI elvégzi az elemzést.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="card p-8 text-center">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Wrench className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Hibakód bevitel
              </h3>
              <p className="text-gray-600">
                Adja meg az OBD-II hibakódokat és írja le a tapasztalt tüneteket magyarul.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="card p-8 text-center">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Zap className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                AI elemzés
              </h3>
              <p className="text-gray-600">
                A rendszer elemzi az adatokat TSB-k, fórum-bejegyzések és szakértői tudás alapján.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="card p-8 text-center">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Shield className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Részletes diagnózis
              </h3>
              <p className="text-gray-600">
                Kapjon részletes diagnózist lehetséges okokkal és javítási javaslatokkal.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gray-100 py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="bg-white rounded-2xl shadow-xl p-8 md:p-12 text-center">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Készen áll a diagnosztikára?
            </h2>
            <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
              Nem kell drága hardver - csak adja meg a hibakódokat és a tüneteket.
              Az AI segít megtalálni a problémát.
            </p>
            <Link
              to="/diagnosis"
              className="btn-primary text-lg px-8 py-4"
            >
              Diagnózis indítása
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
