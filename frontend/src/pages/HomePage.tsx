import { Link } from 'react-router-dom'
import { Wrench, Zap, Shield, ArrowRight, Eye, Bell, ClipboardCheck, Calculator, MessageSquare, MapPin } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { useUpcomingReminders } from '@/services/hooks'

export default function HomePage() {
  const { isAuthenticated } = useAuth()
  const { data: upcomingReminders } = useUpcomingReminders(14)

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
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                to="/diagnosis"
                className="inline-flex items-center gap-2 bg-white text-primary-700 hover:bg-primary-50 px-8 py-4 rounded-lg font-semibold text-lg transition-colors"
              >
                Diagnózis indítása
                <ArrowRight className="h-5 w-5" />
              </Link>
              <Link
                to="/demo"
                className="inline-flex items-center gap-2 bg-white/20 text-white hover:bg-white/30 border border-white/40 px-8 py-4 rounded-lg font-semibold text-lg transition-colors backdrop-blur-sm"
              >
                <Eye className="h-5 w-5" />
                Demo megtekintése
              </Link>
            </div>
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

      {/* New Features Section */}
      <section className="py-24 bg-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Új funkciók
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Fedezze fel az új eszközöket a jármű karbantartásához és javításához.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Link to="/inspection" className="card p-8 text-center hover:shadow-lg transition-shadow group">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <ClipboardCheck className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Műszaki Vizsga Felkészítő
              </h3>
              <p className="text-gray-600 mb-4">
                DTC kódok alapján elemezze, milyen kockázattal járna a műszaki vizsgán.
              </p>
              <ArrowRight className="h-5 w-5 text-primary-600 mx-auto opacity-0 group-hover:opacity-100 transition-opacity" />
            </Link>

            <Link to="/calculator" className="card p-8 text-center hover:shadow-lg transition-shadow group">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Calculator className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Megéri Megjavítani?
              </h3>
              <p className="text-gray-600 mb-4">
                Hasonlítsa össze a javítási költséget a jármű piaci értékével.
              </p>
              <ArrowRight className="h-5 w-5 text-primary-600 mx-auto opacity-0 group-hover:opacity-100 transition-opacity" />
            </Link>

            <Link to="/chat" className="card p-8 text-center hover:shadow-lg transition-shadow group">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <MessageSquare className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                AI Chat Asszisztens
              </h3>
              <p className="text-gray-600 mb-4">
                Beszélgessen az AI-val magyarul a jármű problémáiról.
              </p>
              <ArrowRight className="h-5 w-5 text-primary-600 mx-auto opacity-0 group-hover:opacity-100 transition-opacity" />
            </Link>

            <Link to="/services" className="card p-8 text-center hover:shadow-lg transition-shadow group">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <MapPin className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Szerviz Összehasonlítás
              </h3>
              <p className="text-gray-600 mb-4">
                Keresse meg a legközelebbi szervizeket interaktív térképen.
              </p>
              <ArrowRight className="h-5 w-5 text-primary-600 mx-auto opacity-0 group-hover:opacity-100 transition-opacity" />
            </Link>
          </div>
        </div>
      </section>

      {/* Közelgő emlékeztetők widget — csak bejelentkezett felhasználóknak */}
      {isAuthenticated && upcomingReminders && upcomingReminders.length > 0 && (
        <section className="max-w-4xl mx-auto px-4 pb-12">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <Bell className="h-5 w-5 text-[#2563eb]" />
                Közelgő teendők
              </h3>
              <Link
                to="/garage"
                className="text-sm font-medium text-[#2563eb] hover:underline"
              >
                Összes megtekintése →
              </Link>
            </div>
            <div className="space-y-3">
              {upcomingReminders.slice(0, 3).map((reminder) => (
                <div
                  key={reminder.id}
                  className={`flex items-center justify-between p-3 rounded-xl border ${
                    reminder.urgency === 'overdue'
                      ? 'bg-red-50 border-red-200'
                      : reminder.urgency === 'urgent'
                      ? 'bg-orange-50 border-orange-200'
                      : 'bg-yellow-50 border-yellow-200'
                  }`}
                >
                  <div>
                    <p className="font-medium text-slate-900 text-sm">{reminder.title}</p>
                    <p className="text-xs text-slate-500">
                      {reminder.days_until_due !== null && reminder.days_until_due !== undefined
                        ? reminder.days_until_due < 0
                          ? `${Math.abs(reminder.days_until_due)} napja lejárt`
                          : reminder.days_until_due === 0
                          ? 'Ma esedékes'
                          : `${reminder.days_until_due} nap múlva`
                        : reminder.due_date ?? 'Határidő nincs megadva'}
                    </p>
                  </div>
                  <Link
                    to={`/garage/${reminder.vehicle_id}`}
                    className="text-xs font-semibold text-[#2563eb] hover:underline"
                  >
                    Megtekint →
                  </Link>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

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
