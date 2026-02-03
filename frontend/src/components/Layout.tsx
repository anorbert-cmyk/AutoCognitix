import { Outlet, Link } from 'react-router-dom'
import { Car, Menu, X } from 'lucide-react'
import { useState } from 'react'

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-50">
        <nav className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 justify-between items-center">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2">
              <Car className="h-8 w-8 text-primary-600" />
              <span className="text-xl font-bold text-gray-900">
                AutoCognitix
              </span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-6">
              <Link
                to="/"
                className="text-gray-600 hover:text-gray-900 font-medium"
              >
                Kezdőlap
              </Link>
              <Link
                to="/diagnosis"
                className="text-gray-600 hover:text-gray-900 font-medium"
              >
                Diagnosztika
              </Link>
              <Link
                to="/diagnosis"
                className="btn-primary"
              >
                Új diagnózis
              </Link>
            </div>

            {/* Mobile menu button */}
            <button
              className="md:hidden p-2"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>

          {/* Mobile Navigation */}
          {mobileMenuOpen && (
            <div className="md:hidden py-4 border-t">
              <div className="flex flex-col gap-4">
                <Link
                  to="/"
                  className="text-gray-600 hover:text-gray-900 font-medium"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Kezdőlap
                </Link>
                <Link
                  to="/diagnosis"
                  className="text-gray-600 hover:text-gray-900 font-medium"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Diagnosztika
                </Link>
                <Link
                  to="/diagnosis"
                  className="btn-primary text-center"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Új diagnózis
                </Link>
              </div>
            </div>
          )}
        </nav>
      </header>

      {/* Main content */}
      <main>
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-auto">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-2">
              <Car className="h-6 w-6 text-primary-600" />
              <span className="font-semibold text-gray-900">AutoCognitix</span>
            </div>
            <p className="text-sm text-gray-500">
              © {new Date().getFullYear()} AutoCognitix. AI-alapú gépjármű-diagnosztika.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
