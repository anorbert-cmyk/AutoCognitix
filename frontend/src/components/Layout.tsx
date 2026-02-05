import { Outlet, Link } from 'react-router-dom'
import { Car, Menu, X, History, User, LogOut } from 'lucide-react'
import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const { user, isAuthenticated, logout, isLoading } = useAuth()

  const handleLogout = async () => {
    await logout()
    setUserMenuOpen(false)
  }

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
                Kezdolap
              </Link>
              <Link
                to="/diagnosis"
                className="text-gray-600 hover:text-gray-900 font-medium"
              >
                Diagnosztika
              </Link>
              {isAuthenticated && (
                <Link
                  to="/history"
                  className="text-gray-600 hover:text-gray-900 font-medium flex items-center gap-1"
                >
                  <History className="h-4 w-4" />
                  Tortenelem
                </Link>
              )}
              <Link
                to="/diagnosis"
                className="btn-primary"
              >
                Uj diagnozis
              </Link>

              {/* Auth buttons */}
              {isAuthenticated ? (
                <div className="relative">
                  <button
                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                    className="flex items-center gap-2 text-gray-600 hover:text-gray-900 font-medium"
                  >
                    <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                      <User className="h-5 w-5 text-primary-600" />
                    </div>
                    <span className="hidden lg:inline">
                      {user?.full_name || user?.email}
                    </span>
                  </button>

                  {/* User dropdown menu */}
                  {userMenuOpen && (
                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-50 border">
                      <div className="px-4 py-2 border-b">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {user?.full_name || 'Felhasznalo'}
                        </p>
                        <p className="text-xs text-gray-500 truncate">
                          {user?.email}
                        </p>
                      </div>
                      <Link
                        to="/history"
                        className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                        onClick={() => setUserMenuOpen(false)}
                      >
                        Diagnozis tortenelem
                      </Link>
                      <button
                        onClick={handleLogout}
                        disabled={isLoading}
                        className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-gray-100 flex items-center gap-2"
                      >
                        <LogOut className="h-4 w-4" />
                        Kijelentkezes
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <Link
                    to="/login"
                    className="text-gray-600 hover:text-gray-900 font-medium"
                  >
                    Bejelentkezes
                  </Link>
                  <Link
                    to="/register"
                    className="bg-gray-100 hover:bg-gray-200 text-gray-900 px-4 py-2 rounded-md font-medium transition-colors"
                  >
                    Regisztracio
                  </Link>
                </div>
              )}
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
                  Kezdolap
                </Link>
                <Link
                  to="/diagnosis"
                  className="text-gray-600 hover:text-gray-900 font-medium"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Diagnosztika
                </Link>
                {isAuthenticated && (
                  <Link
                    to="/history"
                    className="text-gray-600 hover:text-gray-900 font-medium flex items-center gap-1"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    <History className="h-4 w-4" />
                    Tortenelem
                  </Link>
                )}
                <Link
                  to="/diagnosis"
                  className="btn-primary text-center"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Uj diagnozis
                </Link>

                {/* Mobile auth section */}
                <div className="pt-4 border-t mt-2">
                  {isAuthenticated ? (
                    <>
                      <div className="px-2 py-2 mb-2">
                        <p className="text-sm font-medium text-gray-900">
                          {user?.full_name || 'Felhasznalo'}
                        </p>
                        <p className="text-xs text-gray-500">{user?.email}</p>
                      </div>
                      <button
                        onClick={() => {
                          handleLogout()
                          setMobileMenuOpen(false)
                        }}
                        className="w-full text-left text-red-600 hover:text-red-700 font-medium flex items-center gap-2"
                      >
                        <LogOut className="h-4 w-4" />
                        Kijelentkezes
                      </button>
                    </>
                  ) : (
                    <>
                      <Link
                        to="/login"
                        className="block text-center py-2 text-gray-600 hover:text-gray-900 font-medium"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        Bejelentkezes
                      </Link>
                      <Link
                        to="/register"
                        className="block text-center py-2 mt-2 bg-gray-100 hover:bg-gray-200 text-gray-900 rounded-md font-medium"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        Regisztracio
                      </Link>
                    </>
                  )}
                </div>
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
              {new Date().getFullYear()} AutoCognitix. AI-alapu gepjarmu-diagnosztika.
            </p>
          </div>
        </div>
      </footer>

      {/* Click outside handler for user menu */}
      {userMenuOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setUserMenuOpen(false)}
        />
      )}
    </div>
  )
}
