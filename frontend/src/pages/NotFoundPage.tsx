import { Link } from 'react-router-dom'
import { Home, Search } from 'lucide-react'

export default function NotFoundPage() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="text-center px-4">
        <h1 className="text-9xl font-bold text-gray-200">404</h1>
        <h2 className="text-2xl font-semibold text-gray-900 mt-4 mb-2">
          Az oldal nem található
        </h2>
        <p className="text-gray-600 mb-8">
          A keresett oldal nem létezik vagy el lett távolítva.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link to="/" className="btn-primary inline-flex items-center gap-2">
            <Home className="h-4 w-4" />
            Kezdőlap
          </Link>
          <Link
            to="/diagnosis"
            className="btn-outline inline-flex items-center gap-2"
          >
            <Search className="h-4 w-4" />
            Új diagnózis
          </Link>
        </div>
      </div>
    </div>
  )
}
