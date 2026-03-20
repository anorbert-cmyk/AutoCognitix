import { MapPin } from 'lucide-react'

interface MapMarker {
  id: string
  name: string
  lat: number
  lng: number
  rating: number
}

interface ServiceMapProps {
  markers: MapMarker[]
  center: { lat: number; lng: number }
  onMarkerClick?: (id: string) => void
}

export default function ServiceMap({ markers, center, onMarkerClick }: ServiceMapProps) {
  return (
    <div className="relative min-h-[400px] overflow-hidden rounded-lg bg-gray-100">
      {/* Map placeholder background */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="flex flex-col items-center gap-2 text-gray-400">
          <MapPin className="h-12 w-12" />
          <span className="text-sm">Térkép betöltése...</span>
          <span className="text-xs text-gray-300">
            Középpont: {center.lat.toFixed(4)}, {center.lng.toFixed(4)}
          </span>
        </div>
      </div>

      {/* Marker list overlay */}
      <div className="relative z-10 flex flex-col gap-2 p-4">
        <div className="mb-2 rounded-md bg-white/90 px-3 py-1.5 text-xs font-medium text-gray-600 shadow-sm backdrop-blur-sm w-fit">
          {markers.length} szerviz a térképen
        </div>

        <div className="flex flex-wrap gap-2">
          {markers.map((marker) => (
            <button
              key={marker.id}
              type="button"
              onClick={() => onMarkerClick?.(marker.id)}
              className="flex items-center gap-1.5 rounded-full bg-white/90 px-3 py-1.5 text-sm shadow-sm backdrop-blur-sm transition-all hover:bg-white hover:shadow-md"
            >
              <MapPin className="h-4 w-4 text-red-500" />
              <span className="font-medium text-gray-800">{marker.name}</span>
              <span className="text-xs text-yellow-600">
                {'★'.repeat(Math.round(marker.rating))}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
