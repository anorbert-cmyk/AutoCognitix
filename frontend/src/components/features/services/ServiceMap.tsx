/**
 * ServiceMap — Leaflet térkép szerviz helyekkel.
 */
import { useEffect } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix Leaflet default marker icons broken by Vite asset hashing
delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

export interface MapMarker {
  id: string
  name: string
  lat: number
  lng: number
  rating: number
  address?: string
  city?: string
}

interface ServiceMapProps {
  markers: MapMarker[]
  center: { lat: number; lng: number }
  selectedId?: string | null
  onMarkerClick?: (id: string) => void
}

/** Flies the map to the selected marker when selectedId changes. */
function FlyToSelected({
  selectedId,
  markers,
}: {
  selectedId: string | null | undefined
  markers: MapMarker[]
}) {
  const map = useMap()
  useEffect(() => {
    if (!selectedId) return
    const found = markers.find((m) => m.id === selectedId)
    if (found) {
      map.flyTo([found.lat, found.lng], 14, { duration: 0.8 })
    }
  }, [selectedId, markers, map])
  return null
}

export default function ServiceMap({
  markers,
  center,
  selectedId,
  onMarkerClick,
}: ServiceMapProps) {
  return (
    <MapContainer
      center={[center.lat, center.lng]}
      zoom={7}
      className="w-full h-full"
      style={{ minHeight: '400px' }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FlyToSelected selectedId={selectedId} markers={markers} />
      {markers.map((marker) => (
        <Marker
          key={marker.id}
          position={[marker.lat, marker.lng]}
          eventHandlers={{ click: () => onMarkerClick?.(marker.id) }}
        >
          <Popup>
            <div className="text-sm min-w-[140px]">
              <div className="font-bold text-slate-900">{marker.name}</div>
              {marker.address && (
                <div className="text-xs text-slate-500 mt-0.5">{marker.address}</div>
              )}
              {marker.city && (
                <div className="text-xs text-slate-500">{marker.city}</div>
              )}
              <div className="text-yellow-600 text-xs mt-1.5 font-medium">
                {'★'.repeat(Math.round(marker.rating))} {marker.rating.toFixed(1)}
              </div>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  )
}
