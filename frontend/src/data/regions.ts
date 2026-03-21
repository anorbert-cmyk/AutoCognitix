/**
 * Hungarian Regions Data
 *
 * Budapest + 19 counties (megyék) with geographic center coordinates.
 * Used for the service shop comparison region selector and map view.
 */

export interface RegionData {
  id: string
  name: string
  county: string
  lat: number
  lng: number
}

// Hungary geographic center
export const HUNGARY_CENTER = {
  lat: 47.1625,
  lng: 19.5033,
} as const

export const HUNGARY_DEFAULT_ZOOM = 7

export const regions: RegionData[] = [
  { id: 'budapest', name: 'Budapest', county: 'Budapest', lat: 47.4979, lng: 19.0402 },
  { id: 'baranya', name: 'Baranya', county: 'Baranya megye', lat: 46.0727, lng: 18.2323 },
  { id: 'bacs-kiskun', name: 'B\u00E1cs-Kiskun', county: 'B\u00E1cs-Kiskun megye', lat: 46.5962, lng: 19.3610 },
  { id: 'bekes', name: 'B\u00E9k\u00E9s', county: 'B\u00E9k\u00E9s megye', lat: 46.7736, lng: 21.0877 },
  { id: 'borsod-abauj-zemplen', name: 'Borsod-Aba\u00FAj-Zempl\u00E9n', county: 'Borsod-Aba\u00FAj-Zempl\u00E9n megye', lat: 48.1035, lng: 20.7784 },
  { id: 'csongrad-csanad', name: 'Csongr\u00E1d-Csan\u00E1d', county: 'Csongr\u00E1d-Csan\u00E1d megye', lat: 46.4167, lng: 20.1500 },
  { id: 'fejer', name: 'Fej\u00E9r', county: 'Fej\u00E9r megye', lat: 47.1896, lng: 18.4104 },
  { id: 'gyor-moson-sopron', name: 'Gy\u0151r-Moson-Sopron', county: 'Gy\u0151r-Moson-Sopron megye', lat: 47.6849, lng: 17.2344 },
  { id: 'hajdu-bihar', name: 'Hajd\u00FA-Bihar', county: 'Hajd\u00FA-Bihar megye', lat: 47.5316, lng: 21.6273 },
  { id: 'heves', name: 'Heves', county: 'Heves megye', lat: 47.8931, lng: 20.3727 },
  { id: 'jasz-nagykun-szolnok', name: 'J\u00E1sz-Nagykun-Szolnok', county: 'J\u00E1sz-Nagykun-Szolnok megye', lat: 47.1621, lng: 20.1825 },
  { id: 'komarom-esztergom', name: 'Kom\u00E1rom-Esztergom', county: 'Kom\u00E1rom-Esztergom megye', lat: 47.5561, lng: 18.4037 },
  { id: 'nograd', name: 'N\u00F3gr\u00E1d', county: 'N\u00F3gr\u00E1d megye', lat: 48.0000, lng: 19.5000 },
  { id: 'pest', name: 'Pest', county: 'Pest megye', lat: 47.4100, lng: 19.2600 },
  { id: 'somogy', name: 'Somogy', county: 'Somogy megye', lat: 46.3542, lng: 17.7589 },
  { id: 'szabolcs-szatmar-bereg', name: 'Szabolcs-Szatm\u00E1r-Bereg', county: 'Szabolcs-Szatm\u00E1r-Bereg megye', lat: 48.1, lng: 21.95 },
  { id: 'tolna', name: 'Tolna', county: 'Tolna megye', lat: 46.6170, lng: 18.5500 },
  { id: 'vas', name: 'Vas', county: 'Vas megye', lat: 47.0867, lng: 16.7250 },
  { id: 'veszprem', name: 'Veszpr\u00E9m', county: 'Veszpr\u00E9m megye', lat: 47.1028, lng: 17.9093 },
  { id: 'zala', name: 'Zala', county: 'Zala megye', lat: 46.8400, lng: 16.8416 },
]

/**
 * Get region by ID
 */
export function getRegionById(id: string): RegionData | undefined {
  return regions.find((r) => r.id === id)
}

/**
 * Get region display name
 */
export function getRegionDisplayName(id: string): string {
  const region = getRegionById(id)
  return region ? region.name : id
}
