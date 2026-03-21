import { useState } from 'react';
import { CheckCircle } from 'lucide-react';

interface FeatureGroup {
  title: string;
  items: string[];
}

const starterFeatures: FeatureGroup[] = [
  {
    title: 'Diagnosztikai Eszközök',
    items: [
      'DTC kód keresés',
      'AI-alapú hibaanalízis',
      'Valós idejű jelentések',
      'Alapvető jármű adatok',
    ],
  },
  {
    title: 'Intelligens Funkciók',
    items: [
      'Egyszerű kezelőfelület',
      'Személyre szabható dashboard',
      'AI-alapú javaslatok',
      'Csapat funkciók',
    ],
  },
  {
    title: 'Felhő Alapú Megoldások',
    items: [
      'Bárhonnan elérhető',
      'Automatikus mentések',
      'Skálázható tárhely',
      'Erős biztonsági protokollok',
    ],
  },
];

const proFeatures: FeatureGroup[] = [
  {
    title: 'Diagnosztikai Eszközök',
    items: [
      'Minden Alap funkció +',
      'Együttműködési eszközök',
      'Egyedi márkajelzés',
      'Prioritásos támogatás',
    ],
  },
  {
    title: 'Intelligens Funkciók',
    items: [
      'Haladó analitika',
      'API hozzáférés',
      'Felhasználói tevékenység követés',
      'Dedikált ügyfélmenedzser',
    ],
  },
  {
    title: 'Továbbfejlesztett Biztonság',
    items: [
      'Kétfaktoros hitelesítés',
      'Adattitkosítás',
      'Megfelelőség-figyelés',
      'Incidenskezelési terv',
    ],
  },
];

const STARTER_MONTHLY = '4.990';
const STARTER_YEARLY = '3.490';
const PRO_MONTHLY = '14.990';
const PRO_YEARLY = '10.490';

export default function PricingPage() {
  const [isYearly, setIsYearly] = useState(false);

  const starterPrice = isYearly ? STARTER_YEARLY : STARTER_MONTHLY;
  const proPrice = isYearly ? PRO_YEARLY : PRO_MONTHLY;

  return (
    <div className="min-h-screen bg-white">
      {/* SECTION 1: Plan Cards */}
      <section className="px-4 py-16 md:py-24">
        <div className="mx-auto max-w-5xl text-center">
          <h1 className="font-serif text-4xl font-bold text-gray-900 md:text-5xl">
            Válaszd ki a legjobb csomagot!
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-gray-500">
            Válaszd ki az igényeidnek megfelelő csomagot, amely jelentősen
            javítja az élményedet.
          </p>

          {/* Toggle */}
          <div className="mt-10 flex items-center justify-center gap-4">
            <span
              className={`text-sm font-medium ${!isYearly ? 'text-gray-900' : 'text-gray-400'}`}
            >
              Havi
            </span>
            <button
              type="button"
              role="switch"
              aria-checked={isYearly}
              onClick={() => setIsYearly(!isYearly)}
              className={`relative inline-flex h-7 w-14 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ${
                isYearly ? 'bg-[#E8654A]' : 'bg-gray-300'
              }`}
              aria-label="Havi/Éves váltás"
            >
              <span
                className={`pointer-events-none inline-block h-6 w-6 transform rounded-full bg-white shadow ring-0 transition-transform duration-200 ${
                  isYearly ? 'translate-x-7' : 'translate-x-0'
                }`}
              />
            </button>
            <span
              className={`text-sm font-medium ${isYearly ? 'text-gray-900' : 'text-gray-400'}`}
            >
              Éves
            </span>
            <span className="rounded-full bg-[#E8654A] px-3 py-1 text-xs font-semibold text-white">
              30% kedvezmény
            </span>
          </div>

          {/* Plan Cards */}
          <div className="mt-12 grid gap-8 md:grid-cols-2">
            {/* Starter Card */}
            <div className="relative rounded-2xl border border-gray-200 bg-[#faf5f0] p-8 text-left">
              <span className="absolute right-6 top-6 rounded-full bg-[#E8654A] px-4 py-1 text-xs font-semibold text-white">
                Legnépszerűbb
              </span>
              <h3 className="font-serif text-2xl font-bold text-gray-900">
                Alap
              </h3>
              <p className="mt-2 text-sm text-gray-500">
                Egyéni felhasználóknak és kis csapatoknak
              </p>
              <div className="mt-6">
                <span className="font-serif text-4xl font-bold text-gray-900">
                  {starterPrice} Ft
                </span>
                <span className="text-gray-500">/hó</span>
              </div>
              <ul className="mt-8 space-y-3">
                {starterFeatures.flatMap((group) =>
                  group.items.map((item) => (
                    <li key={item} className="flex items-start gap-3">
                      <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-[#E8654A]" />
                      <span className="text-sm text-gray-700">{item}</span>
                    </li>
                  )),
                )}
              </ul>
              <button
                type="button"
                onClick={() => { window.location.href = '/auth/register'; }}
                className="mt-8 w-full rounded-lg bg-gray-900 px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-gray-800"
              >
                Választom
              </button>
            </div>

            {/* Professional Card */}
            <div className="rounded-2xl bg-[#1a1a1a] p-8 text-left">
              <h3 className="font-serif text-2xl font-bold text-white">
                Profi
              </h3>
              <p className="mt-2 text-sm text-gray-400">
                Növekvő csapatoknak, akiknek fejlettebb funkciók kellenek
              </p>
              <div className="mt-6">
                <span className="font-serif text-4xl font-bold text-white">
                  {proPrice} Ft
                </span>
                <span className="text-gray-400">/hó</span>
              </div>
              <ul className="mt-8 space-y-3">
                {proFeatures.flatMap((group) =>
                  group.items.map((item) => (
                    <li key={item} className="flex items-start gap-3">
                      <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-green-400" />
                      <span className="text-sm text-gray-300">{item}</span>
                    </li>
                  )),
                )}
              </ul>
              <button
                type="button"
                onClick={() => { window.location.href = '/auth/register'; }}
                className="mt-8 w-full rounded-lg border-2 border-white px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-white hover:text-gray-900"
              >
                Választom
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: Feature Comparison */}
      <section className="bg-white px-4 py-16 md:py-24">
        <div className="mx-auto max-w-5xl text-center">
          <h2 className="font-serif text-4xl font-bold text-gray-900 md:text-5xl">
            Mit kapsz a csomagban?
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-gray-500">
            Bármelyik csomagot is választod, élvezni fogod a platform innovatív
            eszközeit!
          </p>

          {/* Comparison Columns */}
          <div className="mt-12 grid gap-8 md:grid-cols-2">
            {/* Alap Csomag */}
            <div className="rounded-2xl border border-gray-200 bg-[#faf5f0] p-8 text-left">
              <h3 className="font-serif text-2xl font-bold text-gray-900">
                Alap Csomag
              </h3>
              <div className="mt-2">
                <span className="font-serif text-3xl font-bold text-gray-900">
                  {starterPrice} Ft
                </span>
                <span className="text-gray-500">/hó</span>
              </div>
              <div className="mt-8 space-y-8">
                {starterFeatures.map((group) => (
                  <div key={group.title}>
                    <h4 className="mb-3 text-sm font-bold uppercase tracking-wide text-gray-900">
                      {group.title}
                    </h4>
                    <ul className="space-y-2">
                      {group.items.map((item) => (
                        <li key={item} className="flex items-start gap-3">
                          <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-[#E8654A]" />
                          <span className="text-sm text-gray-700">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>

            {/* Profi Csomag */}
            <div className="rounded-2xl border border-gray-200 bg-[#faf5f0] p-8 text-left">
              <h3 className="font-serif text-2xl font-bold text-gray-900">
                Profi Csomag
              </h3>
              <div className="mt-2">
                <span className="font-serif text-3xl font-bold text-gray-900">
                  {proPrice} Ft
                </span>
                <span className="text-gray-500">/hó</span>
              </div>
              <div className="mt-8 space-y-8">
                {proFeatures.map((group) => (
                  <div key={group.title}>
                    <h4 className="mb-3 text-sm font-bold uppercase tracking-wide text-gray-900">
                      {group.title}
                    </h4>
                    <ul className="space-y-2">
                      {group.items.map((item) => (
                        <li key={item} className="flex items-start gap-3">
                          <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-[#E8654A]" />
                          <span className="text-sm text-gray-700">{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
