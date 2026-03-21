import React from 'react';

interface ChangelogEntry {
  version: string;
  date: string;
  isNew?: boolean;
  title: string;
  description: string;
  highlights: string[];
}

const changelogData: ChangelogEntry[] = [
  {
    version: 'v1.4.0',
    date: '2026. márc. 21.',
    isNew: true,
    title: 'Biztonsági Audit & CI Javítások',
    description:
      'Átfogó biztonsági felülvizsgálat és CI pipeline javítások.',
    highlights: [
      'MyPy hibák javítása (27 → 0)',
      'Bandit biztonsági figyelmeztetések elhárítása (8 HIGH → 0)',
      'MD5 cache kulcsok biztonságossá tétele',
    ],
  },
  {
    version: 'v1.3.0',
    date: '2026. márc. 20.',
    title: '4 Új Funkció — Chat, Kalkulátor, Szervizek, Műszaki Vizsga',
    description:
      'Négy teljesen új modul az AI diagnosztikai platformhoz.',
    highlights: [
      'AI Chat asszisztens magyar nyelven',
      'Jármű értékbecslő kalkulátor',
      'Szerviz-összehasonlító térkép',
      'Műszaki vizsga előkészítő',
    ],
  },
  {
    version: 'v1.2.0',
    date: '2026. márc. 1.',
    title: 'Demo Hiba Szimuláció & Valós Árak',
    description:
      'P0300 szimulációs demó oldal valós alkatrészárakkal.',
    highlights: [
      'VW Golf VII 1.4 TSI P0300 demo szimuláció',
      'Valós árak: Bárdi Autó, Uni Autó, AUTODOC',
      '6 alkatrész bolt-specifikus árösszehasonlítás',
    ],
  },
  {
    version: 'v1.1.0',
    date: '2026. feb. 9.',
    title: 'Bővített Diagnosztikai Jelentés',
    description:
      'LLM prompt újraírás és alkatrészár integráció.',
    highlights: [
      'PartsPriceService integráció a diagnosis pipeline-ba',
      'Mérési értékek és szerszám ajánlások',
      'Gyökérok elemzés szekció',
    ],
  },
  {
    version: 'v1.0.0',
    date: '2026. feb. 8.',
    title: 'Platform Indulás — Adatbázis Feltöltés',
    description:
      'Első stabil kiadás a teljes adatbázis háttérrel.',
    highlights: [
      '26,816 Neo4j node (járművek, DTC, visszahívások)',
      '35,000+ Qdrant vektor (HuBERT embeddings)',
      'NHTSA API integráció',
    ],
  },
];

const ChangelogPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div className="pt-16 pb-12 text-center px-4">
        <h1 className="font-serif text-5xl md:text-6xl text-[#1a1a1a] mb-4">
          Changelog
        </h1>
        <p className="text-[#666] text-lg max-w-xl mx-auto leading-relaxed">
          Folyamatosan fejlesztjük a platformot. Nézd meg a legújabb funkciók
          listáját!
        </p>
      </div>

      {/* Timeline */}
      <div className="max-w-3xl mx-auto px-4 pb-24">
        <div className="relative">
          {/* Vertical dashed line */}
          <div
            className="absolute left-[15px] md:left-[19px] top-2 bottom-0 w-0"
            style={{
              borderLeft: '2px dashed #d4c9be',
            }}
          />

          <div className="space-y-10">
            {changelogData.map((entry) => (
              <div key={entry.version} className="relative flex gap-6 md:gap-8">
                {/* Timeline circle */}
                <div className="relative z-10 flex-shrink-0 mt-1">
                  <div
                    className="w-[32px] h-[32px] md:w-[40px] md:h-[40px] rounded-full flex items-center justify-center"
                    style={{ backgroundColor: '#E8654A' }}
                  >
                    <div className="w-[12px] h-[12px] md:w-[14px] md:h-[14px] rounded-full bg-white" />
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  {/* Version, date, badge row */}
                  <div className="flex flex-wrap items-center gap-3 mb-3">
                    <span
                      className="text-lg font-bold"
                      style={{ color: '#E8654A' }}
                    >
                      {entry.version}
                    </span>
                    <span className="text-sm text-[#888]">{entry.date}</span>
                    {entry.isNew && (
                      <span
                        className="text-xs font-semibold px-3 py-0.5 rounded-full border"
                        style={{
                          color: '#E8654A',
                          borderColor: '#E8654A',
                        }}
                      >
                        Új Verzió
                      </span>
                    )}
                  </div>

                  {/* Card */}
                  <div
                    className="rounded-xl p-5 md:p-6"
                    style={{ backgroundColor: '#faf5f0' }}
                  >
                    <h3 className="text-[#1a1a1a] font-bold text-base md:text-lg mb-2">
                      {entry.title}
                    </h3>
                    <p className="text-[#555] text-sm md:text-base mb-4 leading-relaxed">
                      {entry.description}
                    </p>
                    <ul className="space-y-1.5">
                      {entry.highlights.map((item, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2 text-sm md:text-base text-[#333]"
                        >
                          <span
                            className="mt-[7px] flex-shrink-0 w-1.5 h-1.5 rounded-full"
                            style={{ backgroundColor: '#E8654A' }}
                          />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChangelogPage;
