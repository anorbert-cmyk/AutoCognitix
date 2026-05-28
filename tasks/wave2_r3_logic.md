# Wave2 R3 — Logic

## A) Alias teljesség
- severity: LOW (stylistic)
- finding: A hiányzónak tűnő AMG/SRT/VTR egyike sem önálló NHTSA brand —
  AMG recall-ok "Mercedes-Benz", SRT-k "Dodge"/"Chrysler" alatt indexálódnak
  az NHTSA adatbázisban, a VTR Citroen trim, nem márka. A coverage tehát teljes.
  A "smart" → "smart" (lowercase) szándékos és helyes: a Mercedes-Benz Group
  hivatalos brand-styleguide-ja szerint a márkanév csupa kisbetűs, és az NHTSA
  is ezt használja — a `.title()` fallback "Smart"-ot adna, ami 0 hit-et hozna.
  Az új self-alias-ok (BMW/GMC/MG/DS/FCA/KIA) a fő .title()-érzékeny acronymokat
  lefedik. Apróság: a "KIA" valójában title-case ("Kia") NHTSA-ban, így a
  self-alias defensive de nem szükséges — `.title()` amúgy is "Kia"-t ad.

## B) RETURNING semantika
- severity: LOW (style)
- finding: A `bind.execute(sa.text("DELETE ... RETURNING id"))` SQLAlchemy 2.0-ban
  `CursorResult`-ot ad vissza, amelyet iterálva `Row` objektumokat kapunk —
  `row[0]` indexelés helyesen működik, a listcomp **funkcionálisan korrekt**.
  Tisztább alternatíva lenne `result.scalars().all()` (egyszeri oszlop esetén
  idiomatikusabb és Pylance-friendly), majd `[str(uid) for uid in ids]`, de
  ez stilisztikai preferencia. Nincs runtime bug. UUID `str()` konverzió is OK,
  mivel `users.id` UUID típusú és str()-ben formázódik a print-hez.
