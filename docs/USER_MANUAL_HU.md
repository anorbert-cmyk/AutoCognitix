# AutoCognitix Felhasznaloi Kezikonyv

**AI-alapu jarmudisagnosztikai platform magyar nyelvtamogatassal**

## Tartalomjegyzek
- [Bevezetes](#bevezetes)
- [Regisztracio es bejelentkezes](#regisztracio-es-bejelentkezes)
- [DTC kod kereses](#dtc-kod-kereses)
- [Diagnosztika inditasa](#diagnosztika-inditasa)
- [Eredmenyek ertelmezese](#eredmenyek-ertelmezese)
- [Elozmenyek kezelese](#elozmenyek-kezelese)
- [Jarmuinformaciok](#jarmuinformaciok)
- [Gyakori kerdesek](#gyakori-kerdesek)

---

## Bevezetes

Az AutoCognitix egy mestersges intelligenciaval tamogatott jarmudisagnosztikai platform, amely segit a jarmuhibak azonositasaban es megoldasaban. A platform teljes magyar nyelvtamogatast nyujt, es nem igenyel specialis diagnosztikai hardvert - a hibakodokat es tuneteket kezzel adhatja meg.

### Fo funkciok

- **DTC kod kereses**: Atfogo OBD-II hibakod adatbazis magyar leirasokkal
- **AI diagnosztika**: Mestersges intelligenciaval tamogatott hibaelelmzes
- **Tunetleiras**: Termeszetes nyelvu tunetfeldolgozas magyarul
- **Javitasi javaslatok**: Reszletes javitasi utmutatok koltsegbecslessel
- **Jarmuinformaciok**: VIN dekodolas, visszahivasok, panaszok

---

## Regisztracio es bejelentkezes

### Regisztracio

1. Nyissa meg a platform fooldal (https://your-domain.com)
2. Kattintson a **"Regisztracio"** gombra
3. Toltse ki az adatokat:
   - **Email cim**: Ervenyes email cim (kesobb megerositesre kerul)
   - **Jelszo**: Minimum 8 karakter, tartalmazzon nagybetut, kisbetut, szamot es specialis karaktert
   - **Teljes nev**: Opcionalis
4. Kattintson a **"Regisztracio"** gombra

### Bejelentkezes

1. Kattintson a **"Bejelentkezes"** gombra
2. Adja meg az email cimet es jelszot
3. Kattintson a **"Bejelentkezes"** gombra

### Elfelejtett jelszo

1. Kattintson a **"Elfelejtett jelszo"** linkre a bejelentkezesi oldalon
2. Adja meg a regisztralt email cimet
3. Kovesse az emailben kapott utasitasokat

---

## DTC kod kereses

A DTC (Diagnostic Trouble Code) kereso segitsegevel gyorsan megtalalja a hibakodok jelenteset.

### Kereses inditasa

1. Navigaljon a **"DTC kereses"** oldalra
2. Irja be a hibakodot vagy a hiba leidasat a keresomezobe
3. Valasszon opcionalis szuruket:
   - **Kategoria**: Hajtaslanc (P), Karosszeria (B), Alvaz (C), Halozat (U)
   - **Gyarto**: Gyartospecifikus kodok szurese

### Hibakod formatumok

| Prefix | Kategoria | Pelda | Leiras |
|--------|-----------|-------|--------|
| P | Hajtaslanc | P0101 | Motor, valto, emissziu |
| B | Karosszeria | B1234 | Legzsakok, klima, vilagitas |
| C | Alvaz | C0567 | ABS, kormanyzas, felfuggesztes |
| U | Halozat | U0100 | Kommunikacio, vezerlok |

### Reszletes informaciok megtekintese

Kattintson egy hibakodra a reszletes informaciok megtekinteshez:

- **Leiras**: Magyar es angol nyelvu leiras
- **Tunetek**: Tipikus tunetek listaja
- **Lehetseges okok**: Gyakori okok felsorolasa
- **Diagnosztikai lepesek**: Javasolt ellenorzesi lepesek
- **Kapcsolodo kodok**: Osszetartozo hibakodok

---

## Diagnosztika inditasa

A fo diagnosztikai funkcio AI segitsegevel elemzi a jarmuproblemakat.

### 1. lepes: Jarmuadatok megadasa

Toltse ki a jarmu alapadatait:

- **Gyarto** (kotelezo): Pl. Volkswagen, BMW, Toyota
- **Modell** (kotelezo): Pl. Golf, 3-as sorozat, Corolla
- **Evjarat** (kotelezo): 1980-2030 kozotti ev
- **Motor** (opcionalis): Pl. 2.0 TSI, 2.0d
- **VIN** (opcionalis): 17 karakteres jarmuazonisito szam

### 2. lepes: Hibakodok megadasa

Adja meg az olvasott hibakodokat:

- Irjon be 1-20 hibakodot
- Hasznaljon standard formatumot (pl. P0101)
- A rendszer ellenorzi a kodok ervenysesget
- Az automatikus kiegeszites segit a helyes kodok megadasaban

**Pelda hibakodok:**
```
P0101, P0171
```

### 3. lepes: Tunetek leirasa

Irja le a tapasztalt tuneteket magyarul:

**Pelda:**
> "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas megott. A problema telen rosszabb, es neha Check Engine lampa is felgyullad."

**Tippek a jo tunetleirashoz:**
- Legyen konkreet es reszletes
- Emlitse meg, mikor jelentkezik a hiba (hidegben, melegedve, terhelesnel)
- Irja le az esetleges hangokat, rezgeseket
- Emlitse meg, mikor kezdodott a problema

### 4. lepes: Diagnosztika futtatasa

1. Ellenorizze a megadott adatokat
2. Kattintson az **"Elemzes inditasa"** gombra
3. Varjon az eredmenyre (altalaban 10-30 masodperc)

---

## Eredmenyek ertelmezese

### Diagnozis attekintese

A diagnozis eredmeny tartalmazza:

#### Valoszinu okok

Rangsorolt lista a lehetseges hibaokkrol:

| Mezo | Leiras |
|------|--------|
| **Cim** | Rovid megnevezes |
| **Leiras** | Reszletes magyarazat magyarul |
| **Megbizhatosag** | 0-100% kozotti ertek |
| **Kapcsolodo kodok** | A problemahoz kapcsolodo DTC kodok |
| **Erintett alkatreszek** | Lehetseges hibas komponensek |

**Megbizhatosagi szintek:**
- **80-100%**: Nagyon valoszinu ok
- **60-80%**: Valoszinu ok
- **40-60%**: Lehetseges ok
- **0-40%**: Kevesbe valoszinu, de ellenorizendo

#### Javitasi javaslatok

Minden valoszinu okhoz tartoznak javitasi javaslatok:

| Mezo | Leiras |
|------|--------|
| **Cim** | Javitas megnevezese |
| **Leiras** | Reszletes javitasi utmutato |
| **Becsult koltseg** | Minimum-maximum tartomany (HUF) |
| **Nehezsegi szint** | Kezdo, kozephalado, halado, szakember |
| **Szukseges alkatreszek** | Lista a szukseges reszekkrol |
| **Becsult ido** | Munkaido percben |

**Nehezsegi szintek:**
- **Kezdo**: Otthon elvegezhet, alap szerszamokkal
- **Kozephalado**: Tapasztalat es jobb szerszamok szuksegesek
- **Halado**: Szakertelmet igenyel
- **Szakember**: Szervizben kell elvegezni

#### Forrasok

A diagnozis alapjat kepezo informaciok:

- **Adatbazis**: OBD-II szabvany adatbazis
- **Forum**: Kozossegi tapasztalatok
- **Video**: Oktatovideok
- **Gyari**: Gyartoi technikai bulletinek

---

## Elozmenyek kezelese

### Elozmenyek megtekintese

1. Navigaljon az **"Elozmenyek"** oldalra
2. Lathatja az osszes korabbi diagnozist

### Szures es kereses

Szurje az elozmenyeket:
- **Gyarto**: Pl. Volkswagen
- **Modell**: Pl. Golf
- **Evjarat**: Pl. 2018
- **Hibakod**: Pl. P0101
- **Datum**: Idotartomany valasztasa

### Statisztikak

A statisztikak oldalon lathatja:
- Osszes diagnozis szama
- Atlagos megbizhatosagi szint
- Legtobbet diagnosztizalt jarmuvek
- Leggyakoribb hibakodok
- Havi bontasu diagnozisszam

### Diagnozis torlese

1. Nyissa meg a diagnoszt az elozmenyekbol
2. Kattintson a **"Torles"** gombra
3. Erositse meg a torlest

**Megjegyzes:** A torles "soft delete", igy az adatok kesobb visszaallithatoak.

---

## Jarmuinformaciok

### VIN dekodolas

A VIN (Vehicle Identification Number) dekodolas segit azonositani a jarmut.

1. Navigaljon a **"VIN dekodolas"** oldalra
2. Irja be a 17 karakteres VIN-t
3. Kattintson a **"Dekodolas"** gombra

**Eredmeny:**
- Gyarto es modell
- Evjarat
- Motorvaltozat
- Sebessegvalto tipusa
- Hajtaslanc
- Karosszeria tipusa
- Uzemanyag tipusa
- Gyartasi hely

### Visszahivasok lekerdezese

1. Adja meg a jarmu adatait (gyarto, modell, evjarat)
2. Kattintson a **"Visszahivasok"** gombra
3. Lathatja a hivatalos visszahivasokat (NHTSA adatbazis)

**Fontos:** A visszahivasok foleg amerikai piacra vonatkoznak.

### Panaszok megtekintese

1. Adja meg a jarmu adatait
2. Kattintson a **"Panaszok"** gombra
3. Lathatja a bejelentett panaszokat

---

## Gyakori kerdesek

### Altalanos

**K: Kell diagnosztikai eszkoz a platform hasznalathoz?**
V: Nem, a hibakodokat es tuneteket kezzel adhatja meg. Ha van OBD-II olvasoja, onnan masolhatja be a kodokat.

**K: Milyen pontosak a diagnozisok?**
V: A diagnozisok AI-alapuak es adatbazisra tamaszkodnak. Minden esetben szakember altal torteno ellenorzest javaslunk a javitas elott.

**K: Ingyenes a platform?**
V: Az alapfunkciok ingyenesek. Bovitett funkciok elofizeteses modellben erhetek el.

### Diagnosztika

**K: Miert kap alacsony megbizhatosagi szintet a diagnozis?**
V: Az alacsony megbizhatosag okai lehetnek:
- Nem eleg reszletes tunetleiras
- Ritka vagy egyedi hiba
- Tobb lehetseges ok egyforma valoszinuseggel

**K: Mit tegyek, ha a diagnozis nem segit?**
V:
1. Adjon meg reszletesebb tunetleirast
2. Ellenorizze, hogy a hibakodok helyesek-e
3. Keressen fel szakszerviz

**K: Letarolodnak a diagnosisaim?**
V: Igen, bejelentkezett felhasznaloknal minden diagnozis mentesre kerul az elozmenyekbe.

### Hibakodok

**K: Mi a kulonbseg a generikus es gyartospecifikus kodok kozott?**
V:
- **Generikus (P0xxx)**: Szabvanyos OBD-II kodok, minden jarmure ervenysek
- **Gyartospecifikus (P1xxx)**: Adott gyarto altal definialt kodok

**K: A hibakod torlese megoldja a problemat?**
V: Nem, a hibakod torlese csak az eszkoz memorjajabol torli a jelzest. A moge\tes problema megmarad.

### Felhasznaloi fiok

**K: Hogyan torhetem a fiokomat?**
V: Keresse meg a support csapatot email-ben a fiok torlesehez.

**K: Elfelejtette jelszavam, es nincs hozzaferesem az emailhez.**
V: Ebben az esetben keresse a support csapatot szemelyes azonositas utan.

---

## Tamogatas

Ha segitesre van szuksege:

- **Email**: support@autocognitix.hu
- **Dokumentacio**: https://docs.autocognitix.hu

---

## Verziotortenet

| Verzio | Datum | Ujdonsagok |
|--------|-------|------------|
| 0.1.0 | 2024-02 | Elso kiadás |
