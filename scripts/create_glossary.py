#!/usr/bin/env python3
"""
Hungarian Automotive Terminology Glossary Generator.

Creates and manages a comprehensive trilingual (HU/EN/DE) automotive glossary
for use in translation validation, symptom parsing, and search.

Usage:
    python scripts/create_glossary.py --generate    # Generate full glossary
    python scripts/create_glossary.py --search "motor"  # Search terms
    python scripts/create_glossary.py --stats       # Show statistics
    python scripts/create_glossary.py --validate    # Validate glossary
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLOSSARY_DIR = PROJECT_ROOT / "data" / "glossary"
GLOSSARY_FILE = GLOSSARY_DIR / "automotive_glossary_hu.json"
INDEX_FILE = GLOSSARY_DIR / "glossary_index.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GlossaryTerm:
    """Represents a single glossary term."""

    def __init__(
        self,
        hungarian: str,
        english: str,
        german: str = "",
        definition: str = "",
        abbreviations: List[str] = None,
        category: str = "general",
        subcategory: str = "",
        tags: List[str] = None,
        related_terms: List[str] = None,
    ):
        self.hungarian = hungarian
        self.english = english
        self.german = german
        self.definition = definition
        self.abbreviations = abbreviations or []
        self.category = category
        self.subcategory = subcategory
        self.tags = tags or []
        self.related_terms = related_terms or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hungarian": self.hungarian,
            "english": self.english,
            "german": self.german,
            "definition": self.definition,
            "abbreviations": self.abbreviations,
            "category": self.category,
            "subcategory": self.subcategory,
            "tags": self.tags,
            "related_terms": self.related_terms,
        }


class AutomotiveGlossary:
    """Manages the automotive terminology glossary."""

    CATEGORIES = [
        "engine",
        "transmission",
        "electrical",
        "brake",
        "suspension",
        "body",
        "tools",
        "specifications",
        "dtc",
        "symptoms",
        "repairs",
        "slang",
    ]

    def __init__(self):
        self.terms: Dict[str, GlossaryTerm] = {}
        self.index: Dict[str, Set[str]] = defaultdict(set)

    def add_term(self, term: GlossaryTerm) -> None:
        """Add a term to the glossary."""
        key = term.hungarian.lower()
        self.terms[key] = term

        # Index by all languages
        self._index_term(term.hungarian, key)
        self._index_term(term.english, key)
        if term.german:
            self._index_term(term.german, key)
        for abbr in term.abbreviations:
            self._index_term(abbr, key)

    def _index_term(self, text: str, key: str) -> None:
        """Index a term for searching."""
        words = re.findall(r'\w+', text.lower())
        for word in words:
            self.index[word].add(key)

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for terms matching the query."""
        query_lower = query.lower()
        words = re.findall(r'\w+', query_lower)

        # Find matching keys
        matching_keys: Set[str] = set()
        for word in words:
            if word in self.index:
                matching_keys.update(self.index[word])

        # Also check exact matches
        for key, term in self.terms.items():
            if (query_lower in term.hungarian.lower() or
                query_lower in term.english.lower() or
                query_lower in term.german.lower()):
                matching_keys.add(key)

        # Return results
        results = []
        for key in list(matching_keys)[:limit]:
            if key in self.terms:
                results.append(self.terms[key].to_dict())

        return results

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all terms in a category."""
        return [
            term.to_dict()
            for term in self.terms.values()
            if term.category == category
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get glossary statistics."""
        category_counts = defaultdict(int)
        for term in self.terms.values():
            category_counts[term.category] += 1

        return {
            "total_terms": len(self.terms),
            "categories": dict(category_counts),
            "index_words": len(self.index),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export glossary to dictionary."""
        return {
            "metadata": {
                "version": "2.0",
                "language": "hu",
                "languages": ["hu", "en", "de"],
                "description": "Comprehensive Hungarian automotive terminology glossary",
                "created": datetime.now(timezone.utc).isoformat(),
                "terms_count": len(self.terms),
                "categories": self.CATEGORIES,
            },
            "terms": {k: v.to_dict() for k, v in self.terms.items()},
        }

    def save(self, filepath: Path = GLOSSARY_FILE) -> None:
        """Save glossary to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        # Save index
        index_data = {k: list(v) for k, v in self.index.items()}
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(self.terms)} terms to {filepath}")

    @classmethod
    def load(cls, filepath: Path = GLOSSARY_FILE) -> "AutomotiveGlossary":
        """Load glossary from JSON file."""
        glossary = cls()

        if not filepath.exists():
            logger.warning(f"Glossary file not found: {filepath}")
            return glossary

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, term_data in data.get("terms", {}).items():
            term = GlossaryTerm(**term_data)
            glossary.terms[key] = term

        # Rebuild index
        for key, term in glossary.terms.items():
            glossary._index_term(term.hungarian, key)
            glossary._index_term(term.english, key)
            if term.german:
                glossary._index_term(term.german, key)
            for abbr in term.abbreviations:
                glossary._index_term(abbr, key)

        logger.info(f"Loaded {len(glossary.terms)} terms from {filepath}")
        return glossary


def generate_engine_terms() -> List[GlossaryTerm]:
    """Generate engine-related terms."""
    return [
        # Core engine components
        GlossaryTerm("motor", "engine", "Motor", "A jarmű hajtását biztosító erőforrás", ["MOT"], "engine", "core"),
        GlossaryTerm("hengerblokk", "engine block", "Motorblock", "A motor fő szerkezeti eleme, amely a hengereket tartalmazza", [], "engine", "core"),
        GlossaryTerm("hengerfej", "cylinder head", "Zylinderkopf", "A hengerblokk felső záró eleme, tartalmazza a szelepeket", [], "engine", "core"),
        GlossaryTerm("dugattyú", "piston", "Kolben", "A hengerben mozgó elem, amely a nyomást hajtóerővé alakítja", [], "engine", "core"),
        GlossaryTerm("hajtókar", "connecting rod", "Pleuelstange", "A dugattyút a főtengellyel összekötő elem", [], "engine", "core"),
        GlossaryTerm("főtengely", "crankshaft", "Kurbelwelle", "A dugattyúk alternáló mozgását forgó mozgássá alakító tengely", [], "engine", "core"),
        GlossaryTerm("lendkerék", "flywheel", "Schwungrad", "A motor egyenletes járását biztosító tehetetlenségi tömeg", [], "engine", "core"),
        GlossaryTerm("vezérműtengely", "camshaft", "Nockenwelle", "A szelepek nyitását és zárását vezérlő tengely", [], "engine", "core"),
        GlossaryTerm("szelep", "valve", "Ventil", "A henger be- és kiáramlását szabályozó elem", [], "engine", "valvetrain"),
        GlossaryTerm("szívószelep", "intake valve", "Einlassventil", "A levegő-üzemanyag keverék beengedését szabályozó szelep", [], "engine", "valvetrain"),
        GlossaryTerm("kipufogószelep", "exhaust valve", "Auslassventil", "Az égéstermékek kiengedését szabályozó szelep", [], "engine", "valvetrain"),
        GlossaryTerm("szelepszár", "valve stem", "Ventilschaft", "A szelep nyél része", [], "engine", "valvetrain"),
        GlossaryTerm("szeleptányér", "valve head", "Ventilteller", "A szelep tömítő felülete", [], "engine", "valvetrain"),
        GlossaryTerm("szelepvezető", "valve guide", "Ventilführung", "A szelep mozgását vezető persely", [], "engine", "valvetrain"),
        GlossaryTerm("szeleprugó", "valve spring", "Ventilfeder", "A szelep zárását biztosító rugó", [], "engine", "valvetrain"),
        GlossaryTerm("szelepülék", "valve seat", "Ventilsitz", "A szelep tömítő felülete a hengerfejben", [], "engine", "valvetrain"),
        GlossaryTerm("bütykös tengely", "camshaft", "Nockenwelle", "Vezérműtengely más elnevezése", [], "engine", "valvetrain"),
        GlossaryTerm("vezérműszíj", "timing belt", "Zahnriemen", "A vezérmű és főtengely szinkronizálását biztosító szíj", [], "engine", "timing"),
        GlossaryTerm("vezérműlánc", "timing chain", "Steuerkette", "A vezérmű és főtengely szinkronizálását biztosító lánc", [], "engine", "timing"),
        GlossaryTerm("feszítőgörgő", "tensioner pulley", "Spannrolle", "A szíj vagy lánc feszességét biztosító görgő", [], "engine", "timing"),
        GlossaryTerm("vezetőgörgő", "idler pulley", "Umlenkrolle", "A szíj vezetését segítő görgő", [], "engine", "timing"),
        GlossaryTerm("vízpumpa", "water pump", "Wasserpumpe", "A hűtőfolyadék keringetését végző szivattyú", [], "engine", "cooling"),
        GlossaryTerm("hűtő", "radiator", "Kühler", "A hűtőfolyadék hűtését végző hőcserélő", [], "engine", "cooling"),
        GlossaryTerm("termosztát", "thermostat", "Thermostat", "A hűtőfolyadék hőmérsékletét szabályozó elem", [], "engine", "cooling"),
        GlossaryTerm("hűtőventillátor", "cooling fan", "Kühlerlüfter", "A hűtő hűtését segítő ventilátor", [], "engine", "cooling"),
        GlossaryTerm("hűtőfolyadék", "coolant", "Kühlmittel", "A motor hűtését végző folyadék", [], "engine", "cooling"),
        GlossaryTerm("fagyálló", "antifreeze", "Frostschutzmittel", "A hűtőfolyadék fagyását megakadályozó adalék", [], "engine", "cooling"),
        GlossaryTerm("kiegyenlítőtartály", "expansion tank", "Ausgleichsbehälter", "A hűtőfolyadék tágulását kompenzáló tartály", [], "engine", "cooling"),
        GlossaryTerm("hűtőtömlő", "radiator hose", "Kühlerschlauch", "A hűtőrendszer elemeit összekötő tömlő", [], "engine", "cooling"),
        GlossaryTerm("olajszivattyú", "oil pump", "Ölpumpe", "A motorolaj keringetését végző szivattyú", [], "engine", "lubrication"),
        GlossaryTerm("olajszűrő", "oil filter", "Ölfilter", "A motorolaj szűrését végző elem", [], "engine", "lubrication"),
        GlossaryTerm("olajteknő", "oil pan", "Ölwanne", "A motorolaj tárolására szolgáló tartály", [], "engine", "lubrication"),
        GlossaryTerm("olajnyomás", "oil pressure", "Öldruck", "A kenőolaj nyomása a rendszerben", [], "engine", "lubrication"),
        GlossaryTerm("motorolaj", "engine oil", "Motoröl", "A motor kenését biztosító olaj", [], "engine", "lubrication"),
        GlossaryTerm("olajpálca", "dipstick", "Ölmessstab", "Az olajszint ellenőrzésére szolgáló pálca", [], "engine", "lubrication"),
        GlossaryTerm("főcsapágy", "main bearing", "Hauptlager", "A főtengely csapágya", [], "engine", "bearings"),
        GlossaryTerm("hajtókarcsapágy", "rod bearing", "Pleuellager", "A hajtókar csapágya", [], "engine", "bearings"),
        GlossaryTerm("tengelykapcsoló", "clutch", "Kupplung", "A motor és váltó közötti erőátvitelt megszakító szerkezet", [], "engine", "drivetrain"),
        GlossaryTerm("kuplung", "clutch", "Kupplung", "Tengelykapcsoló köznapi elnevezése", [], "engine", "drivetrain"),
        GlossaryTerm("kuplungtárcsa", "clutch disc", "Kupplungsscheibe", "A tengelykapcsoló súrlódó eleme", [], "engine", "drivetrain"),
        GlossaryTerm("nyomócsapágy", "release bearing", "Ausrücklager", "A kuplung kioldását végző csapágy", [], "engine", "drivetrain"),
        GlossaryTerm("szívócsonk", "intake manifold", "Ansaugkrümmer", "A levegőt a hengerekbe vezető csonk", [], "engine", "intake"),
        GlossaryTerm("kipufogócsonk", "exhaust manifold", "Abgaskrümmer", "Az égéstermékeket elvezető csonk", [], "engine", "exhaust"),
        GlossaryTerm("katalizátor", "catalytic converter", "Katalysator", "A káros anyagok átalakítását végző elem", ["KAT"], "engine", "exhaust"),
        GlossaryTerm("kipufogó", "exhaust pipe", "Auspuff", "Az égéstermékek elvezetésére szolgáló cső", [], "engine", "exhaust"),
        GlossaryTerm("hangtompító", "muffler", "Schalldämpfer", "A kipufogó zajának csökkentésére szolgáló elem", [], "engine", "exhaust"),
        GlossaryTerm("lambdaszonda", "oxygen sensor", "Lambdasonde", "A kipufogógáz oxigéntartalmát mérő szenzor", ["O2"], "engine", "sensors"),
        GlossaryTerm("turbófeltöltő", "turbocharger", "Turbolader", "A kipufogógáz energiáját felhasználó feltöltő", ["TURBO"], "engine", "forced_induction"),
        GlossaryTerm("töltőnyomás", "boost pressure", "Ladedruck", "A turbó által létrehozott túlnyomás", [], "engine", "forced_induction"),
        GlossaryTerm("intercooler", "intercooler", "Ladeluftkühler", "A feltöltött levegő hűtésére szolgáló hőcserélő", [], "engine", "forced_induction"),
        GlossaryTerm("kompresszor", "supercharger", "Kompressor", "Mechanikusan hajtott feltöltő", [], "engine", "forced_induction"),
        GlossaryTerm("wastegate", "wastegate", "Wastegate", "A töltőnyomást szabályozó szelep", [], "engine", "forced_induction"),
        GlossaryTerm("befecskendező", "fuel injector", "Einspritzdüse", "Az üzemanyagot a hengerbe juttató elem", ["INJ"], "engine", "fuel"),
        GlossaryTerm("injektor", "injector", "Injektor", "Befecskendező köznapi elnevezése", [], "engine", "fuel"),
        GlossaryTerm("üzemanyag-szivattyú", "fuel pump", "Kraftstoffpumpe", "Az üzemanyag szállítását végző szivattyú", [], "engine", "fuel"),
        GlossaryTerm("üzemanyagszűrő", "fuel filter", "Kraftstofffilter", "Az üzemanyag szűrését végző elem", [], "engine", "fuel"),
        GlossaryTerm("nyomásszabályzó", "pressure regulator", "Druckregler", "A rendszer nyomását szabályozó elem", [], "engine", "fuel"),
        GlossaryTerm("közös nyomócsöves rendszer", "common rail", "Common Rail", "Modern dízel befecskendező rendszer", ["CR"], "engine", "fuel"),
        GlossaryTerm("porlasztó", "nozzle", "Düse", "A befecskendező üzemanyagot porlasztó része", [], "engine", "fuel"),
        GlossaryTerm("fojtószelep", "throttle body", "Drosselklappe", "A levegő mennyiségét szabályozó szelep", [], "engine", "intake"),
        GlossaryTerm("pillangószelep", "butterfly valve", "Drosselklappe", "Fojtószelep más elnevezése", [], "engine", "intake"),
        GlossaryTerm("levegőszűrő", "air filter", "Luftfilter", "A szívott levegő szűrését végző elem", [], "engine", "intake"),
        GlossaryTerm("légtömegmérő", "mass air flow sensor", "Luftmassenmesser", "A beszívott levegő mennyiségét mérő szenzor", ["MAF"], "engine", "sensors"),
        GlossaryTerm("gyújtógyertya", "spark plug", "Zündkerze", "A keverék begyújtását végző elem", [], "engine", "ignition"),
        GlossaryTerm("gyújtótekercs", "ignition coil", "Zündspule", "A gyújtáshoz szükséges nagyfeszültséget előállító elem", [], "engine", "ignition"),
        GlossaryTerm("gyújtáselosztó", "distributor", "Verteiler", "A gyújtást a hengerekhez szétosztó elem", [], "engine", "ignition"),
        GlossaryTerm("izzítógyertya", "glow plug", "Glühkerze", "Dízel motor előmelegítését végző elem", [], "engine", "ignition"),
        GlossaryTerm("fordulatszám", "RPM", "Drehzahl", "A motor percenkénti fordulatszáma", ["RPM"], "engine", "parameters"),
        GlossaryTerm("nyomaték", "torque", "Drehmoment", "A motor által kifejtett forgató erő", ["Nm"], "engine", "parameters"),
        GlossaryTerm("lóerő", "horsepower", "PS", "A motor teljesítményének mértékegysége", ["LE", "HP", "PS"], "engine", "parameters"),
        GlossaryTerm("kilowatt", "kilowatt", "Kilowatt", "Teljesítmény SI mértékegysége", ["kW"], "engine", "parameters"),
        GlossaryTerm("hengerűrtartalom", "displacement", "Hubraum", "A motor összes hengerének térfogata", ["ccm"], "engine", "parameters"),
        GlossaryTerm("kompresszióviszony", "compression ratio", "Verdichtungsverhältnis", "A henger térfogatának aránya", [], "engine", "parameters"),
        GlossaryTerm("alapjárat", "idle", "Leerlauf", "A motor üresjárati fordulatszáma", [], "engine", "parameters"),
        GlossaryTerm("üresjárat", "idle speed", "Leerlaufdrehzahl", "A motor terhelés nélküli fordulatszáma", [], "engine", "parameters"),
        GlossaryTerm("hengersor", "cylinder bank", "Zylinderbank", "V vagy boxer motorok egy oldali hengerei", [], "engine", "core"),
        GlossaryTerm("soros motor", "inline engine", "Reihenmotor", "Sorba rendezett hengerekkel rendelkező motor", [], "engine", "types"),
        GlossaryTerm("V-motor", "V-engine", "V-Motor", "V alakban elrendezett hengerekkel rendelkező motor", [], "engine", "types"),
        GlossaryTerm("boxermotor", "boxer engine", "Boxermotor", "Szemben fekvő hengerekkel rendelkező motor", [], "engine", "types"),
        GlossaryTerm("dízelmotor", "diesel engine", "Dieselmotor", "Kompressziógyújtású motor", [], "engine", "types"),
        GlossaryTerm("benzinmotor", "gasoline engine", "Benzinmotor", "Szikragyújtású motor", [], "engine", "types"),
        GlossaryTerm("hibrid", "hybrid", "Hybrid", "Kombinált hajtású jármű", ["HEV", "PHEV"], "engine", "types"),
        GlossaryTerm("elektromos motor", "electric motor", "Elektromotor", "Elektromos hajtású motor", ["EV"], "engine", "types"),
        GlossaryTerm("EGR szelep", "EGR valve", "AGR-Ventil", "Kipufogógáz visszavezetés szelep", ["EGR"], "engine", "emissions"),
        GlossaryTerm("részecskeszűrő", "particulate filter", "Partikelfilter", "A kipufogógáz szilárd részecskéit kiszűrő elem", ["DPF", "GPF"], "engine", "emissions"),
        GlossaryTerm("SCR rendszer", "SCR system", "SCR-System", "Szelektív katalitikus redukciós rendszer", ["SCR"], "engine", "emissions"),
        GlossaryTerm("AdBlue", "AdBlue", "AdBlue", "Karbamid alapú adalék a NOx csökkentésére", [], "engine", "emissions"),
        GlossaryTerm("szimering", "oil seal", "Wellendichtring", "Forgó tengely tömítése", [], "engine", "seals"),
        GlossaryTerm("tömítés", "gasket", "Dichtung", "Két felület közötti tömítő elem", [], "engine", "seals"),
        GlossaryTerm("hengerfej-tömítés", "head gasket", "Zylinderkopfdichtung", "A hengerblokk és hengerfej közötti tömítés", [], "engine", "seals"),
        GlossaryTerm("szelepfedél-tömítés", "valve cover gasket", "Ventildeckeldichtung", "A szelepfedél tömítése", [], "engine", "seals"),
        GlossaryTerm("olajteknő-tömítés", "oil pan gasket", "Ölwannendichtung", "Az olajteknő tömítése", [], "engine", "seals"),
        GlossaryTerm("gyűrűk", "piston rings", "Kolbenringe", "A dugattyú tömítő és olajlehúzó gyűrűi", [], "engine", "core"),
        GlossaryTerm("dugattyúcsapszeg", "piston pin", "Kolbenbolzen", "A dugattyút a hajtókarral összekötő csapszeg", [], "engine", "core"),
        GlossaryTerm("forgattyúsház", "crankcase", "Kurbelgehäuse", "A főtengelyt befogadó motor alsó része", [], "engine", "core"),
        GlossaryTerm("szelepemelő", "valve lifter", "Ventilstößel", "A bütykös tengely és szelep közötti elem", [], "engine", "valvetrain"),
        GlossaryTerm("hidraulikus szelepemelő", "hydraulic lifter", "Hydrostößel", "Automatikus szelepjáték-beállítású emelő", [], "engine", "valvetrain"),
        GlossaryTerm("kipufogó kollektor", "exhaust header", "Fächerkrümmer", "Teljesítménynövelő kipufogócsonk", [], "engine", "exhaust"),
    ]


def generate_transmission_terms() -> List[GlossaryTerm]:
    """Generate transmission-related terms."""
    return [
        GlossaryTerm("sebességváltó", "transmission", "Getriebe", "Az erőátvitel fokozatait biztosító szerkezet", [], "transmission", "core"),
        GlossaryTerm("váltó", "gearbox", "Getriebe", "Sebességváltó köznapi elnevezése", [], "transmission", "core"),
        GlossaryTerm("manuális váltó", "manual transmission", "Schaltgetriebe", "Kézi kapcsolású sebességváltó", ["MT"], "transmission", "types"),
        GlossaryTerm("automata váltó", "automatic transmission", "Automatikgetriebe", "Automatikus sebességváltó", ["AT"], "transmission", "types"),
        GlossaryTerm("CVT váltó", "CVT transmission", "CVT-Getriebe", "Folyamatosan változtatható áttételű váltó", ["CVT"], "transmission", "types"),
        GlossaryTerm("DSG váltó", "dual clutch transmission", "Doppelkupplungsgetriebe", "Dupla kuplungos automatizált váltó", ["DSG", "DCT"], "transmission", "types"),
        GlossaryTerm("félautomata váltó", "semi-automatic transmission", "Halbautomatikgetriebe", "Kuplung nélküli, de váltáshoz pedál", [], "transmission", "types"),
        GlossaryTerm("fogaskerék", "gear", "Zahnrad", "Az erőátvitelt biztosító fogazott kerék", [], "transmission", "components"),
        GlossaryTerm("fogaskerékpár", "gear pair", "Zahnradpaar", "Két összekapcsolt fogaskerék", [], "transmission", "components"),
        GlossaryTerm("szinkrongyűrű", "synchronizer ring", "Synchronring", "A fokozatok szinkronizálását végző gyűrű", [], "transmission", "components"),
        GlossaryTerm("váltókar", "gear lever", "Schalthebel", "A sebességfokozatok kapcsolására szolgáló kar", [], "transmission", "controls"),
        GlossaryTerm("váltórudazat", "shift linkage", "Schaltgestänge", "A váltókar és váltó közötti mechanikus kapcsolat", [], "transmission", "controls"),
        GlossaryTerm("váltókábel", "shift cable", "Schaltzug", "A váltókar és váltó közötti bowden", [], "transmission", "controls"),
        GlossaryTerm("differenciálmű", "differential", "Differentialgetriebe", "A hajtott kerekek fordulatszám-kiegyenlítése", ["DIFF"], "transmission", "drivetrain"),
        GlossaryTerm("kardántengely", "drive shaft", "Kardanwelle", "A hajtóerőt továbbító tengely", [], "transmission", "drivetrain"),
        GlossaryTerm("féltengelyek", "half shafts", "Antriebswellen", "A differenciálmű és kerekek közötti tengelyek", [], "transmission", "drivetrain"),
        GlossaryTerm("homokinetikus csukló", "CV joint", "Gleichlaufgelenk", "Állandó sebességű csukló", ["CV"], "transmission", "drivetrain"),
        GlossaryTerm("gumiharang", "CV boot", "Achsmanschette", "A homokinetikus csukló védőburkolata", [], "transmission", "drivetrain"),
        GlossaryTerm("váltóolaj", "transmission fluid", "Getriebeöl", "A sebességváltó kenőanyaga", ["ATF"], "transmission", "fluids"),
        GlossaryTerm("ATF olaj", "ATF fluid", "ATF-Öl", "Automata váltó folyadék", ["ATF"], "transmission", "fluids"),
        GlossaryTerm("nyomatékváltó", "torque converter", "Drehmomentwandler", "Az automata váltó hidrodinamikus kapcsolóeleme", [], "transmission", "automatic"),
        GlossaryTerm("reteszelő kuplung", "lock-up clutch", "Wandlerüberbrückungskupplung", "A nyomatékváltó hatásfokát növelő kuplung", [], "transmission", "automatic"),
        GlossaryTerm("bolygómű", "planetary gear set", "Planetengetriebe", "Automata váltók fogaskerék-rendszere", [], "transmission", "automatic"),
        GlossaryTerm("lamellás kuplung", "clutch pack", "Lamellenkupplung", "Automata váltó súrlódó eleme", [], "transmission", "automatic"),
        GlossaryTerm("váltószelep-ház", "valve body", "Ventilgehäuse", "Az automata váltó hidraulikus vezérlése", [], "transmission", "automatic"),
        GlossaryTerm("szelektorkart", "selector lever", "Wählhebel", "Az automata váltó üzemmód-választója", [], "transmission", "automatic"),
        GlossaryTerm("fokozat", "gear ratio", "Übersetzung", "A fogaskerekek áttételi viszonya", [], "transmission", "parameters"),
        GlossaryTerm("végáttétel", "final drive ratio", "Achsübersetzung", "A differenciálmű áttétele", [], "transmission", "parameters"),
        GlossaryTerm("hátrameneti fokozat", "reverse gear", "Rückwärtsgang", "A hátrafelé haladást biztosító fokozat", [], "transmission", "gears"),
        GlossaryTerm("üres fokozat", "neutral", "Leerlauf", "Nincs bekapcsolt fokozat", ["N"], "transmission", "gears"),
        GlossaryTerm("parkoló fokozat", "park", "Parkstufe", "Az automata váltó rögzítő állása", ["P"], "transmission", "gears"),
        GlossaryTerm("összkerékhajtás", "all-wheel drive", "Allradantrieb", "Négykerék-meghajtás", ["AWD", "4WD"], "transmission", "drivetrain"),
        GlossaryTerm("elsőkerék-hajtás", "front-wheel drive", "Vorderradantrieb", "Első kerekek hajtása", ["FWD"], "transmission", "drivetrain"),
        GlossaryTerm("hátsókerék-hajtás", "rear-wheel drive", "Hinterradantrieb", "Hátsó kerekek hajtása", ["RWD"], "transmission", "drivetrain"),
        GlossaryTerm("osztómű", "transfer case", "Verteilergetriebe", "Az összkerékhajtás erőelosztója", [], "transmission", "drivetrain"),
        GlossaryTerm("csúszó kuplungos differenciál", "limited slip differential", "Sperrdifferential", "Korlátozott csúszású differenciálmű", ["LSD"], "transmission", "drivetrain"),
        GlossaryTerm("kuplung pedál", "clutch pedal", "Kupplungspedal", "A kuplung működtetésére szolgáló pedál", [], "transmission", "controls"),
        GlossaryTerm("kuplung főhenger", "clutch master cylinder", "Kupplungsgeberzylinder", "A kuplung hidraulikájának főhengere", [], "transmission", "clutch"),
        GlossaryTerm("kuplung munkahenger", "clutch slave cylinder", "Kupplungsnehmerzylinder", "A kuplung hidraulikájának munkahengere", [], "transmission", "clutch"),
        GlossaryTerm("kuplunglamella", "clutch friction disc", "Kupplungslamelle", "A kuplung súrlódó tárcsája", [], "transmission", "clutch"),
        GlossaryTerm("nyomólap", "pressure plate", "Druckplatte", "A kuplung nyomóeleme", [], "transmission", "clutch"),
        GlossaryTerm("kioldócsapágy", "throw-out bearing", "Ausrücklager", "A kuplung kioldását végző csapágy", [], "transmission", "clutch"),
        GlossaryTerm("kettős tömegű lendkerék", "dual mass flywheel", "Zweimassenschwungrad", "Rezgéscsillapítós lendkerék", ["DMF"], "transmission", "clutch"),
    ]


def generate_electrical_terms() -> List[GlossaryTerm]:
    """Generate electrical system terms."""
    return [
        GlossaryTerm("akkumulátor", "battery", "Batterie", "Az elektromos energia tárolására szolgáló elem", ["AKKU"], "electrical", "power"),
        GlossaryTerm("generátor", "alternator", "Lichtmaschine", "Az akkumulátor töltését végző generátor", [], "electrical", "power"),
        GlossaryTerm("dinamó", "generator", "Generator", "Régi típusú töltő berendezés", [], "electrical", "power"),
        GlossaryTerm("önindító", "starter motor", "Anlasser", "A motor indítását végző elektromotor", [], "electrical", "starting"),
        GlossaryTerm("indítómotor", "starter", "Starter", "Önindító más elnevezése", [], "electrical", "starting"),
        GlossaryTerm("bendix", "starter drive", "Einrückgetriebe", "Az önindító bekapcsolódó fogaskeréke", [], "electrical", "starting"),
        GlossaryTerm("indítórelé", "starter relay", "Anlasserrelais", "Az önindító kapcsoló reléje", [], "electrical", "starting"),
        GlossaryTerm("gyújtáskapcsoló", "ignition switch", "Zündschloss", "A gyújtás és indítás kapcsolója", [], "electrical", "ignition"),
        GlossaryTerm("motorvezérlő egység", "engine control unit", "Motorsteuergerät", "A motor elektronikus vezérlője", ["ECU", "ECM"], "electrical", "control"),
        GlossaryTerm("vezérlőegység", "control module", "Steuergerät", "Elektronikus vezérlő modul", [], "electrical", "control"),
        GlossaryTerm("hajtáslánc vezérlő", "powertrain control module", "Antriebssteuergerät", "Motor és váltó közös vezérlője", ["PCM"], "electrical", "control"),
        GlossaryTerm("sebességváltó vezérlő", "transmission control module", "Getriebesteuergerät", "Az automata váltó vezérlője", ["TCM", "TCU"], "electrical", "control"),
        GlossaryTerm("karosszéria vezérlő", "body control module", "Karosseriesteuergerät", "A karosszéria elektromos funkcióinak vezérlője", ["BCM"], "electrical", "control"),
        GlossaryTerm("ABS vezérlő", "ABS control module", "ABS-Steuergerät", "A blokkolásgátló rendszer vezérlője", [], "electrical", "control"),
        GlossaryTerm("légzsák vezérlő", "airbag control module", "Airbag-Steuergerät", "A légzsákok vezérlője", ["ACM", "SRS"], "electrical", "control"),
        GlossaryTerm("biztosíték", "fuse", "Sicherung", "Túláram elleni védelem", [], "electrical", "protection"),
        GlossaryTerm("biztosítéktábla", "fuse box", "Sicherungskasten", "A biztosítékok gyűjtőhelye", [], "electrical", "protection"),
        GlossaryTerm("relé", "relay", "Relais", "Elektromágneses kapcsoló", [], "electrical", "switching"),
        GlossaryTerm("kapcsoló", "switch", "Schalter", "Elektromos áramkör megszakító", [], "electrical", "switching"),
        GlossaryTerm("érzékelő", "sensor", "Sensor", "Fizikai mennyiséget elektromos jellé alakító elem", [], "electrical", "sensors"),
        GlossaryTerm("szenzor", "sensor", "Sensor", "Érzékelő más elnevezése", [], "electrical", "sensors"),
        GlossaryTerm("jeladó", "sender", "Geber", "Jelet küldő elem", [], "electrical", "sensors"),
        GlossaryTerm("hőmérséklet-érzékelő", "temperature sensor", "Temperaturfühler", "Hőmérsékletet mérő szenzor", [], "electrical", "sensors"),
        GlossaryTerm("nyomásérzékelő", "pressure sensor", "Drucksensor", "Nyomást mérő szenzor", [], "electrical", "sensors"),
        GlossaryTerm("pozícióérzékelő", "position sensor", "Positionssensor", "Helyzetet érzékelő szenzor", [], "electrical", "sensors"),
        GlossaryTerm("fordulatszám-érzékelő", "speed sensor", "Drehzahlsensor", "Fordulatszámot mérő szenzor", [], "electrical", "sensors"),
        GlossaryTerm("főtengely pozíció szenzor", "crankshaft position sensor", "Kurbelwellensensor", "A főtengely helyzetét érzékelő szenzor", ["CKP"], "electrical", "sensors"),
        GlossaryTerm("vezérműtengely pozíció szenzor", "camshaft position sensor", "Nockenwellensensor", "A vezérműtengely helyzetét érzékelő szenzor", ["CMP"], "electrical", "sensors"),
        GlossaryTerm("kopogásérzékelő", "knock sensor", "Klopfsensor", "A motor kopogását érzékelő szenzor", ["KS"], "electrical", "sensors"),
        GlossaryTerm("feszültség", "voltage", "Spannung", "Elektromos potenciálkülönbség", ["V"], "electrical", "parameters"),
        GlossaryTerm("áramerősség", "current", "Stromstärke", "Elektromos töltések áramlása", ["A"], "electrical", "parameters"),
        GlossaryTerm("ellenállás", "resistance", "Widerstand", "Az áram áramlását akadályozó tényező", ["Ohm"], "electrical", "parameters"),
        GlossaryTerm("rövidzárlat", "short circuit", "Kurzschluss", "Nem kívánt elektromos kapcsolat", [], "electrical", "faults"),
        GlossaryTerm("szakadás", "open circuit", "Unterbrechung", "Megszakadt elektromos kapcsolat", [], "electrical", "faults"),
        GlossaryTerm("testzárlat", "ground fault", "Masseschluss", "Rövidzárlat a testre", [], "electrical", "faults"),
        GlossaryTerm("vezeték", "wire", "Kabel", "Elektromos vezető", [], "electrical", "wiring"),
        GlossaryTerm("kábelköteg", "wiring harness", "Kabelbaum", "Vezetékek összefogott csoportja", [], "electrical", "wiring"),
        GlossaryTerm("csatlakozó", "connector", "Stecker", "Vezetékek összekapcsolására szolgáló elem", [], "electrical", "wiring"),
        GlossaryTerm("dugasz", "plug", "Stecker", "Csatlakozó apa oldala", [], "electrical", "wiring"),
        GlossaryTerm("aljzat", "socket", "Buchse", "Csatlakozó anya oldala", [], "electrical", "wiring"),
        GlossaryTerm("test", "ground", "Masse", "Elektromos referencia pont", ["GND"], "electrical", "wiring"),
        GlossaryTerm("földelés", "grounding", "Erdung", "Test kapcsolat kialakítása", [], "electrical", "wiring"),
        GlossaryTerm("CAN busz", "CAN bus", "CAN-Bus", "Járműves kommunikációs hálózat", ["CAN"], "electrical", "communication"),
        GlossaryTerm("LIN busz", "LIN bus", "LIN-Bus", "Egyszerűsített kommunikációs hálózat", ["LIN"], "electrical", "communication"),
        GlossaryTerm("OBD csatlakozó", "OBD port", "OBD-Anschluss", "Diagnosztikai csatlakozó", ["OBD", "OBD-II"], "electrical", "diagnostics"),
        GlossaryTerm("diagnosztika", "diagnostics", "Diagnose", "Hibakeresési eljárás", [], "electrical", "diagnostics"),
        GlossaryTerm("hibakód", "fault code", "Fehlercode", "Tárolt hiba azonosítója", ["DTC"], "electrical", "diagnostics"),
        GlossaryTerm("hibalámpa", "warning light", "Warnleuchte", "Hibát jelző fény", ["MIL"], "electrical", "indicators"),
        GlossaryTerm("műszerfal", "dashboard", "Armaturenbrett", "A műszerek elhelyezésére szolgáló panel", [], "electrical", "indicators"),
        GlossaryTerm("kilométeróra", "odometer", "Kilometerzähler", "A megtett távolság számlálója", [], "electrical", "indicators"),
        GlossaryTerm("sebességmérő", "speedometer", "Tachometer", "A sebesség kijelzője", [], "electrical", "indicators"),
        GlossaryTerm("fordulatszámmérő", "tachometer", "Drehzahlmesser", "A motor fordulatszámának kijelzője", [], "electrical", "indicators"),
        GlossaryTerm("üzemanyagszint-jelző", "fuel gauge", "Tankanzeige", "Az üzemanyag mennyiségének kijelzője", [], "electrical", "indicators"),
        GlossaryTerm("hőfokmérő", "temperature gauge", "Temperaturanzeige", "A hőmérséklet kijelzője", [], "electrical", "indicators"),
        GlossaryTerm("fényszóró", "headlight", "Scheinwerfer", "Elülső világítás", [], "electrical", "lighting"),
        GlossaryTerm("tompított fény", "low beam", "Abblendlicht", "Alacsony fényerősségű főfény", [], "electrical", "lighting"),
        GlossaryTerm("távolsági fény", "high beam", "Fernlicht", "Erős fényerősségű főfény", [], "electrical", "lighting"),
        GlossaryTerm("helyzetjelző", "parking light", "Standlicht", "Gyenge fényerejű jelzőfény", [], "electrical", "lighting"),
        GlossaryTerm("irányjelző", "turn signal", "Blinker", "Kanyarodási irány jelzése", [], "electrical", "lighting"),
        GlossaryTerm("féklámpa", "brake light", "Bremslicht", "A fékezést jelző lámpa", [], "electrical", "lighting"),
        GlossaryTerm("hátsó lámpa", "tail light", "Rücklicht", "Hátsó világítás", [], "electrical", "lighting"),
        GlossaryTerm("tolatólámpa", "reverse light", "Rückfahrlicht", "A tolatást jelző lámpa", [], "electrical", "lighting"),
        GlossaryTerm("ködlámpa", "fog light", "Nebelscheinwerfer", "Rossz látási viszonyokhoz", [], "electrical", "lighting"),
        GlossaryTerm("LED", "LED", "LED", "Fénykibocsátó dióda", ["LED"], "electrical", "lighting"),
        GlossaryTerm("xenon", "xenon", "Xenon", "Gázkisüléses fényforrás", ["HID"], "electrical", "lighting"),
        GlossaryTerm("halogén", "halogen", "Halogen", "Halogéntöltésű izzó", [], "electrical", "lighting"),
        GlossaryTerm("ablaktörlő motor", "wiper motor", "Scheibenwischermotor", "Az ablaktörlő meghajtása", [], "electrical", "accessories"),
        GlossaryTerm("ablaktörlő", "wiper", "Scheibenwischer", "Az ablak tisztítására szolgáló eszköz", [], "electrical", "accessories"),
        GlossaryTerm("ablakmosó", "washer", "Scheibenwaschanlage", "Az ablak tisztító folyadékot permetező rendszer", [], "electrical", "accessories"),
        GlossaryTerm("elektromos ablakemelő", "power window", "elektrischer Fensterheber", "Elektromosan működtetett ablak", [], "electrical", "accessories"),
        GlossaryTerm("központi zár", "central locking", "Zentralverriegelung", "Az ajtók központi zárása", [], "electrical", "accessories"),
        GlossaryTerm("riasztó", "alarm", "Alarmanlage", "Betörés elleni védelem", [], "electrical", "security"),
        GlossaryTerm("immobilizer", "immobilizer", "Wegfahrsperre", "Indításgátló rendszer", [], "electrical", "security"),
    ]


def generate_brake_terms() -> List[GlossaryTerm]:
    """Generate brake system terms."""
    return [
        GlossaryTerm("fékrendszer", "brake system", "Bremsanlage", "A jármű lassítását és megállítását biztosító rendszer", [], "brake", "system"),
        GlossaryTerm("tárcsafék", "disc brake", "Scheibenbremse", "Féktárcsát használó fékrendszer", [], "brake", "types"),
        GlossaryTerm("dobfék", "drum brake", "Trommelbremse", "Fékdobot használó fékrendszer", [], "brake", "types"),
        GlossaryTerm("féktárcsa", "brake disc", "Bremsscheibe", "A fékbetétek által megfogott forgó tárcsa", [], "brake", "components"),
        GlossaryTerm("fékdob", "brake drum", "Bremstrommel", "A fékpofa által megfogott forgó dob", [], "brake", "components"),
        GlossaryTerm("féknyereg", "brake caliper", "Bremssattel", "A fékbetéteket a tárcsához szorító szerkezet", [], "brake", "components"),
        GlossaryTerm("fékbetét", "brake pad", "Bremsbelag", "A féktárcsát megfogó súrlódó elem", [], "brake", "components"),
        GlossaryTerm("fékpofa", "brake shoe", "Bremsbacke", "A fékdobot megfogó súrlódó elem", [], "brake", "components"),
        GlossaryTerm("fékmunkahenger", "brake cylinder", "Radbremszylinder", "A féknyomást mechanikus erővé alakító henger", [], "brake", "hydraulics"),
        GlossaryTerm("fék főhenger", "brake master cylinder", "Hauptbremszylinder", "A fékpedál által működtetett hidraulikus henger", [], "brake", "hydraulics"),
        GlossaryTerm("fékfolyadék", "brake fluid", "Bremsflüssigkeit", "A fékrendszer hidraulikus közege", ["DOT"], "brake", "fluids"),
        GlossaryTerm("fékerő-szabályozó", "brake force regulator", "Bremskraftregler", "A féknyomást szabályozó elem", [], "brake", "control"),
        GlossaryTerm("fékerősítő", "brake booster", "Bremskraftverstärker", "A pedálerőt felerősítő vákuumos szerkezet", [], "brake", "assist"),
        GlossaryTerm("szervófék", "power brake", "Servobremse", "Fékerősítővel szerelt fék", [], "brake", "assist"),
        GlossaryTerm("fékpedál", "brake pedal", "Bremspedal", "A fék működtetésére szolgáló pedál", [], "brake", "controls"),
        GlossaryTerm("kézifék", "parking brake", "Handbremse", "Rögzítőfék kézi működtetéssel", [], "brake", "parking"),
        GlossaryTerm("rögzítőfék", "parking brake", "Feststellbremse", "A jármű álló helyzetben tartása", [], "brake", "parking"),
        GlossaryTerm("elektromos rögzítőfék", "electric parking brake", "elektrische Parkbremse", "Elektromosan működtetett rögzítőfék", ["EPB"], "brake", "parking"),
        GlossaryTerm("ABS", "ABS", "ABS", "Blokkolásgátló rendszer", ["ABS"], "brake", "electronic"),
        GlossaryTerm("blokkolásgátló", "anti-lock braking system", "Antiblockiersystem", "A kerekek blokkolását megakadályozó rendszer", ["ABS"], "brake", "electronic"),
        GlossaryTerm("ESP", "ESP", "ESP", "Elektronikus menetstabilizáló", ["ESP", "ESC"], "brake", "electronic"),
        GlossaryTerm("menetstabilizáló", "stability control", "Stabilitätskontrolle", "A jármű stabilitását biztosító rendszer", ["ESC"], "brake", "electronic"),
        GlossaryTerm("kipörgésgátló", "traction control", "Antriebsschlupfregelung", "A kerekek kipörgését megakadályozó rendszer", ["ASR", "TCS"], "brake", "electronic"),
        GlossaryTerm("fékasszisztens", "brake assist", "Bremsassistent", "Vészfékezést segítő rendszer", ["BAS"], "brake", "electronic"),
        GlossaryTerm("féktávolság", "braking distance", "Bremsweg", "A megállásig megtett távolság", [], "brake", "parameters"),
        GlossaryTerm("fékezési teljesítmény", "braking performance", "Bremsleistung", "A fékrendszer hatékonysága", [], "brake", "parameters"),
        GlossaryTerm("fékvezeték", "brake line", "Bremsleitung", "A fékfolyadékot vezető cső", [], "brake", "hydraulics"),
        GlossaryTerm("féktömlő", "brake hose", "Bremsschlauch", "Rugalmas fékvezeték", [], "brake", "hydraulics"),
        GlossaryTerm("légtelenítés", "bleeding", "Entlüften", "A levegő eltávolítása a fékrendszerből", [], "brake", "service"),
        GlossaryTerm("fékpor", "brake dust", "Bremsstaub", "A fékbetét kopásából származó por", [], "brake", "wear"),
        GlossaryTerm("fékcsikorgás", "brake squeal", "Bremsenquietschen", "A fékek nyikorgó hangja", [], "brake", "symptoms"),
        GlossaryTerm("fékrezgés", "brake vibration", "Bremsenruckeln", "Fékezéskor érezhető rezgés", [], "brake", "symptoms"),
        GlossaryTerm("ABS szenzor", "ABS sensor", "ABS-Sensor", "A kerékfordulatot érzékelő szenzor", [], "brake", "sensors"),
        GlossaryTerm("kerékfordulatszám-érzékelő", "wheel speed sensor", "Raddrehzahlsensor", "ABS szenzor más elnevezése", ["WSS"], "brake", "sensors"),
        GlossaryTerm("hidraulikus egység", "hydraulic unit", "Hydraulikeinheit", "Az ABS/ESP hidraulikus része", ["HCU"], "brake", "electronic"),
        GlossaryTerm("regeneratív fékezés", "regenerative braking", "Rekuperationsbremse", "Energia-visszanyerő fékezés", [], "brake", "hybrid"),
    ]


def generate_suspension_terms() -> List[GlossaryTerm]:
    """Generate suspension system terms."""
    return [
        GlossaryTerm("futómű", "suspension", "Fahrwerk", "A jármű alváza és kerekei közötti rendszer", [], "suspension", "system"),
        GlossaryTerm("felfüggesztés", "suspension", "Aufhängung", "Futómű más elnevezése", [], "suspension", "system"),
        GlossaryTerm("lengéscsillapító", "shock absorber", "Stoßdämpfer", "A rugózás csillapítását végző elem", [], "suspension", "damping"),
        GlossaryTerm("rugó", "spring", "Feder", "A terhelést rugalmasan felvevő elem", [], "suspension", "springs"),
        GlossaryTerm("tekercsrugó", "coil spring", "Schraubenfeder", "Spirálisan tekert rugó", [], "suspension", "springs"),
        GlossaryTerm("laprugó", "leaf spring", "Blattfeder", "Réteges szerkezetű rugó", [], "suspension", "springs"),
        GlossaryTerm("légrugó", "air spring", "Luftfeder", "Levegővel töltött rugó", [], "suspension", "springs"),
        GlossaryTerm("torziós rugó", "torsion bar", "Drehstabfeder", "Csavarással rugózó rúd", [], "suspension", "springs"),
        GlossaryTerm("stabilizátor", "anti-roll bar", "Stabilisator", "Az oldalra dőlést csökkentő rúd", [], "suspension", "stabilizer"),
        GlossaryTerm("stabilizátor szilent", "stabilizer bushing", "Stabilisatorbuchse", "A stabilizátor gumi ágyazása", [], "suspension", "stabilizer"),
        GlossaryTerm("stabilizátor összekötő", "stabilizer link", "Stabilisatorstange", "A stabilizátort a futóműhöz kötő elem", [], "suspension", "stabilizer"),
        GlossaryTerm("lengőkar", "control arm", "Querlenker", "A kereket az alvázhoz kötő kar", [], "suspension", "arms"),
        GlossaryTerm("háromszögkar", "wishbone", "Dreieckslenker", "Háromszög alakú lengőkar", [], "suspension", "arms"),
        GlossaryTerm("hosszlengőkar", "trailing arm", "Längslenker", "Hosszirányú lengőkar", [], "suspension", "arms"),
        GlossaryTerm("keresztlengőkar", "lateral arm", "Querlenker", "Keresztirányú lengőkar", [], "suspension", "arms"),
        GlossaryTerm("gömbfej", "ball joint", "Kugelgelenk", "Gömbcsuklós összekötés", [], "suspension", "joints"),
        GlossaryTerm("szilentblokk", "bushing", "Silentbuchse", "Gumi ágyazású csukló", [], "suspension", "bushings"),
        GlossaryTerm("persely", "bushing", "Buchse", "Csapágyazó hüvely", [], "suspension", "bushings"),
        GlossaryTerm("kerékcsapágy", "wheel bearing", "Radlager", "A kerék forgását biztosító csapágy", [], "suspension", "bearings"),
        GlossaryTerm("kerékagy", "wheel hub", "Radnabe", "A kereket hordozó agy", [], "suspension", "hub"),
        GlossaryTerm("kerékcsap", "spindle", "Achsschenkel", "A kerék forgástengelye", [], "suspension", "hub"),
        GlossaryTerm("kormánymű", "steering gear", "Lenkgetriebe", "A kormánykerék mozgását átvivő szerkezet", [], "suspension", "steering"),
        GlossaryTerm("fogasléces kormánymű", "rack and pinion", "Zahnstangenlenkung", "Fogasléces kormányáttétel", [], "suspension", "steering"),
        GlossaryTerm("szervokormány", "power steering", "Servolenkung", "Rásegítéses kormányzás", [], "suspension", "steering"),
        GlossaryTerm("elektromos szervokormány", "electric power steering", "elektrische Servolenkung", "Elektromos rásegítésű kormány", ["EPS"], "suspension", "steering"),
        GlossaryTerm("hidraulikus szervokormány", "hydraulic power steering", "hydraulische Servolenkung", "Hidraulikus rásegítésű kormány", [], "suspension", "steering"),
        GlossaryTerm("kormányrúd", "steering shaft", "Lenkwelle", "A kormánykerék és kormánymű közötti tengely", [], "suspension", "steering"),
        GlossaryTerm("kormányösszekötő", "tie rod", "Spurstange", "A kormányművet a kerékkel összekötő rúd", [], "suspension", "steering"),
        GlossaryTerm("kormánygömbfej", "tie rod end", "Spurstangenkopf", "A kormányösszekötő gömbfeje", [], "suspension", "steering"),
        GlossaryTerm("csapágyazás", "bearing assembly", "Lagereinheit", "Csapágyas egység", [], "suspension", "bearings"),
        GlossaryTerm("első felfüggesztés", "front suspension", "Vorderachse", "Az első tengely felfüggesztése", [], "suspension", "location"),
        GlossaryTerm("hátsó felfüggesztés", "rear suspension", "Hinterachse", "A hátsó tengely felfüggesztése", [], "suspension", "location"),
        GlossaryTerm("független felfüggesztés", "independent suspension", "Einzelradaufhängung", "Kerekeként független rugózás", [], "suspension", "types"),
        GlossaryTerm("merev tengely", "solid axle", "Starrachse", "Mindkét kereket összekötő merev tengely", [], "suspension", "types"),
        GlossaryTerm("McPherson", "MacPherson strut", "McPherson-Federbein", "Elterjedt első felfüggesztési típus", [], "suspension", "types"),
        GlossaryTerm("többlengőkaros", "multi-link", "Mehrlenkerachse", "Több lengőkart használó felfüggesztés", [], "suspension", "types"),
        GlossaryTerm("kerékállás", "wheel alignment", "Radstellung", "A kerekek beállítási paraméterei", [], "suspension", "alignment"),
        GlossaryTerm("nyomtáv", "toe", "Spur", "A kerekek hosszirányú dőlése", [], "suspension", "alignment"),
        GlossaryTerm("összefutás", "toe-in", "Vorspur", "A kerekek előre dőlése", [], "suspension", "alignment"),
        GlossaryTerm("széttartás", "toe-out", "Nachspur", "A kerekek hátra dőlése", [], "suspension", "alignment"),
        GlossaryTerm("kerékdőlés", "camber", "Sturz", "A kerék függőlegestől való dőlése", [], "suspension", "alignment"),
        GlossaryTerm("utánfutás", "caster", "Nachlauf", "A kormánytengely dőlésszöge", [], "suspension", "alignment"),
        GlossaryTerm("rugóút", "suspension travel", "Federweg", "A felfüggesztés mozgástartománya", [], "suspension", "parameters"),
        GlossaryTerm("hasmagasság", "ground clearance", "Bodenfreiheit", "Az alváz és talaj közötti távolság", [], "suspension", "parameters"),
        GlossaryTerm("adaptív futómű", "adaptive suspension", "adaptives Fahrwerk", "Elektronikusan állítható futómű", [], "suspension", "electronic"),
        GlossaryTerm("aktív futómű", "active suspension", "aktives Fahrwerk", "Aktívan szabályozott felfüggesztés", [], "suspension", "electronic"),
        GlossaryTerm("légfelfüggesztés", "air suspension", "Luftfederung", "Légrugókkal szerelt futómű", [], "suspension", "types"),
    ]


def generate_body_terms() -> List[GlossaryTerm]:
    """Generate body and chassis terms."""
    return [
        GlossaryTerm("karosszéria", "body", "Karosserie", "A jármű külső burkolata", [], "body", "structure"),
        GlossaryTerm("alváz", "chassis", "Fahrgestell", "A jármű teherhordó váza", [], "body", "structure"),
        GlossaryTerm("önhordó karosszéria", "unibody", "selbsttragende Karosserie", "Alváz nélküli teherhordó karosszéria", [], "body", "structure"),
        GlossaryTerm("vázszerkezet", "frame", "Rahmen", "A jármű teherhordó kerete", [], "body", "structure"),
        GlossaryTerm("ajtó", "door", "Tür", "A jármű bejárata", [], "body", "panels"),
        GlossaryTerm("motorháztető", "hood", "Motorhaube", "A motortér fedele", [], "body", "panels"),
        GlossaryTerm("csomagtartó fedél", "trunk lid", "Kofferraumdeckel", "A csomagtér fedele", [], "body", "panels"),
        GlossaryTerm("sárvédő", "fender", "Kotflügel", "A kerekeket fedő lemez", [], "body", "panels"),
        GlossaryTerm("lökhárító", "bumper", "Stoßstange", "Ütközés elleni védelem", [], "body", "panels"),
        GlossaryTerm("küszöb", "sill", "Schweller", "Az ajtók alatti párkány", [], "body", "structure"),
        GlossaryTerm("A-oszlop", "A-pillar", "A-Säule", "A szélvédő melletti oszlop", [], "body", "structure"),
        GlossaryTerm("B-oszlop", "B-pillar", "B-Säule", "Az első és hátsó ajtó közötti oszlop", [], "body", "structure"),
        GlossaryTerm("C-oszlop", "C-pillar", "C-Säule", "A hátsó szélvédő melletti oszlop", [], "body", "structure"),
        GlossaryTerm("tető", "roof", "Dach", "A jármű felső fedele", [], "body", "panels"),
        GlossaryTerm("napfénytető", "sunroof", "Schiebedach", "Nyitható tetőablak", [], "body", "panels"),
        GlossaryTerm("szélvédő", "windshield", "Windschutzscheibe", "Elülső ablak", [], "body", "glass"),
        GlossaryTerm("hátsó szélvédő", "rear window", "Heckscheibe", "Hátsó ablak", [], "body", "glass"),
        GlossaryTerm("oldalsó ablak", "side window", "Seitenfenster", "Oldalsó ablak", [], "body", "glass"),
        GlossaryTerm("visszapillantó tükör", "rearview mirror", "Rückspiegel", "Belső tükör", [], "body", "mirrors"),
        GlossaryTerm("külső tükör", "side mirror", "Außenspiegel", "Oldalsó tükör", [], "body", "mirrors"),
        GlossaryTerm("ülés", "seat", "Sitz", "Ülőhely", [], "body", "interior"),
        GlossaryTerm("első ülés", "front seat", "Vordersitz", "Elülső ülés", [], "body", "interior"),
        GlossaryTerm("hátsó ülés", "rear seat", "Rücksitz", "Hátsó ülés", [], "body", "interior"),
        GlossaryTerm("kormánykerék", "steering wheel", "Lenkrad", "A kormányzásra szolgáló kerék", [], "body", "interior"),
        GlossaryTerm("kesztyűtartó", "glove box", "Handschuhfach", "Az utasoldali tárolórekesz", [], "body", "interior"),
        GlossaryTerm("középkonzol", "center console", "Mittelkonsole", "A két ülés közötti konzol", [], "body", "interior"),
        GlossaryTerm("ülésfűtés", "seat heater", "Sitzheizung", "Az ülés fűtése", [], "body", "comfort"),
        GlossaryTerm("klíma", "air conditioning", "Klimaanlage", "Hűtő berendezés", ["A/C"], "body", "comfort"),
        GlossaryTerm("fűtés", "heating", "Heizung", "Fűtő berendezés", [], "body", "comfort"),
        GlossaryTerm("szellőzés", "ventilation", "Lüftung", "Levegő keringetés", [], "body", "comfort"),
        GlossaryTerm("légzsák", "airbag", "Airbag", "Ütközésvédelmi párna", ["SRS"], "body", "safety"),
        GlossaryTerm("biztonsági öv", "seat belt", "Sicherheitsgurt", "Utas rögzítő öv", [], "body", "safety"),
        GlossaryTerm("övfeszítő", "belt pretensioner", "Gurtstraffer", "Az öv megfeszítése ütközéskor", [], "body", "safety"),
        GlossaryTerm("fejtámla", "headrest", "Kopfstütze", "A fej támasztéka", [], "body", "safety"),
        GlossaryTerm("gyerekülés rögzítés", "ISOFIX", "ISOFIX", "Gyerekülés szabványos rögzítése", ["ISOFIX"], "body", "safety"),
        GlossaryTerm("deformációs zóna", "crumple zone", "Knautschzone", "Ütközési energia elnyelő zóna", [], "body", "safety"),
        GlossaryTerm("horganyzás", "galvanizing", "Verzinkung", "Korrózió elleni fémréteg", [], "body", "materials"),
        GlossaryTerm("festés", "paint", "Lackierung", "A karosszéria bevonata", [], "body", "finish"),
        GlossaryTerm("alapozó", "primer", "Grundierung", "Festék alatti réteg", [], "body", "finish"),
        GlossaryTerm("fényes lakk", "clear coat", "Klarlack", "Védő lakkréteg", [], "body", "finish"),
        GlossaryTerm("rozsda", "rust", "Rost", "Vas oxidációja", [], "body", "damage"),
        GlossaryTerm("korrózió", "corrosion", "Korrosion", "Fém lebomlása", [], "body", "damage"),
        GlossaryTerm("horpadás", "dent", "Delle", "Behorpadt felület", [], "body", "damage"),
        GlossaryTerm("karcolás", "scratch", "Kratzer", "Felületi sérülés", [], "body", "damage"),
        GlossaryTerm("alumínium karosszéria", "aluminum body", "Aluminiumkarosserie", "Könnyűfém karosszéria", [], "body", "materials"),
        GlossaryTerm("szénszál", "carbon fiber", "Kohlefaser", "Nagy szilárdságú könnyű anyag", ["CFRP"], "body", "materials"),
        GlossaryTerm("műanyag", "plastic", "Kunststoff", "Polimer anyag", [], "body", "materials"),
        GlossaryTerm("üvegszál", "fiberglass", "Glasfaser", "Üvegszálas műanyag", ["GFK"], "body", "materials"),
    ]


def generate_tool_terms() -> List[GlossaryTerm]:
    """Generate tool names."""
    return [
        GlossaryTerm("villáskulcs", "open-end wrench", "Maulschlüssel", "Két végén nyitott kulcs", [], "tools", "wrenches"),
        GlossaryTerm("csillagkulcs", "box wrench", "Ringschlüssel", "Zárt végű kulcs", [], "tools", "wrenches"),
        GlossaryTerm("kombinált kulcs", "combination wrench", "Gabelringschlüssel", "Villás és csillag végű kulcs", [], "tools", "wrenches"),
        GlossaryTerm("racsnis kulcs", "ratchet wrench", "Ratschenschlüssel", "Racsnis mechanikájú kulcs", [], "tools", "wrenches"),
        GlossaryTerm("nyomatékkulcs", "torque wrench", "Drehmomentschlüssel", "Meghatározott erővel húzó kulcs", [], "tools", "wrenches"),
        GlossaryTerm("imbuszkulcs", "Allen key", "Inbusschlüssel", "Belső hatszögű kulcs", [], "tools", "wrenches"),
        GlossaryTerm("torx kulcs", "Torx key", "Torx-Schlüssel", "Csillag alakú kulcs", ["TORX"], "tools", "wrenches"),
        GlossaryTerm("csavarhúzó", "screwdriver", "Schraubendreher", "Csavarok be- és kihajtására", [], "tools", "screwdrivers"),
        GlossaryTerm("lapos csavarhúzó", "flat screwdriver", "Schlitzschraubendreher", "Egyenes pengéjű csavarhúzó", [], "tools", "screwdrivers"),
        GlossaryTerm("kereszt csavarhúzó", "Phillips screwdriver", "Kreuzschlitzschraubendreher", "Keresztfejű csavarhúzó", [], "tools", "screwdrivers"),
        GlossaryTerm("fogó", "pliers", "Zange", "Megfogó szerszám", [], "tools", "pliers"),
        GlossaryTerm("kombinált fogó", "combination pliers", "Kombizange", "Általános célú fogó", [], "tools", "pliers"),
        GlossaryTerm("vízpumpafogó", "water pump pliers", "Wasserpumpenzange", "Állítható fogó", [], "tools", "pliers"),
        GlossaryTerm("csípőfogó", "diagonal pliers", "Seitenschneider", "Vágófogó", [], "tools", "pliers"),
        GlossaryTerm("hegyes fogó", "needle-nose pliers", "Spitzzange", "Hosszú orrú fogó", [], "tools", "pliers"),
        GlossaryTerm("biztosítógyűrű fogó", "snap ring pliers", "Seegeringzange", "Biztosítógyűrűkhöz", [], "tools", "pliers"),
        GlossaryTerm("kalapács", "hammer", "Hammer", "Ütő szerszám", [], "tools", "hammers"),
        GlossaryTerm("gumikalapács", "rubber mallet", "Gummihammer", "Puha fejű kalapács", [], "tools", "hammers"),
        GlossaryTerm("csúszókalapács", "slide hammer", "Gleithammer", "Húzó szerszám", [], "tools", "hammers"),
        GlossaryTerm("emelő", "jack", "Wagenheber", "Jármű emelésére", [], "tools", "lifting"),
        GlossaryTerm("krokodil emelő", "floor jack", "Rangierwagenheber", "Alacsony emelő", [], "tools", "lifting"),
        GlossaryTerm("állvány", "jack stand", "Unterstellbock", "Tartóállvány", [], "tools", "lifting"),
        GlossaryTerm("emelőpad", "lift", "Hebebühne", "Jármű emelő platform", [], "tools", "lifting"),
        GlossaryTerm("csörlő", "winch", "Seilwinde", "Húzó eszköz", [], "tools", "lifting"),
        GlossaryTerm("diagnosztikai készülék", "diagnostic tool", "Diagnosegerät", "Hibakód olvasó eszköz", ["OBD"], "tools", "diagnostics"),
        GlossaryTerm("multiméter", "multimeter", "Multimeter", "Elektromos mérőműszer", [], "tools", "diagnostics"),
        GlossaryTerm("oszcilloszkóp", "oscilloscope", "Oszilloskop", "Jelforma megjelenítő", [], "tools", "diagnostics"),
        GlossaryTerm("kompressziómérő", "compression tester", "Kompressionsprüfer", "Henger tömörségmérő", [], "tools", "diagnostics"),
        GlossaryTerm("üzemanyag-nyomásmérő", "fuel pressure gauge", "Kraftstoffdruckprüfer", "Benzin nyomás mérő", [], "tools", "diagnostics"),
        GlossaryTerm("endoszkóp", "borescope", "Endoskop", "Belső vizsgáló kamera", [], "tools", "diagnostics"),
        GlossaryTerm("olajleszívó", "oil extractor", "Ölabsauger", "Olaj eltávolító", [], "tools", "service"),
        GlossaryTerm("üzemanyag tartály", "fuel caddy", "Kraftstofftank", "Hordozható tank", [], "tools", "service"),
        GlossaryTerm("töltőberendezés", "battery charger", "Ladegerät", "Akkumulátor töltő", [], "tools", "electrical"),
        GlossaryTerm("indítókábel", "jumper cables", "Starthilfekabel", "Bikázó kábel", [], "tools", "electrical"),
        GlossaryTerm("hegesztőgép", "welder", "Schweißgerät", "Fém összeillesztésre", [], "tools", "metalwork"),
        GlossaryTerm("sarokcsiszoló", "angle grinder", "Winkelschleifer", "Flexelő", [], "tools", "metalwork"),
        GlossaryTerm("fúrógép", "drill", "Bohrmaschine", "Lyukfúró eszköz", [], "tools", "metalwork"),
        GlossaryTerm("menetfúró", "tap", "Gewindebohrer", "Belső menet készítő", [], "tools", "metalwork"),
        GlossaryTerm("menetmetsző", "die", "Schneideisen", "Külső menet készítő", [], "tools", "metalwork"),
        GlossaryTerm("lehúzó", "puller", "Abzieher", "Alkatrész lehúzó", [], "tools", "specialty"),
        GlossaryTerm("csapágylehúzó", "bearing puller", "Lagerabzieher", "Csapágy eltávolító", [], "tools", "specialty"),
        GlossaryTerm("fékdugattyú visszanyomó", "brake piston tool", "Bremskolbenrücksteller", "Fékdugattyú benyomó", [], "tools", "specialty"),
        GlossaryTerm("gyertyakulcs", "spark plug socket", "Zündkerzenschlüssel", "Gyújtógyertya szerelő", [], "tools", "specialty"),
        GlossaryTerm("olajszűrő kulcs", "oil filter wrench", "Ölfilterschlüssel", "Olajszűrő leszerelő", [], "tools", "specialty"),
        GlossaryTerm("gömbcsukló prés", "ball joint press", "Kugelgelenkpresse", "Gömbcsukló szerelő", [], "tools", "specialty"),
        GlossaryTerm("hézagmérő", "feeler gauge", "Fühlerlehre", "Rés méretének mérése", [], "tools", "measuring"),
        GlossaryTerm("tolómérő", "caliper", "Schieblehre", "Pontos méret mérése", [], "tools", "measuring"),
        GlossaryTerm("mikrométer", "micrometer", "Mikrometer", "Nagyon pontos mérés", [], "tools", "measuring"),
        GlossaryTerm("mérőóra", "dial gauge", "Messuhr", "Ütés mérés", [], "tools", "measuring"),
    ]


def generate_specification_terms() -> List[GlossaryTerm]:
    """Generate technical specification terms."""
    return [
        GlossaryTerm("járműazonosító szám", "vehicle identification number", "Fahrzeugidentnummer", "A jármű egyedi azonosítója", ["VIN"], "specifications", "identification"),
        GlossaryTerm("rendszám", "license plate", "Kennzeichen", "A jármű hatósági jelzése", [], "specifications", "identification"),
        GlossaryTerm("típusbizonyítvány", "type approval", "Typgenehmigung", "A jármű jóváhagyási dokumentuma", [], "specifications", "documents"),
        GlossaryTerm("forgalmi engedély", "registration", "Zulassungsbescheinigung", "A jármű üzemeltetési engedélye", [], "specifications", "documents"),
        GlossaryTerm("műszaki vizsga", "technical inspection", "Hauptuntersuchung", "Időszakos műszaki felülvizsgálat", ["TÜV"], "specifications", "inspection"),
        GlossaryTerm("teljesítmény", "power", "Leistung", "A motor kimenő teljesítménye", ["kW", "LE"], "specifications", "engine"),
        GlossaryTerm("forgatónyomaték", "torque", "Drehmoment", "A motor forgató ereje", ["Nm"], "specifications", "engine"),
        GlossaryTerm("hengerűrtartalom", "displacement", "Hubraum", "A motor térfogata", ["ccm", "L"], "specifications", "engine"),
        GlossaryTerm("fogyasztás", "fuel consumption", "Kraftstoffverbrauch", "Üzemanyag felhasználás", ["l/100km"], "specifications", "consumption"),
        GlossaryTerm("kibocsátás", "emission", "Emission", "Károsanyag kibocsátás", ["g/km"], "specifications", "emission"),
        GlossaryTerm("szén-dioxid kibocsátás", "CO2 emission", "CO2-Emission", "Szén-dioxid kibocsátás", ["CO2"], "specifications", "emission"),
        GlossaryTerm("Euro norma", "Euro standard", "Euro-Norm", "Kibocsátási szabvány", ["EURO"], "specifications", "emission"),
        GlossaryTerm("önsúly", "curb weight", "Leergewicht", "A jármű üres tömege", ["kg"], "specifications", "weight"),
        GlossaryTerm("megengedett össztömeg", "gross vehicle weight", "zulässiges Gesamtgewicht", "Maximális terhelhetőség", ["GVW"], "specifications", "weight"),
        GlossaryTerm("hasznos teher", "payload", "Nutzlast", "Szállítható teher", ["kg"], "specifications", "weight"),
        GlossaryTerm("vontatási tömeg", "towing capacity", "Anhängelast", "Vontatható teher", ["kg"], "specifications", "weight"),
        GlossaryTerm("tengelytáv", "wheelbase", "Radstand", "A tengelyek távolsága", ["mm"], "specifications", "dimensions"),
        GlossaryTerm("hosszúság", "length", "Länge", "A jármű hossza", ["mm"], "specifications", "dimensions"),
        GlossaryTerm("szélesség", "width", "Breite", "A jármű szélessége", ["mm"], "specifications", "dimensions"),
        GlossaryTerm("magasság", "height", "Höhe", "A jármű magassága", ["mm"], "specifications", "dimensions"),
        GlossaryTerm("nyomtáv", "track width", "Spurweite", "A kerekek távolsága", ["mm"], "specifications", "dimensions"),
        GlossaryTerm("gumiméret", "tire size", "Reifengröße", "A gumiabroncs mérete", [], "specifications", "tires"),
        GlossaryTerm("felnméret", "rim size", "Felgengröße", "A felni mérete", [], "specifications", "tires"),
        GlossaryTerm("légnyomás", "tire pressure", "Reifendruck", "A gumiabroncs nyomása", ["bar", "PSI"], "specifications", "tires"),
        GlossaryTerm("olajkapacitás", "oil capacity", "Ölmenge", "A motor olajtérfogata", ["L"], "specifications", "fluids"),
        GlossaryTerm("hűtőfolyadék kapacitás", "coolant capacity", "Kühlmittelmenge", "A hűtőrendszer térfogata", ["L"], "specifications", "fluids"),
        GlossaryTerm("üzemanyagtartály", "fuel tank capacity", "Tankinhalt", "A tank térfogata", ["L"], "specifications", "fluids"),
        GlossaryTerm("gyorsulás", "acceleration", "Beschleunigung", "0-100 km/h idő", ["s"], "specifications", "performance"),
        GlossaryTerm("végsebességhatárleszállítás", "top speed", "Höchstgeschwindigkeit", "Maximális sebesség", ["km/h"], "specifications", "performance"),
        GlossaryTerm("SAE osztály", "SAE grade", "SAE-Klasse", "Olaj viszkozitási besorolás", ["SAE"], "specifications", "oil"),
        GlossaryTerm("API besorolás", "API rating", "API-Klassifikation", "Olaj minőségi besorolás", ["API"], "specifications", "oil"),
        GlossaryTerm("ACEA besorolás", "ACEA rating", "ACEA-Klassifikation", "Európai olaj besorolás", ["ACEA"], "specifications", "oil"),
        GlossaryTerm("viszkozitás", "viscosity", "Viskosität", "Az olaj sűrűsége", [], "specifications", "oil"),
        GlossaryTerm("oktánszám", "octane rating", "Oktanzahl", "Benzin minőségi mutatója", ["RON"], "specifications", "fuel"),
        GlossaryTerm("cetánszám", "cetane rating", "Cetanzahl", "Dízel minőségi mutatója", [], "specifications", "fuel"),
        GlossaryTerm("fékfolyadék típus", "brake fluid type", "Bremsflüssigkeitstyp", "A fékfolyadék specifikációja", ["DOT"], "specifications", "fluids"),
        GlossaryTerm("hűtőfolyadék típus", "coolant type", "Kühlmitteltyp", "A fagyálló specifikációja", [], "specifications", "fluids"),
        GlossaryTerm("szervizintervallum", "service interval", "Wartungsintervall", "Két szerviz közötti idő/km", [], "specifications", "service"),
        GlossaryTerm("garancia", "warranty", "Garantie", "Gyártói garancia", [], "specifications", "warranty"),
    ]


def generate_dtc_terms() -> List[GlossaryTerm]:
    """Generate DTC-related terms."""
    return [
        GlossaryTerm("hibakód", "diagnostic trouble code", "Fehlercode", "A rendszer által tárolt hiba azonosítója", ["DTC"], "dtc", "general"),
        GlossaryTerm("aktuális hiba", "current fault", "aktueller Fehler", "Jelenleg fennálló hiba", [], "dtc", "status"),
        GlossaryTerm("tárolt hiba", "stored fault", "gespeicherter Fehler", "Korábban előfordult hiba", [], "dtc", "status"),
        GlossaryTerm("függőben lévő hiba", "pending fault", "anstehender Fehler", "Még nem megerősített hiba", [], "dtc", "status"),
        GlossaryTerm("állandó hiba", "permanent fault", "permanenter Fehler", "Nem törölhető hiba", [], "dtc", "status"),
        GlossaryTerm("hajtáslánc kód", "powertrain code", "Antriebsstrangcode", "P kezdetű hibakód", ["P"], "dtc", "categories"),
        GlossaryTerm("futómű kód", "chassis code", "Fahrwerkscode", "C kezdetű hibakód", ["C"], "dtc", "categories"),
        GlossaryTerm("karosszéria kód", "body code", "Karosseriecode", "B kezdetű hibakód", ["B"], "dtc", "categories"),
        GlossaryTerm("hálózati kód", "network code", "Netzwerkcode", "U kezdetű hibakód", ["U"], "dtc", "categories"),
        GlossaryTerm("generikus kód", "generic code", "generischer Code", "Szabványos OBD-II kód", [], "dtc", "types"),
        GlossaryTerm("gyártóspecifikus kód", "manufacturer code", "herstellerspezifischer Code", "Gyártó egyedi kódja", [], "dtc", "types"),
        GlossaryTerm("hibatörlés", "clear codes", "Fehler löschen", "A tárolt hibák törlése", [], "dtc", "actions"),
        GlossaryTerm("hibaolvasás", "read codes", "Fehler auslesen", "A tárolt hibák kiolvasása", [], "dtc", "actions"),
        GlossaryTerm("freeze frame", "freeze frame", "Standbild", "A hiba pillanatában rögzített adatok", [], "dtc", "data"),
        GlossaryTerm("élő adatok", "live data", "Live-Daten", "Valós idejű szenzor adatok", [], "dtc", "data"),
        GlossaryTerm("PID", "parameter ID", "Parameter-ID", "Szenzor adat azonosító", ["PID"], "dtc", "data"),
        GlossaryTerm("hibalámpa", "malfunction indicator lamp", "Motorkontrollleuchte", "Motor hiba jelzőlámpa", ["MIL"], "dtc", "indicators"),
        GlossaryTerm("check engine", "check engine", "Motorkontrolle", "Motor hibajelző", [], "dtc", "indicators"),
        GlossaryTerm("readiness", "readiness", "Bereitschaft", "Emissziós rendszer tesztek állapota", [], "dtc", "obd"),
        GlossaryTerm("drive cycle", "drive cycle", "Fahrzyklus", "A tesztek elvégzéséhez szükséges vezetés", [], "dtc", "obd"),
        GlossaryTerm("Mode 1", "Mode 1", "Modus 1", "Élő adatok olvasása", [], "dtc", "modes"),
        GlossaryTerm("Mode 2", "Mode 2", "Modus 2", "Freeze frame adatok", [], "dtc", "modes"),
        GlossaryTerm("Mode 3", "Mode 3", "Modus 3", "Tárolt hibakódok", [], "dtc", "modes"),
        GlossaryTerm("Mode 4", "Mode 4", "Modus 4", "Hibakódok törlése", [], "dtc", "modes"),
        GlossaryTerm("Mode 5", "Mode 5", "Modus 5", "Oxigénszenzor teszt", [], "dtc", "modes"),
        GlossaryTerm("Mode 6", "Mode 6", "Modus 6", "Fedélzeti tesztek eredményei", [], "dtc", "modes"),
        GlossaryTerm("Mode 7", "Mode 7", "Modus 7", "Függőben lévő hibák", [], "dtc", "modes"),
        GlossaryTerm("Mode 9", "Mode 9", "Modus 9", "Jármű információk", [], "dtc", "modes"),
        GlossaryTerm("áramköri hiba", "circuit malfunction", "Stromkreisfehler", "Elektromos áramkör hibája", [], "dtc", "fault_types"),
        GlossaryTerm("jelszint magas", "signal high", "Signal hoch", "A jel a megengedettnél magasabb", [], "dtc", "fault_types"),
        GlossaryTerm("jelszint alacsony", "signal low", "Signal niedrig", "A jel a megengedettnél alacsonyabb", [], "dtc", "fault_types"),
        GlossaryTerm("nincs jel", "no signal", "kein Signal", "Jel hiánya", [], "dtc", "fault_types"),
        GlossaryTerm("szakaszos hiba", "intermittent", "sporadisch", "Időszakosan jelentkező hiba", [], "dtc", "fault_types"),
        GlossaryTerm("tartományon kívül", "out of range", "außerhalb des Bereichs", "Megengedett tartományon kívüli érték", [], "dtc", "fault_types"),
        GlossaryTerm("plauzibilitás hiba", "implausible", "unplausibel", "Valószínűtlen érték", [], "dtc", "fault_types"),
        GlossaryTerm("teljesítmény hiba", "performance", "Leistung", "A komponens nem megfelelő működése", [], "dtc", "fault_types"),
        GlossaryTerm("beragadt", "stuck", "festsitzend", "Mozgásképtelen állapot", [], "dtc", "fault_types"),
    ]


def generate_symptom_terms() -> List[GlossaryTerm]:
    """Generate symptom-related terms."""
    return [
        GlossaryTerm("gyújtáskimaradás", "misfire", "Fehlzündung", "A motor egyik hengerében elmarad a gyújtás", [], "symptoms", "engine"),
        GlossaryTerm("alapjárat ingadozás", "rough idle", "unruhiger Leerlauf", "Egyenetlen üresjárati fordulatszám", [], "symptoms", "engine"),
        GlossaryTerm("nehéz indulás", "hard starting", "schwerer Start", "A motor nehezen indul be", [], "symptoms", "engine"),
        GlossaryTerm("nem indul", "no start", "springt nicht an", "A motor egyáltalán nem indul", [], "symptoms", "engine"),
        GlossaryTerm("leáll", "stalling", "Abwürgen", "A motor váratlanul leáll", [], "symptoms", "engine"),
        GlossaryTerm("teljesítményvesztés", "power loss", "Leistungsverlust", "Csökkent motor teljesítmény", [], "symptoms", "engine"),
        GlossaryTerm("gyenge gyorsulás", "poor acceleration", "schlechte Beschleunigung", "Lassú gyorsulás", [], "symptoms", "engine"),
        GlossaryTerm("kopogás", "knocking", "Klopfen", "Fémesen csörgő hang a motorból", [], "symptoms", "engine"),
        GlossaryTerm("kotyogás", "rattling", "Rasseln", "Laza alkatrészek hangja", [], "symptoms", "engine"),
        GlossaryTerm("sípolás", "whistling", "Pfeifen", "Levegő szivárgás hangja", [], "symptoms", "engine"),
        GlossaryTerm("füstölés", "smoking", "Rauchen", "Füst a kipufogóból", [], "symptoms", "exhaust"),
        GlossaryTerm("fekete füst", "black smoke", "schwarzer Rauch", "Dús keverék jele", [], "symptoms", "exhaust"),
        GlossaryTerm("fehér füst", "white smoke", "weißer Rauch", "Hűtőfolyadék égés jele", [], "symptoms", "exhaust"),
        GlossaryTerm("kék füst", "blue smoke", "blauer Rauch", "Olajfogyasztás jele", [], "symptoms", "exhaust"),
        GlossaryTerm("szivárgás", "leak", "Leck", "Folyadék szivárgás", [], "symptoms", "fluids"),
        GlossaryTerm("olajfogyás", "oil consumption", "Ölverbrauch", "Túlzott olajfogyasztás", [], "symptoms", "fluids"),
        GlossaryTerm("hűtőfolyadék fogyás", "coolant loss", "Kühlmittelverlust", "Hűtőfolyadék eltűnése", [], "symptoms", "fluids"),
        GlossaryTerm("túlmelegedés", "overheating", "Überhitzung", "A motor túl magas hőmérséklete", [], "symptoms", "cooling"),
        GlossaryTerm("rezgés", "vibration", "Vibration", "Nem kívánt rezgés", [], "symptoms", "general"),
        GlossaryTerm("zaj", "noise", "Geräusch", "Rendellenes hang", [], "symptoms", "general"),
        GlossaryTerm("csikorgás", "squealing", "Quietschen", "Éles hang fékezéskor vagy induláskor", [], "symptoms", "brakes"),
        GlossaryTerm("rángatás", "jerking", "Ruckeln", "Hirtelen lökések haladás közben", [], "symptoms", "transmission"),
        GlossaryTerm("húzás", "pulling", "Ziehen", "A jármű egyik irányba húz", [], "symptoms", "steering"),
        GlossaryTerm("kormány remegés", "steering wheel vibration", "Lenkradflattern", "A kormány rezeg", [], "symptoms", "steering"),
        GlossaryTerm("nehéz kormányzás", "heavy steering", "schwere Lenkung", "A kormány nehezen forgatható", [], "symptoms", "steering"),
        GlossaryTerm("fékpedál süllyedés", "brake pedal sinking", "Bremspedal sinkt", "A fékpedál lassan süllyed", [], "symptoms", "brakes"),
        GlossaryTerm("puha fékpedál", "soft brake pedal", "weiches Bremspedal", "A fékpedál nem ellenáll", [], "symptoms", "brakes"),
        GlossaryTerm("égett szag", "burning smell", "Brandgeruch", "Égett szag a járműből", [], "symptoms", "general"),
        GlossaryTerm("benzinszag", "gasoline smell", "Benzingeruch", "Üzemanyag szag", [], "symptoms", "fuel"),
        GlossaryTerm("villogó lámpa", "flashing light", "blinkende Leuchte", "Figyelmeztető lámpa villog", [], "symptoms", "indicators"),
        GlossaryTerm("folyamatos lámpa", "steady light", "dauerhafte Leuchte", "Figyelmeztető lámpa folyamatosan ég", [], "symptoms", "indicators"),
        GlossaryTerm("üzemanyag fogyasztás növekedés", "increased fuel consumption", "erhöhter Kraftstoffverbrauch", "Megnövekedett fogyasztás", [], "symptoms", "fuel"),
        GlossaryTerm("váltási nehézség", "shifting difficulty", "Schaltschwierigkeiten", "Nehéz váltás", [], "symptoms", "transmission"),
        GlossaryTerm("ugráló fordulatszám", "surging RPM", "schwankende Drehzahl", "Ingadozó fordulatszám", [], "symptoms", "engine"),
        GlossaryTerm("holtjáték", "play", "Spiel", "Nem kívánt mozgás", [], "symptoms", "suspension"),
    ]


def generate_repair_terms() -> List[GlossaryTerm]:
    """Generate repair procedure terms."""
    return [
        GlossaryTerm("csere", "replacement", "Austausch", "Alkatrész kicserélése", [], "repairs", "actions"),
        GlossaryTerm("javítás", "repair", "Reparatur", "Alkatrész helyreállítása", [], "repairs", "actions"),
        GlossaryTerm("beállítás", "adjustment", "Einstellung", "Paraméter módosítása", [], "repairs", "actions"),
        GlossaryTerm("tisztítás", "cleaning", "Reinigung", "Szennyeződés eltávolítása", [], "repairs", "actions"),
        GlossaryTerm("ellenőrzés", "inspection", "Prüfung", "Állapot vizsgálata", [], "repairs", "actions"),
        GlossaryTerm("mérés", "measurement", "Messung", "Érték megállapítása", [], "repairs", "actions"),
        GlossaryTerm("diagnosztika", "diagnosis", "Diagnose", "Hiba megállapítása", [], "repairs", "actions"),
        GlossaryTerm("szétszerelés", "disassembly", "Demontage", "Alkatrészek szétbontása", [], "repairs", "actions"),
        GlossaryTerm("összeszerelés", "assembly", "Montage", "Alkatrészek összeépítése", [], "repairs", "actions"),
        GlossaryTerm("beszerelés", "installation", "Einbau", "Alkatrész behelyezése", [], "repairs", "actions"),
        GlossaryTerm("kiszerelés", "removal", "Ausbau", "Alkatrész eltávolítása", [], "repairs", "actions"),
        GlossaryTerm("felújítás", "rebuild", "Überholung", "Teljes helyreállítás", [], "repairs", "actions"),
        GlossaryTerm("regenerálás", "remanufacturing", "Aufarbeitung", "Használt alkatrész felújítása", [], "repairs", "actions"),
        GlossaryTerm("olajcsere", "oil change", "Ölwechsel", "Motorolaj cseréje", [], "repairs", "service"),
        GlossaryTerm("szűrőcsere", "filter change", "Filterwechsel", "Szűrők cseréje", [], "repairs", "service"),
        GlossaryTerm("fékfolyadék csere", "brake fluid change", "Bremsflüssigkeitswechsel", "Fékfolyadék cseréje", [], "repairs", "service"),
        GlossaryTerm("hűtőfolyadék csere", "coolant change", "Kühlmittelwechsel", "Hűtőfolyadék cseréje", [], "repairs", "service"),
        GlossaryTerm("váltóolaj csere", "transmission fluid change", "Getriebeölwechsel", "Váltóolaj cseréje", [], "repairs", "service"),
        GlossaryTerm("gyertyacsere", "spark plug change", "Zündkerzenwechsel", "Gyújtógyertya cseréje", [], "repairs", "service"),
        GlossaryTerm("vezérműszíj csere", "timing belt change", "Zahnriemenwechsel", "Vezérműszíj cseréje", [], "repairs", "service"),
        GlossaryTerm("fékbetét csere", "brake pad change", "Bremsbelagwechsel", "Fékbetétek cseréje", [], "repairs", "brakes"),
        GlossaryTerm("féktárcsa csere", "brake disc change", "Bremsscheibenwechsel", "Féktárcsák cseréje", [], "repairs", "brakes"),
        GlossaryTerm("féknyereg felújítás", "caliper rebuild", "Bremssattelüberholung", "Féknyereg helyreállítása", [], "repairs", "brakes"),
        GlossaryTerm("légtelenítés", "bleeding", "Entlüften", "Levegő eltávolítása", [], "repairs", "brakes"),
        GlossaryTerm("kuplung csere", "clutch replacement", "Kupplungswechsel", "Kuplung cseréje", [], "repairs", "transmission"),
        GlossaryTerm("kerékcsapágy csere", "wheel bearing replacement", "Radlagerwechsel", "Kerékcsapágy cseréje", [], "repairs", "suspension"),
        GlossaryTerm("lengéscsillapító csere", "shock absorber replacement", "Stoßdämpferwechsel", "Lengéscsillapító cseréje", [], "repairs", "suspension"),
        GlossaryTerm("gömbfej csere", "ball joint replacement", "Kugelgelenkaustausch", "Gömbfej cseréje", [], "repairs", "suspension"),
        GlossaryTerm("kerékállítás", "wheel alignment", "Achsvermessung", "Futómű beállítás", [], "repairs", "suspension"),
        GlossaryTerm("kiegyensúlyozás", "balancing", "Auswuchten", "Kerék kiegyensúlyozása", [], "repairs", "tires"),
        GlossaryTerm("gumiszerelés", "tire mounting", "Reifenmontage", "Gumiabroncs felszerelése", [], "repairs", "tires"),
        GlossaryTerm("defektjavítás", "puncture repair", "Reifenreparatur", "Defektes gumi javítása", [], "repairs", "tires"),
        GlossaryTerm("ECU programozás", "ECU programming", "Steuergeräteprogrammierung", "Vezérlőegység programozás", [], "repairs", "electrical"),
        GlossaryTerm("kódolás", "coding", "Codierung", "Vezérlőegység beállítása", [], "repairs", "electrical"),
        GlossaryTerm("adaptáció", "adaptation", "Adaption", "Rendszer betanítása", [], "repairs", "electrical"),
        GlossaryTerm("hegesztés", "welding", "Schweißen", "Fém összekötése hővel", [], "repairs", "bodywork"),
        GlossaryTerm("fényezés", "painting", "Lackierung", "Festés", [], "repairs", "bodywork"),
        GlossaryTerm("karosszériajavítás", "body repair", "Karosseriereparatur", "Karosszéria helyreállítása", [], "repairs", "bodywork"),
        GlossaryTerm("üvegcsere", "glass replacement", "Glasersatz", "Ablak cseréje", [], "repairs", "bodywork"),
    ]


def generate_slang_terms() -> List[GlossaryTerm]:
    """Generate common mechanic slang terms."""
    return [
        GlossaryTerm("dögös", "dead", "tot", "Nem működő alkatrész", [], "slang", "condition"),
        GlossaryTerm("tönkrement", "busted", "kaputt", "Elromlott", [], "slang", "condition"),
        GlossaryTerm("elfáradt", "worn out", "verschlissen", "Elkopott", [], "slang", "condition"),
        GlossaryTerm("meghalt", "died", "gestorben", "Működésképtelen lett", [], "slang", "condition"),
        GlossaryTerm("szétesett", "fell apart", "auseinandergefallen", "Tönkrement", [], "slang", "condition"),
        GlossaryTerm("lejárt", "expired", "abgelaufen", "Élettartamot meghaladta", [], "slang", "condition"),
        GlossaryTerm("kilyukadt", "punctured", "durchlöchert", "Lyukas lett", [], "slang", "condition"),
        GlossaryTerm("megette a rozsda", "rusted out", "durchgerostet", "Teljesen átrozsdásodott", [], "slang", "condition"),
        GlossaryTerm("berozsdált", "seized up", "festgerostet", "Rozsdától megakadt", [], "slang", "condition"),
        GlossaryTerm("beeresztett", "leaked", "ausgelaufen", "Szivárog", [], "slang", "condition"),
        GlossaryTerm("beégett", "burnt", "verbrannt", "Égéstől sérült", [], "slang", "condition"),
        GlossaryTerm("kipöccent", "popped out", "herausgesprungen", "Kijött a helyéről", [], "slang", "condition"),
        GlossaryTerm("gáz adása", "step on it", "Gas geben", "Gyorsítás", [], "slang", "driving"),
        GlossaryTerm("fékbe állni", "hit the brakes", "bremsen", "Erősen fékezni", [], "slang", "driving"),
        GlossaryTerm("bikázás", "jump start", "Starthilfe", "Indítás külső áramforrásból", [], "slang", "service"),
        GlossaryTerm("meghúzni", "tighten", "festziehen", "Megszorítani", [], "slang", "repair"),
        GlossaryTerm("kilazult", "loosened", "gelockert", "Meglazult", [], "slang", "condition"),
        GlossaryTerm("beszorult", "jammed", "eingeklemmt", "Beragadt", [], "slang", "condition"),
        GlossaryTerm("nyekergés", "squeaking", "Quietschen", "Nyikorgó hang", [], "slang", "symptoms"),
        GlossaryTerm("dörömbölés", "thumping", "Poltern", "Tompa ütődő hang", [], "slang", "symptoms"),
        GlossaryTerm("zörgés", "clanking", "Klappern", "Fémes csörgés", [], "slang", "symptoms"),
        GlossaryTerm("szisszenés", "hissing", "Zischen", "Levegő vagy gőz hang", [], "slang", "symptoms"),
        GlossaryTerm("morgás", "growling", "Brummen", "Mély morgó hang", [], "slang", "symptoms"),
        GlossaryTerm("sivítás", "screeching", "Kreischen", "Éles, fémesen hang", [], "slang", "symptoms"),
        GlossaryTerm("kattogás", "clicking", "Klicken", "Ismétlődő kattanó hang", [], "slang", "symptoms"),
        GlossaryTerm("pöfögés", "sputtering", "Stottern", "Egyenetlen motorhang", [], "slang", "symptoms"),
        GlossaryTerm("átfordulás", "rollover", "Überschlag", "Átborulás", [], "slang", "accident"),
        GlossaryTerm("karambol", "crash", "Unfall", "Ütközés", [], "slang", "accident"),
        GlossaryTerm("totálkár", "total loss", "Totalschaden", "Gazdaságilag javíthatatlan", [], "slang", "accident"),
        GlossaryTerm("roncs", "wreck", "Wrack", "Súlyosan sérült jármű", [], "slang", "condition"),
        GlossaryTerm("veterán", "classic", "Oldtimer", "Öreg de értékes autó", [], "slang", "general"),
        GlossaryTerm("tragacs", "beater", "Schrottkarre", "Rossz állapotú autó", [], "slang", "general"),
        GlossaryTerm("szürkeimport", "grey import", "Grauimport", "Nem hivatalos importból", [], "slang", "general"),
        GlossaryTerm("gyári", "OEM", "Original", "Eredeti gyári alkatrész", ["OEM"], "slang", "parts"),
        GlossaryTerm("utángyártott", "aftermarket", "Nachbau", "Nem gyári alkatrész", [], "slang", "parts"),
        GlossaryTerm("bontott", "used", "gebraucht", "Használt alkatrész bontóból", [], "slang", "parts"),
        GlossaryTerm("kínai", "Chinese", "chinesisch", "Olcsó, gyenge minőségű alkatrész", [], "slang", "parts"),
        GlossaryTerm("fuser", "hack job", "Pfusch", "Rossz minőségű javítás", [], "slang", "repair"),
        GlossaryTerm("kontár", "hack", "Pfuscher", "Rossz minőségű szerelő", [], "slang", "general"),
        GlossaryTerm("szaki", "pro", "Fachmann", "Tapasztalt szerelő", [], "slang", "general"),
    ]


def generate_glossary() -> AutomotiveGlossary:
    """Generate the complete glossary."""
    glossary = AutomotiveGlossary()

    # Generate all terms
    generators = [
        ("engine", generate_engine_terms),
        ("transmission", generate_transmission_terms),
        ("electrical", generate_electrical_terms),
        ("brake", generate_brake_terms),
        ("suspension", generate_suspension_terms),
        ("body", generate_body_terms),
        ("tools", generate_tool_terms),
        ("specifications", generate_specification_terms),
        ("dtc", generate_dtc_terms),
        ("symptoms", generate_symptom_terms),
        ("repairs", generate_repair_terms),
        ("slang", generate_slang_terms),
    ]

    for category, generator in generators:
        terms = generator()
        for term in terms:
            glossary.add_term(term)
        logger.info(f"Added {len(terms)} {category} terms")

    return glossary


def validate_glossary(glossary: AutomotiveGlossary) -> Dict[str, Any]:
    """Validate the glossary for completeness and quality."""
    issues = []

    for key, term in glossary.terms.items():
        if not term.hungarian:
            issues.append(f"Missing Hungarian: {key}")
        if not term.english:
            issues.append(f"Missing English: {key}")
        if not term.definition:
            issues.append(f"Missing definition: {key}")

    return {
        "total_terms": len(glossary.terms),
        "issues_count": len(issues),
        "issues": issues[:20],  # First 20 issues
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hungarian Automotive Terminology Glossary"
    )
    parser.add_argument("--generate", action="store_true", help="Generate full glossary")
    parser.add_argument("--search", type=str, help="Search for terms")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--validate", action="store_true", help="Validate glossary")
    parser.add_argument("--category", type=str, help="List terms in category")

    args = parser.parse_args()

    if args.generate:
        logger.info("Generating comprehensive glossary...")
        glossary = generate_glossary()
        glossary.save()
        stats = glossary.get_stats()
        print(f"\nGenerated {stats['total_terms']} terms in {len(stats['categories'])} categories")
        for cat, count in sorted(stats['categories'].items()):
            print(f"  {cat}: {count}")

    elif args.search:
        glossary = AutomotiveGlossary.load()
        results = glossary.search(args.search)
        print(f"\nFound {len(results)} results for '{args.search}':\n")
        for r in results[:20]:
            print(f"  HU: {r['hungarian']}")
            print(f"  EN: {r['english']}")
            print(f"  DE: {r['german']}")
            print(f"  Def: {r['definition']}")
            print()

    elif args.stats:
        glossary = AutomotiveGlossary.load()
        stats = glossary.get_stats()
        print(f"\nGlossary Statistics:")
        print(f"  Total terms: {stats['total_terms']}")
        print(f"  Index words: {stats['index_words']}")
        print(f"\n  By category:")
        for cat, count in sorted(stats['categories'].items()):
            print(f"    {cat}: {count}")

    elif args.validate:
        glossary = AutomotiveGlossary.load()
        result = validate_glossary(glossary)
        print(f"\nValidation Results:")
        print(f"  Total terms: {result['total_terms']}")
        print(f"  Issues found: {result['issues_count']}")
        if result['issues']:
            print(f"\n  Sample issues:")
            for issue in result['issues']:
                print(f"    - {issue}")

    elif args.category:
        glossary = AutomotiveGlossary.load()
        terms = glossary.get_by_category(args.category)
        print(f"\nTerms in '{args.category}' ({len(terms)}):\n")
        for t in terms:
            print(f"  {t['hungarian']} - {t['english']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
