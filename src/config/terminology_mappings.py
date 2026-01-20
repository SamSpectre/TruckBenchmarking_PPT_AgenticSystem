"""
Worldwide OEM Terminology Mappings for E-Powertrain Spec Extraction

This module provides comprehensive terminology mappings for extracting
electric vehicle specifications from OEM websites worldwide.

Supports: English, German, French, Spanish, Italian, Dutch, Swedish,
          Norwegian, Danish, Polish, Portuguese, Japanese (romanized)

OEMs covered: MAN, Mercedes-Benz, Volvo, Scania, DAF, IVECO, Renault,
              Nikola, BYD, Hyundai, Hino, Mitsubishi Fuso, Isuzu, etc.
"""

# =====================================================================
# BATTERY CAPACITY TERMINOLOGY (kWh)
# =====================================================================
BATTERY_CAPACITY_TERMS = {
    # English - Standard terms
    "battery capacity", "battery", "battery pack", "battery packs",
    "energy capacity", "energy content", "energy storage", "kwh",
    "usable capacity", "total capacity", "net capacity", "gross capacity",
    "battery energy", "pack capacity", "cell capacity",

    # English - Additional variations
    "energy storage system", "ess", "traction battery",
    "traction battery capacity", "accumulator", "accumulator capacity",
    "battery system", "battery module", "battery modules",
    "installed capacity", "installed battery capacity",
    "nominal capacity", "nominal battery capacity",
    "available energy", "usable energy", "energy reservoir",
    "power storage", "storage capacity", "onboard energy",

    # German - Standard terms
    "batteriekapazität", "batterie", "akkukapazität", "akku",
    "energieinhalt", "energiespeicher", "batteriepack", "batteriepacks",
    "nutzbare kapazität", "gesamtkapazität", "netto-kapazität",

    # German - Additional variations
    "traktionsbatterie", "energiespeichersystem",
    "akkumulator", "speicherkapazität", "installierte kapazität",

    # French
    "capacité batterie", "capacité de la batterie", "batterie",
    "contenu énergétique", "stockage d'énergie", "capacité énergétique",

    # Spanish
    "capacidad de batería", "batería", "capacidad energética",
    "contenido energético", "almacenamiento de energía",

    # Italian
    "capacità batteria", "batteria", "contenuto energetico",
    "accumulo di energia", "capacità energetica",

    # Dutch
    "batterijcapaciteit", "batterij", "accu", "accucapaciteit",
    "energieinhoud", "energieopslag",

    # Swedish
    "batterikapacitet", "batteri", "energiinnehåll", "energilagring",

    # Norwegian
    "batterikapasitet", "batteri", "energiinnhold",

    # Portuguese
    "capacidade da bateria", "bateria", "conteúdo energético",

    # Technical abbreviations
    "li-ion", "lfp", "nmc", "ncm", "nca",
}

# =====================================================================
# MOTOR POWER TERMINOLOGY (kW)
# =====================================================================
MOTOR_POWER_TERMS = {
    # English - Standard terms
    "motor power", "power", "drive power", "electric motor",
    "continuous power", "peak power", "max power", "maximum power",
    "rated power", "output power", "motor output", "drive output",
    "propulsion power", "traction power", "e-motor", "electric drive",
    "powertrain output", "system power", "total power",
    "continuous output", "nominal power", "net power",

    # English - E-Axle terminology (integrated motor-axle systems)
    # Used by Volvo, Scania, ZF, and others for integrated e-drive axles
    "e-axle power", "eaxle power", "electric axle power",
    "e-axle output", "eAxle power", "eAxle output", "e-drive power",
    "axle power", "drive axle power", "integrated axle power",
    "rear axle power", "electric drive axle power",

    # German - Standard terms
    "motorleistung", "leistung", "antriebsleistung", "elektromotor",
    "dauerleistung", "spitzenleistung", "maximalleistung", "nennleistung",
    "antriebsausgabe", "e-antrieb", "elektrischer antrieb",
    "systemleistung", "gesamtleistung",

    # German - E-Axle terms
    "e-achse leistung", "elektroachse leistung",
    "antriebsachse leistung", "elektrische achse leistung",

    # French
    "puissance moteur", "puissance", "puissance de traction",
    "moteur électrique", "puissance nominale", "puissance maximale",
    "puissance continue", "puissance de pointe", "puissance nette",

    # Spanish
    "potencia del motor", "potencia", "potencia de tracción",
    "motor eléctrico", "potencia nominal", "potencia máxima",
    "potencia continua", "potencia pico",

    # Italian
    "potenza motore", "potenza", "potenza di trazione",
    "motore elettrico", "potenza nominale", "potenza massima",
    "potenza continua", "potenza di picco",

    # Dutch
    "motorvermogen", "vermogen", "aandrijfvermogen",
    "elektromotor", "nominaal vermogen", "piekvermogen",
    "continu vermogen", "maximaal vermogen",

    # Swedish
    "motoreffekt", "effekt", "driveffekt", "elmotor",
    "kontinuerlig effekt", "toppeffekt", "maxeffekt",

    # Unit indicators (important for extraction)
    "kw", "ps", "hp", "cv", "bhp", "pk",  # PS/hp need conversion
}

# =====================================================================
# MOTOR TORQUE TERMINOLOGY (Nm)
# =====================================================================
MOTOR_TORQUE_TERMS = {
    # English - Standard terms
    "torque", "motor torque", "drive torque", "max torque",
    "maximum torque", "peak torque", "wheel torque",
    "output torque", "rated torque", "nominal torque",

    # English - E-Axle terminology (integrated motor-axle systems)
    # Used by Volvo, Scania, ZF, and others for integrated e-drive axles
    "e-axle torque", "eaxle torque", "electric axle torque",
    "e-axle output torque", "eAxle torque", "e-drive torque",
    "rear axle torque", "drive axle torque", "axle torque",
    "integrated axle torque", "electric drive axle torque",

    # German - Standard terms
    "drehmoment", "motordrehmoment", "antriebsdrehmoment",
    "maximales drehmoment", "max. drehmoment",

    # German - E-Axle terms
    "e-achse drehmoment", "elektroachse drehmoment",
    "antriebsachse drehmoment", "elektrische achse drehmoment",

    # French
    "couple", "couple moteur", "couple maximal", "couple de traction",

    # Spanish
    "par motor", "torque", "par máximo", "par de torsión",

    # Italian
    "coppia", "coppia motore", "coppia massima",

    # Dutch
    "koppel", "motorkoppel", "maximaal koppel", "aandrijfkoppel",

    # Swedish
    "vridmoment", "motorvridmoment", "maxvridmoment",

    # Unit indicators
    "nm", "newton-metre", "newton-meter", "n·m", "n.m",
}

# =====================================================================
# RANGE TERMINOLOGY (km)
# =====================================================================
RANGE_TERMS = {
    # English
    "range", "driving range", "electric range", "battery range",
    "maximum range", "total range", "real-world range", "wltp range",
    "single charge range", "distance", "autonomy", "travel distance",

    # German
    "reichweite", "fahrreichweite", "elektrische reichweite",
    "maximale reichweite", "gesamtreichweite", "wltp-reichweite",
    "einfache reichweite", "aktionsradius",

    # French
    "autonomie", "autonomie électrique", "distance parcourue",
    "autonomie maximale", "rayon d'action",

    # Spanish
    "autonomía", "alcance", "autonomía eléctrica", "rango",
    "autonomía máxima", "distancia de conducción",

    # Italian
    "autonomia", "raggio d'azione", "autonomia elettrica",
    "autonomia massima", "percorrenza",

    # Dutch
    "bereik", "actieradius", "rijbereik", "elektrisch bereik",
    "maximaal bereik", "range",

    # Swedish
    "räckvidd", "körsträcka", "elektrisk räckvidd", "maxräckvidd",

    # Norwegian
    "rekkevidde", "kjørelengde", "elektrisk rekkevidde",

    # Unit indicators
    "km", "kilometers", "kilometres", "mi", "miles",
}

# =====================================================================
# CHARGING POWER TERMINOLOGY (kW)
# =====================================================================
DC_CHARGING_TERMS = {
    # English - Standard terms
    "dc charging", "fast charging", "rapid charging", "quick charging",
    "dc fast charging", "charging power", "charge power", "dc power",
    "ccs", "ccs2", "combo", "combo2", "dc charging power",
    "maximum charging", "max charging", "peak charging",
    "charging capacity", "charger power",

    # English - High-power and depot charging variations
    "hpc", "high power charging", "high-power charging",
    "ultrafast charging", "ultra-fast charging", "ultra fast",
    "depot charging", "depot charger", "overnight charging",
    "opportunity charging", "en-route charging",
    "combined charging system", "ccs1", "ccs type 2",
    "dc combo", "combo charging", "fast charge capability",

    # German - Standard terms
    "dc-laden", "schnellladen", "schnellladung", "ladeleistung",
    "dc-ladeleistung", "gleichstromladen", "ccs-laden",
    "maximale ladeleistung", "ladekapazität",

    # German - High-power variations
    "hochleistungsladen", "hpc-laden", "schnellladesystem",
    "depotladen", "nachtladen",

    # French
    "charge rapide", "charge dc", "puissance de charge",
    "recharge rapide", "charge ccs", "puissance de recharge",

    # Spanish
    "carga rápida", "carga dc", "potencia de carga",
    "recarga rápida", "carga ccs", "capacidad de carga",

    # Italian
    "ricarica rapida", "ricarica dc", "potenza di ricarica",
    "ricarica veloce", "ricarica ccs",

    # Dutch
    "snelladen", "dc-laden", "laadvermogen", "snellaadvermogen",
    "ccs-laden", "laadcapaciteit",

    # Swedish
    "snabbladdning", "dc-laddning", "laddeffekt", "laddkapacitet",
}

MCS_CHARGING_TERMS = {
    # English
    "mcs", "megawatt charging", "megawatt charging system",
    "high power charging", "hpc", "mega charging", "mw charging",
    "megawatt", "ultra-fast charging",

    # German
    "mcs", "megawatt-laden", "megawatt charging system",
    "hochleistungsladen", "hpc",

    # French
    "mcs", "charge mégawatt", "recharge haute puissance",

    # Spanish
    "mcs", "carga megavatio", "carga de alta potencia",

    # Italian
    "mcs", "ricarica megawatt", "ricarica ad alta potenza",

    # Indicators for MCS (750+ kW systems)
    "750 kw", "750kw", "1 mw", "1mw", "megawatt",
}

# =====================================================================
# CHARGING TIME TERMINOLOGY (minutes)
# =====================================================================
CHARGING_TIME_TERMS = {
    # English
    "charging time", "charge time", "recharge time", "charging duration",
    "time to charge", "minutes to 80%", "10-80%", "20-80%", "0-80%",
    "fast charge time", "dc charge time", "full charge time",

    # German
    "ladezeit", "ladedauer", "aufladezeit", "schnellladezeit",
    "zeit zum laden", "minuten bis 80%",

    # French
    "temps de charge", "durée de charge", "temps de recharge",
    "temps de charge rapide",

    # Spanish
    "tiempo de carga", "duración de carga", "tiempo de recarga",

    # Italian
    "tempo di ricarica", "durata ricarica", "tempo di carica",

    # Dutch
    "laadtijd", "oplaadtijd", "snellaadtijd",

    # Swedish
    "laddtid", "laddningstid", "snabbladdningstid",

    # Unit indicators
    "min", "mins", "minutes", "minuten", "minutos", "minuti",
}

# =====================================================================
# WEIGHT TERMINOLOGY (kg)
# =====================================================================
GVW_TERMS = {
    # English - Standard terms (truck ALONE weight, typically 18-28t)
    "gvw", "gross vehicle weight", "gvwr", "gross vehicle weight rating",
    "permissible weight", "maximum weight", "total weight",
    "vehicle weight", "technical weight", "max laden weight",
    "permissible total weight", "ptw",

    # English - Additional variations
    "technically permissible weight", "laden weight",
    "operating weight", "maximum operating weight",
    "authorized weight", "maximum authorized weight",
    "maximum permissible weight", "permissible laden weight",

    # German - Standard terms
    "zulässiges gesamtgewicht", "gesamtgewicht", "technisches gewicht",
    "zgg", "höchstzulässiges gesamtgewicht", "fahrzeuggewicht",

    # German - Additional variations
    "technisch zulässiges gesamtgewicht", "tzgg",
    "betriebsgewicht", "zulässige gesamtmasse",

    # French
    "ptac", "poids total autorisé en charge", "poids total",
    "masse maximale", "poids brut",

    # Spanish
    "pma", "peso máximo autorizado", "peso bruto vehicular",
    "peso total", "masa máxima",

    # Italian
    "mtt", "massa totale a terra", "peso complessivo",
    "peso lordo", "massa massima",

    # Dutch
    "ggm", "toegestaan totaalgewicht", "maximaal gewicht",
    "technisch gewicht",

    # Swedish
    "totalvikt", "bruttovikt", "maxvikt", "tjänstevikt",
}

GCW_TERMS = {
    # English - Standard terms (truck + trailer COMBINED, typically 40-44t)
    "gcw", "gross combination weight", "gcwr", "gross combined weight",
    "train weight", "combination weight", "with trailer",
    "maximum train weight", "combined weight", "articulated weight",

    # English - Additional variations
    # NOTE: Values 40-44t for semitrailer trucks are typically GCW, NOT GVW!
    "tractor-trailer weight", "road train weight", "total train mass",
    "combination mass", "total combination weight",
    "maximum combination weight", "gross train weight",

    # German - Standard terms
    "zulässiges zuggesamtgewicht", "zuggesamtgewicht", "zgg",
    "gesamtzuggewicht", "sattelzuggewicht",

    # German - Additional variations
    "maximales zuggewicht", "zulässige gesamtmasse zug",

    # French
    "ptra", "poids total roulant autorisé", "poids en charge",
    "masse en charge", "poids combiné",

    # Spanish
    "mma", "masa máxima autorizada del conjunto",
    "peso combinado", "peso del tren",

    # Italian
    "mmc", "massa massima a carico", "peso combinato",
    "massa complessiva",

    # Dutch
    "gcm", "maximale treingewicht", "combinatiegewicht",
    "totaal treingewicht",

    # Swedish
    "totalvikt släp", "bruttovikt kombination", "tågvikt",
}

PAYLOAD_TERMS = {
    # English
    "payload", "payload capacity", "cargo capacity", "load capacity",
    "carrying capacity", "useful load", "net payload",

    # German
    "nutzlast", "ladekapazität", "tragfähigkeit", "zuladung",
    "netto-nutzlast",

    # French
    "charge utile", "capacité de charge", "charge maximale",

    # Spanish
    "carga útil", "capacidad de carga", "carga máxima",

    # Italian
    "portata utile", "capacità di carico", "carico utile",

    # Dutch
    "laadvermogen", "nuttige lading", "draagvermogen",

    # Swedish
    "lastkapacitet", "nyttolast", "lastförmåga",
}

# =====================================================================
# UNIT CONVERSION FACTORS
# =====================================================================
UNIT_CONVERSIONS = {
    # Power conversions to kW
    "ps_to_kw": 0.7355,      # 1 PS = 0.7355 kW (metric horsepower)
    "hp_to_kw": 0.7457,      # 1 hp = 0.7457 kW (imperial horsepower)
    "cv_to_kw": 0.7355,      # 1 CV = 0.7355 kW (French/Spanish)
    "pk_to_kw": 0.7355,      # 1 pk = 0.7355 kW (Dutch)
    "bhp_to_kw": 0.7457,     # 1 bhp = 0.7457 kW (brake horsepower)

    # Weight conversions to kg
    "t_to_kg": 1000,         # 1 tonne = 1000 kg
    "ton_to_kg": 1000,       # 1 metric ton = 1000 kg
    "lb_to_kg": 0.4536,      # 1 pound = 0.4536 kg
    "lbs_to_kg": 0.4536,

    # Distance conversions to km
    "mi_to_km": 1.6093,      # 1 mile = 1.6093 km
    "miles_to_km": 1.6093,
}

# =====================================================================
# OEM-SPECIFIC NAMING PATTERNS
# =====================================================================
OEM_PATTERNS = {
    # MAN Truck & Bus
    "man": {
        "models": ["etgx", "etgs", "etgl", "etgm", "lion"],
        "configs": ["4x2", "6x2", "6x4", "8x4", "sattel", "chassis"],
    },
    # Mercedes-Benz / Daimler Truck
    "mercedes": {
        "models": ["eactros", "eeconic", "genh2", "esprinter"],
        "configs": ["4x2", "6x2", "6x4", "rigid", "tractor"],
    },
    # Volvo Trucks
    "volvo": {
        "models": ["fh electric", "fm electric", "fmx electric", "fe electric", "fl electric", "vnr electric"],
        "configs": ["4x2", "6x2", "6x4", "rigid", "tractor"],
    },
    # Scania
    "scania": {
        "models": ["bev", "25 p", "35 p", "45 r"],
        "configs": ["4x2", "6x2", "6x4", "rigid", "tractor"],
    },
    # DAF
    "daf": {
        "models": ["xf electric", "xd electric", "xb electric", "cf electric", "lf electric"],
        "configs": ["4x2", "6x2", "fa", "fan", "ftp"],
    },
    # IVECO
    "iveco": {
        "models": ["s-eway", "s-way", "nikola tre", "daily electric", "eurocargo electric"],
        "configs": ["4x2", "6x2", "np", "lng", "cng"],
    },
    # Renault Trucks
    "renault": {
        "models": ["e-tech d", "e-tech c", "e-tech t", "master z.e."],
        "configs": ["4x2", "6x2", "rigid", "tractor"],
    },
    # BYD
    "byd": {
        "models": ["8tt", "etm6"],
        "configs": ["4x2", "6x4", "day cab", "sleeper"],
    },
    # Hyundai / Xcient
    "hyundai": {
        "models": ["xcient fuel cell", "xcient electric", "elec city"],
        "configs": ["4x2", "6x4", "tractor", "cargo"],
    },
    # Nikola
    "nikola": {
        "models": ["tre bev", "tre fcev", "two"],
        "configs": ["4x2", "6x4", "day cab", "sleeper"],
    },
    # Freightliner / Daimler NA
    "freightliner": {
        "models": ["ecascadia", "em2"],
        "configs": ["4x2", "6x4", "day cab", "sleeper"],
    },
}

# =====================================================================
# SEMANTIC EQUIVALENCES
# These terms mean THE SAME THING but are named differently by OEMs
# =====================================================================
SEMANTIC_EQUIVALENCES = {
    # Motor torque equivalences
    # Different OEMs use different terms for the same measurement
    "motor_torque_nm": [
        "motor torque",          # Standard term
        "e-axle torque",         # Volvo, Scania - integrated motor-axle
        "eaxle torque",          # Alternative spelling
        "electric axle torque",  # Full form
        "drive torque",          # Generic
        "wheel torque",          # At wheels
        "output torque",         # Motor output
        "axle torque",           # Generic axle
        "drehmoment",            # German
        "e-achse drehmoment",    # German e-axle
    ],

    # Motor power equivalences
    "motor_power_kw": [
        "motor power",           # Standard term
        "e-axle power",          # Volvo, Scania - integrated motor-axle
        "eaxle power",           # Alternative spelling
        "electric axle power",   # Full form
        "drive power",           # Generic
        "propulsion power",      # Traction
        "system power",          # Total output
        "axle power",            # Generic axle
        "motorleistung",         # German
        "e-achse leistung",      # German e-axle
    ],

    # Battery capacity equivalences
    "battery_capacity_kwh": [
        "battery capacity",      # Standard term
        "energy storage",        # ESS terminology
        "traction battery",      # Industrial term
        "accumulator",           # Technical term
        "battery pack",          # Pack level
        "batteriekapazität",     # German
        "energiespeicher",       # German ESS
    ],

    # Charging power equivalences
    "dc_charging_kw": [
        "dc charging",           # Standard term
        "fast charging",         # Common term
        "ccs charging",          # Standard type
        "hpc",                   # High power charging
        "depot charging",        # Fleet term
        "schnellladen",          # German
    ],
}


def build_semantic_equivalence_prompt() -> str:
    """
    Build a semantic equivalence section for the LLM extraction prompt.

    This teaches the LLM that different terms mean the same thing
    and should be extracted to the same field.
    """
    prompt_parts = []

    prompt_parts.append("""
**SEMANTIC EQUIVALENCES - These terms mean THE SAME THING:**

Different OEMs use different terminology for identical measurements.
Extract ALL of these variations into the standard field:
""")

    for field, equivalents in SEMANTIC_EQUIVALENCES.items():
        equiv_str = " = ".join(f'"{e}"' for e in equivalents[:5])
        prompt_parts.append(f"- {equiv_str} → **{field}**")

    prompt_parts.append("""
**IMPORTANT E-AXLE NOTE:**
OEMs like Volvo, Scania, and ZF use "e-axle" terminology for integrated motor-axle systems.
"e-axle torque" IS "motor torque" - extract it into motor_torque_nm!
"e-axle power" IS "motor power" - extract it into motor_power_kw!
""")

    return "\n".join(prompt_parts)


# =====================================================================
# HELPER FUNCTION: Check if term matches any in a set
# =====================================================================
def matches_terminology(text: str, term_set: set) -> bool:
    """Check if text contains any term from the terminology set."""
    text_lower = text.lower()
    return any(term in text_lower for term in term_set)


def get_all_terms_for_field(field_name: str) -> set:
    """Get all terminology terms for a specific field."""
    field_mapping = {
        "battery_capacity_kwh": BATTERY_CAPACITY_TERMS,
        "motor_power_kw": MOTOR_POWER_TERMS,
        "motor_torque_nm": MOTOR_TORQUE_TERMS,
        "range_km": RANGE_TERMS,
        "dc_charging_kw": DC_CHARGING_TERMS,
        "mcs_charging_kw": MCS_CHARGING_TERMS,
        "charging_time_minutes": CHARGING_TIME_TERMS,
        "gvw_kg": GVW_TERMS,
        "gcw_kg": GCW_TERMS,
        "payload_capacity_kg": PAYLOAD_TERMS,
    }
    return field_mapping.get(field_name, set())


def build_terminology_prompt() -> str:
    """
    Build a comprehensive terminology mapping section for the LLM extraction prompt.

    Returns a formatted string that can be inserted into the extraction prompt.
    """
    prompt_parts = []

    prompt_parts.append("""
**WORLDWIDE TERMINOLOGY MAPPING - Match ANY of these terms to our fields:**

The same specification can be written in many different ways across worldwide OEMs.
Match ALL of these variations to extract the correct data.

""")

    field_terms = [
        ("battery_capacity_kwh", BATTERY_CAPACITY_TERMS, "Battery capacity in kWh"),
        ("motor_power_kw", MOTOR_POWER_TERMS, "Motor power in kW"),
        ("motor_torque_nm", MOTOR_TORQUE_TERMS, "Motor torque in Nm"),
        ("range_km", RANGE_TERMS, "Driving range in km"),
        ("dc_charging_kw", DC_CHARGING_TERMS, "DC/CCS charging power in kW"),
        ("mcs_charging_kw", MCS_CHARGING_TERMS, "MCS/Megawatt charging power in kW"),
        ("charging_time_minutes", CHARGING_TIME_TERMS, "Charging time in minutes"),
        ("gvw_kg", GVW_TERMS, "Gross Vehicle Weight in kg"),
        ("gcw_kg", GCW_TERMS, "Gross Combination Weight in kg"),
        ("payload_capacity_kg", PAYLOAD_TERMS, "Payload capacity in kg"),
    ]

    for field, terms, description in field_terms:
        # Get a representative sample of terms (not all, to keep prompt reasonable)
        sample_terms = sorted(list(terms))[:15]
        terms_str = ", ".join(f'"{t}"' for t in sample_terms)
        prompt_parts.append(f"**{field}** ({description}):\n  {terms_str}, ...\n")

    prompt_parts.append("""
**UNIT CONVERSIONS - Apply automatically:**
- PS/CV/pk → kW: multiply by 0.7355
- hp/bhp → kW: multiply by 0.7457
- tonnes/t → kg: multiply by 1000
- miles → km: multiply by 1.6093
""")

    return "\n".join(prompt_parts)
