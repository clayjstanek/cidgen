"""Notional platform, emitter and engine library (spec Section 3.2). All values invented.

Each class has a 'blue' variant (flown by friendly forces) and a 'red' variant (flown by
adversaries). Neutral and civil classes have a single 'civil' variant. Observable NCTR
signatures (emitter code, engine code, LIDAR dimensions) come from the variant; with
probability rho_variant the observed code is drawn from the pooled variants instead,
so rho_variant = 1 makes NCTR uninformative about allegiance.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformClass:
    code: str
    tier: int
    name: str
    rcs_median_m2: float
    speed_mps: tuple[float, float]
    alt_km: tuple[float, float]
    length_m: float
    span_m: float
    tail_m: float                       # tailplane / rotor-hub-to-tail span
    aspect_rcs: tuple[float, float, float]   # multipliers at nose, beam, tail aspect
    swerling: int                       # 1 or 3
    nctr: str                           # 'jem' | 'rotor' | 'prop' | 'micro' | 'wingbeat' | 'none'
    capability: float                   # c_j in the threat-priority label
    # per-variant signature codes
    emitters: dict = field(default_factory=dict)   # variant -> emitter code or None
    engines: dict = field(default_factory=dict)    # variant -> engine/rotor code or None
    dim_scale: dict = field(default_factory=dict)  # variant -> LIDAR dimension multiplier


PLATFORMS: dict[str, PlatformClass] = {
    # ---------------- Tier 1 ----------------
    "U1": PlatformClass("U1", 1, "Small UAS, Group 1", 0.01, (0, 25), (0.0, 0.4), 0.5, 0.6, 0.3,
                        (1.0, 1.3, 1.0), 1, "micro", 0.3,
                        emitters={"blue": "E13", "red": "E12", "civil": "E12"},
                        engines={"blue": "M4", "red": "M4", "civil": "M4"},
                        dim_scale={"blue": 1.0, "red": 1.08, "civil": 0.95}),
    "U2": PlatformClass("U2", 1, "Small UAS, Group 2", 0.05, (0, 45), (0.0, 1.0), 1.5, 2.5, 0.8,
                        (0.8, 1.6, 1.0), 1, "micro", 0.5,
                        emitters={"blue": "E13", "red": "E14", "civil": "E12"},
                        engines={"blue": "M6", "red": "MP", "civil": "M6"},
                        dim_scale={"blue": 1.0, "red": 1.1, "civil": 0.9}),
    "K":  PlatformClass("K", 1, "Bird", 0.01, (5, 20), (0.02, 0.5), 0.3, 0.8, 0.1,
                        (0.7, 1.5, 0.7), 1, "wingbeat", 0.0,
                        emitters={"none": None}, engines={"none": "WB"}, dim_scale={"none": 1.0}),
    # ---------------- Tier 2 ----------------
    "A": PlatformClass("A", 2, "Fighter", 3.0, (150, 550), (0.1, 14), 15, 11, 5.5,
                       (0.5, 3.0, 1.0), 1, "jem", 0.6,
                       emitters={"blue": "E1", "red": "E2"}, engines={"blue": "J1", "red": "J2"},
                       dim_scale={"blue": 1.0, "red": 1.07}),
    "B": PlatformClass("B", 2, "Strike / attack", 5.0, (150, 300), (0.05, 10), 17, 13, 6.5,
                       (0.6, 3.0, 1.2), 1, "jem", 1.0,
                       emitters={"blue": "E3", "red": "E4"}, engines={"blue": "J3", "red": "J4"},
                       dim_scale={"blue": 1.0, "red": 0.94}),
    "R": PlatformClass("R", 2, "Rotary wing", 8.0, (0, 80), (0.02, 3), 17, 16, 3.0,
                       (0.8, 2.0, 1.0), 1, "rotor", 0.6,
                       emitters={"blue": "E10", "red": "E11"}, engines={"blue": "H4", "red": "H5"},
                       dim_scale={"blue": 1.0, "red": 1.1}),
    "C": PlatformClass("C", 2, "Large military (AEW, tanker, jammer)", 60.0, (180, 260), (6, 12), 45, 40, 14,
                       (0.6, 2.5, 1.0), 3, "jem", 0.3,
                       emitters={"blue": "E5", "red": "E6"}, engines={"blue": "J5", "red": "J5"},
                       dim_scale={"blue": 1.0, "red": 1.05}),
    "D": PlatformClass("D", 2, "Larger UAS, Group 3-5", 0.5, (40, 120), (0.3, 8), 8, 15, 4,
                       (0.7, 2.0, 1.0), 1, "prop", 0.4,
                       emitters={"blue": "E7", "red": "E8"}, engines={"blue": "P1", "red": "P2"},
                       dim_scale={"blue": 1.0, "red": 0.9}),
    "E": PlatformClass("E", 2, "Airliner / transport", 40.0, (200, 260), (8, 12), 40, 36, 13,
                       (0.6, 2.5, 1.0), 3, "jem", 0.0,
                       emitters={"civil": "E9"}, engines={"civil": "J6"}, dim_scale={"civil": 1.0}),
    "F": PlatformClass("F", 2, "General aviation", 2.0, (50, 90), (0.3, 4), 9, 11, 3.5,
                       (0.7, 2.0, 1.0), 1, "prop", 0.0,
                       emitters={"civil": None}, engines={"civil": "P0"}, dim_scale={"civil": 1.0}),
    # ---------------- Tier 3 (hypersonic profile player, cidgen/hypersonic_profiles.py) ----------------
    # Generic placeholder. Signature values are NOT estimates: set the RCS per profile with rcs_dbsm /
    # rcs_mod_db in the scenario YAML. Kinematics come entirely from the profile's segments.
    "G": PlatformClass("G", 3, "HGV", 0.1, (1500, 6000), (30, 70), 4.5, 1, 0.1,
                       (1.0, 10.0, 3.5), 1, "none", 1.0,
                       emitters={"red": None}, engines={"red": None}, dim_scale={"red": 1.0}),
    "H": PlatformClass("H", 3, "Tier 3 high-speed threat (generic placeholder)", 1.0, (0, 10000), (0.0, 100.0),
                       1.0, 1.0, 1.0, (1.0, 1.0, 1.0), 1, "none", 1.0,
                       emitters={"red": None}, engines={"red": None}, dim_scale={"red": 1.0}),
    "X": PlatformClass("X", 3, "HCM", 0.1, (1000, 2100), (.1, 28), 7, .8, 1,
                       (1.0, 10.0, 3.5), 1, "none", 1.0,
                       emitters={"red": None}, engines={"red": None}, dim_scale={"red": 1.0}),
    "Y": PlatformClass("Y", 3, "ABM", 0.1, (1800, 2400), (30, 50), 7, .8, 1,
                       (1.0, 10.0, 3.5), 1, "none", 1.0,
                       emitters={"red": None}, engines={"red": None}, dim_scale={"red": 1.0}),
    "Z": PlatformClass("Z", 3, "MarV", 0.1, (25, 120), (.1, 28), 7, .8, 1,
                       (1.0, 10.0, 3.5), 1, "none", 1.0,
                       emitters={"red": None}, engines={"red": None}, dim_scale={"red": 1.0}),
}

# Notional emitter catalogue: code -> (band, PRI class, scan type).
EMITTERS = {
    "E1": (3, "staggered", "track"),  "E2": (3, "jittered", "track"),
    "E3": (3, "stable", "sector"),    "E4": (4, "jittered", "sector"),
    "E5": (2, "stable", "circular"),  "E6": (2, "staggered", "circular"),
    "E7": (4, "stable", "sector"),    "E8": (4, "jittered", "sector"),
    "E9": (5, "stable", "sector"),    "E10": (3, "stable", "sector"),
    "E11": (3, "jittered", "sector"), "E12": (1, "stable", "none"),
    "E13": (1, "staggered", "none"),  "E14": (1, "jittered", "none"),
    "EJ": (3, "jittered", "none"),    # standoff jammer
}
EMITTER_CODES = sorted(EMITTERS)
ENGINE_CODES = sorted({c for p in PLATFORMS.values() for c in p.engines.values() if c})
CLASS_CODES = list(PLATFORMS)

TIER1 = [c for c, p in PLATFORMS.items() if p.tier == 1]
TIER2 = [c for c, p in PLATFORMS.items() if p.tier == 2]
TIER3 = [c for c, p in PLATFORMS.items() if p.tier == 3]   # hypersonic (Tier 3)
EOIR_CLASSES = ["U1", "U2", "K", "other"]
