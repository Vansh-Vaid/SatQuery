"""Shared remote-sensing vocabulary: land-cover classes, synonyms, spatial words."""
from __future__ import annotations

import re

# Canonical classes produced by the algorithmic tools. Order = display order.
CLASSES = ("water", "vegetation", "built_up", "bare_soil")
CLASS_LABELS = {"water": "water", "vegetation": "vegetation", "built_up": "built-up area",
                "bare_soil": "bare soil", "cloud": "cloud", "change": "changed area"}
CLASS_COLORS = {"water": (30, 120, 255), "vegetation": (40, 200, 60), "built_up": (230, 60, 60),
                "bare_soil": (210, 170, 90), "cloud": (240, 240, 240), "change": (255, 0, 200)}

# Phrase -> canonical class. Longest phrases are matched first.
SYNONYMS: dict[str, str] = {
    # water
    "water body": "water", "water bodies": "water", "waterbody": "water", "water": "water", "river": "water",
    "rivers": "water", "lake": "water", "lakes": "water", "pond": "water", "ponds": "water", "sea": "water",
    "reservoir": "water", "flood": "water", "flooded": "water", "inundated": "water", "inundation": "water",
    "coast": "water", "canal": "water", "wetland": "water", "wetlands": "water", "water-covered": "water",
    "marine": "water", "ocean": "water", "stream": "water",
    # vegetation
    "vegetation": "vegetation", "vegetated": "vegetation", "forest": "vegetation", "forests": "vegetation",
    "trees": "vegetation", "tree": "vegetation", "woodland": "vegetation", "grass": "vegetation",
    "grassland": "vegetation", "meadow": "vegetation", "pasture": "vegetation", "pastures": "vegetation",
    "crop": "vegetation", "crops": "vegetation", "cropland": "vegetation", "farmland": "vegetation",
    "agricultural": "vegetation", "agriculture": "vegetation", "arable land": "vegetation",
    "fields": "vegetation", "field": "vegetation", "green": "vegetation", "greenery": "vegetation",
    "low vegetation": "vegetation", "shrub": "vegetation", "shrubs": "vegetation", "park": "vegetation",
    "orchard": "vegetation", "vineyard": "vegetation", "vineyards": "vegetation",
    # built-up
    "built-up": "built_up", "built up": "built_up", "builtup": "built_up", "urban": "built_up",
    "city": "built_up", "town": "built_up", "settlement": "built_up", "settlements": "built_up",
    "building": "built_up", "buildings": "built_up", "residential": "built_up", "industrial": "built_up",
    "commercial": "built_up", "houses": "built_up", "house": "built_up", "infrastructure": "built_up",
    "construction": "built_up", "impervious": "built_up", "urban fabric": "built_up", "roads": "built_up",
    "road": "built_up", "airport": "built_up", "port": "built_up", "developed": "built_up",
    # bare soil
    "bare soil": "bare_soil", "bare land": "bare_soil", "bare ground": "bare_soil", "soil": "bare_soil",
    "barren": "bare_soil", "desert": "bare_soil", "sand": "bare_soil", "beach": "bare_soil",
    "nonvegetated ground surface": "bare_soil", "ground surface": "bare_soil", "bare": "bare_soil",
    "mine": "bare_soil", "quarry": "bare_soil", "dump": "bare_soil",
    # cloud
    "cloud": "cloud", "clouds": "cloud", "cloudy": "cloud",
}
_SORTED = sorted(SYNONYMS, key=len, reverse=True)


def find_classes(text: str) -> list[str]:
    """Canonical classes mentioned in ``text`` in order of first appearance (deduplicated)."""
    t = " " + re.sub(r"[^a-z\- ]", " ", text.lower()) + " "
    hits: list[tuple[int, str]] = []
    taken = [False] * len(t)
    for phrase in _SORTED:
        for m in re.finditer(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", t):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            hits.append((m.start(), SYNONYMS[phrase]))
    out: list[str] = []
    for _, c in sorted(hits):
        if c not in out:
            out.append(c)
    return out


# ------------------------------------------------------------ spatial phrasing
def direction_of(cy: float, cx: float, h: int, w: int) -> str:
    """Compass phrase for a point (image north = up)."""
    ry, rx = cy / max(h, 1), cx / max(w, 1)
    v = "north" if ry < 1 / 3 else "south" if ry > 2 / 3 else ""
    hz = "west" if rx < 1 / 3 else "east" if rx > 2 / 3 else ""
    if not v and not hz:
        return "centre"
    return f"{v}-{hz}" if v and hz else (v or hz)


SPATIAL_QUALIFIERS = {
    "largest": "largest", "biggest": "largest", "main": "largest", "major": "largest",
    "smallest": "smallest", "tiny": "smallest", "small": "smallest",
    "north": "north", "northern": "north", "top": "north", "upper": "north",
    "south": "south", "southern": "south", "bottom": "south", "lower": "south",
    "west": "west", "western": "west", "left": "west",
    "east": "east", "eastern": "east", "right": "east",
    "centre": "centre", "center": "centre", "central": "centre", "middle": "centre",
    "all": "all", "every": "all", "each": "all",
}


def find_qualifiers(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    out: list[str] = []
    for w in words:
        q = SPATIAL_QUALIFIERS.get(w)
        if q and q not in out:
            out.append(q)
    return out
