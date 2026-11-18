"""Events as land-cover class changes: the vocabulary change_mapper uses to answer any before/after question.

An event is data, not a tool: a word pattern, a readable name, the Dynamic World classes it adds and/or the
classes it removes. A new kind of event is one more entry here; the trained land-cover model is the same.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

VEG = frozenset({"trees", "grass", "crops", "scrub"})


@dataclass(frozen=True)
class Event:
    key: str
    name: str
    gain: frozenset = frozenset()     # classes that appear (after, not before)
    loss: frozenset = frozenset()     # classes that disappear (before, not after)
    pattern: str = ""


EVENTS = [
    Event("flood", "flooding (land newly under water)", frozenset({"water", "flooded_vegetation"}),
          pattern=r"flood|inundat|submerg|waterlogg|deluge"),
    Event("deforestation", "forest loss", loss=frozenset({"trees"}),
          pattern=r"deforest|forest loss|logging|clear[- ]?cut|tree (cover )?loss|trees? (were |was )?(cut|felled|removed|lost)"),
    Event("fire", "vegetation turned to bare ground (the pattern of fire scars)", frozenset({"bare"}), VEG,
          pattern=r"wild ?fire|fire|burn"),
    Event("urban", "new built-up area", frozenset({"built"}),
          pattern=r"urban|construction|new buildings?|built[- ]up|development|sprawl|settlement"),
    Event("drought", "water loss", loss=frozenset({"water"}),
          pattern=r"drought|dried|dry(ing)? up|water loss|receded"),
    Event("farming", "new cropland", frozenset({"crops"}), pattern=r"cropland|farm|agricult|cultivat"),
    Event("snowmelt", "snow and ice loss", loss=frozenset({"snow_ice"}), pattern=r"snow ?melt|glacier|ice loss"),
]

# words for single classes (any question about one class: "did the trees decrease?")
CLASS_WORDS = {
    "water": r"water|river|lake|sea|reservoir|pond",
    "trees": r"trees?|forest|woods?|woodland",
    "grass": r"grass|grassland|meadow|pasture",
    "flooded_vegetation": r"wetland|marsh|swamp|flooded vegetation|mangrove",
    "crops": r"crops?|cropland|farmland|fields?",
    "scrub": r"scrub|shrubs?|bush",
    "built": r"built|buildings?|urban|town|city|settlement|roads?",
    "bare": r"bare|soil|sand|desert|rock",
    "snow_ice": r"snow|ice|glacier",
}
LABELS = {"water": "water", "trees": "trees", "grass": "grass", "flooded_vegetation": "flooded vegetation",
          "crops": "crops", "scrub": "scrub", "built": "built-up area", "bare": "bare ground", "snow_ice": "snow and ice"}


def match_event(query: str) -> Event | None:
    q = query.lower()
    return next((e for e in EVENTS if re.search(rf"\b({e.pattern})", q)), None)


def match_classes(query: str) -> list[str]:
    q = query.lower()
    return [c for c, p in CLASS_WORDS.items() if re.search(rf"\b({p})\b", q)]
