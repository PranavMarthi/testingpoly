"""
Location inference engine.

Three-stage pipeline:
  1. NLP entity extraction (spaCy GPE/LOC/FAC/ORG)
  2. Domain heuristics (sports teams, political figures, institutions, buildings)
  3. Optional LLM fallback for unresolved markets

Each stage produces LocationCandidate objects with confidence scores.
Candidates are merged and deduplicated, with the highest confidence kept.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except Exception:
    spacy = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment,misc]
    SPACY_AVAILABLE = False

from polymarket_geo.config import get_settings
from polymarket_geo.event_resolver import get_event_resolver
from polymarket_geo.gazetteer import get_matcher
from polymarket_geo.models import (
    InferenceMethod,
    LocationCandidate,
    LocationType,
    MarketInferenceResult,
)

logger = logging.getLogger(__name__)

# ── Lazy spaCy model loading ──────────────────────────────────────────

_nlp = None


def get_nlp():
    """Load spaCy model lazily. Returns None if spaCy is unavailable."""
    global _nlp
    if _nlp is None:
        if not SPACY_AVAILABLE:
            logger.warning("spaCy is not available — NLP entity extraction disabled. "
                           "Heuristics-only mode.")
            return None
        model_name = get_settings().inference.spacy_model
        logger.info("Loading spaCy model: %s", model_name)
        try:
            _nlp = spacy.load(model_name)
        except Exception as e:
            logger.warning("Failed to load spaCy model '%s': %s. "
                           "NLP entity extraction disabled.", model_name, e)
            return None
    return _nlp


# ══════════════════════════════════════════════════════════════════════
# DOMAIN HEURISTIC KNOWLEDGE BASES
# ══════════════════════════════════════════════════════════════════════

# Sports teams -> city mapping (major US leagues + some international)
SPORTS_TEAMS: dict[str, tuple[str, LocationType]] = {
    # NBA
    "hawks": ("Atlanta, GA", LocationType.CITY),
    "celtics": ("Boston, MA", LocationType.CITY),
    "nets": ("Brooklyn, NY", LocationType.CITY),
    "hornets": ("Charlotte, NC", LocationType.CITY),
    "bulls": ("Chicago, IL", LocationType.CITY),
    "cavaliers": ("Cleveland, OH", LocationType.CITY),
    "cavs": ("Cleveland, OH", LocationType.CITY),
    "mavericks": ("Dallas, TX", LocationType.CITY),
    "mavs": ("Dallas, TX", LocationType.CITY),
    "nuggets": ("Denver, CO", LocationType.CITY),
    "pistons": ("Detroit, MI", LocationType.CITY),
    "warriors": ("San Francisco, CA", LocationType.CITY),
    "rockets": ("Houston, TX", LocationType.CITY),
    "pacers": ("Indianapolis, IN", LocationType.CITY),
    "clippers": ("Los Angeles, CA", LocationType.CITY),
    "lakers": ("Los Angeles, CA", LocationType.CITY),
    "grizzlies": ("Memphis, TN", LocationType.CITY),
    "heat": ("Miami, FL", LocationType.CITY),
    "bucks": ("Milwaukee, WI", LocationType.CITY),
    "timberwolves": ("Minneapolis, MN", LocationType.CITY),
    "wolves": ("Minneapolis, MN", LocationType.CITY),
    "pelicans": ("New Orleans, LA", LocationType.CITY),
    "knicks": ("New York, NY", LocationType.CITY),
    "thunder": ("Oklahoma City, OK", LocationType.CITY),
    "magic": ("Orlando, FL", LocationType.CITY),
    "76ers": ("Philadelphia, PA", LocationType.CITY),
    "sixers": ("Philadelphia, PA", LocationType.CITY),
    "suns": ("Phoenix, AZ", LocationType.CITY),
    "trail blazers": ("Portland, OR", LocationType.CITY),
    "blazers": ("Portland, OR", LocationType.CITY),
    "kings": ("Sacramento, CA", LocationType.CITY),
    "spurs": ("San Antonio, TX", LocationType.CITY),
    "raptors": ("Toronto, ON, Canada", LocationType.CITY),
    "jazz": ("Salt Lake City, UT", LocationType.CITY),
    "wizards": ("Washington, DC", LocationType.CITY),

    # NFL
    "falcons": ("Atlanta, GA", LocationType.CITY),
    "ravens": ("Baltimore, MD", LocationType.CITY),
    "bills": ("Buffalo, NY", LocationType.CITY),
    "panthers": ("Charlotte, NC", LocationType.CITY),
    "bears": ("Chicago, IL", LocationType.CITY),
    "bengals": ("Cincinnati, OH", LocationType.CITY),
    "browns": ("Cleveland, OH", LocationType.CITY),
    "cowboys": ("Dallas, TX", LocationType.CITY),
    "broncos": ("Denver, CO", LocationType.CITY),
    "lions": ("Detroit, MI", LocationType.CITY),
    "packers": ("Green Bay, WI", LocationType.CITY),
    "texans": ("Houston, TX", LocationType.CITY),
    "colts": ("Indianapolis, IN", LocationType.CITY),
    "jaguars": ("Jacksonville, FL", LocationType.CITY),
    "chiefs": ("Kansas City, MO", LocationType.CITY),
    "raiders": ("Las Vegas, NV", LocationType.CITY),
    "chargers": ("Los Angeles, CA", LocationType.CITY),
    "rams": ("Los Angeles, CA", LocationType.CITY),
    "dolphins": ("Miami, FL", LocationType.CITY),
    "vikings": ("Minneapolis, MN", LocationType.CITY),
    "patriots": ("Foxborough, MA", LocationType.CITY),
    "saints": ("New Orleans, LA", LocationType.CITY),
    "giants": ("East Rutherford, NJ", LocationType.CITY),
    "jets": ("East Rutherford, NJ", LocationType.CITY),
    "eagles": ("Philadelphia, PA", LocationType.CITY),
    "steelers": ("Pittsburgh, PA", LocationType.CITY),
    "49ers": ("San Francisco, CA", LocationType.CITY),
    "seahawks": ("Seattle, WA", LocationType.CITY),
    "buccaneers": ("Tampa, FL", LocationType.CITY),
    "bucs": ("Tampa, FL", LocationType.CITY),
    "titans": ("Nashville, TN", LocationType.CITY),
    "commanders": ("Washington, DC", LocationType.CITY),

    # MLB
    "braves": ("Atlanta, GA", LocationType.CITY),
    "orioles": ("Baltimore, MD", LocationType.CITY),
    "red sox": ("Boston, MA", LocationType.CITY),
    "cubs": ("Chicago, IL", LocationType.CITY),
    "white sox": ("Chicago, IL", LocationType.CITY),
    "reds": ("Cincinnati, OH", LocationType.CITY),
    "guardians": ("Cleveland, OH", LocationType.CITY),
    "rockies": ("Denver, CO", LocationType.CITY),
    "tigers": ("Detroit, MI", LocationType.CITY),
    "astros": ("Houston, TX", LocationType.CITY),
    "royals": ("Kansas City, MO", LocationType.CITY),
    "angels": ("Anaheim, CA", LocationType.CITY),
    "dodgers": ("Los Angeles, CA", LocationType.CITY),
    "marlins": ("Miami, FL", LocationType.CITY),
    "brewers": ("Milwaukee, WI", LocationType.CITY),
    "twins": ("Minneapolis, MN", LocationType.CITY),
    "mets": ("New York, NY", LocationType.CITY),
    "yankees": ("New York, NY", LocationType.CITY),
    "athletics": ("Oakland, CA", LocationType.CITY),
    "phillies": ("Philadelphia, PA", LocationType.CITY),
    "pirates": ("Pittsburgh, PA", LocationType.CITY),
    "padres": ("San Diego, CA", LocationType.CITY),
    "mariners": ("Seattle, WA", LocationType.CITY),
    "cardinals": ("St. Louis, MO", LocationType.CITY),
    "rays": ("St. Petersburg, FL", LocationType.CITY),
    "rangers": ("Arlington, TX", LocationType.CITY),
    "blue jays": ("Toronto, ON, Canada", LocationType.CITY),
    "nationals": ("Washington, DC", LocationType.CITY),

    # NHL
    "bruins": ("Boston, MA", LocationType.CITY),
    "sabres": ("Buffalo, NY", LocationType.CITY),
    "flames": ("Calgary, AB, Canada", LocationType.CITY),
    "hurricanes": ("Raleigh, NC", LocationType.CITY),
    "blackhawks": ("Chicago, IL", LocationType.CITY),
    "avalanche": ("Denver, CO", LocationType.CITY),
    "red wings": ("Detroit, MI", LocationType.CITY),
    "oilers": ("Edmonton, AB, Canada", LocationType.CITY),
    "maple leafs": ("Toronto, ON, Canada", LocationType.CITY),
    "canadiens": ("Montreal, QC, Canada", LocationType.CITY),
    "penguins": ("Pittsburgh, PA", LocationType.CITY),
    "flyers": ("Philadelphia, PA", LocationType.CITY),
    "canucks": ("Vancouver, BC, Canada", LocationType.CITY),
    "golden knights": ("Las Vegas, NV", LocationType.CITY),
    "kraken": ("Seattle, WA", LocationType.CITY),
    "predators": ("Nashville, TN", LocationType.CITY),
    "lightning": ("Tampa, FL", LocationType.CITY),
    "capitals": ("Washington, DC", LocationType.CITY),

    # Soccer / International
    "manchester united": ("Manchester, UK", LocationType.CITY),
    "man united": ("Manchester, UK", LocationType.CITY),
    "manchester city": ("Manchester, UK", LocationType.CITY),
    "man city": ("Manchester, UK", LocationType.CITY),
    "liverpool": ("Liverpool, UK", LocationType.CITY),
    "arsenal": ("London, UK", LocationType.CITY),
    "chelsea": ("London, UK", LocationType.CITY),
    "tottenham": ("London, UK", LocationType.CITY),
    "real madrid": ("Madrid, Spain", LocationType.CITY),
    "barcelona": ("Barcelona, Spain", LocationType.CITY),
    "bayern munich": ("Munich, Germany", LocationType.CITY),
    "bayern": ("Munich, Germany", LocationType.CITY),
    "psg": ("Paris, France", LocationType.CITY),
    "paris saint-germain": ("Paris, France", LocationType.CITY),
    "juventus": ("Turin, Italy", LocationType.CITY),
    "inter milan": ("Milan, Italy", LocationType.CITY),
    "ac milan": ("Milan, Italy", LocationType.CITY),
}

# Prefixed team names (e.g., "Atlanta Hawks", "Los Angeles Lakers")
PREFIXED_TEAMS: dict[str, tuple[str, LocationType]] = {
    "atlanta hawks": ("Atlanta, GA", LocationType.CITY),
    "atlanta falcons": ("Atlanta, GA", LocationType.CITY),
    "atlanta braves": ("Atlanta, GA", LocationType.CITY),
    "los angeles lakers": ("Los Angeles, CA", LocationType.CITY),
    "los angeles clippers": ("Los Angeles, CA", LocationType.CITY),
    "los angeles rams": ("Los Angeles, CA", LocationType.CITY),
    "los angeles chargers": ("Los Angeles, CA", LocationType.CITY),
    "los angeles dodgers": ("Los Angeles, CA", LocationType.CITY),
    "la lakers": ("Los Angeles, CA", LocationType.CITY),
    "la clippers": ("Los Angeles, CA", LocationType.CITY),
    "new york knicks": ("New York, NY", LocationType.CITY),
    "new york yankees": ("New York, NY", LocationType.CITY),
    "new york mets": ("New York, NY", LocationType.CITY),
    "new york giants": ("East Rutherford, NJ", LocationType.CITY),
    "new york jets": ("East Rutherford, NJ", LocationType.CITY),
    "golden state warriors": ("San Francisco, CA", LocationType.CITY),
    "green bay packers": ("Green Bay, WI", LocationType.CITY),
    "kansas city chiefs": ("Kansas City, MO", LocationType.CITY),
    "san francisco 49ers": ("San Francisco, CA", LocationType.CITY),
    "san diego padres": ("San Diego, CA", LocationType.CITY),
    "tampa bay buccaneers": ("Tampa, FL", LocationType.CITY),
    "tampa bay rays": ("St. Petersburg, FL", LocationType.CITY),
    "tampa bay lightning": ("Tampa, FL", LocationType.CITY),
    "new england patriots": ("Foxborough, MA", LocationType.CITY),
    "new orleans saints": ("New Orleans, LA", LocationType.CITY),
    "new orleans pelicans": ("New Orleans, LA", LocationType.CITY),
    "oklahoma city thunder": ("Oklahoma City, OK", LocationType.CITY),
    "salt lake city jazz": ("Salt Lake City, UT", LocationType.CITY),
    "utah jazz": ("Salt Lake City, UT", LocationType.CITY),
    "portland trail blazers": ("Portland, OR", LocationType.CITY),
    "toronto raptors": ("Toronto, ON, Canada", LocationType.CITY),
    "toronto blue jays": ("Toronto, ON, Canada", LocationType.CITY),
    "toronto maple leafs": ("Toronto, ON, Canada", LocationType.CITY),
}

# Political figures -> associated locations
POLITICAL_FIGURES: dict[str, list[tuple[str, float, LocationType]]] = {
    "trump": [
        ("Washington, DC", 0.45, LocationType.CITY),
        ("Palm Beach, FL", 0.35, LocationType.CITY),  # Mar-a-Lago
    ],
    "biden": [
        ("Washington, DC", 0.55, LocationType.CITY),
        ("Wilmington, DE", 0.25, LocationType.CITY),
    ],
    "harris": [
        ("Washington, DC", 0.50, LocationType.CITY),
    ],
    "desantis": [
        ("Tallahassee, FL", 0.60, LocationType.CITY),
    ],
    "newsom": [
        ("Sacramento, CA", 0.60, LocationType.CITY),
    ],
    "zelensky": [
        ("Kyiv, Ukraine", 0.70, LocationType.CITY),
    ],
    "putin": [
        ("Moscow, Russia", 0.70, LocationType.CITY),
    ],
    "xi jinping": [
        ("Beijing, China", 0.70, LocationType.CITY),
    ],
    "macron": [
        ("Paris, France", 0.70, LocationType.CITY),
    ],
    "starmer": [
        ("London, UK", 0.65, LocationType.CITY),
    ],
    "modi": [
        ("New Delhi, India", 0.70, LocationType.CITY),
    ],
    "netanyahu": [
        ("Jerusalem, Israel", 0.70, LocationType.CITY),
    ],
}

# Institutions -> headquarters location
INSTITUTIONS: dict[str, tuple[str, LocationType]] = {
    "fed": ("Washington, DC", LocationType.BUILDING),
    "federal reserve": ("Washington, DC", LocationType.BUILDING),
    "fomc": ("Washington, DC", LocationType.BUILDING),
    "sec": ("Washington, DC", LocationType.BUILDING),
    "fda": ("Silver Spring, MD", LocationType.BUILDING),
    "pentagon": ("Arlington, VA", LocationType.BUILDING),
    "cia": ("Langley, VA", LocationType.BUILDING),
    "fbi": ("Washington, DC", LocationType.BUILDING),
    "congress": ("Washington, DC", LocationType.BUILDING),
    "senate": ("Washington, DC", LocationType.BUILDING),
    "house of representatives": ("Washington, DC", LocationType.BUILDING),
    "supreme court": ("Washington, DC", LocationType.BUILDING),
    "scotus": ("Washington, DC", LocationType.BUILDING),
    "white house": ("Washington, DC", LocationType.BUILDING),
    "european central bank": ("Frankfurt, Germany", LocationType.BUILDING),
    "ecb": ("Frankfurt, Germany", LocationType.BUILDING),
    "bank of england": ("London, UK", LocationType.BUILDING),
    "boe": ("London, UK", LocationType.BUILDING),
    "bank of japan": ("Tokyo, Japan", LocationType.BUILDING),
    "boj": ("Tokyo, Japan", LocationType.BUILDING),
    "imf": ("Washington, DC", LocationType.BUILDING),
    "world bank": ("Washington, DC", LocationType.BUILDING),
    "united nations": ("New York, NY", LocationType.BUILDING),
    "un": ("New York, NY", LocationType.BUILDING),
    "nato": ("Brussels, Belgium", LocationType.BUILDING),
    "eu": ("Brussels, Belgium", LocationType.BUILDING),
    "european union": ("Brussels, Belgium", LocationType.BUILDING),
    "parliament": ("London, UK", LocationType.BUILDING),
    "doj": ("Washington, DC", LocationType.BUILDING),
    "department of justice": ("Washington, DC", LocationType.BUILDING),
}

# Named buildings/landmarks -> fixed location
BUILDINGS: dict[str, tuple[str, float, float, LocationType]] = {
    "mar-a-lago": ("Palm Beach, FL", 26.6774, -80.0368, LocationType.BUILDING),
    "mar a lago": ("Palm Beach, FL", 26.6774, -80.0368, LocationType.BUILDING),
    "maralago": ("Palm Beach, FL", 26.6774, -80.0368, LocationType.BUILDING),
    "white house": ("Washington, DC", 38.8977, -77.0365, LocationType.BUILDING),
    "capitol": ("Washington, DC", 38.8899, -77.0091, LocationType.BUILDING),
    "capitol hill": ("Washington, DC", 38.8899, -77.0091, LocationType.BUILDING),
    "kremlin": ("Moscow, Russia", 55.7520, 37.6175, LocationType.BUILDING),
    "buckingham palace": ("London, UK", 51.5014, -0.1419, LocationType.BUILDING),
    "downing street": ("London, UK", 51.5034, -0.1276, LocationType.BUILDING),
    "wall street": ("New York, NY", 40.7060, -74.0088, LocationType.BUILDING),
    "silicon valley": ("San Jose, CA", 37.3875, -122.0575, LocationType.BUILDING),
    "pentagon": ("Arlington, VA", 38.8719, -77.0563, LocationType.BUILDING),
    "camp david": ("Thurmont, MD", 39.6481, -77.4650, LocationType.BUILDING),
}

# Keywords that indicate global/no-specific-location markets
GLOBAL_KEYWORDS = {
    "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
    "solana", "sol", "xrp", "dogecoin", "doge",
    "global", "worldwide", "world",
    "ai", "artificial intelligence", "chatgpt", "openai",
    "spacex", "mars", "moon", "space",
    "inflation rate",  # generic unless qualified by country
    "interest rate",   # generic unless qualified
}

# Governance / policy keywords that imply a seat-of-government location.
POLICY_KEYWORDS = {
    "act", "bill", "signed into law", "law", "legislation", "legislative",
    "congress", "senate", "house", "parliament", "prime minister",
    "president", "federal", "executive order", "regulation", "regulatory",
    "supreme court", "court ruling", "judge", "justice", "ministry",
    "cabinet", "agency", "department", "sec", "fda", "doj", "treasury",
    "clarity act", "genius act", "appropriations", "budget", "spending bill",
}

# Default capitals for country-level policy events.
COUNTRY_CAPITALS: dict[str, str] = {
    "united states": "Washington, DC",
    "usa": "Washington, DC",
    "us": "Washington, DC",
    "canada": "Ottawa, Canada",
    "united kingdom": "London, United Kingdom",
    "uk": "London, United Kingdom",
    "france": "Paris, France",
    "germany": "Berlin, Germany",
    "italy": "Rome, Italy",
    "spain": "Madrid, Spain",
    "japan": "Tokyo, Japan",
    "china": "Beijing, China",
    "india": "New Delhi, India",
    "russia": "Moscow, Russia",
    "brazil": "Brasilia, Brazil",
    "australia": "Canberra, Australia",
    "mexico": "Mexico City, Mexico",
    "south korea": "Seoul, South Korea",
    "north korea": "Pyongyang, North Korea",
    "israel": "Jerusalem, Israel",
    "iran": "Tehran, Iran",
    "turkey": "Ankara, Turkey",
    "ukraine": "Kyiv, Ukraine",
}

# Sport-related vs match patterns (detect "X vs Y" or "X at Y")
MATCH_PATTERN = re.compile(
    r"(.+?)\s+(?:vs\.?|versus|at|@)\s+(.+?)(?:\s|$|,|\?|!)",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════

class LocationInferenceEngine:
    """
    Multi-stage location inference.
    Combines NLP entities, domain heuristics, and optional LLM fallback.
    """

    def __init__(self) -> None:
        self.settings = get_settings().inference
        self.min_confidence = self.settings.min_confidence
        self.max_candidates = self.settings.max_candidates

    def infer(self, condition_id: str, question: str, description: Optional[str] = None) -> MarketInferenceResult:
        """
        Run full inference pipeline on a market.
        Returns MarketInferenceResult with 0..N location candidates.
        """
        text = question
        if description:
            text = f"{question} {description}"
        text_lower = text.lower()

        candidates: list[LocationCandidate] = []

        # Stage 0: Check for global/non-geographic markets
        if self._is_global_market(text, text_lower):
            return MarketInferenceResult(
                condition_id=condition_id,
                locations=[
                    LocationCandidate(
                        location_name="Global / No specific location",
                        location_type=LocationType.GLOBAL,
                        confidence=0.90,
                        reason="Market topic is global (crypto/tech/space)",
                        inference_method=InferenceMethod.HEURISTIC,
                    )
                ],
                has_location=False,
                is_global=True,
            )

        # Stage 1: NLP entity extraction
        nlp_candidates = self._extract_nlp_entities(text)
        candidates.extend(nlp_candidates)

        # Stage 1.5: Gazetteer matching (works without spaCy)
        gazetteer_candidates = self._extract_gazetteer(text)
        candidates.extend(gazetteer_candidates)

        # Stage 2: Domain heuristics (often higher confidence than NLP alone)
        heuristic_candidates = self._apply_heuristics(text, text_lower)
        candidates.extend(heuristic_candidates)

        # Stage 2.5: Governance/policy inference (capital-city defaults)
        policy_candidates = self._extract_policy_locations(text, text_lower, candidates)
        candidates.extend(policy_candidates)

        # Stage 3: Event venue inference (live lookup + strict not_available fallback)
        event_candidates = self._extract_event_locations(text, text_lower, candidates)
        candidates.extend(event_candidates)

        # Merge and deduplicate
        candidates = self._merge_candidates(candidates)

        # Apply ambiguity penalty if too many candidates
        candidates = self._apply_ambiguity_penalty(candidates)

        # Sort by confidence descending, truncate
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        candidates = [c for c in candidates if c.confidence >= self.min_confidence]
        candidates = candidates[: self.max_candidates]

        has_location = any(c.location_type != LocationType.GLOBAL for c in candidates)
        has_not_available = any(c.location_name == "not_available" for c in candidates)

        return MarketInferenceResult(
            condition_id=condition_id,
            locations=candidates,
            has_location=has_location,
            is_global=(not has_location and not has_not_available),
        )

    # ── Stage 0: Global detection ─────────────────────────────────────

    def _is_global_market(self, text: str, text_lower: str) -> bool:
        """Check if the market is clearly about a global/non-geographic topic."""
        # Count how many global keywords appear
        hits = sum(1 for kw in GLOBAL_KEYWORDS if kw in text_lower)
        # Must have at least one global keyword and no strong geographic signals
        if hits == 0:
            return False

        # Check for institution/building/political keywords that override global
        for inst in INSTITUTIONS:
            pattern = r'\b' + re.escape(inst) + r'\b'
            if re.search(pattern, text_lower):
                return False  # Has a geographic institution, not global
        for building in BUILDINGS:
            if building in text_lower:
                return False
        for figure in POLITICAL_FIGURES:
            pattern = r'\b' + re.escape(figure) + r'\b'
            if re.search(pattern, text_lower):
                return False

        # Check gazetteer for any geographic match
        matcher = get_matcher()
        geo_matches = matcher.find_all(text)
        # Filter out overly broad region matches
        specific_matches = [
            (surface, entry) for surface, entry in geo_matches
            if entry.location_type in ("city", "country", "state")
        ]
        if specific_matches:
            return False

        # If the text has any unknown proper nouns, avoid classifying as global.
        # This prevents false globals like "Will the US acquire part of Greenland..."
        # when a place isn't yet in our gazetteer.
        unknown_proper_nouns = matcher.find_unknown_proper_nouns(text)
        if unknown_proper_nouns:
            return False

        # Check that we DON'T have geographic entities via NLP
        nlp = get_nlp()
        if nlp is not None:
            doc = nlp(text_lower[:500])  # cap to avoid long inference
            geo_ents = [ent for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
            if len(geo_ents) > 0:
                return False

        return True

    # ── Stage 1: NLP ──────────────────────────────────────────────────

    def _extract_nlp_entities(self, text: str) -> list[LocationCandidate]:
        """Extract GPE/LOC/FAC entities using spaCy NER."""
        nlp = get_nlp()
        if nlp is None:
            return []  # NLP unavailable, skip this stage

        doc = nlp(text[:1000])  # cap input length

        candidates = []
        seen = set()

        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC", "FAC"):
                continue

            name = ent.text.strip()
            name_lower = name.lower()

            if name_lower in seen or len(name) < 2:
                continue
            seen.add(name_lower)

            # Base confidence from entity type
            if ent.label_ == "GPE":
                base_conf = 0.70
                loc_type = LocationType.CITY
            elif ent.label_ == "LOC":
                base_conf = 0.55
                loc_type = LocationType.CITY
            else:  # FAC
                base_conf = 0.50
                loc_type = LocationType.BUILDING

            # Boost if entity appears in the question (title) vs description
            if name_lower in text[:200].lower():
                base_conf = min(base_conf + 0.10, 0.95)

            candidates.append(LocationCandidate(
                location_name=name,
                location_type=loc_type,
                confidence=round(base_conf, 2),
                reason=f"NLP entity: '{name}' ({ent.label_})",
                inference_method=InferenceMethod.NLP,
            ))

        return candidates

    # ── Stage 1.5: Gazetteer ──────────────────────────────────────────

    def _extract_gazetteer(self, text: str) -> list[LocationCandidate]:
        """
        Match text against a gazetteer of countries, cities, states, demonyms.
        Works without spaCy. Provides coordinates directly from the gazetteer.
        """
        matcher = get_matcher()
        matches = matcher.find_all(text)

        candidates = []
        for surface_form, entry in matches:
            # Map gazetteer location_type to our LocationType enum
            type_map = {
                "city": LocationType.CITY,
                "country": LocationType.COUNTRY,
                "state": LocationType.STATE,
                "region": LocationType.COUNTRY,  # treat regions as country-level
            }
            loc_type = type_map.get(entry.location_type, LocationType.CITY)

            # Confidence based on type specificity
            if entry.location_type == "city":
                base_conf = 0.75
            elif entry.location_type == "country":
                base_conf = 0.70
            elif entry.location_type == "state":
                base_conf = 0.65
            else:  # region
                base_conf = 0.45

            # Boost if the match is a direct place name vs a demonym
            # (demonyms like "Japanese" are less direct than "Japan")
            if surface_form.lower() == entry.canonical_name.split(",")[0].lower():
                base_conf = min(base_conf + 0.05, 0.95)

            # Slight boost if it appears early in the text (likely in the title)
            if surface_form.lower() in text[:100].lower():
                base_conf = min(base_conf + 0.05, 0.95)

            candidates.append(LocationCandidate(
                location_name=entry.canonical_name,
                location_type=loc_type,
                confidence=round(base_conf, 2),
                reason=f"Gazetteer match: '{surface_form}' -> {entry.canonical_name}",
                inference_method=InferenceMethod.HEURISTIC,
                latitude=entry.latitude,
                longitude=entry.longitude,
            ))

        # Catch-all fallback: only add unknown proper nouns when there is geo context
        # nearby (in/at/from/near/around). This avoids false positives like
        # "Oscars" or "Best Picture Winner" being treated as locations.
        unknown_proper_nouns = matcher.find_unknown_proper_nouns(text)
        geo_context = re.search(r"\b(in|at|from|near|around|outside|inside|within|across)\b", text.lower())
        for noun in unknown_proper_nouns:
            if not geo_context:
                continue
            candidates.append(LocationCandidate(
                location_name=noun,
                location_type=LocationType.CITY,
                confidence=0.42,
                reason=f"Proper-noun location fallback: '{noun}' (not in gazetteer)",
                inference_method=InferenceMethod.HEURISTIC,
            ))

        return candidates

    def _extract_policy_locations(
        self,
        text: str,
        text_lower: str,
        existing_candidates: list[LocationCandidate],
    ) -> list[LocationCandidate]:
        """
        Infer seat-of-government locations for law/policy markets.

        Why: many prediction markets are policy/regulatory questions that imply
        a jurisdiction even if no explicit city is mentioned.
        Example: "Clarity Act signed into law in 2026?" -> Washington, DC.
        """
        # Trigger only on governance/policy language.
        if not any(kw in text_lower for kw in POLICY_KEYWORDS):
            return []

        candidates: list[LocationCandidate] = []
        existing_names = {c.location_name.lower() for c in existing_candidates}

        # If we already inferred countries, map them to capitals.
        countries_in_text = []
        matcher = get_matcher()
        for surface, entry in matcher.find_all(text):
            if entry.location_type == "country":
                countries_in_text.append(entry.canonical_name.lower())

        dedup_countries = set(countries_in_text)
        for country in dedup_countries:
            capital = COUNTRY_CAPITALS.get(country)
            if not capital:
                continue
            if capital.lower() in existing_names:
                continue

            candidates.append(LocationCandidate(
                location_name=capital,
                location_type=LocationType.CITY,
                confidence=0.62,
                reason=f"Policy jurisdiction heuristic: country '{country}' -> capital '{capital}'",
                inference_method=InferenceMethod.HEURISTIC,
            ))

        # If no country was explicit, default to US federal center for US-policy cues.
        us_policy_cues = (
            "clarity act", "genius act", "congress", "senate", "house", "federal",
            "sec", "fda", "doj", "treasury", "white house", "supreme court",
            "u.s.", "usa", " united states", " us ",
        )
        if not candidates and any(cue in f" {text_lower} " for cue in us_policy_cues):
            if "washington, dc" not in existing_names:
                candidates.append(LocationCandidate(
                    location_name="Washington, DC",
                    location_type=LocationType.CITY,
                    confidence=0.60,
                    reason="Policy jurisdiction heuristic: US federal legislation/regulation -> Washington, DC",
                    inference_method=InferenceMethod.HEURISTIC,
                ))

        # Generic fallback for policy questions with no explicit country:
        # still avoid returning "global" by assigning the most likely seat.
        if not candidates and "washington, dc" not in existing_names:
            candidates.append(LocationCandidate(
                location_name="Washington, DC",
                location_type=LocationType.CITY,
                confidence=0.45,
                reason="Policy fallback: unresolved jurisdiction defaults to likely federal seat",
                inference_method=InferenceMethod.HEURISTIC,
            ))

        return candidates

    def _extract_event_locations(
        self,
        text: str,
        text_lower: str,
        existing_candidates: list[LocationCandidate],
    ) -> list[LocationCandidate]:
        """
        Resolve event venues (Oscars, Grammys, Cannes, etc.).

        Rules:
        - If near-term and venue is publicly available, return resolved location.
        - If far-term or unresolved, return strict not_available marker.
        - Do not fabricate historical defaults.
        """
        resolver = get_event_resolver()
        result = resolver.resolve(text)
        if result is None:
            return []

        # Avoid duplicate event location when explicit geo already exists
        existing_names = {c.location_name.lower() for c in existing_candidates}

        if result.status == "confirmed" and result.city:
            loc_name = result.city if not result.country else f"{result.city}, {result.country}"
            if loc_name.lower() in existing_names:
                return []

            reason = "Event venue lookup: confirmed"
            if result.venue_name:
                reason += f" venue '{result.venue_name}'"
            if result.source_url:
                reason += f" (source: {result.source_url})"

            return [
                LocationCandidate(
                    location_name=loc_name,
                    location_type=LocationType.CITY,
                    confidence=max(0.55, min(result.confidence, 0.9)),
                    reason=reason,
                    inference_method=InferenceMethod.HEURISTIC,
                    latitude=result.latitude,
                    longitude=result.longitude,
                )
            ]

        # Strict fallback requested by user: not_available
        return [
            LocationCandidate(
                location_name="not_available",
                location_type=LocationType.GLOBAL,
                confidence=max(0.25, min(result.confidence, 0.5)),
                reason=result.reason or "Event venue not publicly available",
                inference_method=InferenceMethod.HEURISTIC,
            )
        ]

    # ── Stage 2: Domain heuristics ────────────────────────────────────

    def _apply_heuristics(self, text: str, text_lower: str) -> list[LocationCandidate]:
        """Apply domain-specific rules for sports, politics, institutions, buildings."""
        candidates = []

        # 2a: Named buildings (highest priority — exact coordinates known)
        for building, (loc_name, lat, lon, loc_type) in BUILDINGS.items():
            if building in text_lower:
                candidates.append(LocationCandidate(
                    location_name=loc_name,
                    location_type=loc_type,
                    confidence=0.90,
                    reason=f"Building heuristic: '{building}' -> {loc_name}",
                    inference_method=InferenceMethod.HEURISTIC,
                    latitude=lat,
                    longitude=lon,
                ))

        # 2b: Institutions
        for inst, (loc_name, loc_type) in INSTITUTIONS.items():
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(inst) + r'\b'
            if re.search(pattern, text_lower):
                candidates.append(LocationCandidate(
                    location_name=loc_name,
                    location_type=loc_type,
                    confidence=0.65,
                    reason=f"Institution heuristic: '{inst}' HQ -> {loc_name}",
                    inference_method=InferenceMethod.HEURISTIC,
                ))

        # 2c: Sports team match patterns ("X vs Y")
        match = MATCH_PATTERN.search(text)
        if match:
            team_a = match.group(1).strip().lower()
            team_b = match.group(2).strip().lower()
            for team_text in (team_a, team_b):
                loc = self._resolve_team(team_text)
                if loc:
                    candidates.append(loc)

        # 2d: General sports team mentions (even outside "vs" patterns)
        for team_key, (loc_name, loc_type) in PREFIXED_TEAMS.items():
            if team_key in text_lower:
                candidates.append(LocationCandidate(
                    location_name=loc_name,
                    location_type=loc_type,
                    confidence=0.75,
                    reason=f"Sports team heuristic: '{team_key}' -> {loc_name}",
                    inference_method=InferenceMethod.HEURISTIC,
                ))

        # Also check short team names if not already found via prefixed
        found_locs = {c.location_name for c in candidates}
        for team_key, (loc_name, loc_type) in SPORTS_TEAMS.items():
            if loc_name in found_locs:
                continue
            pattern = r'\b' + re.escape(team_key) + r'\b'
            if re.search(pattern, text_lower):
                candidates.append(LocationCandidate(
                    location_name=loc_name,
                    location_type=loc_type,
                    confidence=0.65,
                    reason=f"Sports team heuristic: '{team_key}' -> {loc_name}",
                    inference_method=InferenceMethod.HEURISTIC,
                ))

        # 2e: Political figures
        for figure, locations in POLITICAL_FIGURES.items():
            pattern = r'\b' + re.escape(figure) + r'\b'
            if re.search(pattern, text_lower):
                for loc_name, conf, loc_type in locations:
                    candidates.append(LocationCandidate(
                        location_name=loc_name,
                        location_type=loc_type,
                        confidence=round(conf, 2),
                        reason=f"Political figure heuristic: '{figure}' -> {loc_name}",
                        inference_method=InferenceMethod.HEURISTIC,
                    ))

        return candidates

    def _resolve_team(self, team_text: str) -> Optional[LocationCandidate]:
        """Try to resolve a team name string to a location."""
        team_text = team_text.strip().lower()

        # Try prefixed names first (more specific)
        if team_text in PREFIXED_TEAMS:
            loc_name, loc_type = PREFIXED_TEAMS[team_text]
            return LocationCandidate(
                location_name=loc_name,
                location_type=loc_type,
                confidence=0.80,
                reason=f"Sports match: '{team_text}' -> {loc_name}",
                inference_method=InferenceMethod.HEURISTIC,
            )

        # Try short team names
        for key, (loc_name, loc_type) in SPORTS_TEAMS.items():
            if key in team_text:
                return LocationCandidate(
                    location_name=loc_name,
                    location_type=loc_type,
                    confidence=0.70,
                    reason=f"Sports match: '{team_text}' contains '{key}' -> {loc_name}",
                    inference_method=InferenceMethod.HEURISTIC,
                )

        return None

    # ── Merge & Confidence ────────────────────────────────────────────

    @staticmethod
    def _merge_key(name: str) -> str:
        """
        Normalize a location name into a merge key so that
        'Atlanta, GA' and 'Atlanta, GA, USA' merge together.
        """
        key = name.lower().strip()
        # Strip trailing country qualifiers
        for suffix in (", usa", ", united states", ", us",
                       ", uk", ", united kingdom", ", canada"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                break
        return key

    def _merge_candidates(self, candidates: list[LocationCandidate]) -> list[LocationCandidate]:
        """
        Merge duplicate location candidates.
        When multiple signals point to the same location, boost confidence.
        When signals conflict, keep both but don't boost.
        """
        merged: dict[str, LocationCandidate] = {}

        for c in candidates:
            key = self._merge_key(c.location_name)
            if key in merged:
                existing = merged[key]
                # Combine confidence: union of independent signals
                # P(A or B) = P(A) + P(B) - P(A)*P(B)
                combined = existing.confidence + c.confidence - (existing.confidence * c.confidence)
                combined = min(round(combined, 2), 0.99)

                reasons = existing.reason
                if c.reason not in reasons:
                    reasons = f"{reasons}; {c.reason}"

                # Use the more specific location type
                loc_type = c.location_type if c.location_type != LocationType.CITY else existing.location_type

                # Prefer pre-geocoded coordinates
                lat = c.latitude if c.latitude else existing.latitude
                lon = c.longitude if c.longitude else existing.longitude

                merged[key] = LocationCandidate(
                    location_name=existing.location_name,
                    location_type=loc_type,
                    confidence=combined,
                    reason=reasons,
                    inference_method=existing.inference_method,
                    latitude=lat,
                    longitude=lon,
                )
            else:
                merged[key] = c

        return list(merged.values())

    def _apply_ambiguity_penalty(self, candidates: list[LocationCandidate]) -> list[LocationCandidate]:
        """
        If there are many candidates, reduce confidence for all (market is ambiguous).
        1 candidate: no penalty
        2 candidates: -5% each
        3+ candidates: -10% each
        """
        n = len(candidates)
        if n <= 1:
            return candidates

        penalty = 0.05 if n == 2 else 0.10

        return [
            LocationCandidate(
                location_name=c.location_name,
                location_type=c.location_type,
                confidence=max(round(c.confidence - penalty, 2), 0.05),
                reason=c.reason + f" [ambiguity penalty: -{penalty:.0%}, {n} candidates]",
                inference_method=c.inference_method,
                latitude=c.latitude,
                longitude=c.longitude,
            )
            for c in candidates
        ]


# ══════════════════════════════════════════════════════════════════════
# LLM FALLBACK (Optional)
# ══════════════════════════════════════════════════════════════════════

async def llm_infer_location(
    question: str,
    description: Optional[str] = None,
) -> list[LocationCandidate]:
    """
    Use an LLM to infer locations from market text.
    Returns structured LocationCandidate list.
    Only called when NLP + heuristics produce no results and LLM is enabled.
    """
    settings = get_settings().inference
    if not settings.llm_enabled or not settings.llm_api_key:
        return []

    import json as json_mod

    prompt = f"""Analyze this prediction market question and extract geographic locations.
Return a JSON array of objects with these fields:
- location_name: human-readable location (e.g., "Atlanta, GA" or "London, UK")
- location_type: one of "city", "state", "country", "building", "arena", "global"
- confidence: float 0-1 indicating how strongly the market relates to this location
- reason: brief explanation

If the market has no geographic relevance, return:
[{{"location_name": "Global / No specific location", "location_type": "global", "confidence": 0.9, "reason": "No geographic relevance"}}]

Market question: {question}
{"Market description: " + description if description else ""}

Return ONLY the JSON array, no other text."""

    try:
        if settings.llm_provider == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=settings.llm_api_key)
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            raw_text = response.choices[0].message.content.strip()
        elif settings.llm_provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.llm_api_key)
            response = await client.messages.create(
                model=settings.llm_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()
        else:
            logger.warning("Unknown LLM provider: %s", settings.llm_provider)
            return []

        # Parse JSON response
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0]

        parsed = json_mod.loads(raw_text)
        candidates = []
        for item in parsed:
            try:
                loc_type = LocationType(item.get("location_type", "city"))
            except ValueError:
                loc_type = LocationType.CITY

            candidates.append(LocationCandidate(
                location_name=item["location_name"],
                location_type=loc_type,
                confidence=min(float(item.get("confidence", 0.5)), 0.85),  # cap LLM confidence
                reason=f"LLM: {item.get('reason', 'inferred by LLM')}",
                inference_method=InferenceMethod.LLM,
            ))

        return candidates

    except Exception as e:
        logger.error("LLM inference failed: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════════
# BATCH INFERENCE
# ══════════════════════════════════════════════════════════════════════

async def infer_locations_batch(
    markets: list[dict],
) -> list[MarketInferenceResult]:
    """
    Run inference on a batch of markets.
    Uses synchronous NLP + heuristics first, then async LLM fallback if needed.
    """
    engine = LocationInferenceEngine()
    settings = get_settings().inference
    results = []

    for market in markets:
        result = engine.infer(
            condition_id=market["condition_id"],
            question=market["question"],
            description=market.get("description"),
        )

        # LLM fallback for markets with no locations
        if not result.has_location and not result.is_global and settings.llm_enabled:
            logger.debug("No NLP/heuristic locations for %s, trying LLM",
                         market["condition_id"])
            llm_candidates = await llm_infer_location(
                market["question"], market.get("description")
            )
            if llm_candidates:
                result = MarketInferenceResult(
                    condition_id=market["condition_id"],
                    locations=llm_candidates,
                    has_location=any(
                        c.location_type != LocationType.GLOBAL for c in llm_candidates
                    ),
                    is_global=all(
                        c.location_type == LocationType.GLOBAL for c in llm_candidates
                    ),
                )

        results.append(result)

    return results
