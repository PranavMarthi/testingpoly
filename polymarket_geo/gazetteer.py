"""
Gazetteer-based geographic entity extraction.

A large dictionary of known place names (countries, capitals, major cities,
US states, demonyms, and regions) that can be matched against text using
word-boundary regex. This works without spaCy or any ML model.

Design:
  - Every entry maps a surface form (what appears in text) to a canonical
    location name + type + coordinates.
  - Demonyms ("Japanese", "French", "Somali") map to their country.
  - US state names and abbreviations are included.
  - Matching uses pre-compiled word-boundary regexes for speed.
  - Case-insensitive matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GazetteerEntry:
    canonical_name: str        # "Paris, France" or "Somalia"
    location_type: str         # city, country, state, region
    latitude: Optional[float] = None
    longitude: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════
# GAZETTEER DATA
# ══════════════════════════════════════════════════════════════════════

# Each key is a lowercase surface form. Values are GazetteerEntry objects.
# Multiple surface forms can point to the same canonical location.

_RAW_GAZETTEER: dict[str, GazetteerEntry] = {}


def _add(surface_forms: list[str], canonical: str, loc_type: str,
         lat: Optional[float] = None, lon: Optional[float] = None):
    entry = GazetteerEntry(canonical, loc_type, lat, lon)
    for form in surface_forms:
        _RAW_GAZETTEER[form.lower()] = entry


# ── Countries + demonyms + capitals ───────────────────────────────────

_add(["afghanistan", "afghan"], "Afghanistan", "country", 33.94, 67.71)
_add(["kabul"], "Kabul, Afghanistan", "city", 34.53, 69.17)
_add(["albania", "albanian"], "Albania", "country", 41.15, 20.17)
_add(["algeria", "algerian"], "Algeria", "country", 28.03, 1.66)
_add(["argentina", "argentine", "argentinian"], "Argentina", "country", -38.42, -63.62)
_add(["buenos aires"], "Buenos Aires, Argentina", "city", -34.60, -58.38)
_add(["armenia", "armenian"], "Armenia", "country", 40.07, 45.04)
_add(["australia", "australian", "aussie"], "Australia", "country", -25.27, 133.78)
_add(["sydney"], "Sydney, Australia", "city", -33.87, 151.21)
_add(["melbourne"], "Melbourne, Australia", "city", -37.81, 144.96)
_add(["austria", "austrian"], "Austria", "country", 47.52, 14.55)
_add(["vienna"], "Vienna, Austria", "city", 48.21, 16.37)
_add(["azerbaijan", "azerbaijani", "azeri"], "Azerbaijan", "country", 40.14, 47.58)
_add(["bahrain", "bahraini"], "Bahrain", "country", 26.07, 50.55)
_add(["bangladesh", "bangladeshi"], "Bangladesh", "country", 23.68, 90.36)
_add(["belarus", "belarusian"], "Belarus", "country", 53.71, 27.95)
_add(["belgium", "belgian"], "Belgium", "country", 50.50, 4.47)
_add(["brussels"], "Brussels, Belgium", "city", 50.85, 4.35)
_add(["bolivia", "bolivian"], "Bolivia", "country", -16.29, -63.59)
_add(["bosnia", "bosnian", "bosnia and herzegovina"], "Bosnia and Herzegovina", "country", 43.92, 17.68)
_add(["brazil", "brazilian"], "Brazil", "country", -14.24, -51.93)
_add(["rio de janeiro", "rio"], "Rio de Janeiro, Brazil", "city", -22.91, -43.17)
_add(["sao paulo", "são paulo"], "São Paulo, Brazil", "city", -23.55, -46.63)
_add(["bulgaria", "bulgarian"], "Bulgaria", "country", 42.73, 25.49)
_add(["cambodia", "cambodian", "khmer"], "Cambodia", "country", 12.57, 104.99)
_add(["cameroon", "cameroonian"], "Cameroon", "country", 7.37, 12.35)
_add(["canada", "canadian"], "Canada", "country", 56.13, -106.35)
_add(["ottawa"], "Ottawa, Canada", "city", 45.42, -75.70)
_add(["toronto"], "Toronto, Canada", "city", 43.65, -79.38)
_add(["vancouver"], "Vancouver, Canada", "city", 49.28, -123.12)
_add(["montreal"], "Montreal, Canada", "city", 45.50, -73.57)
_add(["chile", "chilean"], "Chile", "country", -35.68, -71.54)
_add(["santiago"], "Santiago, Chile", "city", -33.45, -70.67)
_add(["china", "chinese"], "China", "country", 35.86, 104.20)
_add(["beijing"], "Beijing, China", "city", 39.90, 116.41)
_add(["shanghai"], "Shanghai, China", "city", 31.23, 121.47)
_add(["hong kong"], "Hong Kong", "city", 22.32, 114.17)
_add(["colombia", "colombian"], "Colombia", "country", 4.57, -74.30)
_add(["bogota", "bogotá"], "Bogotá, Colombia", "city", 4.71, -74.07)
_add(["congo", "congolese", "drc"], "Democratic Republic of the Congo", "country", -4.04, 21.76)
_add(["costa rica", "costa rican"], "Costa Rica", "country", 9.75, -83.75)
_add(["croatia", "croatian"], "Croatia", "country", 45.10, 15.20)
_add(["cuba", "cuban"], "Cuba", "country", 21.52, -77.78)
_add(["havana"], "Havana, Cuba", "city", 23.11, -82.37)
_add(["cyprus", "cypriot"], "Cyprus", "country", 35.13, 33.43)
_add(["czech republic", "czech", "czechia"], "Czech Republic", "country", 49.82, 15.47)
_add(["prague"], "Prague, Czech Republic", "city", 50.08, 14.44)
_add(["denmark", "danish", "dane", "danes"], "Denmark", "country", 56.26, 9.50)
_add(["copenhagen"], "Copenhagen, Denmark", "city", 55.68, 12.57)
_add(["dominican republic", "dominican"], "Dominican Republic", "country", 18.74, -70.16)
_add(["ecuador", "ecuadorian"], "Ecuador", "country", -1.83, -78.18)
_add(["egypt", "egyptian"], "Egypt", "country", 26.82, 30.80)
_add(["cairo"], "Cairo, Egypt", "city", 30.04, 31.24)
_add(["el salvador", "salvadoran", "salvadorean"], "El Salvador", "country", 13.79, -88.90)
_add(["estonia", "estonian"], "Estonia", "country", 58.60, 25.01)
_add(["ethiopia", "ethiopian"], "Ethiopia", "country", 9.15, 40.49)
_add(["addis ababa"], "Addis Ababa, Ethiopia", "city", 9.02, 38.75)
_add(["finland", "finnish", "finn", "finns"], "Finland", "country", 61.92, 25.75)
_add(["helsinki"], "Helsinki, Finland", "city", 60.17, 24.94)
_add(["france", "french"], "France", "country", 46.23, 2.21)
_add(["paris"], "Paris, France", "city", 48.86, 2.35)
_add(["marseille"], "Marseille, France", "city", 43.30, 5.37)
_add(["lyon"], "Lyon, France", "city", 45.76, 4.84)
_add(["georgia"], "Georgia (country)", "country", 42.32, 43.36)  # country, not US state
_add(["georgian"], "Georgia (country)", "country", 42.32, 43.36)
_add(["tbilisi"], "Tbilisi, Georgia", "city", 41.72, 44.79)
_add(["germany", "german"], "Germany", "country", 51.17, 10.45)
_add(["berlin"], "Berlin, Germany", "city", 52.52, 13.41)
_add(["munich"], "Munich, Germany", "city", 48.14, 11.58)
_add(["frankfurt"], "Frankfurt, Germany", "city", 50.11, 8.68)
_add(["ghana", "ghanaian"], "Ghana", "country", 7.95, -1.02)
_add(["greece", "greek"], "Greece", "country", 39.07, 21.82)
_add(["athens"], "Athens, Greece", "city", 37.98, 23.73)
_add(["guatemala", "guatemalan"], "Guatemala", "country", 15.78, -90.23)
_add(["haiti", "haitian"], "Haiti", "country", 18.97, -72.29)
_add(["honduras", "honduran"], "Honduras", "country", 15.20, -86.24)
_add(["hungary", "hungarian"], "Hungary", "country", 47.16, 19.50)
_add(["budapest"], "Budapest, Hungary", "city", 47.50, 19.04)
_add(["iceland", "icelandic", "icelander"], "Iceland", "country", 64.96, -19.02)
_add(["reykjavik"], "Reykjavik, Iceland", "city", 64.15, -21.94)
_add(["india", "indian"], "India", "country", 20.59, 78.96)
_add(["new delhi", "delhi"], "New Delhi, India", "city", 28.61, 77.21)
_add(["mumbai"], "Mumbai, India", "city", 19.08, 72.88)
_add(["indonesia", "indonesian"], "Indonesia", "country", -0.79, 113.92)
_add(["jakarta"], "Jakarta, Indonesia", "city", -6.21, 106.85)
_add(["iran", "iranian", "persian"], "Iran", "country", 32.43, 53.69)
_add(["tehran"], "Tehran, Iran", "city", 35.69, 51.39)
_add(["iraq", "iraqi"], "Iraq", "country", 33.22, 43.68)
_add(["baghdad"], "Baghdad, Iraq", "city", 33.31, 44.37)
_add(["ireland", "irish"], "Ireland", "country", 53.14, -7.69)
_add(["dublin"], "Dublin, Ireland", "city", 53.35, -6.26)
_add(["israel", "israeli"], "Israel", "country", 31.05, 34.85)
_add(["jerusalem"], "Jerusalem, Israel", "city", 31.77, 35.23)
_add(["tel aviv"], "Tel Aviv, Israel", "city", 32.09, 34.78)
_add(["italy", "italian"], "Italy", "country", 41.87, 12.57)
_add(["rome"], "Rome, Italy", "city", 41.90, 12.50)
_add(["milan"], "Milan, Italy", "city", 45.46, 9.19)
_add(["naples"], "Naples, Italy", "city", 40.85, 14.27)
_add(["ivory coast", "cote d'ivoire", "ivorian"], "Ivory Coast", "country", 7.54, -5.55)
_add(["jamaica", "jamaican"], "Jamaica", "country", 18.11, -77.30)
_add(["japan", "japanese"], "Japan", "country", 36.20, 138.25)
_add(["tokyo"], "Tokyo, Japan", "city", 35.68, 139.65)
_add(["osaka"], "Osaka, Japan", "city", 34.69, 135.50)
_add(["jordan", "jordanian"], "Jordan", "country", 30.59, 36.24)
_add(["amman"], "Amman, Jordan", "city", 31.95, 35.93)
_add(["kazakhstan", "kazakh"], "Kazakhstan", "country", 48.02, 66.92)
_add(["kenya", "kenyan"], "Kenya", "country", -0.02, 37.91)
_add(["nairobi"], "Nairobi, Kenya", "city", -1.29, 36.82)
_add(["north korea", "north korean", "dprk"], "North Korea", "country", 40.34, 127.51)
_add(["pyongyang"], "Pyongyang, North Korea", "city", 39.04, 125.76)
_add(["south korea", "south korean", "korean"], "South Korea", "country", 35.91, 127.77)
_add(["seoul"], "Seoul, South Korea", "city", 37.57, 126.98)
_add(["kosovo", "kosovar"], "Kosovo", "country", 42.60, 20.90)
_add(["kuwait", "kuwaiti"], "Kuwait", "country", 29.31, 47.48)
_add(["kyrgyzstan", "kyrgyz"], "Kyrgyzstan", "country", 41.20, 74.77)
_add(["laos", "lao", "laotian"], "Laos", "country", 19.86, 102.50)
_add(["latvia", "latvian"], "Latvia", "country", 56.88, 24.60)
_add(["lebanon", "lebanese"], "Lebanon", "country", 33.85, 35.86)
_add(["beirut"], "Beirut, Lebanon", "city", 33.89, 35.50)
_add(["libya", "libyan"], "Libya", "country", 26.34, 17.23)
_add(["tripoli"], "Tripoli, Libya", "city", 32.90, 13.18)
_add(["lithuania", "lithuanian"], "Lithuania", "country", 55.17, 23.88)
_add(["luxembourg", "luxembourgish"], "Luxembourg", "country", 49.82, 6.13)
_add(["madagascar", "malagasy"], "Madagascar", "country", -18.77, 46.87)
_add(["malaysia", "malaysian", "malay"], "Malaysia", "country", 4.21, 101.98)
_add(["kuala lumpur"], "Kuala Lumpur, Malaysia", "city", 3.14, 101.69)
_add(["mali", "malian"], "Mali", "country", 17.57, -4.00)
_add(["mexico", "mexican"], "Mexico", "country", 23.63, -102.55)
_add(["mexico city"], "Mexico City, Mexico", "city", 19.43, -99.13)
_add(["moldova", "moldovan"], "Moldova", "country", 47.41, 28.37)
_add(["mongolia", "mongolian"], "Mongolia", "country", 46.86, 103.85)
_add(["montenegro", "montenegrin"], "Montenegro", "country", 42.71, 19.37)
_add(["morocco", "moroccan"], "Morocco", "country", 31.79, -7.09)
_add(["mozambique", "mozambican"], "Mozambique", "country", -18.67, 35.53)
_add(["myanmar", "burmese", "burma"], "Myanmar", "country", 21.91, 95.96)
_add(["nepal", "nepalese", "nepali"], "Nepal", "country", 28.39, 84.12)
_add(["netherlands", "dutch", "holland"], "Netherlands", "country", 52.13, 5.29)
_add(["amsterdam"], "Amsterdam, Netherlands", "city", 52.37, 4.90)
_add(["new zealand", "kiwi"], "New Zealand", "country", -40.90, 174.89)
_add(["auckland"], "Auckland, New Zealand", "city", -36.85, 174.76)
_add(["nicaragua", "nicaraguan"], "Nicaragua", "country", 12.87, -85.21)
_add(["niger", "nigerien"], "Niger", "country", 17.61, 8.08)
_add(["nigeria", "nigerian"], "Nigeria", "country", 9.08, 8.68)
_add(["lagos"], "Lagos, Nigeria", "city", 6.52, 3.38)
_add(["norway", "norwegian"], "Norway", "country", 60.47, 8.47)
_add(["oslo"], "Oslo, Norway", "city", 59.91, 10.75)
_add(["oman", "omani"], "Oman", "country", 21.47, 55.98)
_add(["pakistan", "pakistani"], "Pakistan", "country", 30.38, 69.35)
_add(["islamabad"], "Islamabad, Pakistan", "city", 33.69, 73.04)
_add(["karachi"], "Karachi, Pakistan", "city", 24.86, 67.01)
_add(["palestine", "palestinian", "gaza", "west bank"], "Palestine", "country", 31.95, 35.23)
_add(["panama", "panamanian"], "Panama", "country", 8.54, -80.78)
_add(["paraguay", "paraguayan"], "Paraguay", "country", -23.44, -58.44)
_add(["peru", "peruvian"], "Peru", "country", -9.19, -75.02)
_add(["lima"], "Lima, Peru", "city", -12.05, -77.04)
_add(["philippines", "filipino", "philippine"], "Philippines", "country", 12.88, 121.77)
_add(["manila"], "Manila, Philippines", "city", 14.60, 120.98)
_add(["poland", "polish", "pole", "poles"], "Poland", "country", 51.92, 19.15)
_add(["warsaw"], "Warsaw, Poland", "city", 52.23, 21.01)
_add(["portugal", "portuguese"], "Portugal", "country", 39.40, -8.22)
_add(["lisbon"], "Lisbon, Portugal", "city", 38.72, -9.14)
_add(["qatar", "qatari"], "Qatar", "country", 25.35, 51.18)
_add(["doha"], "Doha, Qatar", "city", 25.29, 51.53)
_add(["romania", "romanian"], "Romania", "country", 45.94, 24.97)
_add(["bucharest"], "Bucharest, Romania", "city", 44.43, 26.10)
_add(["russia", "russian"], "Russia", "country", 61.52, 105.32)
_add(["moscow"], "Moscow, Russia", "city", 55.76, 37.62)
_add(["st. petersburg", "saint petersburg"], "Saint Petersburg, Russia", "city", 59.93, 30.32)
_add(["rwanda", "rwandan"], "Rwanda", "country", -1.94, 29.87)
_add(["saudi arabia", "saudi"], "Saudi Arabia", "country", 23.89, 45.08)
_add(["riyadh"], "Riyadh, Saudi Arabia", "city", 24.71, 46.68)
_add(["senegal", "senegalese"], "Senegal", "country", 14.50, -14.45)
_add(["serbia", "serbian"], "Serbia", "country", 44.02, 21.01)
_add(["belgrade"], "Belgrade, Serbia", "city", 44.79, 20.47)
_add(["singapore", "singaporean"], "Singapore", "country", 1.35, 103.82)
_add(["slovakia", "slovak"], "Slovakia", "country", 48.67, 19.70)
_add(["slovenia", "slovenian", "slovene"], "Slovenia", "country", 46.15, 14.99)
_add(["somalia", "somali", "somalian"], "Somalia", "country", 5.15, 46.20)
_add(["mogadishu"], "Mogadishu, Somalia", "city", 2.05, 45.32)
_add(["south africa", "south african"], "South Africa", "country", -30.56, 22.94)
_add(["johannesburg"], "Johannesburg, South Africa", "city", -26.20, 28.05)
_add(["cape town"], "Cape Town, South Africa", "city", -33.92, 18.42)
_add(["spain", "spanish", "spaniard"], "Spain", "country", 40.46, -3.75)
_add(["madrid"], "Madrid, Spain", "city", 40.42, -3.70)
_add(["barcelona"], "Barcelona, Spain", "city", 41.39, 2.17)
_add(["sri lanka", "sri lankan"], "Sri Lanka", "country", 7.87, 80.77)
_add(["sudan", "sudanese"], "Sudan", "country", 12.86, 30.22)
_add(["khartoum"], "Khartoum, Sudan", "city", 15.50, 32.56)
_add(["south sudan", "south sudanese"], "South Sudan", "country", 6.88, 31.31)
_add(["sweden", "swedish", "swede", "swedes"], "Sweden", "country", 60.13, 18.64)
_add(["stockholm"], "Stockholm, Sweden", "city", 59.33, 18.07)
_add(["switzerland", "swiss"], "Switzerland", "country", 46.82, 8.23)
_add(["zurich", "zürich"], "Zurich, Switzerland", "city", 47.38, 8.54)
_add(["geneva"], "Geneva, Switzerland", "city", 46.20, 6.14)
_add(["syria", "syrian"], "Syria", "country", 34.80, 38.99)
_add(["damascus"], "Damascus, Syria", "city", 33.51, 36.29)
_add(["taiwan", "taiwanese"], "Taiwan", "country", 23.70, 120.96)
_add(["taipei"], "Taipei, Taiwan", "city", 25.03, 121.57)
_add(["tajikistan", "tajik"], "Tajikistan", "country", 38.86, 71.28)
_add(["tanzania", "tanzanian"], "Tanzania", "country", -6.37, 34.89)
_add(["thailand", "thai"], "Thailand", "country", 15.87, 100.99)
_add(["bangkok"], "Bangkok, Thailand", "city", 13.76, 100.50)
_add(["tunisia", "tunisian"], "Tunisia", "country", 33.89, 9.54)
_add(["turkey", "turkish"], "Turkey", "country", 38.96, 35.24)
_add(["istanbul"], "Istanbul, Turkey", "city", 41.01, 28.98)
_add(["ankara"], "Ankara, Turkey", "city", 39.93, 32.87)
_add(["turkmenistan", "turkmen"], "Turkmenistan", "country", 38.97, 59.56)
_add(["uganda", "ugandan"], "Uganda", "country", 1.37, 32.29)
_add(["ukraine", "ukrainian"], "Ukraine", "country", 48.38, 31.17)
_add(["kyiv", "kiev"], "Kyiv, Ukraine", "city", 50.45, 30.52)
_add(["united arab emirates", "uae", "emirati", "dubai", "abu dhabi"], "United Arab Emirates", "country", 23.42, 53.85)
_add(["united kingdom", "uk", "britain", "british", "great britain"], "United Kingdom", "country", 55.38, -3.44)
_add(["london"], "London, United Kingdom", "city", 51.51, -0.13)
_add(["manchester"], "Manchester, United Kingdom", "city", 53.48, -2.24)
_add(["birmingham"], "Birmingham, United Kingdom", "city", 52.49, -1.90)
_add(["edinburgh"], "Edinburgh, United Kingdom", "city", 55.95, -3.19)
_add(["scotland", "scottish", "scots"], "Scotland, United Kingdom", "region", 56.49, -4.20)
_add(["wales", "welsh"], "Wales, United Kingdom", "region", 52.13, -3.78)
_add(["england", "english"], "England, United Kingdom", "region", 52.36, -1.17)
_add(["northern ireland"], "Northern Ireland, United Kingdom", "region", 54.79, -6.49)
_add(["uruguay", "uruguayan"], "Uruguay", "country", -32.52, -55.77)
_add(["uzbekistan", "uzbek"], "Uzbekistan", "country", 41.38, 64.59)
_add(["venezuela", "venezuelan"], "Venezuela", "country", 6.42, -66.59)
_add(["caracas"], "Caracas, Venezuela", "city", 10.48, -66.90)
_add(["vietnam", "vietnamese"], "Vietnam", "country", 14.06, 108.28)
_add(["hanoi"], "Hanoi, Vietnam", "city", 21.03, 105.85)
_add(["yemen", "yemeni"], "Yemen", "country", 15.55, 48.52)
_add(["zambia", "zambian"], "Zambia", "country", -13.13, 27.85)
_add(["zimbabwe", "zimbabwean"], "Zimbabwe", "country", -19.02, 29.15)

# ── U.S. states ───────────────────────────────────────────────────────

_add(["alabama"], "Alabama, USA", "state", 32.32, -86.90)
_add(["alaska"], "Alaska, USA", "state", 63.59, -154.49)
_add(["arizona"], "Arizona, USA", "state", 34.05, -111.09)
_add(["arkansas"], "Arkansas, USA", "state", 35.20, -91.83)
_add(["california", "californian"], "California, USA", "state", 36.78, -119.42)
_add(["colorado"], "Colorado, USA", "state", 39.55, -105.78)
_add(["connecticut"], "Connecticut, USA", "state", 41.60, -72.76)
_add(["delaware"], "Delaware, USA", "state", 38.91, -75.53)
_add(["florida", "floridian"], "Florida, USA", "state", 27.66, -81.52)
_add(["hawaii", "hawaiian"], "Hawaii, USA", "state", 19.90, -155.58)
_add(["idaho"], "Idaho, USA", "state", 44.07, -114.74)
_add(["illinois"], "Illinois, USA", "state", 40.63, -89.40)
_add(["indiana"], "Indiana, USA", "state", 40.27, -86.13)
_add(["iowa"], "Iowa, USA", "state", 41.88, -93.10)
_add(["kansas"], "Kansas, USA", "state", 39.01, -98.48)
_add(["kentucky"], "Kentucky, USA", "state", 37.84, -84.27)
_add(["louisiana"], "Louisiana, USA", "state", 30.98, -91.96)
_add(["maine"], "Maine, USA", "state", 45.25, -69.45)
_add(["maryland"], "Maryland, USA", "state", 39.05, -76.64)
_add(["massachusetts"], "Massachusetts, USA", "state", 42.41, -71.38)
_add(["michigan"], "Michigan, USA", "state", 44.31, -85.60)
_add(["minnesota"], "Minnesota, USA", "state", 46.73, -94.69)
_add(["mississippi"], "Mississippi, USA", "state", 32.35, -89.40)
_add(["missouri"], "Missouri, USA", "state", 37.96, -91.83)
_add(["montana"], "Montana, USA", "state", 46.88, -110.36)
_add(["nebraska"], "Nebraska, USA", "state", 41.49, -99.90)
_add(["nevada"], "Nevada, USA", "state", 38.80, -116.42)
_add(["new hampshire"], "New Hampshire, USA", "state", 43.19, -71.57)
_add(["new jersey"], "New Jersey, USA", "state", 40.06, -74.41)
_add(["new mexico"], "New Mexico, USA", "state", 34.52, -105.87)
_add(["new york"], "New York, USA", "state", 40.71, -74.01)  # also matches city
_add(["north carolina"], "North Carolina, USA", "state", 35.76, -79.02)
_add(["north dakota"], "North Dakota, USA", "state", 47.55, -101.00)
_add(["ohio"], "Ohio, USA", "state", 40.42, -82.91)
_add(["oklahoma"], "Oklahoma, USA", "state", 35.47, -97.52)
_add(["oregon"], "Oregon, USA", "state", 43.80, -120.55)
_add(["pennsylvania"], "Pennsylvania, USA", "state", 41.20, -77.19)
_add(["rhode island"], "Rhode Island, USA", "state", 41.58, -71.48)
_add(["south carolina"], "South Carolina, USA", "state", 33.84, -81.16)
_add(["south dakota"], "South Dakota, USA", "state", 43.97, -99.90)
_add(["tennessee"], "Tennessee, USA", "state", 35.52, -86.58)
_add(["texas", "texan"], "Texas, USA", "state", 31.97, -99.90)
_add(["utah"], "Utah, USA", "state", 39.32, -111.09)
_add(["vermont"], "Vermont, USA", "state", 44.56, -72.58)
_add(["virginia"], "Virginia, USA", "state", 37.43, -78.66)
_add(["west virginia"], "West Virginia, USA", "state", 38.60, -80.95)
_add(["wisconsin"], "Wisconsin, USA", "state", 43.78, -88.79)
_add(["wyoming"], "Wyoming, USA", "state", 43.08, -107.29)

# ── Major US cities (not already in states or sports teams) ───────────

_add(["atlanta"], "Atlanta, GA, USA", "city", 33.75, -84.39)
_add(["austin"], "Austin, TX, USA", "city", 30.27, -97.74)
_add(["baltimore"], "Baltimore, MD, USA", "city", 39.29, -76.61)
_add(["boston"], "Boston, MA, USA", "city", 42.36, -71.06)
_add(["charlotte"], "Charlotte, NC, USA", "city", 35.23, -80.84)
_add(["chicago"], "Chicago, IL, USA", "city", 41.88, -87.63)
_add(["cincinnati"], "Cincinnati, OH, USA", "city", 39.10, -84.51)
_add(["cleveland"], "Cleveland, OH, USA", "city", 41.50, -81.69)
_add(["columbus"], "Columbus, OH, USA", "city", 39.96, -83.00)
_add(["dallas"], "Dallas, TX, USA", "city", 32.78, -96.80)
_add(["denver"], "Denver, CO, USA", "city", 39.74, -104.99)
_add(["detroit"], "Detroit, MI, USA", "city", 42.33, -83.05)
_add(["el paso"], "El Paso, TX, USA", "city", 31.76, -106.49)
_add(["fort worth"], "Fort Worth, TX, USA", "city", 32.76, -97.33)
_add(["houston"], "Houston, TX, USA", "city", 29.76, -95.37)
_add(["indianapolis"], "Indianapolis, IN, USA", "city", 39.77, -86.16)
_add(["jacksonville"], "Jacksonville, FL, USA", "city", 30.33, -81.66)
_add(["kansas city"], "Kansas City, MO, USA", "city", 39.10, -94.58)
_add(["las vegas"], "Las Vegas, NV, USA", "city", 36.17, -115.14)
_add(["los angeles"], "Los Angeles, CA, USA", "city", 34.05, -118.24)
_add(["louisville"], "Louisville, KY, USA", "city", 38.25, -85.76)
_add(["memphis"], "Memphis, TN, USA", "city", 35.15, -90.05)
_add(["miami"], "Miami, FL, USA", "city", 25.76, -80.19)
_add(["milwaukee"], "Milwaukee, WI, USA", "city", 43.04, -87.91)
_add(["minneapolis"], "Minneapolis, MN, USA", "city", 44.98, -93.27)
_add(["nashville"], "Nashville, TN, USA", "city", 36.16, -86.78)
_add(["new orleans"], "New Orleans, LA, USA", "city", 29.95, -90.07)
_add(["oklahoma city"], "Oklahoma City, OK, USA", "city", 35.47, -97.52)
_add(["omaha"], "Omaha, NE, USA", "city", 41.26, -95.94)
_add(["orlando"], "Orlando, FL, USA", "city", 28.54, -81.38)
_add(["philadelphia"], "Philadelphia, PA, USA", "city", 39.95, -75.17)
_add(["phoenix"], "Phoenix, AZ, USA", "city", 33.45, -112.07)
_add(["pittsburgh"], "Pittsburgh, PA, USA", "city", 40.44, -80.00)
_add(["portland"], "Portland, OR, USA", "city", 45.52, -122.68)
_add(["raleigh"], "Raleigh, NC, USA", "city", 35.78, -78.64)
_add(["sacramento"], "Sacramento, CA, USA", "city", 38.58, -121.49)
_add(["salt lake city"], "Salt Lake City, UT, USA", "city", 40.76, -111.89)
_add(["san antonio"], "San Antonio, TX, USA", "city", 29.42, -98.49)
_add(["san diego"], "San Diego, CA, USA", "city", 32.72, -117.16)
_add(["san francisco"], "San Francisco, CA, USA", "city", 37.77, -122.42)
_add(["san jose"], "San Jose, CA, USA", "city", 37.34, -121.89)
_add(["seattle"], "Seattle, WA, USA", "city", 47.61, -122.33)
_add(["st. louis", "saint louis"], "St. Louis, MO, USA", "city", 38.63, -90.20)
_add(["tampa"], "Tampa, FL, USA", "city", 27.95, -82.46)
_add(["tucson"], "Tucson, AZ, USA", "city", 32.22, -110.93)
_add(["virginia beach"], "Virginia Beach, VA, USA", "city", 36.85, -75.98)
_add(["washington d.c.", "washington, d.c.", "washington dc", "d.c."], "Washington, DC, USA", "city", 38.91, -77.04)

# ── Regions / geopolitical areas ──────────────────────────────────────

_add(["middle east", "mideast"], "Middle East", "region", 29.0, 41.0)
_add(["europe", "european"], "Europe", "region", 54.53, 15.26)
_add(["asia", "asian"], "Asia", "region", 34.05, 100.62)
_add(["africa", "african"], "Africa", "region", -8.78, 34.51)
_add(["south america", "south american"], "South America", "region", -8.78, -55.49)
_add(["latin america", "latin american"], "Latin America", "region", 14.60, -90.55)
_add(["southeast asia"], "Southeast Asia", "region", 2.50, 112.50)
_add(["east asia", "east asian"], "East Asia", "region", 35.86, 104.20)
_add(["central america", "central american"], "Central America", "region", 12.77, -85.60)
_add(["caribbean"], "Caribbean", "region", 21.47, -78.66)
_add(["balkans", "balkan"], "Balkans", "region", 41.50, 21.50)
_add(["scandinavia", "scandinavian"], "Scandinavia", "region", 61.52, 13.38)
_add(["arctic"], "Arctic", "region", 90.0, 0.0)
_add(["antarctic", "antarctica"], "Antarctica", "region", -82.86, -135.00)
_add(["sahel"], "Sahel, Africa", "region", 14.50, 0.0)
_add(["crimea", "crimean"], "Crimea", "region", 44.95, 34.10)
_add(["donbas", "donbass"], "Donbas, Ukraine", "region", 48.00, 37.80)
_add(["kashmir", "kashmiri"], "Kashmir", "region", 34.08, 74.80)
_add(["tibet", "tibetan"], "Tibet, China", "region", 29.65, 91.10)
_add(["xinjiang", "uyghur"], "Xinjiang, China", "region", 41.12, 85.23)
_add(["kurdistan", "kurdish"], "Kurdistan", "region", 36.40, 44.40)

# ── Territories, dependencies, and disputed areas ─────────────────────

_add(["greenland", "greenlandic"], "Greenland", "country", 71.71, -42.60)
_add(["puerto rico", "puerto rican"], "Puerto Rico", "country", 18.22, -66.59)
_add(["guam", "guamanian"], "Guam", "country", 13.44, 144.79)
_add(["u.s. virgin islands", "usvi"], "U.S. Virgin Islands", "country", 18.34, -64.93)
_add(["american samoa"], "American Samoa", "country", -14.27, -170.13)
_add(["bermuda", "bermudian"], "Bermuda", "country", 32.32, -64.76)
_add(["cayman islands"], "Cayman Islands", "country", 19.31, -81.25)
_add(["gibraltar"], "Gibraltar", "country", 36.14, -5.35)
_add(["falkland islands", "falklands", "malvinas"], "Falkland Islands", "country", -51.80, -59.17)
_add(["faroe islands", "faroese"], "Faroe Islands", "country", 61.89, -6.91)
_add(["french polynesia", "tahiti"], "French Polynesia", "country", -17.68, -149.41)
_add(["new caledonia"], "New Caledonia", "country", -20.90, 165.62)
_add(["guadeloupe"], "Guadeloupe", "country", 16.27, -61.55)
_add(["martinique"], "Martinique", "country", 14.64, -61.02)
_add(["reunion", "réunion"], "Réunion", "country", -21.12, 55.54)
_add(["macau", "macao"], "Macau", "city", 22.20, 113.54)
_add(["northern mariana islands", "saipan"], "Northern Mariana Islands", "country", 15.18, 145.75)
_add(["curacao", "curaçao"], "Curaçao", "country", 12.17, -68.98)
_add(["aruba", "aruban"], "Aruba", "country", 12.51, -69.97)
_add(["isle of man"], "Isle of Man", "country", 54.24, -4.55)
_add(["channel islands", "jersey", "guernsey"], "Channel Islands", "country", 49.21, -2.13)
_add(["samoa", "samoan"], "Samoa", "country", -13.76, -172.10)
_add(["tonga", "tongan"], "Tonga", "country", -21.18, -175.20)
_add(["fiji", "fijian"], "Fiji", "country", -17.71, 178.07)
_add(["papua new guinea"], "Papua New Guinea", "country", -6.31, 143.96)
_add(["solomon islands"], "Solomon Islands", "country", -9.43, 160.03)
_add(["vanuatu"], "Vanuatu", "country", -15.38, 166.96)
_add(["timor-leste", "east timor", "timorese"], "Timor-Leste", "country", -8.87, 125.73)
_add(["brunei"], "Brunei", "country", 4.94, 114.95)
_add(["bhutan", "bhutanese"], "Bhutan", "country", 27.51, 90.43)
_add(["maldives", "maldivian"], "Maldives", "country", 3.20, 73.22)
_add(["mauritius", "mauritian"], "Mauritius", "country", -20.35, 57.55)
_add(["seychelles"], "Seychelles", "country", -4.68, 55.49)
_add(["eswatini", "swaziland", "swazi"], "Eswatini", "country", -26.52, 31.47)
_add(["lesotho"], "Lesotho", "country", -29.61, 28.23)
_add(["botswana"], "Botswana", "country", -22.33, 24.68)
_add(["namibia", "namibian"], "Namibia", "country", -22.96, 18.49)
_add(["gabon", "gabonese"], "Gabon", "country", -0.80, 11.61)
_add(["eritrea", "eritrean"], "Eritrea", "country", 15.18, 39.78)
_add(["djibouti", "djiboutian"], "Djibouti", "country", 11.83, 42.59)
_add(["chad", "chadian"], "Chad", "country", 15.45, 18.73)
_add(["burkina faso", "burkinabe"], "Burkina Faso", "country", 12.24, -1.56)
_add(["benin", "beninese"], "Benin", "country", 9.31, 2.32)
_add(["togo", "togolese"], "Togo", "country", 8.62, 1.21)
_add(["sierra leone"], "Sierra Leone", "country", 8.46, -11.78)
_add(["liberia", "liberian"], "Liberia", "country", 6.43, -9.43)
_add(["guinea", "guinean"], "Guinea", "country", 9.95, -9.70)
_add(["guinea-bissau"], "Guinea-Bissau", "country", 11.80, -15.18)
_add(["gambia", "gambian"], "Gambia", "country", 13.44, -15.31)
_add(["cape verde", "cabo verde"], "Cape Verde", "country", 16.00, -24.01)
_add(["comoros", "comorian"], "Comoros", "country", -11.88, 43.87)
_add(["mauritania", "mauritanian"], "Mauritania", "country", 21.01, -10.94)

# ── Commonly referenced places in prediction markets ──────────────────

_add(["silicon valley"], "Silicon Valley, CA, USA", "region", 37.39, -122.06)
_add(["wall street"], "Wall Street, New York, USA", "city", 40.71, -74.01)
_add(["hollywood"], "Hollywood, CA, USA", "city", 34.09, -118.33)
_add(["the pentagon"], "The Pentagon, VA, USA", "city", 38.87, -77.06)
_add(["davos"], "Davos, Switzerland", "city", 46.80, 9.84)
_add(["strait of hormuz", "hormuz"], "Strait of Hormuz", "region", 26.57, 56.25)
_add(["south china sea"], "South China Sea", "region", 12.00, 113.00)
_add(["taiwan strait"], "Taiwan Strait", "region", 24.00, 119.00)
_add(["black sea"], "Black Sea", "region", 43.17, 34.00)
_add(["red sea"], "Red Sea", "region", 20.00, 38.50)
_add(["suez canal", "suez"], "Suez Canal, Egypt", "region", 30.46, 32.34)
_add(["panama canal"], "Panama Canal, Panama", "region", 9.08, -79.68)

# ── Country codes / abbreviations commonly seen in markets ────────────

_add(["u.s.", "u.s.a.", "usa", "united states", "america", "american"], "United States", "country", 37.09, -95.71)


# ══════════════════════════════════════════════════════════════════════
# COMPILED MATCHER
# ══════════════════════════════════════════════════════════════════════

# ── Context-sensitive abbreviations ───────────────────────────────────
# Short codes like "US" that are also common English words.
# These require contextual patterns instead of bare word-boundary matching.
# Each tuple: (compiled_regex, GazetteerEntry)

_CONTEXT_PATTERNS: list[tuple[re.Pattern, GazetteerEntry]] = [
    # "US" — only match when it looks like a country reference, not the pronoun "us"
    # Matches: "the US", "US military", "US-China", "U.S.", start-of-sentence "US "
    (re.compile(
        r'(?:'
        r'(?:the|a)\s+US\b'          # "the US", "a US"
        r'|US[\s-](?:[A-Z])'          # "US military", "US-China" (uppercase follows)
        r'|\bUS\s+(?:military|government|president|congress|senate|forces|troops|economy|'
        r'dollar|election|strike|sanctions|tariff|trade|policy|border|citizen|federal|'
        r'national|supreme|state\s+dept|department|embassy|intelligence|navy|army|'
        r'air\s+force|coast\s+guard|debt|budget|deficit|gdp|inflation|unemployment|'
        r'stock|market|bank|treasury|house|officials?|lawmakers?|voters?|'
        r'acquire|annex|invade|attack|bomb|aid|support)'
        r'|(?:in|from|to|by|for|with|against|of|and)\s+(?:the\s+)?US\b'  # preposition + US
        r'|\bUS\s+(?:and|or|vs\.?|versus)\s'  # "US and China", "US vs China"
        r')',
        re.IGNORECASE,
    ), GazetteerEntry("United States", "country", 37.09, -95.71)),
]


# ── Proper noun extractor ─────────────────────────────────────────────
# Catch-all for capitalized words that might be place names not in the gazetteer.
# This is the safety net so we never miss a location just because it's not listed.

# Words that are capitalized but are NOT locations (common false positives)
_NON_LOCATION_WORDS = {
    # Common English words that appear capitalized at start of sentence
    "will", "what", "how", "when", "where", "who", "why", "which",
    "the", "this", "that", "these", "those", "there",
    "can", "could", "would", "should", "may", "might", "must",
    "do", "does", "did", "has", "have", "had", "is", "are", "was", "were",
    "be", "been", "being", "get", "got", "if", "no", "yes", "not",
    # Prediction market terms
    "super", "bowl", "mvp", "nba", "nfl", "mlb", "nhl", "mls", "ufc",
    "premier", "league", "cup", "championship", "series", "game",
    "election", "vote", "poll", "candidate", "president", "governor",
    "mayor", "senator", "minister", "prime",
    "act", "bill", "law", "code", "statute", "regulation", "rule", "policy", "order",
    "clarity", "genius",
    "price", "rate", "market", "stock", "bond", "index", "gdp",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # People / titles often capitalized
    "trump", "biden", "harris", "obama", "clinton", "bush",
    "elon", "musk", "bezos", "zuckerberg",
    "mr", "mrs", "ms", "dr", "sir", "lord", "king", "queen", "prince", "princess",
    # Crypto/tech (already handled by global keywords)
    "bitcoin", "btc", "ethereum", "eth", "solana", "dogecoin", "xrp",
    "openai", "chatgpt", "google", "apple", "microsoft", "meta", "amazon",
    "tesla", "spacex", "nvidia",
    # Organizations / parties
    "democrat", "democratic", "republican", "gop", "labour", "conservative",
    "liberal", "party", "ldp", "congress", "senate", "parliament",
    "nato", "opec", "sec", "fda", "fbi", "cia", "fed", "imf",
    # Sports terms
    "win", "lose", "beat", "defeat", "score", "points", "goals", "touchdown",
    "playoff", "finals", "semifinals", "world",
    # Other
    "no", "yes", "true", "false", "above", "below", "over", "under",
    "before", "after", "between", "during", "about", "first", "last",
    "new", "old", "next",
}

# Pattern to extract capitalized proper nouns (single or multi-word)
_PROPER_NOUN_RE = re.compile(
    r'\b('
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'  # "Greenland", "North Korea", "San Francisco"
    r')\b'
)


def extract_proper_nouns(text: str) -> list[str]:
    """
    Extract capitalized proper nouns from text that could be locations.
    Filters out known non-location words and already-gazetteer-matched names.
    Returns a list of candidate location strings.
    """
    matches = _PROPER_NOUN_RE.findall(text)

    results = []
    seen = set()
    for match in matches:
        # Skip if every word in the match is a known non-location word
        words = match.lower().split()
        if all(w in _NON_LOCATION_WORDS for w in words):
            continue
        # Skip single-character or very short
        if len(match) < 3:
            continue
        key = match.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(match)

    return results


class GazetteerMatcher:
    """
    Fast word-boundary regex matcher over the gazetteer.

    Strategy:
      - Sort entries longest-first so "New York" matches before "York",
        "South Korea" before "Korea", "North Carolina" before "Carolina".
      - Compile one big alternation regex for speed.
      - Return all non-overlapping matches with their GazetteerEntry.
      - Also run context-sensitive patterns for ambiguous abbreviations like "US".
      - Extract proper nouns as low-confidence fallback candidates.
    """

    def __init__(self):
        # Sort by length descending so longer matches take priority
        self._entries = dict(
            sorted(_RAW_GAZETTEER.items(), key=lambda kv: len(kv[0]), reverse=True)
        )
        # Build individual compiled patterns (more reliable than one giant alternation
        # for overlapping match control)
        self._patterns: list[tuple[re.Pattern, str, GazetteerEntry]] = []
        for surface, entry in self._entries.items():
            # Word boundary matching, case insensitive
            pattern = re.compile(r'\b' + re.escape(surface) + r'\b', re.IGNORECASE)
            self._patterns.append((pattern, surface, entry))

    def find_all(self, text: str) -> list[tuple[str, GazetteerEntry]]:
        """
        Find all gazetteer matches in text.
        Returns list of (matched_surface_form, GazetteerEntry).
        Deduplicates by canonical name (keeps longest surface match).
        """
        results: dict[str, tuple[str, GazetteerEntry]] = {}
        text_lower = text.lower()

        # Standard gazetteer matching
        for pattern, surface, entry in self._patterns:
            if pattern.search(text_lower):
                canonical = entry.canonical_name
                # Keep the longest surface form match per canonical location
                if canonical not in results or len(surface) > len(results[canonical][0]):
                    results[canonical] = (surface, entry)

        # Context-sensitive patterns (e.g., "US" as country)
        for ctx_pattern, entry in _CONTEXT_PATTERNS:
            if ctx_pattern.search(text):
                canonical = entry.canonical_name
                if canonical not in results:
                    results[canonical] = ("US", entry)

        return list(results.values())

    def find_unknown_proper_nouns(self, text: str) -> list[str]:
        """
        Extract proper nouns that are NOT already matched by the gazetteer.
        These are potential location names we don't have in our dictionary.
        """
        # First get what the gazetteer already matched
        known_matches = self.find_all(text)
        known_surfaces = set()
        for surface, entry in known_matches:
            known_surfaces.add(surface.lower())
            # Also add the canonical name parts
            for part in entry.canonical_name.lower().replace(",", "").split():
                known_surfaces.add(part)

        # Extract proper nouns
        proper_nouns = extract_proper_nouns(text)

        # Filter out ones already found by gazetteer
        unknown = []
        for pn in proper_nouns:
            pn_lower = pn.lower()
            # Skip if any word overlaps with a known match
            if pn_lower in known_surfaces:
                continue
            if any(pn_lower in s or s in pn_lower for s in known_surfaces):
                continue
            # Skip if it's in the raw gazetteer (already matched)
            if pn_lower in _RAW_GAZETTEER:
                continue
            unknown.append(pn)

        return unknown


# Singleton matcher instance
_matcher: Optional[GazetteerMatcher] = None


def get_matcher() -> GazetteerMatcher:
    global _matcher
    if _matcher is None:
        _matcher = GazetteerMatcher()
    return _matcher
