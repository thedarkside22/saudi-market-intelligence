"""
utils.py — Runtime company search & price fetcher
===================================================
Loads the pre-built company_registry.json (from setup_registry.py)
and provides fast lookups with fuzzy fallback only for truly novel inputs.
"""

import requests
import streamlit as st
import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from thefuzz import process, fuzz

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "https://app.sahmk.sa/api/v1")
REGISTRY_FILE = "company_registry.json"
SCRIPT_DIR = Path(__file__).resolve().parent


# ═════════════════════════════════════════════════════════
#  SECTION 1 — LOAD PRE-BUILT REGISTRY
# ═════════════════════════════════════════════════════════

@st.cache_data
def _load_registry() -> dict:
    """Load the pre-built registry from setup_registry.py output."""
    candidates = [
        SCRIPT_DIR / REGISTRY_FILE,
        SCRIPT_DIR / "data" / REGISTRY_FILE,
        Path.cwd() / REGISTRY_FILE,
        Path.cwd() / "data" / REGISTRY_FILE,
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            stats = registry.get("stats", {})
            print(
                f"✅ Registry loaded: {stats.get('total_companies', '?')} companies, "
                f"{stats.get('total_search_keys', '?')} search keys"
            )
            return registry

    st.error(
        f"⚠️ {REGISTRY_FILE} not found! "
        f"Run `python setup_registry.py` first to generate it."
    )
    return {
        "companies": {},
        "search_index": {},
        "skeleton_index": {},
        "stats": {},
    }


# ═════════════════════════════════════════════════════════
#  SECTION 2 — TEXT UTILITIES (minimal, for runtime only)
# ═════════════════════════════════════════════════════════

def _has_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)


def _normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


_AR2LAT = {
    "ا": "a", "أ": "a", "إ": "a", "آ": "a", "ٱ": "a",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h",
    "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t",
    "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h",
    "ة": "a", "و": "w", "ي": "y", "ى": "a", "ئ": "y",
    "ؤ": "w",
}


def _arabic_to_latin(text: str) -> str:
    text = _normalize_arabic(text)
    text = re.sub(r"\bال", "", text)
    return "".join(_AR2LAT.get(c, c) for c in text).strip()


def _consonant_skeleton(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[aeiou\s\-_.\'\"\(\)]", "", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    return text


# ═════════════════════════════════════════════════════════
#  SECTION 3 — THE SEARCH ENGINE (5-LAYER PIPELINE)
# ═════════════════════════════════════════════════════════

def get_symbol_from_query(query: str) -> tuple[str | None, str | None]:
    """
    Resolve user input → (symbol, display_name) or (None, None).

    Pipeline:
      Layer 1 — Direct 4-digit symbol code
      Layer 2 — Exact key in pre-built search_index
      Layer 3 — Arabic → transliterate → search index + skeleton
      Layer 4 — Consonant skeleton match
      Layer 5 — Fuzzy match (last resort for truly novel typos)
    """
    registry = _load_registry()
    search_index = registry["search_index"]
    skeleton_index = registry["skeleton_index"]
    companies = registry["companies"]

    raw = query.strip()
    if not raw:
        return None, None

    def _get_name(sym: str) -> str | None:
        co = companies.get(sym, {})
        return co.get("name_en") or co.get("name_ar") or None

    # ── Layer 1: Direct 4-digit symbol ──
    code_match = re.search(r"\b(\d{4})\b", raw)
    if code_match:
        code = code_match.group(1)
        if code in companies:
            return code, _get_name(code)
        if raw.strip() == code:
            return code, None  # Trust user if they typed ONLY a code

    # ── Layer 2: Exact lookup in pre-built index ──
    low = raw.lower().strip()

    if low in search_index:
        sym = search_index[low]
        return sym, _get_name(sym)

    # Try Arabic normalization
    if _has_arabic(raw):
        normalized = _normalize_arabic(raw)
        if normalized in search_index:
            sym = search_index[normalized]
            return sym, _get_name(sym)

    # ── Layer 3: Arabic transliteration bridge ──
    if _has_arabic(raw):
        latin = _arabic_to_latin(raw)

        # 3a: Exact match on transliterated form
        if latin in search_index:
            sym = search_index[latin]
            return sym, _get_name(sym)

        # 3b: Skeleton of transliterated form
        sk = _consonant_skeleton(latin)
        if sk and sk in skeleton_index:
            sym = skeleton_index[sk][0]
            return sym, _get_name(sym)

        # 3c: Fuzzy on transliterated form
        if search_index:
            results = process.extract(
                latin, search_index.keys(),
                scorer=fuzz.WRatio, limit=3
            )
            if results and results[0][1] >= 60:
                sym = search_index[results[0][0]]
                return sym, _get_name(sym)

    # ── Layer 4: Consonant skeleton ──
    query_sk = _consonant_skeleton(low)
    if query_sk and len(query_sk) >= 2 and query_sk in skeleton_index:
        candidates = skeleton_index[query_sk]
        if len(candidates) == 1:
            sym = candidates[0]
            return sym, _get_name(sym)
        else:
            # Multiple collisions — pick best via fuzzy
            candidate_names = {}
            for s in candidates:
                name = _get_name(s) or s
                candidate_names[name.lower()] = s
            best = process.extractOne(low, candidate_names.keys(), scorer=fuzz.WRatio)
            if best and best[1] >= 50:
                sym = candidate_names[best[0]]
                return sym, _get_name(sym)
            sym = candidates[0]
            return sym, _get_name(sym)

    # ── Layer 5: Full fuzzy (last resort) ──
    if not search_index:
        return None, None

    best_name, best_score = None, 0
    for scorer in (fuzz.WRatio, fuzz.token_sort_ratio, fuzz.partial_ratio):
        result = process.extractOne(low, search_index.keys(), scorer=scorer)
        if result and result[1] > best_score:
            best_name, best_score = result[0], result[1]

    if best_score >= 60:
        sym = search_index[best_name]
        return sym, _get_name(sym)

    return None, None


# ═════════════════════════════════════════════════════════
#  SECTION 4 — PRICE FETCHER
# ═════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_stock_price(query: str) -> dict | None:
    """Resolve query → symbol → fetch live price from API."""
    symbol, matched_name = get_symbol_from_query(query)
    if not symbol:
        return None

    api_key = os.getenv("MARKET_API_KEY")
    if not api_key:
        st.warning("❌ MARKET_API_KEY not set.")
        return None

    url = f"{API_BASE_URL}/quote/{symbol}/"
    headers = {
        "X-API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "SahmkApp/1.0",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        st.error(f"❌ API error: {e}")
        return None

    if "quotes" in data and isinstance(data["quotes"], list) and data["quotes"]:
        stock = data["quotes"][0]
    elif isinstance(data, list) and data:
        stock = data[0]
    else:
        stock = data

    change = stock.get("change") or 0
    change_pct = stock.get("change_percent") or 0
    price = stock.get("price", "N/A")

    ui_data = {
        "name": stock.get("name_en") or stock.get("name") or matched_name or "Unknown",
        "symbol": stock.get("symbol", symbol),
        "price": f"{price} SAR",
        "delta": f"{change_pct}%",
        "delta_color": "normal" if change >= 0 else "inverse",
    }

    context_text = (
        f"[OFFICIAL MARKET DATA]\n"
        f"Company: {ui_data['name']}\n"
        f"Symbol: {ui_data['symbol']}\n"
        f"Price: {price} SAR\n"
        f"Change: {change} ({change_pct}%)\n"
        f"Volume: {stock.get('volume', 'N/A')}\n"
        f"Last Update: {stock.get('updated_at', 'N/A')}\n"
    )

    return {"ui": ui_data, "context": context_text}


# ═════════════════════════════════════════════════════════
#  SECTION 5 — DEBUG / TEST (run with: streamlit run utils.py)
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    st.title("🔎 Company Search — Registry Mode")

    registry = _load_registry()
    stats = registry.get("stats", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Companies", stats.get("total_companies", 0))
    c2.metric("Search Keys", stats.get("total_search_keys", 0))
    c3.metric("Skeletons", stats.get("total_skeletons", 0))

    missing = stats.get("companies_missing_arabic", 0)
    if missing:
        st.warning(f"⚠️ {missing} companies still missing Arabic names. Re-run setup_registry.py after adding them.")

    st.divider()

    q = st.text_input("Search (name, code, Arabic…)", "Saudi Aramco")
    if st.button("Search"):
        sym, name = get_symbol_from_query(q)
        if sym:
            st.success(f"✅ **{sym}** — {name}")
            result = fetch_stock_price(q)
            if result:
                st.json(result["ui"])
        else:
            st.error("❌ Not found")

    st.divider()

    # Batch test
    st.subheader("🧪 Batch Tests")
    tests = [
        "2222", "aramco", "أرامكو السعودية", "ارامكو",
        "rajhi", "الراجحي", "1120",
        "stc", "sabic", "سابك",
        "moodn", "maaden", "معادن",
        "bahree", "aramko", "jareer",
        "جرير", "جاهز", "موبايلي",
        "استرا", "بوان", "acwa power",
        "nice one", "bupa", "zain",
    ]
    results = []
    for t in tests:
        sym, name = get_symbol_from_query(t)
        results.append({"Query": t, "Symbol": sym or "❌", "Name": name or "—"})
    st.dataframe(results, use_container_width=True)