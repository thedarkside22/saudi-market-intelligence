"""
Saudi Market Intelligence RAG System
Author: Ziyad Alrasheedi
Date: Feb 2026
Description: Hybrid RAG + Live API system with historical data support

KEY FEATURES:
1. Live price queries → API (current real-time prices)
2. Historical queries → HTML documents (past prices, trends)
3. Analysis queries → PDF documents (annual reports, financials)
4. Smart intent classification for proper routing
"""

import streamlit as st
import os
import requests
import re
from dotenv import load_dotenv

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Saudi Market Intelligence", page_icon="📈", layout="wide")

# --- IMPORTS ---
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA, NVIDIARerank
    from langchain_community.vectorstores import Milvus
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        from langchain_classic.retrievers import ContextualCompressionRetriever
except ImportError as e:
    st.error(f"❌ Import Error: {e}. Please run: pip install langchain-nvidia-ai-endpoints langchain-community pymilvus")
    st.stop()

load_dotenv()


# ══════════════════════════════════════════════════════════
#  SECTION 1: QUERY INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════

def classify_query_intent(query: str) -> str:
    """
    Classify user query into one of four intents:
    - 'live_price': Current/live stock price → Use API
    - 'comparison': Compare historical to current → Use BOTH API + documents
    - 'historical': Historical data/prices → Use documents
    - 'analysis': Company analysis/performance → Use documents
    
    Returns: 'live_price', 'comparison', 'historical', or 'analysis'
    """
    q = query.lower()
    
    # INTENT 1: Comparison queries (check FIRST - most specific)
    comparison_patterns = [
        r"\bfrom\s+.+\s+to\s+(today|now|this\s+day|current)\b",
        r"\bcompare\s+.+\s+to\s+(today|now|current)\b",
        r"\b(one|1)\s+year\s+to\s+(today|now|this\s+day)\b",
        r"\bsince\s+last\s+(year|month).+\s+(today|now)\b",
        r"\bchange\s+from\s+.+\s+to\s+(today|now)\b",
        r"\bcurrent\s+(price\s+)?vs\s+(last|previous)\b",
        r"\bhow\s+much\s+(has|have)\s+.+\s+changed\b", 
        r"\bchange\s+since\b",
        r"\bperformance\s+since\b",
        r"\bمن.+(السنة|الشهر).+اليوم\b",
    ]
    
    for pattern in comparison_patterns:
        if re.search(pattern, q):
            return 'comparison'
    
    # INTENT 2: Historical data indicators (check SECOND)
    historical_patterns = [
        r"\b(last|previous|past)\s+(month|year|quarter|week)\b",
        r"\b(monthly|yearly|annual|quarterly)\s+(report|data|performance)\b",
        r"\b(was|were)\s+the\s+price\b",
        r"\bhistorical\b",
        r"\bhistory\b",
        r"\btrend\b",
        r"\bover\s+time\b",
        r"\bcompare.*(last|previous|past)\b",
        r"\byear.?to.?date\b",
        r"\bytd\b",
        r"\bmtd\b",
        r"\b(الشهر|السنة|الربع)\s+(الماضي|الماضية)\b",
        r"\bالتاريخي\b",
    ]
    
    for pattern in historical_patterns:
        if re.search(pattern, q):
            return 'historical'
    
    # INTENT 3: Live/current price indicators
    live_price_patterns = [
        r"\bcurrent\s+(stock\s+)?price\b",
        r"\bnow\b.*\bprice\b",
        r"\bprice\s+now\b",
        r"\btoday(?:'s)?\s+price\b",
        r"\bprice\s+today\b",
        r"\blive\s+price\b",
        r"\breal.?time\s+price\b",
        r"\bhow\s+much\s+(is|does)\b",
        r"\bwhat(?:'s| is)\s+the\s+(current\s+)?price\b",
        r"\btrading\s+at\b",
        r"\bسعر.*الحالي\b",
        r"\bسعر.*اليوم\b",
        r"\bالسعر\s+الآن\b",
    ]
    
    # Check if it's a pure current price query
    for pattern in live_price_patterns:
        if re.search(pattern, q):
            # Make sure it's not asking for comparison with historical
            comparison_words = ["compare", "vs", "versus", "than", "مقارنة", "change", "changed"]
            if not any(word in q for word in comparison_words):
                return 'live_price'
    
    # INTENT 4: Analysis/company info indicators
    analysis_words = [
        "analysis", "analyze", "performance", "report", "summary",
        "structure", "board", "strategy", "revenue", "profit",
        "dividend", "annual", "financial", "sector", "overview",
        "business", "operations", "management", "forecast",
        "تحليل", "أداء", "تقرير", "ملخص", "هيكل", "استراتيجية"
    ]
    
    if any(word in q for word in analysis_words):
        return 'analysis'
    
    # Default: if mentions "price" but not current → likely historical
    if re.search(r"\bprice\b|\bسعر\b", q):
        return 'historical'
    
    # Everything else → analysis
    return 'analysis'


# ══════════════════════════════════════════════════════════
#  SECTION 2: COMPANY NAME EXTRACTION
# ══════════════════════════════════════════════════════════

_FILLER_WORDS = {
    "summarize", "summary", "tell", "me", "about", "the", "of", "and",
    "what", "is", "are", "how", "for", "its", "it's", "stock", "price",
    "performance", "structure", "company", "share", "shares", "current",
    "now", "today", "this", "that", "please", "can", "you", "give",
    "show", "display", "get", "find", "search", "look", "up",
    "a", "an", "in", "on", "at", "to", "from", "with", "by",
    "board", "revenue", "profit", "dividend", "market", "cap",
    "volume", "change", "daily", "monthly", "yearly", "weekly",
    "do", "does", "did", "will", "would", "could", "should",
    "has", "have", "had", "been", "being", "be",
    "top", "best", "worst", "highest", "lowest", "most", "least",
    "trading", "traded", "trade", "report", "annual", "financial",
    "sector", "index", "overview", "gainer", "loser", "active",
    "ما", "هو", "هي", "عن", "من", "في", "على", "الى", "مع",
    "كم", "كيف", "اين", "متى", "لماذا", "هل", "لي", "لنا",
}


def extract_company_candidates(query: str) -> list[str]:
    """
    Strip filler/intent words to isolate the likely company name.
    """
    words = query.lower().replace("'s", "").replace("'", "").split()
    name_words = [
        w.strip(".,!?;:'\"()[]")
        for w in words
        if w.strip(".,!?;:'\"()[]") not in _FILLER_WORDS
        and len(w.strip(".,!?;:'\"()[]")) > 1
    ]
    if not name_words:
        return []

    candidates = []
    original = " ".join(name_words)
    candidates.append(original)

    normalized = []
    for w in name_words:
        if w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
            normalized.append(w[:-1])
        else:
            normalized.append(w)
    norm_str = " ".join(normalized)
    if norm_str != original:
        candidates.append(norm_str)

    for w in name_words:
        if len(w) > 3:
            candidates.append(w)

    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ══════════════════════════════════════════════════════════
#  SECTION 3: COMPANY RESOLVER (Vector Search)
# ══════════════════════════════════════════════════════════

def resolve_companies(query: str, registry_db) -> list[dict]:
    """
    Search company_registry with multiple query variants.
    """
    if not registry_db:
        return []

    search_queries = [query]
    for candidate in extract_company_candidates(query):
        if candidate.lower().strip() != query.lower().strip():
            search_queries.append(candidate)

    seen = set()
    unique_queries = []
    for sq in search_queries:
        key = sq.lower().strip()
        if key not in seen and len(key) > 1:
            seen.add(key)
            unique_queries.append(sq)

    best_by_symbol = {}

    for sq in unique_queries:
        try:
            results = registry_db.similarity_search_with_score(sq, k=5)
            for doc, score in results:
                if score < 0.25:
                    continue
                symbol = doc.metadata.get("symbol", "")
                if not symbol:
                    continue
                if symbol not in best_by_symbol or score > best_by_symbol[symbol]["score"]:
                    best_by_symbol[symbol] = {
                        "symbol": symbol,
                        "name_en": doc.metadata.get("name_en", ""),
                        "name_ar": doc.metadata.get("name_ar", ""),
                        "sector_en": doc.metadata.get("sector_en", ""),
                        "score": round(score, 4),
                    }
        except Exception:
            pass

    if not best_by_symbol:
        return []

    ranked = sorted(best_by_symbol.values(), key=lambda x: x["score"], reverse=True)
    
    if not ranked: return []
    
    top_score = ranked[0]["score"]
    final = []
    for r in ranked:
        if r["score"] >= top_score - 0.10:
            final.append(r)
        if len(final) >= 2:
            break
    return final


# ══════════════════════════════════════════════════════════
#  SECTION 4: LIVE PRICE FETCHER (API Integration)
# ══════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_live_price(symbol: str) -> dict | None:
    """
    Fetch current stock price from live API.
    """
    api_base = os.getenv("API_BASE_URL", "https://app.sahmk.sa/api/v1")
    api_key = os.getenv("MARKET_API_KEY")
    
    if not api_key:
        st.warning("❌ MARKET_API_KEY not set in .env file")
        return None
    
    url = f"{api_base}/quote/{symbol}/"
    headers = {
        "X-API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "SahmkApp/2.0",
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
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
    
    return {
        "symbol": stock.get("symbol", symbol),
        "name_en": stock.get("name_en") or stock.get("name"),
        "name_ar": stock.get("name"),
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "previous_close": stock.get("previous_close", "N/A"),
        "open": stock.get("open", "N/A"),
        "high": stock.get("high", "N/A"),
        "low": stock.get("low", "N/A"),
        "volume": stock.get("volume", "N/A"),
        "updated_at": stock.get("updated_at", "N/A"),
    }


# ══════════════════════════════════════════════════════════
#  SECTION 5: FOLLOW-UP DETECTION
# ══════════════════════════════════════════════════════════

def is_followup(query: str) -> bool:
    """Detect if query refers to a previously discussed company."""
    signals = [
        "its ", "it's ", "this company", "that company",
        "the same", "their ", "about it", "this stock",
        "that stock", "the stock", "the price", "what about",
    ]
    q = query.lower()
    return any(s in q for s in signals)


# ══════════════════════════════════════════════════════════
#  SECTION 6: DATABASE CONNECTION
# ══════════════════════════════════════════════════════════

@st.cache_resource
def get_db_connection():
    """
    Connects to both Milvus collections and NVIDIA services.
    """
    zilliz_uri = os.getenv("ZILLIZ_CLOUD_URI")
    zilliz_token = os.getenv("ZILLIZ_CLOUD_TOKEN")

    if not zilliz_uri or not zilliz_token:
        return None, None, None, None, "Missing ZILLIZ credentials in .env file"

    try:
        embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", truncate="END")

        # Main reports collection
        vector_db = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": zilliz_uri, "token": zilliz_token, "secure": True},
            collection_name="tadawul_reports",
        )

        # Company registry collection
        registry_db = None
        try:
            registry_db = Milvus(
                embedding_function=embeddings,
                connection_args={"uri": zilliz_uri, "token": zilliz_token, "secure": True},
                collection_name="company_registry",
            )
        except Exception as reg_err:
            st.warning(f"⚠️ Company registry unavailable: {reg_err}")

        # Reranker
        try:
            reranker = NVIDIARerank(
                model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
                top_n=5
            )
        except Exception:
            reranker = None

        # LLM
        llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0.1)

        return vector_db, registry_db, reranker, llm, None

    except Exception as e:
        return None, None, None, None, str(e)


# ══════════════════════════════════════════════════════════
#  SECTION 7: SESSION STATE
# ══════════════════════════════════════════════════════════

for key in ["vector_db", "registry_db", "reranker", "llm", "messages", "last_resolved"]:
    if key not in st.session_state:
        if key == "messages":
            st.session_state[key] = []
        elif key == "last_resolved":
            st.session_state[key] = []
        else:
            st.session_state[key] = None


# ══════════════════════════════════════════════════════════
#  SECTION 8: AUTO CONNECT
# ══════════════════════════════════════════════════════════

if st.session_state.vector_db is None:
    with st.spinner("🔌 Connecting to Tadawul Database & Live API..."):
        db, registry, ranker, llm_model, err = get_db_connection()
        if db:
            st.session_state.vector_db = db
            st.session_state.registry_db = registry
            st.session_state.reranker = ranker
            st.session_state.llm = llm_model
        else:
            st.error(f"❌ Connection Failed: {err}")


# ══════════════════════════════════════════════════════════
#  SECTION 9: SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Control Panel")

    if st.session_state.vector_db is None:
        st.warning("⚠️ Database Disconnected")
        if st.button("🔄 Retry Connection"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.success("✅ Reports DB Online")
        if st.session_state.registry_db:
            st.success("✅ Company Registry Online")
            st.success("✅ Live Price API Ready")
        else:
            st.warning("⚠️ Registry unavailable — limited mode")
        if st.button("🔄 Refresh Connection"):
            st.session_state.vector_db = None
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

    st.divider()
    show_debug = st.toggle("🔍 Show Debug Info", value=False)

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_resolved = []
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════
#  SECTION 10: MAIN APP
# ══════════════════════════════════════════════════════════

st.title("📈 Saudi Market Intelligence Unit")
st.caption("Powered by NVIDIA AI, Tadawul Reports & Live Price API")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt_input = st.chat_input("Ask about Al Rajhi, Aramco, or Market Performance...")

if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    if st.session_state.vector_db is None:
        with st.chat_message("assistant"):
            st.error("🚨 Database not connected. Check sidebar logs.")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🧠 *Analyzing...*")

            try:
                vector_db = st.session_state.vector_db
                registry_db = st.session_state.registry_db
                reranker = st.session_state.reranker

                # STEP 1: Classify intent
                intent = classify_query_intent(prompt_input)
                
                # STEP 2: Follow-up handling
                working_query = prompt_input
                followup = is_followup(prompt_input)
                
                if followup and st.session_state.last_resolved:
                    prior_names = " ".join(
                        co.get("name_en", "") for co in st.session_state.last_resolved
                    )
                    working_query = f"{prompt_input} {prior_names}"

                # STEP 3: Resolve company
                resolved = resolve_companies(working_query, registry_db)
                if not resolved and followup:
                    resolved = st.session_state.get("last_resolved", [])
                
                if resolved:
                    st.session_state.last_resolved = resolved

                # STEP 4: LIVE PRICE FETCH (SAFETY NET)
                live_context_str = ""
                live_data = None
                
                if resolved:
                    co = resolved[0]
                    live_data = fetch_live_price(co["symbol"])
                    
                    if live_data:
                        live_context_str = f"""
*** REAL-TIME MARKET DATA (Use this as absolute truth for 'Current Price'): ***
- Stock: {live_data['name_en']} ({live_data['symbol']})
- Current Price: {live_data['price']} SAR
- Change: {live_data['change']} ({live_data['change_pct']}%)
- Last Update: {live_data['updated_at']}
- High/Low: {live_data['high']} / {live_data['low']}
*** END REAL-TIME DATA ***
"""

                # STEP 5: EXECUTION PATHS

                # PATH A: Pure Live Price Request
                if intent == 'live_price' and resolved and live_data:
                     name = live_data["name_en"]
                     symbol = live_data["symbol"]
                     price = live_data["price"]
                     change = live_data["change"]
                     change_pct = live_data["change_pct"]
                     
                     if change > 0: trend = f"📈 +{change_pct}%"
                     elif change < 0: trend = f"📉 {change_pct}%"
                     else: trend = "➡️ 0.00%"

                     response = f"""The current stock price of **{name}** ({symbol}) is **SAR {price}**.

{trend} (Change: {change:+.2f} SAR)

**Trading Summary:**
- High: SAR {live_data.get('high', 'N/A')}
- Low: SAR {live_data.get('low', 'N/A')}
- Open: SAR {live_data.get('open', 'N/A')}
- Volume: {live_data.get('volume', 'N/A'):,} shares

*Last updated: {live_data.get('updated_at', 'N/A')}*
*Data source: Live Tadawul API*"""
                     
                     message_placeholder.markdown(response)
                     st.session_state.messages.append({"role": "assistant", "content": response})

                # PATH B: RAG (Comparison / Historical / Analysis)
                else:
                    # 1. Retrieve Documents
                    enriched_query = prompt_input
                    if resolved:
                        extras = [co["name_en"] for co in resolved] + [co["symbol"] for co in resolved]
                        enriched_query = f"{prompt_input} {' '.join(extras)}"

                    # Weighting: If analysis, get more PDFs. If history, get more HTML.
                    if intent == 'analysis':
                        docs_pdf = vector_db.similarity_search(enriched_query, k=10, expr="type == 'pdf_report'")
                        docs_html = vector_db.similarity_search(enriched_query, k=5, expr="type == 'html_report'")
                    else:
                        docs_html = vector_db.similarity_search(enriched_query, k=15, expr="type == 'html_report'")
                        docs_pdf = vector_db.similarity_search(enriched_query, k=5, expr="type == 'pdf_report'")
                    
                    combined_docs = docs_html + docs_pdf
                    unique_docs = list({d.page_content: d for d in combined_docs}.values())

                    if reranker:
                        context_docs = reranker.compress_documents(unique_docs, enriched_query)
                    else:
                        context_docs = unique_docs[:7]

                    doc_context_str = "\n\n".join([d.page_content for d in context_docs]) if context_docs else "No relevant documents found."

                    # 2. DYNAMIC INSTRUCTION BLOCK (The Critical Fix)
                    if intent == 'analysis':
                        instruction_block = """
                        **MODE: ANNUAL REPORT ANALYSIS**
                        1. **Primary Source:** Your answer MUST be based on the 'Document Context' (PDFs/Reports) below.
                        2. **Live Data Usage:** Use the 'Real-Time Market Data' ONLY to add context (e.g., "Company X, currently trading at 26.0 SAR...").
                        3. **Do NOT** let the live price override the strategic info in the reports.
                        4. If the user asks about Strategy, Risks, Board, or Financials, extract details from the documents.
                        """
                    elif intent == 'historical':
                         instruction_block = """
                        **MODE: HISTORICAL DATA**
                        1. **Primary Source:** Use 'Document Context' (HTML/PDFs) to find past prices and trends.
                        2. **Live Data Usage:** Use 'Real-Time Market Data' ONLY for comparison (e.g. "It was 20 SAR last year, compared to 26 SAR today").
                        3. **Accuracy:** If the user asks for a specific past date (e.g. Dec 2025) and it is in the docs, quote it exactly.
                        """
                    else:
                        instruction_block = """
                        **MODE: HYBRID COMPARISON**
                        1. Use 'Real-Time Market Data' for TODAY'S price.
                        2. Use 'Document Context' for PAST prices or details.
                        3. Combine them to answer the question (e.g. "Up 10% since last year").
                        """

                    # 3. Prompt Template
                    template = """You are a senior financial analyst for Tadawul.

{live_context}

{instruction_block}

Document Context (Annual Reports & Data):
{doc_context}

Chat History:
{history}

User Question: {question}
"""
                    prompt_template = ChatPromptTemplate.from_template(template)
                    
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
                    
                    response = (prompt_template | st.session_state.llm | StrOutputParser()).invoke({
                        "live_context": live_context_str,
                        "instruction_block": instruction_block,
                        "doc_context": doc_context_str,
                        "history": history_text,
                        "question": prompt_input
                    })
                    
                    if resolved:
                        header = f"🏢 **{resolved[0]['symbol']}** {resolved[0]['name_en']}"
                        message_placeholder.markdown(f"{header}\n\n---\n\n{response}")
                    else:
                        message_placeholder.markdown(response)
                        
                    st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                message_placeholder.error(f"❌ Error: {e}")
                if show_debug:
                    st.exception(e)