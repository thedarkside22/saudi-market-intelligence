#!/usr/bin/env python3
"""
storing_company_registry.py — FIXED VERSION
Drop old data, re-ingest with LangChain-compatible schema
"""

import json
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()


# ─── Config ─────────────────────────────────────────────
REGISTRY_PATH = r"C:\Users\iizey\Downloads\company_registry.json"

ZILLIZ_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN")
COLLECTION_NAME = "company_registry"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBED_DIM = 1024
BATCH_SIZE = 50

# ─── Embedder ──────────────────────────────────────────
embedder = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    nvidia_api_key=NVIDIA_API_KEY,
    truncate="END",
)


# ─── Search Text Builder ───────────────────────────────

def build_search_text(co: dict) -> str:
    parts = []
    for key in ["symbol", "name_en", "name_ar", "sector_en", "sector_ar"]:
        val = co.get(key, "")
        if val:
            parts.append(val)

    for alias in co.get("search_keys", [])[:30]:
        if alias and alias not in parts:
            parts.append(alias)

    return " | ".join(parts)[:1024]


# ─── Main ──────────────────────────────────────────────

def main():
    # 1. Connect
    print("🔌 Connecting to Zilliz...")
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

    # 2. Drop old collection if it exists
    if utility.has_collection(COLLECTION_NAME):
        print(f"🗑️  Dropping old collection '{COLLECTION_NAME}'...")
        utility.drop_collection(COLLECTION_NAME)
        print("   Done.")

    # 3. Create fresh collection with FIXED SCHEMA
    print("📦 Creating fresh collection...")
    schema = CollectionSchema(fields=[
        FieldSchema(name="primary_key", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        # ⚠️ CRITICAL: LangChain requires a field named "text"
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="symbol", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="name_en", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="name_ar", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="sector_en", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="sector_ar", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="aliases", dtype=DataType.VARCHAR, max_length=2048),
    ])

    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create index on vector field
    collection.create_index(
        field_name="vector",
        index_params={
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
        },
    )
    print("   ✅ Collection + index created")

    # 4. Load registry
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    companies = registry["companies"]
    total = len(companies)
    print(f"   📥 {total} companies to ingest\n")

    # Peek at first company
    first_key = next(iter(companies))
    print(f"   🔎 Sample company '{first_key}':")
    for k, v in companies[first_key].items():
        if k != "search_keys":
            print(f"      {k}: {v}")
        else:
            print(f"      search_keys: [{len(v)} items]")
    print()

    # 5. Ingest in batches
    batch_rows = []
    batch_texts = []
    inserted = 0

    for i, (sym, co) in enumerate(companies.items(), 1):
        search_text = build_search_text(co)

        batch_rows.append({
            "symbol": sym,
            "name_en": co.get("name_en", "")[:512],
            "name_ar": co.get("name_ar", "")[:512],
            "sector_en": co.get("sector_en", "")[:256],
            "sector_ar": co.get("sector_ar", "")[:256],
            "aliases": " | ".join(co.get("search_keys", [])[:50])[:2048],
            "text": search_text,  # ← "text" field required by LangChain
        })
        batch_texts.append(search_text)

        if len(batch_rows) >= BATCH_SIZE or i == total:
            # Embed documents
            vectors = embedder.embed_documents(batch_texts)

            for row, vec in zip(batch_rows, vectors):
                row["vector"] = vec

            collection.insert(batch_rows)
            inserted += len(batch_rows)
            print(f"   💾 [{inserted}/{total}] Batch inserted")
            batch_rows = []
            batch_texts = []

    # 6. Finalize
    collection.flush()
    collection.load()

    print(f"\n{'='*50}")
    print(f"  ✅ Ingested {inserted} companies")
    print(f"  Entities: {collection.num_entities}")
    print(f"{'='*50}")

    # 7. Quick sanity test
    print("\n🔍 Quick test: 'aramco'")
    vec = embedder.embed_query("aramco")
    hits = collection.search(
        data=[vec],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["symbol", "name_en", "name_ar", "sector_en", "text"],
    )
    for hit in hits[0]:
        print(f"   {hit.score:.4f}  |  {hit.entity.get('symbol'):>6}  |  {hit.entity.get('name_en'):<30}  |  {hit.entity.get('sector_en')}")

    print("\n🔍 Quick test: 'الراجحي'")
    vec = embedder.embed_query("الراجحي")
    hits = collection.search(
        data=[vec],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["symbol", "name_en", "name_ar", "sector_en"],
    )
    for hit in hits[0]:
        print(f"   {hit.score:.4f}  |  {hit.entity.get('symbol'):>6}  |  {hit.entity.get('name_en'):<30}  |  {hit.entity.get('name_ar')}")

    print("\n✅ Done! Collection is now LangChain-compatible.")
    print("   You can now use it with LangChain's Milvus wrapper.")


if __name__ == "__main__":
    main()