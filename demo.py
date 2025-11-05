import streamlit as st
from elasticsearch import Elasticsearch
import pandas as pd

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
ES_HOST = "https://eden-search-smokestack-2025-11-03-ab4014.es.us-east4.gcp.elastic-cloud.com/"
ES_USER = "harsh"
ES_PASS = "123456"
INDEX_NAME = "superstack-products"

# Connect to Elasticsearch
es = Elasticsearch(
    ES_HOST,
    http_auth=(ES_USER, ES_PASS),
    verify_certs=False
)

# ---------------------------------------------
# STREAMLIT LAYOUT
# ---------------------------------------------
st.set_page_config(page_title="Elastic Product Search 🛍️", layout="wide")
st.title("Elastic Product Search 🛍️")
query = st.text_input("🔍 Search products (semantic & keyword):", "")

# Sidebar filters
st.sidebar.header("Filter Options")

brands = es.search(index=INDEX_NAME, size=0, aggs={"brands": {"terms": {"field": "brand", "size": 20}}})
brand_list = [b["key"] for b in brands["aggregations"]["brands"]["buckets"]]
brand_filter = st.sidebar.multiselect("Brand", brand_list)

categories = es.search(index=INDEX_NAME, size=0, aggs={"cats": {"terms": {"field": "category", "size": 20}}})
category_list = [c["key"] for c in categories["aggregations"]["cats"]["buckets"]]
category_filter = st.sidebar.multiselect("Category", category_list)

availability_filter = st.sidebar.multiselect(
    "Availability", ["Backorder", "Preorder", "In Stock",  "Out of Stock"]
)
price_filter = st.sidebar.slider("Max Price (USD)", 0, 5000, 5000)
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5)

# ---------------------------------------------
# BUILD ELASTIC QUERY
# ---------------------------------------------
def build_query():
    must_clauses = []
    filter_clauses = []

    # Semantic + keyword search
    if query:
        must_clauses.append({
            "bool": {
                "should": [
                    {
                        "semantic": {
                                "field": "semantic_search",
                                "query": query
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["name^4", "title^3", "description^2", "brand", "category", "sub_category"]
                        }
                    }
                ]
            }
        })
    else:
        must_clauses.append({"match_all": {}})

    # Filters
    if brand_filter:
        filter_clauses.append({"terms": {"brand": brand_filter}})
    if category_filter:
        filter_clauses.append({"terms": {"category": category_filter}})
    if availability_filter:
        filter_clauses.append({"terms": {"availability_status": [("_").join(x.split()).lower() for x in availability_filter]}})
    if price_filter < 5000:
        filter_clauses.append({"range": {"price_usd": {"lte": price_filter}}})
    if min_rating > 0:
        filter_clauses.append({"range": {"rating": {"gte": min_rating}}})

    return {
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        },
        "size": 30
    }

# ---------------------------------------------
# FETCH RESULTS
# ---------------------------------------------
def search_products():
    try:
        res = es.search(index=INDEX_NAME, body=build_query())
        hits = [h["_source"] for h in res["hits"]["hits"]]
        return pd.DataFrame(hits)
    except Exception as e:
        st.error(f"Elasticsearch error: {e}")
        return pd.DataFrame()

results = search_products()

# ---------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------
if not results.empty:
    cols = st.columns(3)
    for idx, row in results.iterrows():
        with cols[idx % 3]:
            st.image(row.get("image", ""), use_container_width=True)
            st.subheader(row.get("name", row.get("title", "Unnamed Product")))
            st.caption(
                f"💵 ${row.get('price_usd', 0):,.2f} | ⭐ {row.get('rating', 0)} ({row.get('num_reviews', 0)} reviews)"
            )
            st.write(row.get("description", "")[:150] + "...")
            if row.get("technical_specs"):
                with st.expander("🔧 Technical Specs"):
                    for k, v in row["technical_specs"].items():
                        st.text(f"{k.capitalize()}: {v}")
else:
    st.info("No products found. Try changing your filters or query.")
