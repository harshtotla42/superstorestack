import streamlit as st
from elasticsearch import Elasticsearch
from openai import OpenAI
import pandas as pd

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
ES_HOST = st.secrets["es_host"]
ES_USER = st.secrets["es_user"]
ES_PASS = st.secrets["es_pass"]

# Connect to Elasticsearch
es = Elasticsearch(
    ES_HOST,
    http_auth=(ES_USER, ES_PASS),
    verify_certs=False
)

# Configure OpenAI
openai_client = OpenAI(
    api_key=st.secrets["openai_key"]
)

index_source_fields = {
    "ele-faq": [
        "semantic_search"
    ],
    "ele-support": [
        "semantic_search"
    ]
}

def get_elasticsearch_results(query):
    es_query = {
        "retriever": {
            "rrf": {
                "retrievers": [
                    {
                        "standard": {
                            "query": {
                                "semantic": {
                                    "field": "semantic_search",
                                    "query": query
                                }
                            }
                        }
                    },
                    {
                        "standard": {
                            "query": {
                                "semantic": {
                                    "field": "semantic_search",
                                    "query": query
                                }
                            }
                        }
                    }
                ]
            }
        },
        "highlight": {
            "fields": {
                "semantic_search": {
                    "type": "semantic",
                    "number_of_fragments": 2,
                    "order": "score"
                }
            }
        },
        "size": 5
    }
    result = es.search(index="ele-faq,ele-support", body=es_query)
    return result["hits"]["hits"]

def create_openai_prompt(results):
    context = ""
    for hit in results:
        ## For semantic_text matches, we need to extract the text from the highlighted field
        if "highlight" in hit:
            highlighted_texts = []
            for values in hit["highlight"].values():
                highlighted_texts.extend(values)
            context += "\n --- \n".join(highlighted_texts)
        else:
            context_fields = index_source_fields.get(hit["_index"])
            for source_field in context_fields:
                hit_context = hit["_source"][source_field]
                if hit_context:
                    context += f"{source_field}: {hit_context}\n"
    prompt = f"""
Instructions:

- You are an assistant for question-answering tasks.
- Answer questions truthfully and factually using only the context presented.
- If you don't know the answer, just say that you don't know, don't make up an answer.
- You must always cite the document where the answer was extracted using inline academic citation style [], using the position.
- Use markdown format for code examples.
- You are correct, factual, precise, and reliable.

Context:
{context}

"""
    return prompt

def generate_openai_completion(user_prompt, query):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": user_prompt},
            {"role": "user", "content": query},
        ]
    )
    return response.choices[0].message.content

# ---------------------------------------------
# STREAMLIT LAYOUT
# ---------------------------------------------

st.set_page_config(page_title="Elastic Superstore Stack", layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(["Product Search", "Support Tickets", "Knowledge Base", "Chat"])
with tab1:
    col_sidebar1, col_main1 = st.columns([1, 4])
    with col_sidebar1:
        # Sidebar filters
        st.title("Filter Options")
        brands = es.search(index="superstack-products", size=0, aggs={"brands": {"terms": {"field": "brand", "size": 20}}})
        brand_list = [b["key"] for b in brands["aggregations"]["brands"]["buckets"]]
        brand_filter = st.multiselect("Brand", brand_list)
        
        categories = es.search(index="superstack-products", size=0, aggs={"cats": {"terms": {"field": "category", "size": 20}}})
        category_list = [c["key"] for c in categories["aggregations"]["cats"]["buckets"]]
        category_filter = st.multiselect("Category", category_list)

        availability_filter = st.multiselect(
            "Availability", ["Backorder", "Preorder", "In Stock",  "Out of Stock"]
        )
        price_filter = st.slider("Max Price (USD)", 0, 5000, 5000)
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.5)
    with col_main1:
        st.title("Product Search 🛍️")
        query = st.text_input("🔍 Search products:", "")
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
                "size": 15
            }

        # ---------------------------------------------
        # FETCH RESULTS
        # ---------------------------------------------
        def search_products():
            try:
                res = es.search(index="superstack-products", body=build_query())
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
with tab2:
    col_sidebar2, col_main2 = st.columns([1, 4])
    with col_sidebar2:
        # Sidebar filters
        st.title("Filter Options")
        status = es.search(index="ele-support", size=0, aggs={"status": {"terms": {"field": "status", "size": 10}}})
        status_list = [c["key"] for c in status["aggregations"]["status"]["buckets"]]
        status_filter = st.multiselect("Status", status_list)
        city = es.search(index="ele-support", size=0, aggs={"city": {"terms": {"field": "customer_city", "size": 10}}})
        city_list = [c["key"] for c in city["aggregations"]["city"]["buckets"]]
        city_filter = st.multiselect("City", city_list)

    with col_main2:
        st.title("Support Tickets 🎫")
        query = st.text_input("🔍 Search tickets:", "")
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
                                    "fields": ["id^4", "order^4", "subject^3", "content^2", "internal_notes", "tags"]
                                }
                            }
                        ]
                    }
                })
            else:
                must_clauses.append({"match_all": {}})

            # Filters
            if status_filter:
                filter_clauses.append({"terms": {"status": status_filter}})
            if city_filter:
                filter_clauses.append({"terms": {"customer_city": city_filter}})
            return {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses
                    }
                },
                "size": 15
            }

        # ---------------------------------------------
        # FETCH RESULTS
        # ---------------------------------------------
        def search_tickets():
            try:
                res = es.search(index="ele-support", body=build_query())
                hits = [h["_source"] for h in res["hits"]["hits"]]
                return pd.DataFrame(hits)
            except Exception as e:
                st.error(f"Elasticsearch error: {e}")
                return pd.DataFrame()
        results = search_tickets()

        # ---------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------
        if not results.empty:
            for idx, row in results.iterrows():
                with st.container():
                    st.subheader(f"{row.get("id")} - {row.get("subject")}")
                    st.caption(row.get("status"))
                    st.write(row.get("content"))
                    st.caption(f"Notes -- {row.get("internal_notes")}")
                    st.markdown("---")

        else:
            st.info("No tickets found. Try changing your filters or query.")
with tab3:
    col_sidebar3, col_main3 = st.columns([1, 4])
    with col_sidebar3:
        # Sidebar filters
        st.title("Filter Options")
        tags = es.search(index="ele-faq", size=0, aggs={"tags": {"terms": {"field": "tags", "size": 30}}})
        tags_list = [c["key"] for c in tags["aggregations"]["tags"]["buckets"]]
        tags_filter = st.multiselect("Tags", tags_list)

    with col_main3:
        st.title("Knowledge Base 📚")
        query = st.text_input("🔍 Search knowledge base:", "")
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
                                    "fields": ["title^3", "content^2", "tags"]
                                }
                            }
                        ]
                    }
                })
            else:
                must_clauses.append({"match_all": {}})
            # Filters
            if tags_filter:
                filter_clauses.append({"terms": {"tags": tags_filter}})
            return {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses
                    }
                },
                "size": 15
            }

        # ---------------------------------------------
        # FETCH RESULTS
        # ---------------------------------------------
        def search_knowledge_base():
            try:
                res = es.search(index="ele-faq", body=build_query())
                hits = [h["_source"] for h in res["hits"]["hits"]]
                return pd.DataFrame(hits)
            except Exception as e:
                st.error(f"Elasticsearch error: {e}")
                return pd.DataFrame()
        results = search_knowledge_base()

        # ---------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------
        if not results.empty:
            for idx, row in results.iterrows():
                with st.container():
                    st.subheader(f"{row.get("title")}")
                    st.write(row.get("content"))
                    st.markdown("---")
        else:
            st.info("No articles found. Try changing your filters or query.")
with tab4:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask Suppy, our superstore assistant"):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        results = get_elasticsearch_results(query)

        if results is not None:
            prompt = create_openai_prompt(results)
            response = generate_openai_completion(prompt, query).strip()
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response = "Unable to process your question"
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})