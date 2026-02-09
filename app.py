import streamlit as st
import os
from dotenv import load_dotenv

# --- PAGE CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Saudi Market Intelligence", page_icon="📈", layout="wide")

# --- IMPORTS ---
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA, NVIDIARerank
    from langchain_community.vectorstores import Milvus
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    # Handle LangChain version differences
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        from langchain_classic.retrievers import ContextualCompressionRetriever
except ImportError as e:
    st.error(f"❌ Import Error: {e}. Please run: pip install langchain-nvidia-ai-endpoints langchain-community pymilvus")
    st.stop()

# Load Environment Variables
load_dotenv()

# --- INITIALIZATION FUNCTION ---
@st.cache_resource
def get_db_connection():
    """
    Establishes the connection to Milvus and NVIDIA.
    Cached so it doesn't reconnect on every interaction.
    """
    zilliz_uri = os.getenv("ZILLIZ_CLOUD_URI")
    zilliz_token = os.getenv("ZILLIZ_CLOUD_TOKEN")
    
    if not zilliz_uri or not zilliz_token:
        return None, None, None, "Missing API Keys in .env file"

    try:
        # 1. Embeddings & DB
        embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", truncate="END")
        
        vector_db = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": zilliz_uri, "token": zilliz_token, "secure": True},
            collection_name="tadawul_reports",
        )

        # 2. Reranker
        try:
            reranker = NVIDIARerank(
                model="nvidia/llama-3.2-nv-rerankqa-1b-v2", 
                top_n=5
            )
        except Exception:
            reranker = None # Fallback if reranker fails

        # 3. LLM
        llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", temperature=0.1)

        return vector_db, reranker, llm, None

    except Exception as e:
        return None, None, None, str(e)

# --- ENSURE SESSION STATE EXISTS ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "reranker" not in st.session_state:
    st.session_state.reranker = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 🔌 AUTOMATIC CONNECTION BLOCK ---
# This runs immediately when the app loads
if st.session_state.vector_db is None:
    with st.spinner("🔌 Automatically connecting to Tadawul Database..."):
        db, ranker, llm_model, err = get_db_connection()
        if db:
            st.session_state.vector_db = db
            st.session_state.reranker = ranker
            st.session_state.llm = llm_model
        else:
            st.error(f"❌ Auto-Connection Failed: {err}")

# --- SIDEBAR (STATUS ONLY) ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Check connection status
    if st.session_state.vector_db is None:
        st.warning("⚠️ Database Disconnected")
        # Keep a retry button just in case auto-connect failed
        if st.button("🔄 Retry Connection"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.success("✅ System Online")
        if st.button("🔄 Refresh Connection"):
            st.session_state.vector_db = None
            st.rerun()
            
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN APP LAYOUT ---
st.title("📈 Saudi Market Intelligence Unit")
st.caption("Powered by NVIDIA AI & Tadawul Data")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Chat Input (ALWAYS VISIBLE)
prompt_input = st.chat_input("Ask about Al Rajhi, Aramco, or Market Performance...")

# 3. Handle User Input
if prompt_input:
    # A. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # B. Check if DB is ready
    if st.session_state.vector_db is None:
        with st.chat_message("assistant"):
            st.error("🚨 Database is not connected. Check logs in sidebar.")
    else:
        # C. Generate Answer
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🧠 *Thinking...*")

            try:
                # --- RETRIEVAL LOGIC ---
                vector_db = st.session_state.vector_db
                reranker = st.session_state.reranker
                
                # 1. Hybrid Search
                docs_pdf = vector_db.similarity_search(prompt_input, k=10, expr="type == 'pdf_report'")
                docs_html = vector_db.similarity_search(prompt_input, k=30, expr="type == 'html_report'")
                
                # 2. Combine & Rerank
                combined_docs = docs_pdf + docs_html
                unique_docs = {d.page_content: d for d in combined_docs}.values()
                final_pool = list(unique_docs)
                
                if reranker:
                    context_docs = reranker.compress_documents(final_pool, prompt_input)
                else:
                    context_docs = final_pool[:5]
                
                # 3. LLM Generation
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
                
                template = """You are a senior financial analyst for the Saudi Stock Market (Tadawul).
                Guidelines:
                1. **Source Strategy:** Use HTML data for Prices/Volumes and PDF data for Strategy.
                2. **Language:** ALWAYS answer in the same language as the user.
                3. **Context:** Accurate numbers are critical. Quote prices exactly.

                Conversation History:
                {history}

                Context:
                {context}

                User Question: {question}
                """
                
                prompt_template = ChatPromptTemplate.from_template(template)
                
                chain = (
                    {
                        "context": lambda x: context_docs,
                        "question": RunnablePassthrough(),
                        "history": lambda x: history_text
                    }
                    | prompt_template
                    | st.session_state.llm
                    | StrOutputParser()
                )
                
                response = chain.invoke({"question": prompt_input})
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                message_placeholder.error(f"Error generating response: {e}")