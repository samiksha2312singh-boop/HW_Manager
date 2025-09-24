import streamlit as st
import os
import glob
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup

# Chunking strategy used in this app
#
# Method: **Semantic paragraph-based chunking with a sentence fallback**
#
# How it works:
# 1) Split the document on paragraph boundaries (`\n\n`). We then build chunks
#    by appending whole paragraphs until a size cap is reached (`max_chunk_size`).
# 2) If we end up with a single, oversized paragraph (e.g., a long blob of text),
#    we fall back to splitting by sentences ('. ') to avoid cutting in the middle
#    of words while still respecting the size cap.
#
# Why this method (vs. naive fixed-length splitting):
# - **Preserves meaning and context:** Keeping paragraphs intact keeps related
#   sentences together (titles + bullets + explanations), which improves RAG
#   retrieval quality and reduces “orphaned” facts.
# - **Natural boundaries:** Paragraphs are author-chosen semantic units; using
#   them minimizes cutting mid-thought, which can confuse embeddings and the LLM.
# - **Better answer grounding:** Larger, coherent chunks give the model enough
#   local context to cite correctly without drifting.
# - **Robustness:** The sentence fallback handles pages that are a single giant
#   paragraph (or poorly formatted HTML) without exceeding size limits.
# ──────────────────────────────────────────────────────────────────────────────
# SQLite fix for ChromaDB (must run before chromadb import)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import pysqlite3  # type: ignore
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────────────────────────────────────
CHROMADB_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Providers
# ──────────────────────────────────────────────────────────────────────────────
from openai import OpenAI

# Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# HTML → text + chunking
# ──────────────────────────────────────────────────────────────────────────────
def extract_text_from_html(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return ' '.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        st.error(f"Error extracting text from HTML: {str(e)}")
        return ""

def chunk_document_semantic(text: str, max_chunk_size: int = 3000) -> List[str]:
    paragraphs = text.split('\n\n')
    chunks, current = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) > max_chunk_size and current:
            chunks.append(current.strip())
            current = p
        else:
            current = (current + "\n\n" + p) if current else p
    if current:
        chunks.append(current.strip())

    if len(chunks) == 1 and len(chunks[0]) > max_chunk_size:
        sentences = chunks[0].split('. ')
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) > max_chunk_size and current:
                chunks.append(current.strip())
                current = s
            else:
                current = (current + ". " + s) if current else s
        if current:
            chunks.append(current.strip())
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB (persistent)
# ──────────────────────────────────────────────────────────────────────────────
INDEX_DIR = Path("./vector_db")
INDEX_DIR.mkdir(exist_ok=True)

def _discover_html_files() -> List[str]:
    patterns = ["*.html", "*.htm", "su_org/*.html", "su_org/*.htm", "**/*.html", "**/*.htm"]
    hits = set()
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            if os.path.isfile(f):
                hits.add(os.path.normpath(f))
    return sorted(hits)

def _init_chroma_persistent():
    if not CHROMADB_AVAILABLE:
        st.error("ChromaDB not available. Install: pip install chromadb pysqlite3-binary")
        return None, None
    try:
        client = chromadb.PersistentClient(
            path=str(INDEX_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(
            name="iSchoolCollection",
            metadata={"description": "iSchool student organizations information"}
        )
        return client, collection
    except Exception as e:
        st.warning(f"Persistent Chroma failed ({e}). Falling back to in-memory.")
        try:
            client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
            collection = client.get_or_create_collection(
                name="iSchoolCollection",
                metadata={"description": "iSchool student organizations information"}
            )
            return client, collection
        except Exception as e2:
            st.error(f"Chroma init failed: {e2}")
            return None, None

def _has_index(collection) -> bool:
    try:
        return collection.count() > 0
    except Exception:
        return False

def create_or_load_vector_database(force_rebuild: bool = False):
    client, collection = _init_chroma_persistent()
    if not client or not collection:
        return None

    if _has_index(collection) and not force_rebuild:
        st.success("Loaded existing vector database.")
        return {"client": client, "collection": collection}

    if force_rebuild and _has_index(collection):
        st.info("Clearing existing collection for rebuild…")
        try:
            client.delete_collection("iSchoolCollection")
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name="iSchoolCollection",
            metadata={"description": "iSchool student organizations information"}
        )

    st.info("Building vector database from HTML files…")

    openai_api_key = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY not found in secrets or env.")
        return None
    openai_client = OpenAI(api_key=openai_api_key)

    html_files = _discover_html_files()
    if not html_files:
        st.error("No HTML files found. Put files in project root or ./su_org/")
        return None
    st.write(f"Found {len(html_files)} HTML files")

    documents, metadatas, ids = [], [], []
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for idx, html_file in enumerate(html_files):
        status_text.text(f"Processing {html_file}…")
        try:
            with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
            text = extract_text_from_html(html_content)
            if text and len(text.strip()) > 100:
                chunks = chunk_document_semantic(text, max_chunk_size=3000)
                for cidx, chunk in enumerate(chunks[:2]):  # ≤2 chunks per doc
                    if chunk.strip():
                        documents.append(chunk)
                        metadatas.append({
                            "filename": os.path.basename(html_file),
                            "filepath": os.path.normpath(html_file),
                            "chunk_id": cidx + 1,
                            "total_chunks": min(len(chunks), 2),
                            "document_type": "html",
                        })
                        ids.append(f"{os.path.basename(html_file)}_chunk_{cidx + 1}")
        except Exception as e:
            st.warning(f"Error processing {html_file}: {e}")
        progress_bar.progress((idx + 1) / len(html_files))

    if not documents:
        st.error("No valid text extracted from HTML.")
        return None

    status_text.text("Creating embeddings and storing in vector DB…")
    try:
        batch_size = 16
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start:start+batch_size]
            resp = openai_client.embeddings.create(
                input=[d[:8000] for d in batch_docs],
                model="text-embedding-3-small"
            )
            embs = [d.embedding for d in resp.data]
            collection.add(
                documents=batch_docs,
                metadatas=metadatas[start:start+batch_size],
                embeddings=embs,
                ids=ids[start:start+batch_size]
            )
            st.write(f"Indexed {min(start+batch_size, len(documents))}/{len(documents)} chunks")

        st.success(f"Vector DB ready with {collection.count()} chunks.")
        progress_bar.progress(1.0)
        status_text.text("Vector DB build complete.")
        return {"client": client, "collection": collection}
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

def search_vector_database(vector_db, query: str, n_results: int = 3):
    collection = vector_db['collection']
    openai_api_key = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    try:
        resp = client.embeddings.create(input=query, model="text-embedding-3-small")
        qemb = resp.data[0].embedding
        return collection.query(query_embeddings=[qemb], n_results=n_results)
    except Exception as e:
        st.error(f"Error searching vector database: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────────────────────────────────────
# Map deprecated Groq IDs → current ones
GROQ_MODEL_ALIASES = {
    "llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-70b": "llama-3.3-70b-versatile",
}

def _resolve_groq_model(name: str) -> str:
    base = name.replace("groq/", "")
    return GROQ_MODEL_ALIASES.get(base, base)

def get_llm_response(messages: List[Dict], model_name: str, api_clients: Dict):
    """
    Returns:
      - OpenAI: streaming iterator OR error string
      - Groq / Gemini: string OR error string
    """
    try:
        # OpenAI GPT family
        if model_name.startswith("gpt"):
            client = api_clients.get("openai")
            if not client:
                return "OpenAI client not available. Set OPENAI_API_KEY."

            params = {"model": model_name, "messages": messages, "stream": True}
            if model_name.startswith("gpt-5"):
                params["max_completion_tokens"] = 1000   # GPT-5 style
                # omit temperature (some previews lock it)
            else:
                params["max_tokens"] = 1000
                params["temperature"] = 0.7

            try:
                return client.chat.completions.create(**params)
            except Exception as e:
                err = str(e)
                # Drop temperature if rejected
                if "temperature" in err and "unsupported" in err.lower():
                    params.pop("temperature", None)
                # Flip token kwarg if needed
                if "max_tokens" in err and "unsupported" in err.lower():
                    params.pop("max_tokens", None)
                    params["max_completion_tokens"] = 1000
                elif "max_completion_tokens" in err and "unsupported" in err.lower():
                    params.pop("max_completion_tokens", None)
                    params["max_tokens"] = 1000
                try:
                    return client.chat.completions.create(**params)
                except Exception as e2:
                    return f"Error with {model_name}: {e2}"

        # Groq family (non-stream for simplicity)
        elif (model_name.startswith("llama") or model_name.startswith("mixtral")
              or model_name.startswith("gemma") or model_name.startswith("groq/")
              or model_name.startswith("openai/gpt-oss")):
            client = api_clients.get("groq")
            if not client:
                return "Groq client not available. Set GROQ_API_KEY."

            resolved = _resolve_groq_model(model_name)
            try:
                resp = client.chat.completions.create(
                    model=resolved,
                    messages=messages,
                    temperature=0.7
                )
                if hasattr(resp, "choices") and resp.choices:
                    content = getattr(resp.choices[0].message, "content", None)
                    if content:
                        return content
                return str(resp)
            except Exception as e:
                # Retry without temperature, and also try alias resolution once more
                try:
                    resp = client.chat.completions.create(
                        model=_resolve_groq_model(resolved),
                        messages=messages
                    )
                    if hasattr(resp, "choices") and resp.choices:
                        content = getattr(resp.choices[0].message, "content", None)
                        if content:
                            return content
                    return str(resp)
                except Exception as e2:
                    return f"Error with {model_name}: {e2}"

        # Gemini
        elif model_name.startswith("gemini"):
            client = api_clients.get("gemini")
            if not client:
                return "Gemini client not available. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            prompt = "\n\n".join(
                f"{'System' if m['role']=='system' else 'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in messages
            )
            try:
                response = client.generate_content(prompt)
                return getattr(response, "text", str(response))
            except Exception as e:
                return f"Error with {model_name}: {e}"

        return f"Unsupported model/provider: {model_name}"
    except Exception as e:
        return f"Unexpected error ({model_name}): {e}"

def display_streaming_response(stream, placeholder):
    full = ""
    for chunk in stream:
        if isinstance(chunk, str):
            full += chunk
            placeholder.markdown(full + "▌")
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if content is not None:
            full += content
            placeholder.markdown(full + "▌")
    placeholder.markdown(full)
    return full

def create_rag_prompt(user_question: str, context_docs: List[str], source_files: List[str], conversation_history: List[Dict]) -> str:
    convo = ""
    if conversation_history:
        convo = "Previous conversation:\n"
        for msg in conversation_history[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            convo += f"{role}: {msg['content'][:200]}...\n"
        convo += "\n"

    context_text = "\n\n".join([f"Source {i+1}:\n{doc[:1500]}" for i, doc in enumerate(context_docs)])
    source_list = ", ".join(source_files)

    prompt = f"""You are a helpful assistant for the iSchool, specializing in information about student organizations and academic programs.

{convo}
RELEVANT INFORMATION FROM iSCHOOL DOCUMENTS:
Sources: {source_list}

{context_text}

USER QUESTION: {user_question}

Please answer based on the iSchool information above. If the answer comes from the provided context, say "Based on the iSchool information...". If you add general advice, label it clearly. Be specific and helpful.
Answer:"""
    return prompt

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit app
# ──────────────────────────────────────────────────────────────────────────────
def run():
    st.set_page_config(page_title="HW4 - iSchool Chatbot", page_icon="🎓", layout="wide")
    st.title("HW4 - iSchool Student Organizations Chatbot")
    st.write("Ask questions about iSchool student organizations, programs, and opportunities using RAG-powered AI")

    # API clients
    api_clients: Dict[str, object] = {}

    # OpenAI
    openai_key = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    if openai_key:
        api_clients["openai"] = OpenAI(api_key=openai_key)

    # Groq
    if GROQ_AVAILABLE:
        try:
            groq_key = (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("GROQ_API_KEY")
            if groq_key:
                api_clients["groq"] = Groq(api_key=groq_key)
        except Exception as e:
            st.sidebar.error(f"Groq setup error: {e}")

    # Gemini
    if GEMINI_AVAILABLE:
        try:
            gemini_key = (
                (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None) or
                os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            )
            if gemini_key:
                genai.configure(api_key=gemini_key)
                api_clients["gemini_available"] = True
        except Exception as e:
            st.sidebar.error(f"Gemini setup error: {e}")

    # Sidebar — availability + models
    st.sidebar.header("AI Model Selection")
    st.sidebar.write("**Available APIs:**")
    st.sidebar.write(f"- OpenAI: {'✓' if 'openai' in api_clients else '✗'}")
    st.sidebar.write(f"- Groq: {'✓' if 'groq' in api_clients else '✗'}")
    st.sidebar.write(f"- Gemini: {'✓' if 'gemini_available' in api_clients else '✗'}")

    # Groq current production model IDs (from docs)
    GROQ_DEFAULTS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        # You can also expose Groq’s OpenAI-licensed OSS:
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]

    available_models = []
    if "openai" in api_clients:
        available_models += ["gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    if "groq" in api_clients:
        available_models += GROQ_DEFAULTS
    if "gemini_available" in api_clients:
        available_models += ["gemini-1.5-pro", "gemini-1.5-pro-002", "gemini-1.5-flash"]

    if not available_models:
        st.error("No LLM API keys found. Add OPENAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY in secrets.toml.")
        return

    selected_model = st.sidebar.selectbox("Choose LLM Model:", available_models)
    model_info = {
        "gpt-5": "OpenAI next-gen (if enabled on your key)",
        "gpt-4o": "OpenAI flagship",
        "gpt-4o-mini": "OpenAI efficient",
        "gpt-3.5-turbo": "OpenAI economical",
        "llama-3.3-70b-versatile": "Groq • Llama 3.3 70B (prod)",
        "llama-3.1-8b-instant": "Groq • Llama 3.1 8B instant (prod)",
        "openai/gpt-oss-120b": "Groq • OpenAI GPT-OSS 120B",
        "openai/gpt-oss-20b": "Groq • OpenAI GPT-OSS 20B",
        "gemini-1.5-pro": "Gemini flagship",
        "gemini-1.5-pro-002": "Gemini flagship (newer tag)",
        "gemini-1.5-flash": "Gemini efficient",
    }
    st.sidebar.caption(model_info.get(selected_model, ""))

    # Custom model override
    st.sidebar.divider()
    st.sidebar.subheader("Custom Model (optional)")
    custom_model = st.sidebar.text_input(
        "Enter exact model name (e.g., gpt-5, llama-3.3-70b-versatile, openai/gpt-oss-120b, gemini-1.5-pro-002)",
        value="",
        placeholder="Type here to override the dropdown…"
    )
    if custom_model.strip():
        selected_model = custom_model.strip()
        st.sidebar.caption(f"Using custom model: {selected_model}")

    # Guardrails
    if selected_model.startswith("gpt") and "openai" not in api_clients:
        st.sidebar.warning("OpenAI not configured; GPT models will error.")
    if (selected_model.startswith("llama") or selected_model.startswith("mixtral")
        or selected_model.startswith("gemma") or selected_model.startswith("groq/")
        or selected_model.startswith("openai/gpt-oss")) and "groq" not in api_clients:
        st.sidebar.warning("Groq not configured; Groq models will error.")
    if selected_model.startswith("gemini") and "gemini_available" not in api_clients:
        st.sidebar.warning("Gemini not configured; Gemini models will error.")

    # Memory controls
    st.sidebar.header("Memory Configuration")
    memory_options = {
        "Conversation Buffer (5 Q&A pairs)": 10,
        "Conversation Buffer (3 Q&A pairs)": 6,
        "Conversation Buffer (7 Q&A pairs)": 14,
        "No Memory": 0
    }
    selected_memory = st.sidebar.selectbox("Choose Memory Strategy:", list(memory_options.keys()), index=0)
    max_memory_messages = memory_options[selected_memory]
    st.sidebar.caption(
        f"Storing last {max_memory_messages} messages" if max_memory_messages > 0 else "No conversation history stored"
    )

    st.sidebar.header("Knowledge Base")
    rebuild = st.sidebar.button("Rebuild Vector DB (from HTML)")

    # Init/load vector DB
    if "ischool_vectorDB" not in st.session_state or rebuild:
        with st.spinner("Loading/creating vector database…"):
            vector_db = create_or_load_vector_database(force_rebuild=rebuild)
        if vector_db:
            st.session_state.ischool_vectorDB = vector_db
            st.success("iSchool knowledge base ready.")
        else:
            st.error("Vector DB init failed.")
            return
    else:
        vector_db = st.session_state.ischool_vectorDB

    # Metrics
    try:
        count = vector_db["collection"].count()
        st.sidebar.metric("Knowledge Base Size", f"{count} chunks")
    except Exception as e:
        st.sidebar.error(f"Database error: {e}")

    # Conversation state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Chat UI — render history
    st.header("Chat with iSchool Assistant")
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.caption(f"Sources: {message['sources']}")

    # Helper to produce an answer
    def answer(prompt_text: str):
        st.session_state.conversation_history.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)
        with st.chat_message("assistant"):
            with st.spinner(f"Searching KB and generating with {selected_model}…"):
                try:
                    results = search_vector_database(vector_db, prompt_text, n_results=3)
                    if results and results.get("documents"):
                        context_docs = results["documents"][0]
                        source_files = [m["filename"] for m in results["metadatas"][0]]
                        rag_prompt = create_rag_prompt(
                            prompt_text, context_docs, source_files,
                            st.session_state.conversation_history[:-1]
                        )
                        messages = [{"role": "user", "content": rag_prompt}]

                        # OpenAI (stream)
                        if selected_model.startswith("gpt"):
                            resp = get_llm_response(messages, selected_model, api_clients)
                            if isinstance(resp, str):
                                st.error(resp)
                                full_response = resp
                            elif hasattr(resp, "__iter__"):
                                placeholder = st.empty()
                                full_response = display_streaming_response(resp, placeholder)
                            else:
                                full_response = "Unexpected response type from OpenAI."
                                st.error(full_response)
                            st.markdown(full_response)

                        # Groq (non-stream)
                        elif (selected_model.startswith("llama") or selected_model.startswith("mixtral")
                              or selected_model.startswith("gemma") or selected_model.startswith("groq/")
                              or selected_model.startswith("openai/gpt-oss")):
                            full_response = get_llm_response(messages, selected_model, api_clients)
                            if isinstance(full_response, str) and full_response.lower().startswith("error with"):
                                st.error(full_response)
                            st.markdown(full_response)

                        # Gemini
                        elif selected_model.startswith("gemini"):
                            if GEMINI_AVAILABLE:
                                gen_model = genai.GenerativeModel(selected_model)  # type: ignore
                                local_clients = {"gemini": gen_model}
                                full_response = get_llm_response(messages, selected_model, local_clients)
                                if isinstance(full_response, str) and full_response.lower().startswith("error with"):
                                    st.error(full_response)
                                st.markdown(full_response)
                            else:
                                full_response = "Gemini SDK not installed."
                                st.error(full_response)

                        else:
                            full_response = f"Unsupported provider for model '{selected_model}'."
                            st.error(full_response)

                        unique_sources = sorted(set(source_files))
                        sources_text = ", ".join(unique_sources)
                        st.caption(f"📚 Sources: {sources_text}")
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": sources_text
                        })
                    else:
                        msg = ("I couldn't find specific information about that in the iSchool materials. "
                               "Try asking about student organizations, programs, or other iSchool topics?")
                        st.markdown(msg)
                        st.session_state.conversation_history.append({"role": "assistant", "content": msg})
                except Exception as e:
                    err = f"Error generating response: {e}"
                    st.error(err)
                    st.session_state.conversation_history.append({"role": "assistant", "content": err})

                # Enforce memory buffer
                if max_memory_messages > 0 and len(st.session_state.conversation_history) > max_memory_messages:
                    st.session_state.conversation_history = st.session_state.conversation_history[-max_memory_messages:]

    # Chat input
    user_prompt = st.chat_input("Ask me about iSchool student organizations…")
    if user_prompt:
        answer(user_prompt)

    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

    # Suggested questions
    st.subheader("Suggested Questions for Testing")
    test_qs = [
        "What student organizations are available in the iSchool?",
        "How can I get involved in research opportunities?",
        "What are the requirements for the information science program?",
        "Are there any networking events for students?",
        "What career services are available to iSchool students?"
    ]
    cols = st.columns(len(test_qs))
    for i, q in enumerate(test_qs):
        with cols[i % len(test_qs)]:
            if st.button(q, key=f"eval_{i}"):
                st.session_state.pending_question = q
                st.rerun()

    if "pending_question" in st.session_state:
        q = st.session_state.pending_question
        del st.session_state.pending_question
        answer(q)

    # Debug
    with st.expander("Debug Information"):
        st.write(f"**Selected Model:** {selected_model}")
        st.write(f"**Conversation Length:** {len(st.session_state.conversation_history)}")
        st.write(f"**Memory Buffer Size:** {max_memory_messages}")
        st.write(f"**ChromaDB Available:** {CHROMADB_AVAILABLE}")
        st.write(f"**Groq Available:** {GROQ_AVAILABLE}")
        st.write(f"**Gemini Available:** {GEMINI_AVAILABLE}")
        try:
            st.write(f"**Documents in DB:** {vector_db['collection'].count()}")
        except Exception:
            st.write("**Database Status:** Error accessing collection")

if __name__ == "__main__":
    run()
