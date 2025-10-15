import streamlit as st
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import tiktoken
from datetime import datetime
from openai import OpenAI

# Try importing optional libraries
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("sentence-transformers not available. Install: pip install sentence-transformers")

# ChromaDB for vector storage (reusing from hw4)
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

CHROMADB_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    pass

# Removed sklearn dependency - using ChromaDB's built-in similarity search instead

# ============================================
# NEWS RANKING SYSTEM (Core Logic)
# ============================================

class NewsRanker:
    """Handles news ranking logic for law firm relevance"""
    
    def __init__(self):
        # Law firm specific keywords
        self.legal_keywords = {
            'high_priority': ['lawsuit', 'litigation', 'regulatory', 'compliance', 'SEC', 
                             'CFPB', 'investigation', 'antitrust', 'merger', 'acquisition', 
                             'settlement', 'court', 'legal'],
            'medium_priority': ['patent', 'trademark', 'copyright', 'contract', 'governance', 
                               'ethics', 'fraud', 'violation', 'dispute'],
            'industry_impact': ['bankruptcy', 'IPO', 'restructuring', 'scandal', 'breach', 
                               'cybersecurity', 'privacy', 'GDPR', 'data protection', 'AI regulation']
        }
    
    def calculate_relevance_score(self, news_text: str, date_str: str = None) -> float:
        """Calculate relevance score for a single news item"""
        score = 0.0
        doc_lower = news_text.lower()
        
        # Keyword scoring
        for keyword in self.legal_keywords['high_priority']:
            if keyword in doc_lower:
                score += 3.0
        
        for keyword in self.legal_keywords['medium_priority']:
            if keyword in doc_lower:
                score += 1.5
        
        for keyword in self.legal_keywords['industry_impact']:
            if keyword in doc_lower:
                score += 2.0
        
        # Recency bonus
        if date_str:
            try:
                news_date = pd.to_datetime(date_str)
                days_old = (datetime.now() - news_date).days
                if days_old <= 7:
                    score += 2.0
                elif days_old <= 14:
                    score += 1.0
                elif days_old <= 30:
                    score += 0.5
            except:
                pass
        
        # Financial impact
        import re
        amounts = re.findall(r'\$?(\d+(?:\.\d+)?)\s*(?:billion|million)', doc_lower)
        if amounts:
            try:
                score += min(float(amounts[0]) / 10, 5.0)
            except:
                pass
        
        return score

# ============================================
# VECTOR DATABASE FUNCTIONS (Adapted from hw4)
# ============================================

def init_vector_db(force_rebuild: bool = False):
    """Initialize ChromaDB for news articles"""
    if not CHROMADB_AVAILABLE:
        return None
    
    try:
        # Try persistent client first
        client = chromadb.PersistentClient(
            path="./news_vector_db",
            settings=Settings(anonymized_telemetry=False)
        )
    except:
        # Fallback to ephemeral
        client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
    
    # Get or create collection
    try:
        if force_rebuild:
            client.delete_collection("news_collection")
    except:
        pass
    
    collection = client.get_or_create_collection(
        name="news_collection",
        metadata={"description": "Law firm news articles"}
    )
    
    return {"client": client, "collection": collection}

def index_news_to_vectordb(df: pd.DataFrame, vector_db: Dict, api_key: str):
    """Index news articles into vector database"""
    if not vector_db:
        return False
    
    collection = vector_db["collection"]
    openai_client = OpenAI(api_key=api_key)
    
    documents = []
    metadatas = []
    ids = []
    
    # Process each news article
    for idx, row in df.iterrows():
        doc_text = f"{row['company_name']}: {row['Document']}"
        
        documents.append(doc_text[:8000])  # Limit for embedding
        metadatas.append({
            "company": row['company_name'],
            "date": str(row['Date']),
            "url": row.get('URL', ''),
            "index": idx
        })
        ids.append(f"news_{idx}")
        
        # Batch process
        if len(documents) >= 10 or idx == len(df) - 1:
            try:
                # Create embeddings
                resp = openai_client.embeddings.create(
                    input=documents,
                    model="text-embedding-3-small"
                )
                embeddings = [d.embedding for d in resp.data]
                
                # Add to collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                
                # Clear batch
                documents = []
                metadatas = []
                ids = []
            except Exception as e:
                st.error(f"Error indexing batch: {e}")
    
    return True

def search_news_vectordb(vector_db: Dict, query: str, n_results: int = 10, api_key: str = None):
    """Search vector database for relevant news"""
    if not vector_db or not api_key:
        return None
    
    collection = vector_db["collection"]
    openai_client = OpenAI(api_key=api_key)
    
    try:
        # Create query embedding
        resp = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = resp.data[0].embedding
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return None

# ============================================
# LLM HELPERS (Adapted from hw3/hw4)
# ============================================

def get_llm_response(prompt: str, model: str, api_clients: Dict, stream: bool = True):
    """Get LLM response with multi-vendor support"""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant for a global law firm, specializing in legal and regulatory news analysis."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # OpenAI models
        if model.startswith("gpt"):
            client = api_clients.get("openai")
            if not client:
                return "OpenAI client not available"
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                temperature=0.3,
                max_tokens=1000
            )
            
            if stream:
                return response
            else:
                return response.choices[0].message.content
        
        # Groq models
        elif GROQ_AVAILABLE and (model.startswith("llama") or model.startswith("mixtral")):
            client = api_clients.get("groq")
            if not client:
                return "Groq client not available"
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        
        # Gemini models
        elif GEMINI_AVAILABLE and model.startswith("gemini"):
            client = api_clients.get("gemini")
            if not client:
                return "Gemini client not available"
            
            prompt_text = messages[0]["content"] + "\n\n" + messages[1]["content"]
            response = client.generate_content(prompt_text)
            return response.text
        
        else:
            return f"Unsupported model: {model}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def display_streaming_response(stream, placeholder):
    """Display streaming response from OpenAI"""
    full_response = ""
    for chunk in stream:
        if hasattr(chunk.choices[0].delta, 'content'):
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)
    return full_response

def rank_news_with_llm(news_items: List[Dict], model: str, api_clients: Dict, context: str = "law firm") -> List[Dict]:
    """Use LLM to rank news items by relevance"""
    
    # Prepare news summaries
    news_summaries = []
    for i, item in enumerate(news_items[:30]):  # Limit to prevent token overflow
        summary = f"{i}. {item.get('company', 'Unknown')}: {item.get('headline', '')[:150]}"
        news_summaries.append(summary)
    
    prompt = f"""As an AI assistant for a global {context}, rank these news items by importance and relevance.
Consider: legal implications, regulatory impact, M&A activity, litigation potential, and business significance.

News items:
{chr(10).join(news_summaries)}

Return ONLY a JSON array of indices (0-based) for the top 10 most relevant items, ordered by importance.
Example: [2, 5, 1, 8, 3, 9, 0, 7, 4, 6]"""
    
    response = get_llm_response(prompt, model, api_clients, stream=False)
    
    try:
        # Parse response
        import re
        indices_match = re.search(r'\[[\d,\s]+\]', response)
        if indices_match:
            indices = json.loads(indices_match.group())
            # Return reordered items
            ranked = []
            for idx in indices[:10]:
                if idx < len(news_items):
                    ranked.append(news_items[idx])
            return ranked
    except:
        pass
    
    # Fallback to original order
    return news_items[:10]

# ============================================
# STREAMLIT APP
# ============================================

def run():
    st.set_page_config(
        page_title="HW7 - Law Firm News Bot",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ HW7 - Law Firm News Intelligence Bot")
    st.markdown("AI-powered news analysis and ranking for legal professionals")
    
    # Initialize API clients
    api_clients = {}
    
    # OpenAI
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if openai_key:
        api_clients["openai"] = OpenAI(api_key=openai_key)
    
    # Groq
    if GROQ_AVAILABLE:
        groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if groq_key:
            api_clients["groq"] = Groq(api_key=groq_key)
    
    # Gemini
    if GEMINI_AVAILABLE:
        gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            api_clients["gemini"] = genai.GenerativeModel('gemini-1.5-pro')
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    available_models = []
    if "openai" in api_clients:
        available_models.extend(["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    if "groq" in api_clients:
        available_models.extend(["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
    if "gemini" in api_clients:
        available_models.extend(["gemini-1.5-pro", "gemini-1.5-flash"])
    
    if not available_models:
        st.error("No LLM API keys found. Please configure API keys.")
        return
    
    selected_model = st.sidebar.selectbox("Select LLM Model", available_models)
    
    # Display model info
    model_info = {
        "gpt-4o": "OpenAI flagship - highest accuracy",
        "gpt-4o-mini": "OpenAI efficient - good balance",
        "gpt-3.5-turbo": "OpenAI fast - economical",
        "llama-3.3-70b-versatile": "Groq - open source powerhouse",
        "mixtral-8x7b-32768": "Groq - efficient mixture of experts",
        "gemini-1.5-pro": "Google - multimodal flagship",
        "gemini-1.5-flash": "Google - fast and efficient"
    }
    st.sidebar.caption(model_info.get(selected_model, ""))
    
    # File upload
    st.sidebar.header("📁 Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload News CSV",
        type=['csv'],
        help="Upload the news data CSV file"
    )
    
    # Use RAG option
    use_rag = st.sidebar.checkbox(
        "Use Vector Search (RAG)",
        value=True,
        help="Enable semantic search using embeddings"
    )
    
    # Number of results
    num_results = st.sidebar.slider(
        "Number of Results",
        min_value=5,
        max_value=20,
        value=10
    )
    
    # Load and process data
    if uploaded_file:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(df)} news articles")
        
        # Store in session state
        st.session_state['news_df'] = df
        
        # Initialize components
        if 'news_ranker' not in st.session_state:
            st.session_state['news_ranker'] = NewsRanker()
        
        # Initialize vector DB if using RAG
        if use_rag and CHROMADB_AVAILABLE:
            if 'news_vector_db' not in st.session_state:
                with st.spinner("Initializing vector database..."):
                    vector_db = init_vector_db()
                    if vector_db and openai_key:
                        success = index_news_to_vectordb(df, vector_db, openai_key)
                        if success:
                            st.session_state['news_vector_db'] = vector_db
                            st.sidebar.success("✅ Vector DB ready")
                        else:
                            st.sidebar.error("❌ Vector DB indexing failed")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📰 Most Interesting News", "🔍 Topic Search", "📊 Analysis", "🤖 Model Comparison"])
    
    with tab1:
        st.header("Most Interesting News for Law Firms")
        
        if 'news_df' in st.session_state:
            df = st.session_state['news_df']
            ranker = st.session_state['news_ranker']
            
            if st.button("🔄 Generate Rankings", type="primary"):
                with st.spinner(f"Analyzing with {selected_model}..."):
                    # Calculate relevance scores
                    df['relevance_score'] = df.apply(
                        lambda row: ranker.calculate_relevance_score(
                            row['Document'],
                            row.get('Date')
                        ),
                        axis=1
                    )
                    
                    # Get top candidates
                    candidates = df.nlargest(num_results * 2, 'relevance_score')
                    
                    # Prepare for LLM ranking
                    news_items = []
                    for _, row in candidates.iterrows():
                        news_items.append({
                            'company': row['company_name'],
                            'headline': row['Document'],
                            'date': str(row.get('Date', '')),
                            'url': row.get('URL', ''),
                            'score': row['relevance_score']
                        })
                    
                    # LLM ranking
                    ranked_news = rank_news_with_llm(
                        news_items,
                        selected_model,
                        api_clients
                    )
                    
                    # Display results
                    for i, item in enumerate(ranked_news[:num_results], 1):
                        with st.expander(f"**#{i} - {item['company']}** | Score: {item.get('score', 0):.1f}"):
                            st.write(item['headline'])
                            if item.get('date'):
                                st.caption(f"📅 Date: {item['date']}")
                            if item.get('url'):
                                st.link_button("Read Full Article", item['url'])
                            
                            # Explain relevance
                            relevance_reasons = []
                            headline_lower = item['headline'].lower()
                            if any(kw in headline_lower for kw in ['lawsuit', 'litigation', 'court']):
                                relevance_reasons.append("⚖️ Litigation")
                            if any(kw in headline_lower for kw in ['regulatory', 'compliance', 'SEC', 'CFPB']):
                                relevance_reasons.append("📋 Regulatory")
                            if any(kw in headline_lower for kw in ['merger', 'acquisition', 'M&A']):
                                relevance_reasons.append("🤝 M&A Activity")
                            if any(kw in headline_lower for kw in ['data', 'privacy', 'breach', 'cyber']):
                                relevance_reasons.append("🔒 Data/Privacy")
                            
                            if relevance_reasons:
                                st.info("Relevant for: " + " | ".join(relevance_reasons))
        else:
            st.info("Please upload a news CSV file in the sidebar")
    
    with tab2:
        st.header("Search News by Topic")
        
        search_query = st.text_input(
            "Enter search topic",
            placeholder="e.g., merger acquisition, data breach, regulatory compliance"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            search_button = st.button("🔍 Search", key="search_btn", type="primary")
        
        if search_button and search_query:
            if 'news_df' in st.session_state:
                df = st.session_state['news_df']
                
                with st.spinner("Searching..."):
                    if use_rag and 'news_vector_db' in st.session_state and openai_key:
                        # Vector search
                        results = search_news_vectordb(
                            st.session_state['news_vector_db'],
                            search_query,
                            num_results,
                            openai_key
                        )
                        
                        if results and results.get('metadatas'):
                            st.success(f"Found {len(results['documents'][0])} relevant articles")
                            
                            for i, (doc, metadata) in enumerate(zip(
                                results['documents'][0],
                                results['metadatas'][0]
                            ), 1):
                                with st.container():
                                    st.markdown(f"**#{i} - {metadata.get('company', 'Unknown')}**")
                                    st.write(doc[:300] + "...")
                                    if metadata.get('date'):
                                        st.caption(f"📅 {metadata['date']}")
                                    if metadata.get('url'):
                                        st.link_button("Read More", metadata['url'])
                                    st.divider()
                    else:
                        # Fallback to keyword search
                        query_lower = search_query.lower()
                        mask = df['Document'].str.lower().str.contains(query_lower, na=False)
                        filtered = df[mask].head(num_results)
                        
                        if not filtered.empty:
                            st.success(f"Found {len(filtered)} matching articles")
                            
                            for i, (_, row) in enumerate(filtered.iterrows(), 1):
                                with st.container():
                                    st.markdown(f"**#{i} - {row['company_name']}**")
                                    st.write(row['Document'][:300] + "...")
                                    if 'Date' in row:
                                        st.caption(f"📅 {row['Date']}")
                                    if 'URL' in row:
                                        st.link_button("Read More", row['URL'])
                                    st.divider()
                        else:
                            st.warning("No matching articles found")
            else:
                st.info("Please upload a news CSV file first")
    
    with tab3:
        st.header("📊 News Analysis Dashboard")
        
        if 'news_df' in st.session_state:
            df = st.session_state['news_df']
            ranker = st.session_state['news_ranker']
            
            # Calculate statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Articles", len(df))
                st.metric("Unique Companies", df['company_name'].nunique())
            
            with col2:
                # Count legal-relevant articles
                legal_count = 0
                for _, row in df.iterrows():
                    doc_lower = row['Document'].lower()
                    if any(kw in doc_lower for kw in ranker.legal_keywords['high_priority']):
                        legal_count += 1
                
                st.metric("Legal/Regulatory News", legal_count)
                st.metric("Relevance %", f"{(legal_count/len(df)*100):.1f}%")
            
            with col3:
                # Top companies by news volume
                top_companies = df['company_name'].value_counts().head(5)
                st.markdown("**Top Companies by Coverage:**")
                for company, count in top_companies.items():
                    st.write(f"• {company}: {count} articles")
            
            # Keyword distribution
            st.subheader("📈 Keyword Analysis")
            
            keyword_counts = {
                'Litigation': 0,
                'Regulatory': 0,
                'M&A': 0,
                'Cybersecurity': 0,
                'Financial': 0
            }
            
            for _, row in df.iterrows():
                doc_lower = row['Document'].lower()
                if any(kw in doc_lower for kw in ['lawsuit', 'litigation', 'court']):
                    keyword_counts['Litigation'] += 1
                if any(kw in doc_lower for kw in ['regulatory', 'compliance', 'SEC']):
                    keyword_counts['Regulatory'] += 1
                if any(kw in doc_lower for kw in ['merger', 'acquisition', 'M&A']):
                    keyword_counts['M&A'] += 1
                if any(kw in doc_lower for kw in ['cyber', 'breach', 'security']):
                    keyword_counts['Cybersecurity'] += 1
                if any(kw in doc_lower for kw in ['earnings', 'revenue', 'profit']):
                    keyword_counts['Financial'] += 1
            
            # Display as bar chart (simple text version)
            for category, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(df) * 100
                bar = "█" * int(percentage / 2)
                st.text(f"{category:15} {bar} {count} ({percentage:.1f}%)")
        else:
            st.info("Please upload a news CSV file to see analytics")
    
    with tab4:
        st.header("🤖 Model Comparison Testing")
        
        if 'news_df' in st.session_state:
            df = st.session_state['news_df']
            
            # Model comparison configuration
            st.subheader("Test Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Available Models to Compare:**")
                comparison_models = []
                
                # Group models by vendor
                if "openai" in api_clients:
                    st.write("**OpenAI:**")
                    if st.checkbox("GPT-4o (Expensive - $0.03/1K tokens)", value=True, key="gpt4o"):
                        comparison_models.append(("gpt-4o", "OpenAI", "expensive"))
                    if st.checkbox("GPT-4o-mini (Medium - $0.006/1K tokens)", value=True, key="gpt4omini"):
                        comparison_models.append(("gpt-4o-mini", "OpenAI", "medium"))
                    if st.checkbox("GPT-3.5-turbo (Cheap - $0.002/1K tokens)", value=True, key="gpt35"):
                        comparison_models.append(("gpt-3.5-turbo", "OpenAI", "cheap"))
                
                if "groq" in api_clients:
                    st.write("**Groq:**")
                    if st.checkbox("Llama-3.3-70b (Cheap - $0.0008/1K tokens)", value=True, key="llama"):
                        comparison_models.append(("llama-3.3-70b-versatile", "Groq", "cheap"))
                
                if "gemini" in api_clients:
                    st.write("**Google:**")
                    if st.checkbox("Gemini-1.5-pro (Expensive - $0.007/1K tokens)", value=False, key="gemini_pro"):
                        comparison_models.append(("gemini-1.5-pro", "Google", "expensive"))
                    if st.checkbox("Gemini-1.5-flash (Cheap - $0.001/1K tokens)", value=False, key="gemini_flash"):
                        comparison_models.append(("gemini-1.5-flash", "Google", "cheap"))
            
            with col2:
                st.markdown("**Test Queries:**")
                test_queries = [
                    "Find most interesting legal news",
                    "Show merger and acquisition news",
                    "Regulatory compliance issues",
                    "Cybersecurity incidents",
                    "Litigation news"
                ]
                for query in test_queries:
                    st.write(f"• {query}")
            
            if st.button("🚀 Run Comparison Test", type="primary"):
                if len(comparison_models) < 2:
                    st.error("Please select at least 2 models to compare")
                else:
                    comparison_results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_tests = len(comparison_models) * len(test_queries)
                    current_test = 0
                    
                    # Run tests for each model
                    for model_name, vendor, tier in comparison_models:
                        model_results = {
                            "vendor": vendor,
                            "tier": tier,
                            "response_times": [],
                            "quality_scores": [],
                            "responses": []
                        }
                        
                        for query in test_queries:
                            current_test += 1
                            progress_bar.progress(current_test / total_tests)
                            status_text.text(f"Testing {model_name}: {query[:30]}...")
                            
                            # Prepare test prompt
                            sample_news = df.sample(min(5, len(df)))['Document'].str[:150].to_list()
                            prompt = f"""As a legal expert for a law firm, rank these news items for: {query}
                            
News samples:
{chr(10).join([f'{i+1}. {news}' for i, news in enumerate(sample_news)])}

Rank by legal relevance and explain briefly."""
                            
                            # Measure response time
                            import time
                            start_time = time.time()
                            
                            try:
                                response = get_llm_response(prompt, model_name, api_clients, stream=False)
                                end_time = time.time()
                                
                                response_time = end_time - start_time
                                
                                # Calculate quality scores
                                quality_score = 0
                                response_lower = response.lower() if isinstance(response, str) else ""
                                
                                # Check for legal keywords (quality indicator)
                                legal_keywords = ['legal', 'regulatory', 'compliance', 'litigation', 
                                                'merger', 'acquisition', 'court', 'lawsuit']
                                for keyword in legal_keywords:
                                    if keyword in response_lower:
                                        quality_score += 10
                                
                                # Check if it addresses the query
                                if any(term in response_lower for term in query.lower().split()):
                                    quality_score += 20
                                
                                # Check for structure
                                if len(response) > 100:
                                    quality_score += 20
                                if any(marker in response for marker in ['\n', '1.', '•', '-']):
                                    quality_score += 10
                                
                                quality_score = min(quality_score, 100)
                                
                                model_results["response_times"].append(response_time)
                                model_results["quality_scores"].append(quality_score)
                                model_results["responses"].append(response[:200])
                                
                            except Exception as e:
                                st.warning(f"Error testing {model_name}: {str(e)}")
                                model_results["response_times"].append(999)
                                model_results["quality_scores"].append(0)
                                model_results["responses"].append(f"Error: {str(e)}")
                        
                        # Calculate averages
                        import numpy as np
                        comparison_results[model_name] = {
                            "vendor": vendor,
                            "tier": tier,
                            "avg_response_time": np.mean(model_results["response_times"]),
                            "avg_quality_score": np.mean(model_results["quality_scores"]),
                            "min_time": min(model_results["response_times"]),
                            "max_time": max(model_results["response_times"])
                        }
                    
                    progress_bar.progress(1.0)
                    status_text.text("Comparison complete!")
                    
                    # Display results
                    st.subheader("📊 Comparison Results")
                    
                    # Create results table
                    results_data = []
                    for model, metrics in comparison_results.items():
                        cost_map = {
                            "gpt-4o": 30.0,
                            "gpt-4o-mini": 6.0,
                            "gpt-3.5-turbo": 2.0,
                            "llama-3.3-70b-versatile": 0.8,
                            "mixtral-8x7b-32768": 0.6,
                            "gemini-1.5-pro": 7.0,
                            "gemini-1.5-flash": 1.0
                        }
                        
                        results_data.append({
                            "Model": model,
                            "Vendor": metrics["vendor"],
                            "Type": metrics["tier"],
                            "Avg Response Time": f"{metrics['avg_response_time']:.2f}s",
                            "Quality Score": f"{metrics['avg_quality_score']:.0f}/100",
                            "Cost per 1K queries": f"${cost_map.get(model, 1.0):.2f}",
                            "Speed Range": f"{metrics['min_time']:.1f}s - {metrics['max_time']:.1f}s"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Analysis and recommendations
                    st.subheader("🎯 Analysis & Recommendations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Performance Winners")
                        
                        # Find best performers
                        best_quality = max(comparison_results.items(), 
                                         key=lambda x: x[1]['avg_quality_score'])
                        fastest = min(comparison_results.items(), 
                                    key=lambda x: x[1]['avg_response_time'])
                        
                        st.success(f"**Highest Quality:** {best_quality[0]}")
                        st.write(f"Score: {best_quality[1]['avg_quality_score']:.0f}/100")
                        
                        st.info(f"**Fastest:** {fastest[0]}")
                        st.write(f"Avg time: {fastest[1]['avg_response_time']:.2f}s")
                    
                    with col2:
                        st.markdown("### Cost-Benefit Analysis")
                        
                        # Calculate value scores
                        value_scores = {}
                        cost_map = {
                            "gpt-4o": 30.0, 
                            "gpt-4o-mini": 6.0,
                            "gpt-3.5-turbo": 2.0,
                            "llama-3.3-70b-versatile": 0.8,
                            "mixtral-8x7b-32768": 0.6,
                            "gemini-1.5-pro": 7.0, 
                            "gemini-1.5-flash": 1.0
                        }
                        
                        for model, metrics in comparison_results.items():
                            cost = cost_map.get(model, 1.0)
                            value = metrics['avg_quality_score'] / cost
                            value_scores[model] = value
                        
                        best_value = max(value_scores.items(), key=lambda x: x[1])
                        
                        st.success(f"**Best Value:** {best_value[0]}")
                        st.write(f"Value score: {best_value[1]:.1f} (quality/cost)")
                    
                    # Final recommendations
                    st.markdown("### 📋 Final Recommendations")
                    
                    vendors_used = list(set([m["vendor"] for m in comparison_results.values()]))
                    
                    st.write(f"**Vendors Compared:** {', '.join(vendors_used)}")
                    
                    # Separate by tier
                    expensive_models = {k: v for k, v in comparison_results.items() 
                                      if v['tier'] == 'expensive'}
                    cheap_models = {k: v for k, v in comparison_results.items() 
                                   if v['tier'] == 'cheap'}
                    
                    if expensive_models:
                        best_expensive = max(expensive_models.items(), 
                                           key=lambda x: x[1]['avg_quality_score'])
                        st.write(f"**For critical legal analysis:** {best_expensive[0]} ({best_expensive[1]['vendor']})")
                    
                    if cheap_models:
                        best_cheap = max(cheap_models.items(), 
                                       key=lambda x: value_scores.get(x[0], 0))
                        st.write(f"**For production/scale:** {best_cheap[0]} ({best_cheap[1]['vendor']})")
                    
                    # Winner announcement
                    st.success(f"""
                    **🏆 Overall Winner: {best_value[0]}**
                    - Vendor: {comparison_results[best_value[0]]['vendor']}
                    - Best balance of quality, speed, and cost
                    - Recommended for law firm production use
                    """)
                    
                    # Export results
                    st.download_button(
                        "📥 Download Comparison Results",
                        data=results_df.to_csv(index=False),
                        file_name="model_comparison_results.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Please upload a news CSV file to run model comparisons")
    
    # Debug information
    with st.expander("🔧 Debug Information"):
        st.write(f"**Selected Model:** {selected_model}")
        st.write(f"**APIs Available:** {list(api_clients.keys())}")
        st.write(f"**ChromaDB Available:** {CHROMADB_AVAILABLE}")
        st.write(f"**Sentence Transformers:** {SENTENCE_TRANSFORMERS_AVAILABLE}")
        st.write(f"**Using RAG:** {use_rag}")
        if 'news_df' in st.session_state:
            st.write(f"**Data Loaded:** {len(st.session_state['news_df'])} articles")
        if 'news_vector_db' in st.session_state:
            try:
                count = st.session_state['news_vector_db']['collection'].count()
                st.write(f"**Vectors Indexed:** {count}")
            except:
                pass


if __name__ == "__main__":
    run()