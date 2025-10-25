#!/usr/bin/env python3
"""
SEC 10-Q RAG + Reranking Lab - Simplified Implementation
Works with minimal dependencies
"""

# Fix for ChromaDB SQLite issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
from typing import List, Dict, Tuple
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="SEC 10-Q RAG + Reranking Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if not api_key:
            st.stop()
    return OpenAI(api_key=api_key)

# Initialize ChromaDB
@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB with OpenAI embeddings"""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # Create embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path="./sec_10q_db_simple")
    
    # Get or create collection
    try:
        collection = client.get_collection(
            name="amazon_10q",
            embedding_function=openai_ef
        )
    except:
        collection = client.create_collection(
            name="amazon_10q",
            embedding_function=openai_ef
        )
    
    return collection

def get_amazon_10q_chunks():
    """Get Amazon 10-Q document chunks"""
    
    chunks = [
        # Risk Factors
        {
            "text": """We Face Intense Competition: Our businesses are rapidly evolving and intensely competitive, 
            with many competitors across geographies including cross-border competition. We compete in e-commerce, 
            cloud services, digital content, advertising, and logistics. Competition intensifies with new business 
            models and well-funded competitors. Some competitors have greater resources, longer histories, and more 
            customers. Increased competition may require us to lower prices or increase spending.""",
            "section": "Risk Factors",
            "topic": "Competition",
            "importance": "high"
        },
        {
            "text": """International Operations Risks: International activities are significant to revenues. China 
            and India regulate our business through foreign investment restrictions in internet, retail, and delivery. 
            China-based sellers account for significant third-party seller and advertising revenues. Regulatory 
            restrictions, tariff changes, and trade disputes could adversely affect results. We face risks from 
            local economic conditions, government regulations, and cultural differences.""",
            "section": "Risk Factors",
            "topic": "International",
            "importance": "high"
        },
        {
            "text": """Technology Infrastructure Risks: We experience system interruptions making services unavailable. 
            Systems could be damaged by natural disasters, extreme weather, geopolitical events, cyber attacks, or 
            operational failures. Our systems aren't fully redundant and disaster recovery may be insufficient. 
            We depend on third-party technology and face security breach risks that could expose customer data.""",
            "section": "Risk Factors",
            "topic": "Technology",
            "importance": "high"
        },
        
        # Financial Performance
        {
            "text": """Q2 2025 Revenue Performance: Net sales increased 13% to $167.7 billion in Q2 2025, compared 
            with $148.0 billion in Q2 2024. North America segment sales grew 11% to $100.1 billion. International 
            segment sales increased 16% to $36.8 billion. AWS sales increased 17% to $30.9 billion. Growth reflects 
            increased unit sales, advertising sales, and subscription services.""",
            "section": "Financial Results",
            "topic": "Revenue",
            "importance": "critical"
        },
        {
            "text": """Operating Income Growth: Operating income reached $19.2 billion in Q2 2025, up from $14.7 
            billion in Q2 2024. North America operating income was $7.5 billion (vs $5.1B prior year). International 
            operating income was $1.5 billion (vs $273M prior year). AWS operating income was $10.2 billion (vs $9.3B 
            prior year), representing 53% of total operating income.""",
            "section": "Financial Results",
            "topic": "Operating Income",
            "importance": "critical"
        },
        {
            "text": """Cash Flow Performance: Cash provided by operating activities was $32.5 billion for Q2 2025, 
            compared with $25.3 billion for Q2 2024. Free cash flow for trailing twelve months was $18.2 billion. 
            Cash and marketable securities totaled $93.2 billion as of June 30, 2025. Working capital improvements 
            and increased net income drove cash flow growth.""",
            "section": "Financial Results",
            "topic": "Cash Flow",
            "importance": "high"
        },
        
        # AWS Segment
        {
            "text": """AWS Business Overview: AWS provides compute, storage, database, and AI/ML services globally. 
            Q2 2025 revenue of $30.9 billion represents 17% year-over-year growth. Growth driven by increased customer 
            usage, partially offset by pricing optimizations. AWS serves startups, enterprises, governments, and 
            academic institutions. Represents 18% of total Amazon revenue.""",
            "section": "Segment Information",
            "topic": "AWS",
            "importance": "critical"
        },
        {
            "text": """AWS Infrastructure Investment: Technology and infrastructure costs were $27.2 billion in Q2 
            2025 (vs $22.3B in Q2 2024). Major investments in AI/ML infrastructure including GPUs. Changed server 
            depreciation from 6 to 5 years due to AI technology pace, increasing Q2 depreciation by $280 million. 
            Capital expenditures primarily for AWS data centers.""",
            "section": "Segment Information",
            "topic": "AWS Infrastructure",
            "importance": "high"
        },
        
        # Strategic Investments
        {
            "text": """Anthropic AI Partnership: Amazon invested $1.3 billion in Anthropic convertible notes in Q2 
            2025, with additional $1.4 billion commitment by Q4 2025. Total Anthropic investment fair value 
            approximately $15.1 billion as of June 30, 2025. Includes commercial arrangement for AWS cloud services 
            and AWS AI chip usage. Strategic partnership for generative AI development.""",
            "section": "Investments",
            "topic": "Anthropic AI",
            "importance": "critical"
        },
        {
            "text": """Equity Investment Portfolio: Rivian investment resulted in $388 million valuation gain in Q2 
            2025. Total equity and warrant investments in public companies worth $4.6 billion. Private company 
            investments valued at $6.1 billion. Portfolio aligned with strategic objectives in transportation, 
            AI, and technology sectors.""",
            "section": "Investments",
            "topic": "Equity Portfolio",
            "importance": "medium"
        },
        
        # Guidance
        {
            "text": """Q3 2025 Guidance: Net sales expected between $174.0-$179.5 billion (10-13% growth vs Q3 2024). 
            Operating income expected between $15.5-$20.5 billion (vs $17.4B in Q3 2024). Guidance includes favorable 
            foreign exchange impact of 130 basis points. Assumes no major acquisitions or legal settlements. Reflects 
            continued investment in AWS and fulfillment capacity.""",
            "section": "Guidance",
            "topic": "Q3 2025 Outlook",
            "importance": "critical"
        },
        {
            "text": """2025 Tax Act Impact: The 2025 Tax Act reinstates 100% accelerated depreciation and immediate 
            R&D expensing. Expected to significantly decrease U.S. cash taxes in 2025. Income tax provision will 
            increase due to reduced foreign income deduction. Retroactive application from January 2025.""",
            "section": "Tax Updates",
            "topic": "Tax Changes",
            "importance": "medium"
        }
    ]
    
    return chunks

def load_documents(collection):
    """Load documents into ChromaDB"""
    chunks = get_amazon_10q_chunks()
    
    # Check if already loaded
    if collection.count() > 0:
        return collection
    
    # Prepare documents
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{
        "section": chunk["section"],
        "topic": chunk["topic"],
        "importance": chunk["importance"]
    } for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

def retrieve_chunks(collection, query: str, n_results: int = 6) -> List[Dict]:
    """Retrieve relevant chunks using vector similarity"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i],
            "similarity": 1 - results['distances'][0][i]
        })
    
    return chunks

def rerank_chunks_with_llm(client: OpenAI, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """Use LLM to rerank chunks for better relevance"""
    
    # Prepare reranking prompt
    chunks_text = "\n\n".join([
        f"Chunk {i+1} [{chunk['metadata'].get('topic', 'General')}]:\n{chunk['text'][:300]}..."
        for i, chunk in enumerate(chunks)
    ])
    
    prompt = f"""Score each chunk's relevance to the query on a scale of 1-10.
    Consider direct relevance, specificity, and importance.
    
    Query: {query}
    
    Chunks:
    {chunks_text}
    
    Return ONLY a JSON array of scores: [score1, score2, ...]"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating document relevance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        
        # Parse scores
        scores_text = response.choices[0].message.content.strip()
        scores_text = scores_text.replace('```json', '').replace('```', '').strip()
        scores = json.loads(scores_text)
        
        # Add scores to chunks
        for i, chunk in enumerate(chunks[:len(scores)]):
            chunk['rerank_score'] = scores[i]
        
        # Sort and return top k
        reranked = sorted(chunks[:len(scores)], key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked[:top_k]
        
    except Exception as e:
        st.error(f"Reranking failed: {str(e)}")
        return chunks[:top_k]

def generate_answer(client: OpenAI, query: str, chunks: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate answer using retrieved chunks"""
    
    context = "\n\n".join([
        f"[{chunk['metadata'].get('section')} - {chunk['metadata'].get('topic')}]\n{chunk['text']}"
        for chunk in chunks
    ])
    
    prompt = f"""Based on Amazon's Q2 2025 10-Q filing excerpts below, answer the question.
    Be specific and include numbers when available.
    
    Question: {query}
    
    Context:
    {context}
    
    Answer:"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst specializing in SEC filings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Main Application
def main():
    st.title("📊 SEC 10-Q RAG + Reranking Lab")
    st.markdown("### Amazon Q2 2025 10-Q Filing Analysis")
    st.markdown("*Demonstrating RAG with LLM-based reranking for improved relevance*")
    
    # Initialize components
    client = get_openai_client()
    collection = init_chromadb()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load documents
        if st.button("📄 Load Amazon 10-Q Data"):
            with st.spinner("Loading documents..."):
                collection = load_documents(collection)
                st.success(f"✅ Loaded {collection.count()} chunks")
        
        # Settings
        st.markdown("### Settings")
        model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
        use_reranking = st.checkbox("Enable LLM Reranking", value=True)
        num_retrieve = st.slider("Chunks to retrieve", 3, 10, 6)
        num_rerank = st.slider("Chunks after reranking", 1, 5, 3)
        
        # Example questions
        st.markdown("### 📝 Example Questions")
        examples = [
            "What are Amazon's AI investments?",
            "How did AWS perform in Q2 2025?",
            "What are the main technology risks?",
            "What is the Q3 2025 guidance?",
            "Tell me about international operations risks"
        ]
        
        for ex in examples:
            if st.button(f"→ {ex[:30]}...", key=ex):
                st.session_state.query = ex
    
    # Main query interface
    query = st.text_input(
        "💬 Ask about Amazon's Q2 2025 10-Q:",
        placeholder="e.g., What is Amazon's investment in Anthropic?",
        value=st.session_state.get('query', '')
    )
    
    if query and st.button("🔍 Search", type="primary"):
        # Three-column layout for pipeline stages
        col1, col2, col3 = st.columns(3)
        
        # Step 1: Retrieval
        with col1:
            st.markdown("### 📥 Step 1: Vector Search")
            with st.spinner("Retrieving..."):
                retrieved_chunks = retrieve_chunks(collection, query, num_retrieve)
                
                with st.expander(f"Retrieved {len(retrieved_chunks)} chunks", expanded=True):
                    for i, chunk in enumerate(retrieved_chunks[:3]):
                        st.markdown(f"**#{i+1}** *{chunk['metadata']['topic']}*")
                        st.caption(f"Similarity: {chunk['similarity']:.3f}")
                        st.text(chunk['text'][:150] + "...")
        
        # Step 2: Reranking
        with col2:
            st.markdown("### 🎯 Step 2: Reranking")
            if use_reranking:
                with st.spinner("Reranking..."):
                    reranked_chunks = rerank_chunks_with_llm(client, query, retrieved_chunks, num_rerank)
                    
                    with st.expander(f"Top {len(reranked_chunks)} reranked", expanded=True):
                        for i, chunk in enumerate(reranked_chunks):
                            st.markdown(f"**🏆 #{i+1}** *{chunk['metadata']['topic']}*")
                            if 'rerank_score' in chunk:
                                st.caption(f"Score: {chunk['rerank_score']}/10")
                            st.text(chunk['text'][:150] + "...")
            else:
                reranked_chunks = retrieved_chunks[:num_rerank]
                st.info("Reranking disabled")
        
        # Step 3: Generation
        with col3:
            st.markdown("### 💡 Step 3: Answer")
            with st.spinner("Generating..."):
                final_chunks = reranked_chunks if use_reranking else retrieved_chunks[:num_rerank]
                answer = generate_answer(client, query, final_chunks, model)
                st.success(answer)
        
        # Metrics
        st.markdown("---")
        st.markdown("### 📊 Performance Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retrieved Chunks", len(retrieved_chunks))
        with col2:
            st.metric("After Reranking", len(reranked_chunks) if use_reranking else "N/A")
        with col3:
            if use_reranking and reranked_chunks and 'rerank_score' in reranked_chunks[0]:
                avg_score = sum(c.get('rerank_score', 0) for c in reranked_chunks) / len(reranked_chunks)
                st.metric("Avg Relevance", f"{avg_score:.1f}/10")
            else:
                st.metric("Avg Similarity", f"{sum(c['similarity'] for c in final_chunks)/len(final_chunks):.3f}")

if __name__ == "__main__":
    # Auto-load documents
    collection = init_chromadb()
    if collection.count() == 0:
        collection = load_documents(collection)
    
    main()