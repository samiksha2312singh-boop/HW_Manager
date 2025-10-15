import streamlit as st
import os
import json
from typing import List, Dict, Optional
from openai import OpenAI

# Import the vector DB functions from hw4
from hw4 import (
    create_or_load_vector_database,
    search_vector_database,
    CHROMADB_AVAILABLE
)

# ============ FUNCTION DEFINITION ============
def get_relevant_club_info(query: str, n_results: int = 3) -> str:
    """
    Function that retrieves relevant club/organization information from vector database.
    
    Args:
        query: The user's question or search query
        n_results: Number of relevant documents to retrieve
        
    Returns:
        A formatted string containing relevant information from the knowledge base
    """
    try:
        # Get vector database from session state
        if "ischool_vectorDB" not in st.session_state:
            return json.dumps({"error": "Knowledge base not initialized"})
        
        vector_db = st.session_state.ischool_vectorDB
        
        # Search the vector database
        results = search_vector_database(vector_db, query, n_results=n_results)
        
        if not results or not results.get("documents"):
            return json.dumps({"error": "No relevant information found"})
        
        # Format the results
        context_docs = results["documents"][0]
        source_files = [m["filename"] for m in results["metadatas"][0]]
        
        # Create structured response
        formatted_info = {
            "relevant_information": context_docs,
            "sources": list(set(source_files)),
            "query": query
        }
        
        return json.dumps(formatted_info, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error retrieving information: {str(e)}"})


# OpenAI function definition for the API
CLUB_INFO_FUNCTION = {
    "name": "get_relevant_club_info",
    "description": "Retrieves relevant information about iSchool student organizations, clubs, programs, and opportunities from the knowledge base",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's question or topic to search for in the iSchool knowledge base"
            },
            "n_results": {
                "type": "integer",
                "description": "Number of relevant documents to retrieve (default: 3)",
                "default": 3
            }
        },
        "required": ["query"]
    }
}


# ============ LLM INTERACTION ============
def get_response_with_function_calling(
    client: OpenAI,
    user_input: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> tuple[str, Optional[str]]:
    """
    Get response from OpenAI using function calling for knowledge retrieval.
    
    Returns:
        Tuple of (response_text, sources_text)
    """
    try:
        # Prepare messages with conversation history
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for the iSchool at Syracuse University. "
                          "You specialize in providing information about student organizations, "
                          "academic programs, and opportunities. Use the get_relevant_club_info "
                          "function to retrieve accurate information from the knowledge base."
            }
        ]
        
        # Add conversation history (last 5 exchanges for short-term memory)
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 5 Q&A pairs
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # First API call - let OpenAI decide if it needs to call the function
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=[CLUB_INFO_FUNCTION],
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )
        
        message = response.choices[0].message
        sources = None
        
        # Check if OpenAI wants to call the function
        if message.function_call:
            # Parse function arguments
            function_args = json.loads(message.function_call.arguments)
            query = function_args.get("query", user_input)
            n_results = function_args.get("n_results", 3)
            
            # Call our knowledge retrieval function
            knowledge_result = get_relevant_club_info(query, n_results)
            
            # Parse to extract sources
            try:
                result_data = json.loads(knowledge_result)
                sources = ", ".join(result_data.get("sources", []))
            except:
                pass
            
            # Second API call - provide knowledge and get final response
            messages.append(message.model_dump())  # Add assistant's function call
            messages.append({
                "role": "function",
                "name": "get_relevant_club_info",
                "content": knowledge_result
            })
            
            final_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return final_response.choices[0].message.content, sources
        else:
            # No function call needed - direct response
            return message.content, None
            
    except Exception as e:
        return f"Error generating response: {str(e)}", None


# ============ STREAMLIT APP ============
def run():
    st.set_page_config(
        page_title="HW5 - Enhanced iSchool Chatbot",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 HW5 - Enhanced iSchool Chatbot with Function Calling")
    st.markdown("Ask questions about iSchool organizations using intelligent function-based retrieval")
    
    # Initialize API clients
    openai_api_key = (
        st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    ) or os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.error("❌ OPENAI_API_KEY not found. Please configure it in secrets.toml or environment.")
        return
    
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_options = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        model_options,
        help="Choose which OpenAI model to use"
    )
    
    # Memory configuration
    st.sidebar.markdown("### 🧠 Memory Settings")
    memory_size = st.sidebar.slider(
        "Conversation History (message pairs)",
        min_value=0,
        max_value=10,
        value=5,
        help="Number of previous Q&A pairs to remember"
    )
    
    # Knowledge base management
    st.sidebar.markdown("### 📚 Knowledge Base")
    rebuild = st.sidebar.button("🔄 Rebuild Vector Database")
    
    # Initialize or load vector database
    if "ischool_vectorDB" not in st.session_state or rebuild:
        with st.spinner("Loading knowledge base..."):
            vector_db = create_or_load_vector_database(force_rebuild=rebuild)
            if vector_db:
                st.session_state.ischool_vectorDB = vector_db
                st.sidebar.success("✅ Knowledge base ready")
            else:
                st.error("❌ Failed to initialize knowledge base")
                return
    
    # Display database stats
    try:
        count = st.session_state.ischool_vectorDB["collection"].count()
        st.sidebar.metric("Documents Indexed", f"{count} chunks")
    except:
        st.sidebar.warning("Could not retrieve database stats")
    
    # Initialize conversation history
    if "hw5_conversation" not in st.session_state:
        st.session_state.hw5_conversation = []
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display conversation history
    for message in st.session_state.hw5_conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(f"📚 Sources: {message['sources']}")
    
    # Chat input
    user_input = st.chat_input("Ask me about iSchool organizations...")
    
    if user_input:
        # Add user message to history
        st.session_state.hw5_conversation.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model}..."):
                # Use function calling to get response
                response_text, sources = get_response_with_function_calling(
                    openai_client,
                    user_input,
                    st.session_state.hw5_conversation[:-1],  # Exclude just-added message
                    model=selected_model
                )
                
                # Display response
                st.markdown(response_text)
                
                # Display sources if available
                if sources:
                    st.caption(f"📚 Sources: {sources}")
                
                # Add to conversation history
                st.session_state.hw5_conversation.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources
                })
        
        # Trim conversation history based on memory setting
        max_messages = memory_size * 2  # Each exchange = 2 messages
        if len(st.session_state.hw5_conversation) > max_messages:
            st.session_state.hw5_conversation = st.session_state.hw5_conversation[-max_messages:]
    
    # Action buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.hw5_conversation = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat History"):
            history_json = json.dumps(st.session_state.hw5_conversation, indent=2)
            st.download_button(
                label="Download JSON",
                data=history_json,
                file_name="chat_history.json",
                mime="application/json"
            )
    
    # Example queries
    st.markdown("---")
    st.subheader("💡 Example Questions")
    example_cols = st.columns(3)
    
    examples = [
        "What student organizations focus on data science?",
        "How can I get involved in research at the iSchool?",
        "What networking events are available for students?",
        "Tell me about career services",
        "What are the requirements for the information science program?",
        "Are there any clubs for women in tech?"
    ]
    
    for idx, example in enumerate(examples):
        with example_cols[idx % 3]:
            if st.button(example, key=f"example_{idx}"):
                st.session_state.pending_query = example
                st.rerun()
    
    # Handle pending query from button click
    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query
        st.session_state.hw5_conversation.append({"role": "user", "content": query})
        st.rerun()
    
    # Debug information
    with st.expander("🔍 Debug Information"):
        st.markdown(f"**Selected Model:** {selected_model}")
        st.markdown(f"**Memory Size:** {memory_size} Q&A pairs ({memory_size * 2} messages)")
        st.markdown(f"**Conversation Length:** {len(st.session_state.hw5_conversation)} messages")
        st.markdown(f"**ChromaDB Available:** {CHROMADB_AVAILABLE}")
        
        if st.session_state.hw5_conversation:
            st.markdown("**Last Exchange:**")
            st.json(st.session_state.hw5_conversation[-2:])
    


if __name__ == "__main__":
    run()