
import streamlit as st
import os
import tiktoken
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Optional


# Multi-vendor LLM imports
from openai import OpenAI
import anthropic
import google.generativeai as genai


def fetch_url_content(url: str) -> Optional[str]:
   """Fetch and extract text content from a URL."""
   try:
       headers = {
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
       }
       response = requests.get(url, headers=headers, timeout=10)
       response.raise_for_status()
      
       soup = BeautifulSoup(response.content, 'html.parser')
      
       # Remove script and style elements
       for script in soup(["script", "style"]):
           script.decompose()
      
       # Get text content
       text = soup.get_text()
      
       # Clean up whitespace
       lines = (line.strip() for line in text.splitlines())
       chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
       text = ' '.join(chunk for chunk in chunks if chunk)
      
       return text[:8000]  # Limit to 8000 characters to avoid token limits
      
   except Exception as e:
       st.error(f"Error fetching URL {url}: {str(e)}")
       return None


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
   """Count tokens in text using tiktoken."""
   try:
       if "gpt" in model.lower():
           encoding = tiktoken.encoding_for_model("gpt-4o-mini")
       else:
           # Use cl100k_base as fallback for other models
           encoding = tiktoken.get_encoding("cl100k_base")
       return len(encoding.encode(text))
   except:
       # Fallback estimation
       return len(text) // 4


def manage_conversation_buffer(messages: List[Dict], max_pairs: int = 6) -> List[Dict]:
   """Keep only the last N message pairs plus the system message."""
   if not messages:
       return messages
  
   # Always keep the system message
   system_message = messages[0] if messages[0]["role"] == "system" else None
   conversation_messages = messages[1:] if system_message else messages
  
   # Count complete pairs
   pairs = []
   current_pair = []
  
   for msg in conversation_messages:
       current_pair.append(msg)
      
       if len(current_pair) == 2 and current_pair[0]["role"] == "user" and current_pair[1]["role"] == "assistant":
           pairs.append(current_pair)
           current_pair = []
       elif len(current_pair) == 1 and current_pair[0]["role"] == "user":
           continue
       else:
           current_pair = [msg] if msg["role"] == "user" else []
  
   # Keep only the last max_pairs complete pairs
   kept_pairs = pairs[-max_pairs:] if len(pairs) > max_pairs else pairs
  
   # Flatten pairs back into message list
   kept_messages = []
   for pair in kept_pairs:
       kept_messages.extend(pair)
  
   # Add any incomplete current pair
   if current_pair:
       kept_messages.extend(current_pair)
  
   # Reconstruct final message list
   final_messages = []
   if system_message:
       final_messages.append(system_message)
   final_messages.extend(kept_messages)
  
   return final_messages


def manage_token_buffer(messages: List[Dict], max_tokens: int, model: str = "gpt-4o-mini") -> tuple:
   """Keep messages within token limit."""
   if not messages:
       return messages, 0, 0
  
   system_message = messages[0] if messages[0]["role"] == "system" else None
   conversation_messages = messages[1:] if system_message else messages
  
   system_tokens = count_tokens(system_message["content"], model) + 4 if system_message else 0
  
   selected_messages = []
   current_tokens = system_tokens + 3
   removed_count = 0
  
   for message in reversed(conversation_messages):
       message_tokens = count_tokens(message["content"], model) + 4
      
       if current_tokens + message_tokens <= max_tokens:
           selected_messages.insert(0, message)
           current_tokens += message_tokens
       else:
           removed_count += 1
  
   final_messages = []
   if system_message:
       final_messages.append(system_message)
   final_messages.extend(selected_messages)
  
   return final_messages, current_tokens, removed_count


def create_conversation_summary(messages: List[Dict], client, model: str) -> str:
   """Create a summary of the conversation history."""
   if len(messages) <= 3:  # System message + one exchange
       return ""
  
   # Extract conversation without system message
   conversation_text = ""
   for msg in messages[1:]:  # Skip system message
       role = "Human" if msg["role"] == "user" else "Assistant"
       conversation_text += f"{role}: {msg['content']}\n\n"
  
   summary_prompt = f"""Please provide a concise summary of this conversation, focusing on the key topics discussed and important information shared:


{conversation_text}


Summary:"""
  
   try:
       if "gpt" in model.lower():
           response = client.chat.completions.create(
               model=model,
               messages=[{"role": "user", "content": summary_prompt}],
               max_tokens=200
           )
           return response.choices[0].message.content
       elif "claude" in model.lower():
           response = client.messages.create(
               model=model,
               max_tokens=200,
               messages=[{"role": "user", "content": summary_prompt}]
           )
           return response.content[0].text
       elif "gemini" in model.lower():
           response = client.generate_content(summary_prompt)
           return response.text
   except:
       return "Summary generation failed."
  
   return ""


def get_llm_response(messages: List[Dict], model: str, client, temperature: float = 0.7):
   """Get streaming response from the selected LLM."""
   try:
       if "gpt" in model.lower():
           # OpenAI models
           return client.chat.completions.create(
               model=model,
               messages=messages,
               stream=True,
               temperature=temperature
           )
       elif "claude" in model.lower():
           # Anthropic Claude - convert to Claude format
           claude_messages = []
           system_content = ""
          
           for msg in messages:
               if msg["role"] == "system":
                   system_content = msg["content"]
               else:
                   claude_messages.append({
                       "role": msg["role"],
                       "content": msg["content"]
                   })
          
           return client.messages.stream(
               model=model,
               max_tokens=2000,
               temperature=temperature,
               system=system_content,
               messages=claude_messages
           )
       elif "gemini" in model.lower():
           # Google Gemini
           prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
           return client.generate_content(prompt, stream=True)
   except Exception as e:
       st.error(f"Error generating response: {e}")
       return None


def display_streamed_response(stream, model: str):
   """Display streamed response based on model type."""
   full_response = ""
   response_placeholder = st.empty()
  
   try:
       if "gpt" in model.lower():
           for chunk in stream:
               if chunk.choices[0].delta.content is not None:
                   content = chunk.choices[0].delta.content
                   full_response += content
                   response_placeholder.markdown(full_response + "▌")
       elif "claude" in model.lower():
           with stream as stream_manager:
               for text in stream_manager.text_stream:
                   full_response += text
                   response_placeholder.markdown(full_response + "▌")
       elif "gemini" in model.lower():
           for chunk in stream:
               if chunk.text:
                   full_response += chunk.text
                   response_placeholder.markdown(full_response + "▌")
   except Exception as e:
       st.error(f"Error in streaming: {e}")
       full_response = "Error generating response."
  
   response_placeholder.markdown(full_response)
   return full_response


def create_system_prompt(url1_content: str = "", url2_content: str = "") -> str:
   """Create system prompt with URL content."""
   base_prompt = """You are a helpful assistant that can answer questions based on provided web content and general knowledge.


When answering questions:
1. If the information is available in the provided web content, prioritize that information
2. You can supplement with your general knowledge when helpful
3. Be clear about what information comes from the provided sources vs. your general knowledge
4. Provide comprehensive, well-structured answers"""
  
   if url1_content or url2_content:
       base_prompt += "\n\nYou have access to the following web content:\n"
      
       if url1_content:
           base_prompt += f"\n--- SOURCE 1 ---\n{url1_content}\n"
      
       if url2_content:
           base_prompt += f"\n--- SOURCE 2 ---\n{url2_content}\n"
      
       base_prompt += "\nUse this content to help answer user questions when relevant."
  
   return base_prompt


def run():
   """HW3 - Streaming Chatbot with URL Context"""
   # Page setup
   st.set_page_config(page_title="HW3 - URL Chatbot", page_icon="🌐", layout="wide")
   st.title("HW3 - Multivendor LLM Chatbot with URL Discussion 🌐")
   st.write("Chat with an AI assistant about content from web URLs using different LLMs and memory strategies.")
  
   # Sidebar configuration
   st.sidebar.header("⚙️ Configuration")
  
   # URL inputs
   st.sidebar.subheader("🔗 URLs")
   url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/page1")
   url2 = st.sidebar.text_input("URL 2 (optional)", placeholder="https://example.com/page2")
  
   # LLM selection
   st.sidebar.subheader("🤖 LLM Selection")
  
   # Model options
   model_options = {
       "OpenAI GPT-4o (Flagship)": "gpt-4o",
       "OpenAI GPT-4o-mini (Efficient)": "gpt-4o-mini",
       "Anthropic Claude Haiku": "claude-3-5-haiku-20241022",
       "Google Gemini Pro": "gemini-1.5-pro",
       "Google Gemini Flash": "gemini-1.5-flash"
   }
  
   selected_model_name = st.sidebar.selectbox("Choose LLM Model", list(model_options.keys()))
   selected_model = model_options[selected_model_name]
  
   # Memory type selection
   st.sidebar.subheader("🧠 Memory Strategy")
   memory_options = {
       "Buffer (6 questions)": "buffer_6",
       "Buffer (2,000 tokens)": "token_2000",
       "Conversation Summary": "summary"
   }
  
   memory_type = st.sidebar.selectbox("Choose Memory Type", list(memory_options.keys()))
   memory_strategy = memory_options[memory_type]
  
   # Temperature control
   temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
  
   # Load URLs button
   if st.sidebar.button("🔄 Load URLs"):
       with st.sidebar:
           with st.spinner("Loading URLs..."):
               url1_content = fetch_url_content(url1) if url1 else ""
               url2_content = fetch_url_content(url2) if url2 else ""
              
               st.session_state["url1_content"] = url1_content
               st.session_state["url2_content"] = url2_content
               st.session_state["urls_loaded"] = True
              
               if url1_content:
                   st.success(f"✅ URL 1 loaded ({len(url1_content)} chars)")
               if url2_content:
                   st.success(f"✅ URL 2 loaded ({len(url2_content)} chars)")
  
   # Clear conversation button
   if st.sidebar.button("🗑️ Clear Conversation"):
       for key in ["messages", "conversation_summary", "urls_loaded", "url1_content", "url2_content"]:
           if key in st.session_state:
               del st.session_state[key]
       st.rerun()
  
   # Initialize API clients
   clients = {}
  
   # OpenAI client
   try:
       openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
       if openai_api_key:
           clients["openai"] = OpenAI(api_key=openai_api_key)
   except:
       pass
  
   # Anthropic client
   try:
       anthropic_api_key = st.secrets.get("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
       if anthropic_api_key:
           clients["anthropic"] = anthropic.Anthropic(api_key=anthropic_api_key)
   except:
       pass
  
   # Google client
   try:
       google_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
       if google_api_key:
           genai.configure(api_key=google_api_key)
           clients["google"] = genai.GenerativeModel('gemini-1.5-pro')
   except:
       pass
  
   # Check if we have the required client
   client = None
   if "gpt" in selected_model:
       client = clients.get("openai")
       if not client:
           st.error("🔑 OpenAI API key not found. Please set OPENAI_API_KEY in secrets.toml or environment.")
           return
   elif "claude" in selected_model:
       client = clients.get("anthropic")
       if not client:
           st.error("🔑 Anthropic API key not found. Please set CLAUDE_API_KEY in secrets.toml or environment.")
           return
   elif "gemini" in selected_model:
       client = clients.get("google")
       if not client:
           st.error("🔑 Google API key not found. Please set GEMINI_API_KEY in secrets.toml or environment.")
           return
  
   # Display current configuration
   st.sidebar.subheader("📊 Current Setup")
   st.sidebar.write(f"**Model:** {selected_model_name}")
   st.sidebar.write(f"**Memory:** {memory_type}")
  
   if st.session_state.get("urls_loaded"):
       st.sidebar.write("**URLs:** ✅ Loaded")
       if st.session_state.get("url1_content"):
           st.sidebar.write(f"• URL 1: {len(st.session_state['url1_content'])} chars")
       if st.session_state.get("url2_content"):
           st.sidebar.write(f"• URL 2: {len(st.session_state['url2_content'])} chars")
   else:
       st.sidebar.write("**URLs:** ❌ Not loaded")
  
   # Initialize chat history
   if "messages" not in st.session_state:
       url1_content = st.session_state.get("url1_content", "")
       url2_content = st.session_state.get("url2_content", "")
      
       st.session_state["messages"] = [
           {"role": "system", "content": create_system_prompt(url1_content, url2_content)}
       ]
  
   if "conversation_summary" not in st.session_state:
       st.session_state["conversation_summary"] = ""
  
   # Apply memory management
   if memory_strategy == "buffer_6":
       display_messages = manage_conversation_buffer(st.session_state["messages"], max_pairs=6)
   elif memory_strategy == "token_2000":
       display_messages, current_tokens, removed_count = manage_token_buffer(
           st.session_state["messages"], 2000, selected_model
       )
       st.sidebar.write(f"**Tokens:** {current_tokens}/2000")
       if removed_count > 0:
           st.sidebar.warning(f"Removed {removed_count} old messages")
   else:  # summary
       if len(st.session_state["messages"]) > 5:
           # Create summary and keep only recent messages
           summary = create_conversation_summary(st.session_state["messages"], client, selected_model)
           if summary:
               st.session_state["conversation_summary"] = summary
          
           # Keep system message + summary + last 2 exchanges
           recent_messages = st.session_state["messages"][-4:]  # Last 2 user-assistant pairs
           url1_content = st.session_state.get("url1_content", "")
           url2_content = st.session_state.get("url2_content", "")
          
           system_with_summary = create_system_prompt(url1_content, url2_content)
           if summary:
               system_with_summary += f"\n\nPrevious conversation summary: {summary}"
          
           display_messages = [
               {"role": "system", "content": system_with_summary}
           ] + recent_messages
       else:
           display_messages = st.session_state["messages"]
  
   # Display chat history
   for msg in display_messages:
       if msg["role"] == "user":
           st.chat_message("user").markdown(msg["content"])
       elif msg["role"] == "assistant":
           st.chat_message("assistant").markdown(msg["content"])
  
   # Show initial greeting if no conversation yet
   if len(st.session_state["messages"]) == 1:
       with st.chat_message("assistant"):
           if st.session_state.get("urls_loaded"):
               st.markdown("Hi! I've loaded the web content you provided. What would you like to know about it?")
           else:
               st.markdown("Hi! Please load some URLs in the sidebar, then ask me questions about their content.")
  
   # User input
   if prompt := st.chat_input("Ask a question about the loaded content..."):
       # Add user message
       st.session_state["messages"].append({"role": "user", "content": prompt})
       st.chat_message("user").markdown(prompt)
      
       # Generate assistant response
       with st.chat_message("assistant"):
           with st.spinner("Thinking..."):
               # Prepare messages for API call
               if memory_strategy == "buffer_6":
                   api_messages = manage_conversation_buffer(st.session_state["messages"], max_pairs=6)
               elif memory_strategy == "token_2000":
                   api_messages, _, _ = manage_token_buffer(
                       st.session_state["messages"], 2000, selected_model
                   )
               else:  # summary
                   if len(st.session_state["messages"]) > 6:
                       summary = st.session_state.get("conversation_summary", "")
                       recent_messages = st.session_state["messages"][-2:]  # Just added user message
                       url1_content = st.session_state.get("url1_content", "")
                       url2_content = st.session_state.get("url2_content", "")
                      
                       system_with_summary = create_system_prompt(url1_content, url2_content)
                       if summary:
                           system_with_summary += f"\n\nPrevious conversation summary: {summary}"
                      
                       api_messages = [
                           {"role": "system", "content": system_with_summary}
                       ] + recent_messages
                   else:
                       api_messages = st.session_state["messages"]
              
               # Get streaming response
               stream = get_llm_response(api_messages, selected_model, client, temperature)
              
               if stream:
                   response = display_streamed_response(stream, selected_model)
                   st.session_state["messages"].append({"role": "assistant", "content": response})
               else:
                   st.error("Failed to get response from the model.")
  
   # Example questions for baseball evaluation
   if url1 and "baseball" in url1.lower():
       st.sidebar.subheader("⚾ Example Baseball Questions")
       example_questions = [
           "What are the basic rules of baseball?",
           "How do you score runs in baseball?",
           "What equipment do you need to play baseball?"
       ]
      
       for question in example_questions:
           if st.sidebar.button(question, key=f"example_{hash(question)}"):
               st.session_state["messages"].append({"role": "user", "content": question})
               st.rerun()
  
   # Debug information
   with st.expander("🔍 Debug Information"):
       st.write(f"**Selected Model:** {selected_model}")
       st.write(f"**Memory Strategy:** {memory_strategy}")
       st.write(f"**Total Messages:** {len(st.session_state['messages'])}")
      
       if st.session_state.get("conversation_summary"):
           st.write("**Conversation Summary:**")
           st.write(st.session_state["conversation_summary"])


if __name__ == "__main__":
    run()

