import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import time
import os

# Optional provider SDKs
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import cohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

def read_url_content(url):
    """Read content from a URL and return the text."""
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
        
        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def get_openai_summary(content, summary_type, language, use_advanced=False, api_key=None):
    """Generate summary using OpenAI."""
    try:
        client = OpenAI(api_key=api_key)
        model = "gpt-4" if use_advanced else "gpt-3.5-turbo"
        
        prompt = f"""
        Please provide a {summary_type} summary of the following content in {language}.
        Make sure your response is entirely in {language}.
        
        Content: {content[:4000]}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI: {str(e)}"

def get_claude_summary(content, summary_type, language, use_advanced=False, api_key=None):
    """Generate summary using Claude."""
    if not HAS_ANTHROPIC:
        return "Error: Anthropic library not installed. Run: pip install anthropic"
    
    if not api_key:
        return "Error: Claude API key not provided"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        model = "claude-3-opus-20240229" if use_advanced else "claude-3-haiku-20240307"
        
        prompt = f"""
        Please provide a {summary_type} summary of the following content in {language}.
        Make sure your response is entirely in {language}.
        
        Content: {content[:4000]}
        """
        
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    except Exception as e:
        return f"Error with Claude: {str(e)}"

def get_gemini_summary(content, summary_type, language, use_advanced=False, api_key=None):
    """Generate summary using Gemini."""
    if not HAS_GEMINI:
        return "Error: Google Generative AI library not installed. Run: pip install google-generativeai"
    
    if not api_key:
        return "Error: Gemini API key not provided"
    
    try:
        genai.configure(api_key=api_key)
        model_name = "gemini-pro"  # Adjust based on available models
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Please provide a {summary_type} summary of the following content in {language}.
        Make sure your response is entirely in {language}.
        
        Content: {content[:4000]}
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error with Gemini: {str(e)}"

def get_cohere_summary(content, summary_type, language, use_advanced=False, api_key=None):
    """Generate summary using Cohere."""
    if not HAS_COHERE:
        return "Error: Cohere library not installed. Run: pip install cohere"
    
    if not api_key:
        return "Error: Cohere API key not provided"
    
    try:
        co = cohere.Client(api_key)
        model = "command" if use_advanced else "command-light"
        
        prompt = f"""
        Please provide a {summary_type} summary of the following content in {language}.
        Make sure your response is entirely in {language}.
        
        Content: {content[:4000]}
        """
        
        response = co.generate(
            model=model,
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )
        
        return response.generations[0].text
    except Exception as e:
        return f"Error with Cohere: {str(e)}"

def main():
    st.title("🌐 HW2 - URL Summarizer")
    st.write("Enter a URL to get a summary using different LLMs")
    
    # URL input at the top
    url = st.text_input("Enter URL:", placeholder="https://example.com")
    
    # Sidebar options
    st.sidebar.header("Summary Options")
    
    # Summary type selection
    summary_types = [
        "Brief Summary",
        "Detailed Summary", 
        "Key Points",
        "Executive Summary",
        "Technical Summary"
    ]
    summary_type = st.sidebar.selectbox("Select Summary Type:", summary_types)
    
    # Language selection (at least 3 options as required)
    languages = {
        "English": "English",
        "Spanish": "Spanish (Español)", 
        "French": "French (Français)",
        "German": "German (Deutsch)",
        "Chinese": "Chinese (中文)"
    }
    selected_language = st.sidebar.selectbox("Output Language:", list(languages.keys()))
    language = languages[selected_language]
    
    # LLM selection (Added Cohere as 4th LLM)
    llm_options = ["OpenAI", "Claude", "Gemini", "Cohere"]
    selected_llm = st.sidebar.selectbox("Select LLM:", llm_options)
    
    # Advanced model checkbox
    use_advanced = st.sidebar.checkbox("Use Advanced Model", value=False)
    
    # Model information
    model_info = {
        "OpenAI": {
            "advanced": "GPT-4",
            "standard": "GPT-3.5-turbo"
        },
        "Claude": {
            "advanced": "Claude-3 Opus",
            "standard": "Claude-3 Haiku"
        },
        "Gemini": {
            "advanced": "Gemini Pro",
            "standard": "Gemini Pro"
        },
        "Cohere": {
            "advanced": "Command",
            "standard": "Command-Light"
        }
    }
    
    current_model = model_info[selected_llm]["advanced" if use_advanced else "standard"]
    st.sidebar.info(f"Using: {current_model}")
    
    # API Key Status (using secrets only)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 API Key Status")
    
    # CORRECTED: Get API keys from secrets using KEY NAMES, not actual keys
    try:
        openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
        claude_api_key = st.secrets.get("CLAUDE_API_KEY", None)
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
        cohere_api_key = st.secrets.get("COHERE_API_KEY", None)
    except:
        openai_api_key = claude_api_key = gemini_api_key = cohere_api_key = None
    
    # Show status for selected LLM
    if selected_llm == "OpenAI":
        if openai_api_key:
            st.sidebar.success("✅ OpenAI API key configured")
        else:
            st.sidebar.error("❌ OpenAI API key not found in secrets")
            st.sidebar.info("Add OPENAI_API_KEY to app secrets")
    
    elif selected_llm == "Claude":
        if claude_api_key:
            st.sidebar.success("✅ Claude API key configured")
        else:
            st.sidebar.error("❌ Claude API key not found in secrets")
            st.sidebar.info("Add ANTHROPIC_API_KEY to app secrets")
        
        if HAS_ANTHROPIC:
            st.sidebar.success("✅ Anthropic SDK installed")
        else:
            st.sidebar.error("❌ Anthropic SDK not installed")
            st.sidebar.code("pip install anthropic")
    
    elif selected_llm == "Gemini":
        if gemini_api_key:
            st.sidebar.success("✅ Gemini API key configured")
        else:
            st.sidebar.error("❌ Gemini API key not found in secrets")
            st.sidebar.info("Add GEMINI_API_KEY to app secrets")
            
        if HAS_GEMINI:
            st.sidebar.success("✅ Google AI SDK installed")
        else:
            st.sidebar.error("❌ Google AI SDK not installed")
            st.sidebar.code("pip install google-generativeai")
    
    elif selected_llm == "Cohere":
        if cohere_api_key:
            st.sidebar.success("✅ Cohere API key configured")
        else:
            st.sidebar.error("❌ Cohere API key not found in secrets")
            st.sidebar.info("Add COHERE_API_KEY to app secrets")
            
        if HAS_COHERE:
            st.sidebar.success("✅ Cohere SDK installed")
        else:
            st.sidebar.error("❌ Cohere SDK not installed")
            st.sidebar.code("pip install cohere")
    
    # Process button
    if st.button("Generate Summary", type="primary"):
        if not url:
            st.error("Please enter a URL")
            return
        
        # Check if API key is available for selected LLM
        current_api_key = None
        if selected_llm == "OpenAI":
            current_api_key = openai_api_key
        elif selected_llm == "Claude":
            current_api_key = claude_api_key
        elif selected_llm == "Gemini":
            current_api_key = gemini_api_key
        elif selected_llm == "Cohere":
            current_api_key = cohere_api_key
        
        if not current_api_key:
            st.error(f"❌ {selected_llm} API key not configured in secrets. Please add the required API key to your app secrets.")
            return
        
        # Read URL content
        with st.spinner("Reading URL content..."):
            content = read_url_content(url)
        
        if not content:
            st.error("Could not read content from the URL")
            return
        
        if len(content) < 100:
            st.warning("The content seems very short. The URL might not contain much text.")
        
        # Display original content preview
        with st.expander("Preview of Original Content"):
            st.text(content[:500] + "..." if len(content) > 500 else content)
        
        # Generate summary
        with st.spinner(f"Generating summary using {selected_llm}..."):
            start_time = time.time()
            
            if selected_llm == "OpenAI":
                summary = get_openai_summary(content, summary_type, language, use_advanced, current_api_key)
            elif selected_llm == "Claude":
                summary = get_claude_summary(content, summary_type, language, use_advanced, current_api_key)
            elif selected_llm == "Gemini":
                summary = get_gemini_summary(content, summary_type, language, use_advanced, current_api_key)
            elif selected_llm == "Cohere":
                summary = get_cohere_summary(content, summary_type, language, use_advanced, current_api_key)
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
        
        # Display results
        st.subheader("Summary Results")
        st.write(f"**URL:** {url}")
        st.write(f"**Summary Type:** {summary_type}")
        st.write(f"**Language:** {selected_language}")
        st.write(f"**LLM Used:** {selected_llm} ({current_model})")
        st.write(f"**Processing Time:** {processing_time}s")
        
        st.markdown("### Summary:")
        st.write(summary)
        

   
if __name__ == "__main__":
    main()

