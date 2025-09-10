import streamlit as st

# Import lab files directly (they're in the same directory)
import hw1
import hw2

# Page configuration
st.set_page_config(
    page_title="HW Manager - Samiksha Singh", 
    page_icon="📚",
    layout="wide"
)

# Wrapper functions to call the lab files
def run_hw1():
    # Since lab1.py doesn't have a main() function, we need to execute its content
    # We'll create a simple wrapper that handles the lab1 logic
    import streamlit as st
    from openai import OpenAI
    import PyPDF2
    from io import BytesIO

    def read_pdf(uploaded_file):
        """Read PDF file and extract text content"""
        try:
            uploaded_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None

    # Lab1 content
    st.title("HW1 Document QA - Samiksha Singh")
    st.write(
        "Upload a document below and ask a question about it — GPT will answer! "
        "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
        "Supported file types: .txt and .pdf"
    )

    # Try to get API key from secrets first, then fall back to text input
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
            st.success("✅ Using API key from secrets")
        else:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
    except:
        openai_api_key = st.text_input("OpenAI API Key", type="password")

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    else:
        client = OpenAI(api_key=openai_api_key)
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o-mini"]
        selected_model = st.selectbox("Select GPT Model:", model_options, index=1)

        uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))
        document = None
        
        if uploaded_file:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'txt':
                try:
                    document = uploaded_file.read().decode('utf-8')
                    st.success(f"Successfully loaded text file: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error reading text file: {str(e)}")
                    document = None
            elif file_extension == 'pdf':
                document = read_pdf(uploaded_file)
                if document:
                    st.success(f"Successfully loaded PDF file: {uploaded_file.name}")
                    st.text_area("Document Preview:", document[:500] + "..." if len(document) > 500 else document, height=150, disabled=True)

        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Can you give me a short summary?",
            disabled=not document,
        )

        if document and question:
            if st.button("Generate Answer", type="primary"):
                with st.spinner(f"Generating answer using {selected_model}..."):
                    try:
                        messages = [
                            {
                                "role": "user",
                                "content": f"Here's a document: {document} \n\n---\n\n {question}",
                            }
                        ]

                        stream = client.chat.completions.create(
                            model=selected_model,
                            messages=messages,
                            stream=True,
                        )

                        st.subheader("Answer:")
                        st.write_stream(stream)
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

    # Sidebar content
    st.sidebar.title("Model Comparison Notes")
    st.sidebar.write("""
    **Model Testing Results**:

    **GPT-3.5-turbo:**
    - Fast response time
    - Lower cost
    - Good for basic questions

    **GPT-4:**
    - Better comprehension
    - More detailed answers  
    - Higher cost

    **GPT-4-turbo:**
    - Balanced performance
    - Good speed/quality ratio

    **GPT-4o-mini:**
    - Fastest response
    - Very low cost
    - Good for simple queries
    """)

def run_hw2():
    # Call the main function from lab2
    hw2.main()

# Create navigation with dropdown structure
nav = st.navigation({
    "HW Manager": [
        st.Page(run_hw1, title="HW1 - Document QA", url_path="hw1"),
        st.Page(run_hw2, title="HW2 - URL Summarizer", url_path="hw2", default=True),
    ]
})

# Run the navigation
nav.run()