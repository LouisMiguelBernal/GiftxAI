import streamlit as st
import re
import os
import tempfile
from typing import List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from groq import Groq

# ------------------------------
# APP CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="GiftxAI",
    page_icon="🎁",
    layout="wide"
)

# ------------------------------
# DARK MODE CHRISTMAS-THEMED CSS
# ------------------------------
st.markdown("""
    <style>
    /* Dark Mode Christmas Theme */
    :root {
        --christmas-red: #E63946;
        --christmas-green: #2D6A4F;
        --christmas-gold: #FFB703;
        --dark-bg: #1E1E1E;
        --dark-card: #2D2D2D;
        --dark-text: #E0E0E0;
    }
    
    /* Main Background */
    .stApp {
        background-color: var(--dark-bg);
        color: var(--dark-text);
    }
    
    /* App Title */
    .app-title {
        font-size: 48px;
        font-weight: 700;
        color: var(--christmas-red);
        text-align: center;
        margin-bottom: 8px;
    }
    
    .app-subtitle {
        text-align: center;
        color: var(--christmas-green);
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 24px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--christmas-green) !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #252525;
    }
    
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--christmas-red) !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--dark-text) !important;
        font-weight: 400 !important;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: var(--christmas-red);
        color: white;
        font-weight: 600;
        height: 48px;
        border-radius: 8px;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #D62839;
        box-shadow: 0 4px 8px rgba(230, 57, 70, 0.4);
    }
    
    .stButton > button {
        background-color: var(--dark-card);
        color: var(--christmas-green);
        font-weight: 600;
        height: 45px;
        border-radius: 8px;
        border: 2px solid var(--christmas-green);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: var(--christmas-green);
        color: white;
    }
    
    /* Chat Messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* User Messages */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: var(--christmas-red);
    }
    
    [data-testid="stChatMessage"][data-testid*="user"] p,
    [data-testid="stChatMessage"][data-testid*="user"] span,
    [data-testid="stChatMessage"][data-testid*="user"] div {
        color: white !important;
        font-weight: 400 !important;
        font-style: normal !important;
    }
    
    /* Assistant Messages - FORCE UNIFORM TEXT */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: var(--dark-card);
        border-left: 4px solid var(--christmas-green);
    }
    
    [data-testid="stChatMessage"][data-testid*="assistant"] p,
    [data-testid="stChatMessage"][data-testid*="assistant"] span,
    [data-testid="stChatMessage"][data-testid*="assistant"] div,
    [data-testid="stChatMessage"][data-testid*="assistant"] li,
    [data-testid="stChatMessage"][data-testid*="assistant"] ul,
    [data-testid="stChatMessage"][data-testid*="assistant"] ol,
    [data-testid="stChatMessage"][data-testid*="assistant"] strong,
    [data-testid="stChatMessage"][data-testid*="assistant"] em,
    [data-testid="stChatMessage"][data-testid*="assistant"] b,
    [data-testid="stChatMessage"][data-testid*="assistant"] i {
        color: var(--dark-text) !important;
        font-weight: 400 !important;
        font-style: normal !important;
        line-height: 1.6;
    }
    
    /* Force all text elements to be normal */
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b {
        font-weight: 400 !important;
    }
    
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] i {
        font-style: normal !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: var(--dark-card);
        border: 2px dashed #404040;
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"] label {
        color: var(--dark-text) !important;
        font-weight: 400 !important;
    }
    
    /* Success/Warning/Error Messages */
    .stSuccess {
        background-color: var(--christmas-green);
        color: white;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 400;
    }
    
    .stWarning {
        background-color: var(--christmas-gold);
        color: #1E1E1E;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 400;
    }
    
    .stError {
        background-color: var(--christmas-red);
        color: white;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 400;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--dark-card);
        border-radius: 6px;
        font-weight: 600;
        color: var(--christmas-green);
    }
    
    .streamlit-expanderContent {
        background-color: #2A2A2A;
        color: var(--dark-text);
    }
    
    /* Chat Input */
    .stChatInput input {
        background-color: var(--dark-card);
        color: var(--dark-text);
        border: 1px solid #404040;
        border-radius: 8px;
    }
    
    .stChatInput input::placeholder {
        color: #808080;
    }
    
    /* Tips Container */
    .tips-box {
        background: linear-gradient(135deg, var(--christmas-red), var(--christmas-green));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
    }
    
    .tips-box strong {
        font-size: 20px;
        font-weight: 600;
        display: block;
        margin-bottom: 0.75rem;
    }
    
    .tips-box p {
        font-weight: 400;
    }
    
    /* Processed Files */
    .processed-file {
        background-color: var(--dark-card);
        padding: 0.6rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        color: var(--dark-text);
        font-size: 14px;
        font-weight: 400;
        border-left: 3px solid var(--christmas-green);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--christmas-red) !important;
    }
    
    /* Text elements uniform weight */
    p, span, div, li {
        font-weight: 400 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# GROQ API CLIENT
# ------------------------------
if 'groq_client' not in st.session_state or st.session_state.groq_client is None:
    try:
        st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        st.session_state.groq_client.models.list()
        st.success("✅ Connected to Groq API successfully.")
    except Exception as e:
        st.error(f"❌ Error connecting to Groq: {str(e)}")
        st.session_state.groq_client = None

# ------------------------------
# INITIALIZE SESSION STATE
# ------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def clean_extracted_text(text: str) -> str:
    """Clean extracted text while preserving important spacing"""
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\$(\d+(?:\.\d{2})?)', r'$\1 ', text)
    text = re.sub(r'([a-zA-Z])(\$\d)', r'\1 \2', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = re.sub(r' +', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def is_christmas_related(text: str) -> bool:
    keywords = ['christmas', 'xmas', 'gift', 'present', 'holiday', 
                'santa', 'festive', 'celebration', 'december', 'winter',
                'toy', 'decoration', 'tree', 'wrapping', 'seasonal']
    return any(word in text.lower() for word in keywords)

def create_document_chunks(text: str, filename: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": filename, "chunk": i}) for i, chunk in enumerate(chunks)]

def clean_response_formatting(text: str) -> str:
    """Remove ALL markdown formatting to ensure completely uniform text"""
    # Remove bold markers (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # Protect bullet points at the start of lines
    text = re.sub(r'^(\s*)[-*•]\s+', r'\1BULLETPOINT ', text, flags=re.MULTILINE)
    
    # Remove italic markers
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Restore bullet points
    text = re.sub(r'BULLETPOINT ', '• ', text)
    
    # Remove mid-sentence bullets
    text = re.sub(r':\s*•\s*', ': ', text)
    text = re.sub(r'([a-zA-Z0-9])\s+•\s+', r'\1, ', text)
    
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove strikethrough, code
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Fix dollar sign spacing
    text = re.sub(r'(\$\d+(?:\.\d{2})?)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\$\d)', r'\1 \2', text)
    text = re.sub(r'(\$\d+(?:\.\d{2})?),([a-zA-Z])', r'\1, \2', text)
    text = re.sub(r'(\$\d+(?:\.\d{2})?)\)([a-zA-Z])', r'\1) \2', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def process_documents(uploaded_files):
    all_docs = []
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            text = extract_text_from_pdf(tmp_path)
            text = clean_extracted_text(text)
            os.unlink(tmp_path)
            if not text:
                st.warning(f"No text extracted from {uploaded_file.name}")
                continue
            if not is_christmas_related(text):
                st.warning(f"⚠️ {uploaded_file.name} may not be Christmas-related. Still processing.")
            docs = create_document_chunks(text, uploaded_file.name)
            all_docs.extend(docs)
            st.session_state.processed_files.append(uploaded_file.name)
    if not all_docs:
        st.error("No valid documents to process")
        return None
    with st.spinner("Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if st.session_state.vectorstore is None:
        vectorstore = FAISS.from_documents(all_docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(all_docs)
        vectorstore = st.session_state.vectorstore
    st.success(f"✅ Processed {len(uploaded_files)} document(s) into {len(all_docs)} chunks")
    return vectorstore

def get_relevant_context(question: str, vectorstore, k=8):
    """Get relevant context with intelligent k value based on query type"""
    if any(word in question.lower() for word in ['top', 'most', 'best', 'all', 'list', 'expensive', 'cheapest', 'compare', 'ranking', 'every', 'entire']):
        k = 15
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs

def generate_answer(question: str, context: str, groq_client: Groq) -> str:
    prompt = f"""Answer the question based ONLY on the context provided below about Christmas gifts.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, well-structured answer
- You can use bullet points or numbered lists if it helps organize the information
- Do NOT use bold (**text**) or italic (*text*) formatting
- Use normal spacing and line breaks for readability
- Write naturally and clearly in plain text only
- When listing items (like "top 10 most expensive"), make sure to include ALL relevant items from the context, not just a few
- Double-check that you've captured all the data points requested"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in Christmas gift recommendations. Respond clearly and naturally in plain text only. You can use bullet points, numbered lists, and normal formatting to organize information. However, do NOT use bold or italic text formatting. Keep all text in regular font weight. When asked for lists or rankings, be thorough and include ALL relevant items from the context provided."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1500,
            top_p=1,
            stream=False
        )
        answer = response.choices[0].message.content.strip()
        answer = clean_response_formatting(answer)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

def handle_user_input(user_question: str):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process documents first!")
        return
    with st.spinner("Thinking..."):
        context, source_docs = get_relevant_context(user_question, st.session_state.vectorstore)
        answer = generate_answer(user_question, context, st.session_state.groq_client)
        st.session_state.chat_history.append({
            'question': user_question,
            'answer': answer,
            'sources': source_docs
        })

# ------------------------------
# APP TITLE
# ------------------------------
st.markdown('<div class="app-title">🎁 GiftxAI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">AI-Powered Christmas Gift Recommendations</div>', unsafe_allow_html=True)

# ------------------------------
# SIDEBAR: PDF UPLOAD & PROCESS
# ------------------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files about Christmas gifts",
        type=['pdf'],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            st.session_state.vectorstore = process_documents(uploaded_files)
    
    if st.session_state.processed_files:
        st.header("✅ Processed Files")
        for f in st.session_state.processed_files:
            st.markdown(f'<div class="processed-file">📎 {f}</div>', unsafe_allow_html=True)
    
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

# ------------------------------
# CHAT INTERFACE
# ------------------------------
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message['question'])
        with st.chat_message("assistant"):
            st.write(message['answer'])
            if message.get('sources'):
                with st.expander("📚 View Sources"):
                    for j, doc in enumerate(message['sources'][:3]):
                        st.markdown(f"**Source {j+1}** ({doc.metadata.get('source','Unknown')})")
                        st.text(doc.page_content[:300]+"...")

user_question = st.chat_input("Ask a question about the Christmas gift documents...")
if user_question:
    handle_user_input(user_question)
    st.rerun()

# ------------------------------
# FOOTER: Tips
# ------------------------------
if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="tips-box">
        <strong>💡 How to Use</strong>
        <p>Upload PDF documents with Christmas gifts info • Ask questions about gifts, pricing, or comparisons • Get AI-powered recommendations based on your documents</p>
        </div>
        """,
        unsafe_allow_html=True
    )
