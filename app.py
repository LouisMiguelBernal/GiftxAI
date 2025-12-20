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
# MINIMAL THEME-AGNOSTIC CSS (Works with Streamlit's Dark/Light Toggle)
# ------------------------------
st.markdown("""
    <style>
    /* Christmas Accent Colors - Work in both themes */
    :root {
        --accent-red: #C41E3A;
        --accent-green: #165B33;
    }
    
    /* App Title */
    .app-title {
        font-size: 48px;
        font-weight: 700;
        color: var(--accent-red);
        text-align: center;
        margin-bottom: 8px;
    }
    
    .app-subtitle {
        text-align: center;
        color: var(--accent-green);
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 24px;
    }
    
    /* CRITICAL: Force uniform text in ALL chat messages */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] b,
    [data-testid="stChatMessage"] i {
        font-weight: 400 !important;
        font-style: normal !important;
    }
    
    /* User message accent */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: var(--accent-red);
        opacity: 0.9;
    }
    
    /* Assistant message accent */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        border-left: 4px solid var(--accent-green);
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background-color: var(--accent-red);
        font-weight: 600;
        border-radius: 8px;
        height: 48px;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #A01729;
    }
    
    /* Processed files display */
    .processed-file {
        padding: 0.6rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--accent-green);
        font-weight: 400;
    }
    
    /* Tips box */
    .tips-box {
        background: linear-gradient(135deg, var(--accent-red), var(--accent-green));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
    }
    
    .tips-box strong {
        font-weight: 600;
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
        st.success("✅ Connected to Groq API")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
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
    # Remove bold
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # Protect bullet points
    text = re.sub(r'^(\s*)[-*•]\s+', r'\1BULLETPOINT ', text, flags=re.MULTILINE)
    
    # Remove italic
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Restore bullets
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
    
    # Fix dollar spacing
    text = re.sub(r'(\$\d+(?:\.\d{2})?)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\$\d)', r'\1 \2', text)
    text = re.sub(r'(\$\d+(?:\.\d{2})?),([a-zA-Z])', r'\1, \2', text)
    text = re.sub(r'(\$\d+(?:\.\d{2})?)\)([a-zA-Z])', r'\1) \2', text)
    
    # Clean whitespace
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
    """Intelligent retrieval based on query type"""
    # Power queries get more context
    power_keywords = ['top', 'most', 'best', 'all', 'list', 'expensive', 
                      'cheapest', 'compare', 'ranking', 'every', 'entire']
    if any(word in question.lower() for word in power_keywords):
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
- You can use bullet points or numbered lists to organize information
- Do NOT use bold (**text**) or italic (*text*) formatting
- Write in plain text with normal spacing
- For lists/rankings, include ALL relevant items from the context
- Double-check completeness of your answer"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Christmas gift recommendation assistant. Respond in plain text only - no bold or italic formatting. Use bullet points and lists for organization. Be thorough and include all relevant information from the context."
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
# APP HEADER
# ------------------------------
st.markdown('<div class="app-title">🎁 GiftxAI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">AI-Powered Christmas Gift Recommendations</div>', unsafe_allow_html=True)

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files with gift information",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            st.session_state.vectorstore = process_documents(uploaded_files)
    
    if st.session_state.processed_files:
        st.subheader("✅ Processed Files")
        for f in st.session_state.processed_files:
            st.markdown(f'<div class="processed-file">📎 {f}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Stats
    if st.session_state.vectorstore:
        st.divider()
        st.caption(f"💬 Messages: {len(st.session_state.chat_history)}")
        st.caption(f"📚 Files: {len(st.session_state.processed_files)}")

# ------------------------------
# CHAT INTERFACE
# ------------------------------
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

user_question = st.chat_input("Ask about gifts, pricing, recommendations...")
if user_question:
    handle_user_input(user_question)
    st.rerun()

# ------------------------------
# TIPS (First Time)
# ------------------------------
if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="tips-box">
        <strong>💡 Quick Start Guide</strong><br><br>
        1. Upload PDF catalogs with Christmas gift information<br>
        2. Ask questions about products, prices, or get recommendations<br>
        3. The AI retrieves relevant info and provides accurate answers
        </div>
        """,
        unsafe_allow_html=True
    )
