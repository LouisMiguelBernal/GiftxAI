import streamlit as st
import re
import os
import tempfile
from typing import List
from datetime import datetime
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from groq import Groq
import time

# ------------------------------
# APP CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="GiftxAI - Enterprise RAG System",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# MINIMAL THEME-AGNOSTIC CSS
# ------------------------------
st.markdown("""
    <style>
    :root {
        --accent-red: #C41E3A;
        --accent-green: #165B33;
    }
    
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
    
    /* Force uniform text */
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
    
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: var(--accent-red);
        opacity: 0.9;
    }
    
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        border-left: 4px solid var(--accent-green);
    }
    
    .stButton > button[kind="primary"] {
        background-color: var(--accent-red);
        font-weight: 600;
        border-radius: 8px;
        height: 48px;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #A01729;
    }
    
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-green);
        margin: 0.5rem 0;
    }
    
    .tips-box {
        background: linear-gradient(135deg, var(--accent-red), var(--accent-green));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 2rem;
    }
    
    .feature-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# INITIALIZE SESSION STATE
# ------------------------------
if 'groq_client' not in st.session_state or st.session_state.groq_client is None:
    try:
        st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        st.session_state.groq_client.models.list()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.session_state.groq_client = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_queries': 0,
        'total_documents': 0,
        'avg_response_time': 0,
        'total_chunks': 0
    }

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
    """Extract text from PDF with error handling"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def is_christmas_related(text: str) -> bool:
    """Check if document content is relevant to domain"""
    keywords = ['christmas', 'xmas', 'gift', 'present', 'holiday', 
                'santa', 'festive', 'celebration', 'december', 'winter',
                'toy', 'decoration', 'tree', 'wrapping', 'seasonal']
    return any(word in text.lower() for word in keywords)

def create_document_chunks(text: str, filename: str) -> List[Document]:
    """Split documents into optimized chunks for retrieval"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": filename, "chunk": i}) for i, chunk in enumerate(chunks)]

def clean_response_formatting(text: str) -> str:
    """Remove markdown formatting for uniform text display"""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'^(\s*)[-*•]\s+', r'\1BULLETPOINT ', text, flags=re.MULTILINE)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'BULLETPOINT ', '• ', text)
    text = re.sub(r':\s*•\s*', ': ', text)
    text = re.sub(r'([a-zA-Z0-9])\s+•\s+', r'\1, ', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'(\$\d+(?:\.\d{2})?)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\$\d)', r'\1 \2', text)
    text = re.sub(r'(\$\d+(?:\.\d{2})?),([a-zA-Z])', r'\1, \2', text)
    text = re.sub(r'(\$\d+(?:\.\d{2})?)\)([a-zA-Z])', r'\1) \2', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def process_documents(uploaded_files):
    """Process and index uploaded documents"""
    all_docs = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        text = extract_text_from_pdf(tmp_path)
        text = clean_extracted_text(text)
        os.unlink(tmp_path)
        
        if not text:
            st.warning(f"⚠️ No text extracted from {uploaded_file.name}")
            continue
        
        if not is_christmas_related(text):
            st.warning(f"⚠️ {uploaded_file.name} may not be domain-relevant")
        
        docs = create_document_chunks(text, uploaded_file.name)
        all_docs.extend(docs)
        st.session_state.processed_files.append(uploaded_file.name)
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_docs:
        st.error("❌ No valid documents to process")
        return None
    
    with st.spinner("Creating vector embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    if st.session_state.vectorstore is None:
        vectorstore = FAISS.from_documents(all_docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(all_docs)
        vectorstore = st.session_state.vectorstore
    
    # Update metrics
    st.session_state.metrics['total_documents'] = len(st.session_state.processed_files)
    st.session_state.metrics['total_chunks'] = len(all_docs)
    
    st.success(f"✅ Successfully indexed {len(uploaded_files)} document(s) into {len(all_docs)} chunks")
    return vectorstore

def get_relevant_context(question: str, vectorstore, k=8):
    """Intelligent retrieval with adaptive context window"""
    power_keywords = ['top', 'most', 'best', 'all', 'list', 'expensive', 
                      'cheapest', 'compare', 'ranking', 'every', 'entire']
    
    # Adaptive retrieval based on query complexity
    if any(word in question.lower() for word in power_keywords):
        k = 15
    
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return context, docs, k

def generate_answer(question: str, context: str, groq_client: Groq) -> str:
    """Generate answer using LLM with RAG context"""
    prompt = f"""Answer the question based ONLY on the context provided below about Christmas gifts.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, well-structured answer
- Use bullet points or numbered lists to organize information
- Do NOT use bold (**text**) or italic (*text*) formatting
- Write in plain text with normal spacing
- For lists/rankings, include ALL relevant items from the context
- If the context doesn't contain enough information, say so
- Cite specific details from the context when possible"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant specializing in Christmas gift recommendations. Respond in plain text only - no bold or italic formatting. Use bullet points and lists for organization. Be thorough, accurate, and cite information from the provided context."
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
        return f"Error generating response: {str(e)}"

def handle_user_input(user_question: str):
    """Handle user query with metrics tracking"""
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and process documents first!")
        return
    
    start_time = time.time()
    
    with st.spinner("🔍 Retrieving relevant information..."):
        context, source_docs, k_used = get_relevant_context(
            user_question, 
            st.session_state.vectorstore
        )
    
    with st.spinner("💭 Generating response..."):
        answer = generate_answer(user_question, context, st.session_state.groq_client)
    
    response_time = time.time() - start_time
    
    # Update metrics
    st.session_state.metrics['total_queries'] += 1
    current_avg = st.session_state.metrics['avg_response_time']
    total_queries = st.session_state.metrics['total_queries']
    st.session_state.metrics['avg_response_time'] = (
        (current_avg * (total_queries - 1) + response_time) / total_queries
    )
    
    st.session_state.chat_history.append({
        'question': user_question,
        'answer': answer,
        'sources': source_docs,
        'response_time': response_time,
        'k_used': k_used,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# ------------------------------
# APP HEADER
# ------------------------------
st.markdown('<div class="app-title">🎁 GiftxAI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Enterprise RAG System for Intelligent Gift Recommendations</div>', unsafe_allow_html=True)

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("📄 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload PDF catalogs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents containing gift information"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary"):
            st.session_state.vectorstore = process_documents(uploaded_files)
    
    if st.session_state.processed_files:
        st.divider()
        st.subheader("✅ Indexed Documents")
        for f in st.session_state.processed_files:
            st.markdown(f"📎 {f}")
    
    st.divider()
    
    # System Metrics
    st.subheader("📊 System Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", st.session_state.metrics['total_documents'])
        st.metric("Queries", st.session_state.metrics['total_queries'])
    with col2:
        st.metric("Chunks", st.session_state.metrics['total_chunks'])
        if st.session_state.metrics['avg_response_time'] > 0:
            st.metric("Avg Time", f"{st.session_state.metrics['avg_response_time']:.2f}s")
    
    st.divider()
    
    # Actions
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🔄 Reset System"):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.processed_files = []
        st.session_state.metrics = {
            'total_queries': 0,
            'total_documents': 0,
            'avg_response_time': 0,
            'total_chunks': 0
        }
        st.rerun()
    
    st.divider()
    
    # Technical Info
    with st.expander("⚙️ System Info"):
        st.markdown("""
        **RAG Architecture:**
        - Embedding: all-MiniLM-L6-v2
        - Vector Store: FAISS
        - LLM: Llama 3.3 70B
        - Chunk Size: 800 tokens
        - Retrieval: Adaptive (8-15 chunks)
        """)

# ------------------------------
# CHAT INTERFACE
# ------------------------------
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(message['question'])
    
    with st.chat_message("assistant"):
        st.write(message['answer'])
        
        # Metadata
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.caption(f"⏱️ {message.get('response_time', 0):.2f}s")
        with col2:
            st.caption(f"📚 {message.get('k_used', 0)} chunks")
        with col3:
            st.caption(f"🕐 {message.get('timestamp', 'N/A')}")
        
        # Sources
        if message.get('sources'):
            with st.expander("📚 View Retrieved Sources"):
                for j, doc in enumerate(message['sources'][:3]):
                    st.markdown(f"**Source {j+1}** - {doc.metadata.get('source','Unknown')}")
                    st.text(doc.page_content[:300]+"...")
                    st.divider()

user_question = st.chat_input("Ask about gifts, pricing, recommendations, or comparisons...")
if user_question:
    handle_user_input(user_question)
    st.rerun()

# ------------------------------
# INFO SECTION
# ------------------------------
if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="tips-box">
        <strong>🎯 Enterprise RAG System Features</strong><br><br>
        ✅ Intelligent document indexing & retrieval<br>
        ✅ Adaptive context window (8-15 chunks)<br>
        ✅ Real-time performance metrics<br>
        ✅ Source attribution & transparency<br>
        ✅ Scalable vector storage with FAISS<br>
        ✅ Production-ready LLM integration
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Features showcase
    st.subheader("🌟 Key Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Analytics")
        st.markdown("- Query tracking\n- Response time monitoring\n- Document metrics\n- Usage statistics")
    
    with col2:
        st.markdown("### 🎯 Smart Retrieval")
        st.markdown("- Semantic search\n- Adaptive context\n- Relevance ranking\n- Multi-document support")
    
    with col3:
        st.markdown("### 🔒 Enterprise Ready")
        st.markdown("- Error handling\n- Source verification\n- Clean formatting\n- Scalable architecture")
