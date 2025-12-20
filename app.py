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
        text-align: center;
        margin-bottom: 8px;
    }
    
    .app-title .gift-text {
        color: var(--accent-red);
    }
    
    .app-title .xai-text {
        color: var(--accent-green);
    }
    
    .app-subtitle {
        text-align: center;
        color: #FFB703;
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
    
    .features-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .feature-item {
        flex: 1;
        text-align: center;
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
                      'cheapest', 'compare', 'ranking', 'every', 'entire',
                      'order', 'sorted', 'ranked', 'highest', 'lowest']
    
    # Adaptive retrieval based on query complexity
    if any(word in question.lower() for word in power_keywords):
        k = 20
    
    # For very specific ranking queries (top 10, top 5, etc.)
    top_n_match = re.search(r'top\s+(\d+)', question.lower())
    if top_n_match:
        requested_count = int(top_n_match.group(1))
        k = max(20, requested_count * 2)
    
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return context, docs, k

def validate_and_reformat_response(initial_response: str, user_question: str, groq_client: Groq) -> str:
    """
    Validate and ensure the response is concise, direct, and matches the exact request.
    """
    validation_prompt = f"""You are a precision validator. Review this response and ensure it's CONCISE and ACCURATE.

USER QUESTION: {user_question}

ORIGINAL RESPONSE:
{initial_response}

VALIDATION RULES:

1. COUNT VERIFICATION:
   - If user asks "top 10", response must have EXACTLY 10 items
   - If user asks "top 5", response must have EXACTLY 5 items
   - Remove any extra items beyond the requested count

2. STRUCTURE:
   - Keep introduction to ONE sentence maximum (e.g., "Here are the top 10 most expensive items:")
   - Go DIRECTLY to the numbered list
   - NO additional commentary before or between the list items

3. FORMAT:
   - Use format: "1. Item Name - Price"
   - Remove ALL bold, italic, markdown formatting
   - Consistent spacing and punctuation

4. ACCURACY:
   - Verify items are sorted correctly (highest to lowest for "most expensive")
   - Ensure prices are accurate and properly formatted

OUTPUT REQUIREMENTS:
- Provide ONLY the corrected response
- One-sentence intro + numbered list + nothing else
- No explanations about what you changed

Validate and output now:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a precision editor. Your job is to make responses concise and direct.

Rules:
- Keep introductions to ONE sentence maximum
- Ensure exact count matches user request
- Remove any rambling or extra content
- Maintain only the essential information
- Enforce plain text formatting"""
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            temperature=0.05,
            max_tokens=2000,
            top_p=0.9,
            stream=False
        )
        validated_response = response.choices[0].message.content.strip()
        validated_response = clean_response_formatting(validated_response)
        return validated_response
    except Exception as e:
        return clean_response_formatting(initial_response)

def generate_answer(question: str, context: str, groq_client: Groq) -> str:
    """Generate answer using LLM with RAG context - optimized for concise ranking responses"""
    
    # Detect if this is a ranking/listing query
    is_ranking = any(word in question.lower() for word in ['top', 'most expensive', 'cheapest', 'ranking', 'in order', 'sorted'])
    
    # Extract exact number requested
    requested_count = None
    count_match = re.search(r'top\s+(\d+)', question.lower())
    if count_match:
        requested_count = int(count_match.group(1))
    
    # Streamlined prompt for ranking queries
    if is_ranking and requested_count:
        prompt = f"""Answer this question using ONLY the context provided.

Context:
{context}

Question: {question}

CRITICAL INSTRUCTIONS FOR RANKING/LISTING:

1. EXTRACT & SORT:
   - Find ALL items with prices in the context
   - Sort numerically by price (highest to lowest for "most expensive", lowest to highest for "cheapest")
   - Select EXACTLY {requested_count} items - no more, no less

2. RESPONSE FORMAT (FOLLOW EXACTLY):
   - Line 1: One brief sentence intro (e.g., "Here are the top {requested_count} most expensive items:")
   - Lines 2-{requested_count+1}: The ranked list using "1. Item Name - Price" format
   - NO additional text, commentary, or explanations

3. FORMATTING:
   - Plain text only (NO bold, italic, or markdown)
   - Format: "1. Item Name - Price"
   - Prices: dollar sign + amount (e.g., 649.99)

4. ACCURACY:
   - Double-check sorting is numerical (not alphabetical)
   - Verify count is exactly {requested_count}
   - Use only information from the context

Think step-by-step:
Step 1: Extract all items with prices
Step 2: Sort them numerically
Step 3: Take the top {requested_count}
Step 4: Format as specified

Now provide your answer:"""
    else:
        # Standard prompt for non-ranking queries
        prompt = f"""Answer this question based strictly on the context provided.

Context:
{context}

Question: {question}

INSTRUCTIONS:
1. Be concise and direct - answer the question clearly
2. Use bullet points or numbered lists only when helpful
3. Cite specific details and prices when available
4. Use plain text formatting (NO bold, italic, or markdown)
5. If context lacks information, state this clearly

Provide your answer:"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise, concise AI assistant for gift recommendations.

For ranking queries (top 10, most expensive, etc.):
- Be EXTREMELY concise
- One-sentence intro, then go STRAIGHT to the numbered list
- No fluff, no extra explanations
- Count accuracy is CRITICAL

For general queries:
- Clear, structured answers
- Use context information accurately
- Be helpful but concise

Always use plain text - NO markdown formatting."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000,
            top_p=0.95,
            stream=False
        )
        initial_answer = response.choices[0].message.content.strip()
        
        # Validation step with question context
        validated_answer = validate_and_reformat_response(initial_answer, question, groq_client)
        
        return validated_answer
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
st.markdown('<div class="app-title">🎁 <span class="gift-text">Gift</span><span class="xai-text">xAI</span></div>', unsafe_allow_html=True)
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
        - Retrieval: Adaptive (8-20 chunks)
        - Validation: 2-stage concise response checking
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
        ✅ Adaptive context window (8-20 chunks)<br>
        ✅ Concise ranking responses (no extra items)<br>
        ✅ Real-time performance metrics<br>
        ✅ Source attribution & transparency<br>
        ✅ Scalable vector storage with FAISS<br>
        ✅ Production-ready LLM integration<br>
        ✅ 2-stage response validation for accuracy
        </div>
        """,
        unsafe_allow_html=True
    )
