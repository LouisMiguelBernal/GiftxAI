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
    # Increase retrieval for ranking/comparison queries
    if any(word in question.lower() for word in power_keywords):
        k = 30  # Increased from 20 to 30 for better coverage
    
    # For very specific ranking queries (top 10, top 5, etc.)
    import re
    top_n_match = re.search(r'top\s+(\d+)', question.lower())
    if top_n_match:
        requested_count = int(top_n_match.group(1))
        # Retrieve significantly more chunks to ensure we have enough items
        k = max(40, requested_count * 4)  # 4x multiplier instead of 2x
    
    docs = vectorstore.similarity_search(question, k=k)
    
    # Deduplicate and consolidate information by item name
    seen_items = {}
    unique_docs = []
    
    for doc in docs:
        content = doc.page_content
        # Extract potential item names (simple heuristic)
        lines = content.split('\n')
        item_key = lines[0][:50] if lines else content[:50]
        
        if item_key not in seen_items:
            seen_items[item_key] = doc
            unique_docs.append(doc)
    
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    
    return context, unique_docs, k

def validate_and_reformat_response(initial_response: str, groq_client: Groq, iteration: int = 1) -> str:
    """
    Triple-check validation system with progressively stricter checking.
    Each iteration verifies accuracy, formatting, and numerical ordering.
    """
    validation_prompt = f"""You are a quality assurance validator (ITERATION {iteration}/3). Review and reformat the following response.

ORIGINAL RESPONSE:
{initial_response}

YOUR VALIDATION TASKS:
1. VERIFY ACCURACY:
   - Check if ALL items are extracted from context
   - Ensure prices are correctly associated with items
   - Confirm lists are in STRICT NUMERICAL ORDER (highest to lowest for "expensive", lowest to highest for "cheapest")
   - Verify the count matches EXACTLY (e.g., "top 10" must have EXACTLY 10 items, no more, no less)
   - Check for duplicate items (remove duplicates, keep highest price if ambiguous)

2. VERIFY NUMERICAL SORTING:
   - For "most expensive" or "top" queries: MUST be sorted from HIGHEST to LOWEST price
   - For "cheapest" or "lowest" queries: MUST be sorted from LOWEST to HIGHEST price
   - Double-check EVERY single number is in correct order
   - If ANY item is out of order, RE-SORT the entire list

3. ENFORCE STRICT FORMATTING:
   - Remove ALL bold text (no ** or __)
   - Remove ALL italic text (no * or _)
   - Remove ALL markdown and special formatting
   - Use numbered lists (1. 2. 3.) for rankings
   - Format prices consistently: $amount (e.g., $649.99)
   - Maintain uniform spacing

4. ENSURE CLARITY:
   - Keep logical organization and structure
   - Preserve all factual information exactly
   - Use clean, readable plain text formatting
   - Remove any redundant or confusing elements

5. CRITICAL VERIFICATION (ITERATION {iteration}):
   {"- FIRST PASS: Extract all items and prices, verify completeness" if iteration == 1 else ""}
   {"- SECOND PASS: Verify numerical ordering is PERFECT, check for any sorting errors" if iteration == 2 else ""}
   {"- FINAL PASS: Triple-check sorting, counts, and formatting are all correct" if iteration == 3 else ""}

OUTPUT REQUIREMENTS:
- Provide ONLY the validated, reformatted response
- No explanations or meta-commentary
- Plain text with consistent formatting throughout
- PERFECT numerical ordering

Validate and output the corrected response now."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a precision quality assurance validator (Pass {iteration}/3). 

Your responsibilities:
1. Verify accuracy and completeness of information
2. Ensure PERFECT numerical sorting in rankings (this is CRITICAL)
3. Enforce strict plain text formatting (no markdown)
4. Maintain data integrity while improving clarity
5. Format prices consistently throughout
6. Verify exact counts match requirements

You have a keen eye for detail and catch even subtle sorting errors. You are thorough and methodical."""
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            temperature=0.01,  # Extremely low temperature for maximum accuracy
            max_tokens=2500,
            top_p=0.85,
            stream=False
        )
        validated_response = response.choices[0].message.content.strip()
        validated_response = clean_response_formatting(validated_response)
        return validated_response
    except Exception as e:
        return clean_response_formatting(initial_response)

def generate_answer(question: str, context: str, groq_client: Groq) -> str:
    """Generate enterprise-grade answer using strict RAG discipline with triple validation"""

    master_prompt = f"""
You are an enterprise-grade Retrieval-Augmented Generation (RAG) answer engine with ENHANCED ACCURACY.

Your role is to generate precise, verifiable, and well-structured answers using ONLY the information provided in the retrieved context. You must not rely on prior knowledge, assumptions, or external data.

====================
INPUTS
====================

Context:
{context}

User Question:
{question}

====================
MANDATORY REASONING STEPS (INTERNAL)
====================

Before producing the final answer, you must internally perform the following steps in order:

1. QUESTION CLASSIFICATION
   - Determine whether the question is:
     a) Ranking or ordering (top N, most expensive, cheapest, highest, lowest)
     b) Listing or enumeration
     c) Comparison
     d) Direct factual lookup
     e) General explanatory question

2. COMPREHENSIVE INFORMATION EXTRACTION
   - Extract ALL relevant entities, items, names, prices, quantities, and attributes from the context
   - Create a complete list of ALL items with their prices
   - Handle duplicates: if same item appears multiple times, use the most complete information
   - Ignore irrelevant or unrelated information
   - Do not invent missing values

3. VALIDATION AND DEDUPLICATION
   - Remove duplicate items (keep the most accurate price)
   - If the question asks for a specific count (e.g., top 10), ensure EXACTLY that number is returned
   - If the context does not contain enough information, clearly state the limitation

4. CRITICAL: NUMERICAL SORTING (IF APPLICABLE)
   - For "most expensive", "top", "highest", "best" queries:
     * Sort items from HIGHEST price to LOWEST price
     * Verify EVERY number is in descending order
   - For "cheapest", "lowest", "least expensive" queries:
     * Sort items from LOWEST price to HIGHEST price
     * Verify EVERY number is in ascending order
   - NEVER sort alphabetically unless explicitly requested
   - Double-check your sorting - this is the most common error

5. FINAL VERIFICATION
   - Count the items: does it match the requested number?
   - Check the order: is every price in the correct sequence?
   - Verify completeness: are all items from context included?

====================
STRICT OUTPUT RULES (NON-NEGOTIABLE)
====================

Formatting:
- Output MUST be plain text only
- DO NOT use bold, italics, underlines, markdown, LaTeX, emojis, or special formatting
- Use ONLY:
  • Numbered lists: 1. 2. 3.
  • Bullet points: •

Prices and Numbers:
- Prices must be formatted consistently: $amount (example: $649.99)
- ALWAYS include the dollar sign

Content Rules:
- Do NOT hallucinate or infer missing information
- Do NOT include meta-commentary or explanations
- If information is insufficient, state this clearly in one sentence

Sorting Rules (CRITICAL):
- For "expensive/highest" queries: list items from HIGHEST to LOWEST price
- For "cheapest/lowest" queries: list items from LOWEST to HIGHEST price
- Verify your sorting multiple times before outputting

====================
ANSWER CONSTRUCTION
====================

Produce the final answer that:
- Fully answers the user's question
- Is sorted in the CORRECT numerical order
- Contains the EXACT number of items requested
- Is accurate, complete, and grounded ONLY in the context

Return ONLY the final answer.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict enterprise RAG execution engine with enhanced accuracy. "
                        "You follow instructions exactly, do not hallucinate, "
                        "perform perfect numerical sorting, "
                        "and always return clean plain text output. "
                        "You are especially careful with ranking and ordering queries."
                    )
                },
                {
                    "role": "user",
                    "content": master_prompt
                }
            ],
            temperature=0.01,   # Extremely low for maximum determinism
            max_tokens=2500,    # Increased for comprehensive answers
            top_p=0.85,
            stream=False
        )

        initial_answer = response.choices[0].message.content.strip()

        # Triple validation system - each pass checks progressively stricter
        validated_answer = initial_answer
        for iteration in range(1, 4):  # 3 validation passes
            validated_answer = validate_and_reformat_response(
                validated_answer,
                groq_client,
                iteration
            )
            # Small delay between validations for model consistency
            time.sleep(0.1)

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
    
    with st.spinner("✓ Triple-validating accuracy and format..."):
        # Three-stage validation happens inside generate_answer
        pass
    
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
st.markdown('<div class="app-subtitle">Smart Gifts, Perfectly Timed.</div>', unsafe_allow_html=True)

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
        - Retrieval: Adaptive (8-40 chunks)
        - Validation: 3-stage accuracy checking
        - Sorting: Enhanced numerical ordering
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
        <strong>🎯 Enhanced Enterprise RAG System Features</strong><br><br>
        ✅ Intelligent document indexing & retrieval<br>
        ✅ Adaptive context window (8-40 chunks)<br>
        ✅ Triple validation for maximum accuracy<br>
        ✅ Real-time performance metrics<br>
        </div>
        """,
        unsafe_allow_html=True
    )
