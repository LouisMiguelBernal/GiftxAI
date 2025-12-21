import streamlit as st
import re
import os
import tempfile
from typing import List, Tuple
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
    
    st.session_state.metrics['total_documents'] = len(st.session_state.processed_files)
    st.session_state.metrics['total_chunks'] = len(all_docs)
    
    st.success(f"✅ Successfully indexed {len(uploaded_files)} document(s) into {len(all_docs)} chunks")
    return vectorstore

def add_conversational_memory(question: str, chat_history: list, max_history: int = 3) -> str:
    """Add conversational context for follow-up questions"""
    if not chat_history or len(chat_history) == 0:
        return question
    
    follow_up_indicators = ['that', 'it', 'them', 'those', 'this', 'these', 
                            'the first', 'the last', 'previous', 'earlier',
                            'tell me more', 'what about', 'how about']
    
    is_follow_up = any(indicator in question.lower() for indicator in follow_up_indicators)
    
    if not is_follow_up:
        return question
    
    recent_context = []
    for msg in chat_history[-max_history:]:
        recent_context.append(f"Previous Q: {msg['question']}")
        recent_context.append(f"Previous A: {msg['answer'][:200]}...")
    
    contextualized_question = f"""Given this recent conversation context:

{chr(10).join(recent_context)}

Current question: {question}

[Note: Interpret pronouns and references based on the conversation history]"""
    
    return contextualized_question

def get_relevant_context_enhanced(question: str, vectorstore, k=8) -> Tuple[str, List[Document], int, str]:
    """Enhanced retrieval with query understanding and context optimization"""
    
    query_patterns = {
        'ranking': ['top', 'most', 'best', 'highest', 'lowest', 'cheapest', 'expensive'],
        'comparison': ['compare', 'versus', 'vs', 'difference between', 'better'],
        'recommendation': ['recommend', 'suggest', 'looking for', 'need', 'want', 'gift for'],
        'specific': ['what is', 'tell me about', 'describe', 'explain', 'how much'],
        'listing': ['all', 'every', 'list', 'show me', 'available']
    }
    
    question_lower = question.lower()
    query_type = 'general'
    
    for qtype, keywords in query_patterns.items():
        if any(keyword in question_lower for keyword in keywords):
            query_type = qtype
            break
    
    k_adaptive = {
        'ranking': 20,
        'comparison': 12,
        'recommendation': 15,
        'listing': 20,
        'specific': 8,
        'general': 10
    }
    
    k = k_adaptive.get(query_type, 10)
    
    top_n_match = re.search(r'top\s+(\d+)', question_lower)
    if top_n_match:
        requested_count = int(top_n_match.group(1))
        k = max(k, requested_count * 3)
    
    docs = vectorstore.similarity_search(question, k=k)
    
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    return context, docs, k, query_type

def validate_and_enhance_response(initial_response: str, original_question: str, groq_client: Groq) -> str:
    """Advanced validation that ensures quality while preserving natural language"""
    
    validation_prompt = f"""You are a quality assurance specialist for AI responses. Review and enhance this response while maintaining its natural, helpful tone.

ORIGINAL QUESTION:
{original_question}

GENERATED RESPONSE:
{initial_response}

====================
VALIDATION CHECKLIST
====================

1. ACCURACY & COMPLETENESS
   ✓ Does it fully answer the question?
   ✓ Are all numbers, prices, and facts correct?
   ✓ Is the information logically consistent?
   ✓ For rankings/lists: Is the ordering correct?

2. NATURAL LANGUAGE QUALITY
   ✓ Does it sound natural and conversational?
   ✓ Is it helpful without being robotic?
   ✓ Does it show understanding of user intent?
   ✓ Is the tone appropriate and engaging?

3. FORMATTING CONSISTENCY
   ✓ Plain text only (no markdown/bold/italic)
   ✓ Prices formatted consistently: $amount
   ✓ Appropriate use of lists vs. paragraphs
   ✓ Clean spacing and structure

4. RESPONSE INTELLIGENCE
   ✓ Provides context for numbers and facts
   ✓ Acknowledges limitations naturally if any
   ✓ Adds helpful insights beyond raw data
   ✓ Appropriate level of detail

====================
YOUR TASK
====================

If the response meets all criteria: Return it as-is (with only format cleaning if needed).

If improvements are needed: Enhance the response while:
- Keeping all factual information accurate
- Maintaining a natural, helpful tone
- Ensuring proper formatting
- Preserving the conversational quality

OUTPUT: Provide ONLY the final, validated response. No explanations or meta-commentary."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a meticulous quality assurance specialist who ensures AI responses are accurate, natural, and helpful. You maintain high standards while preserving the conversational quality that makes responses engaging."
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            temperature=0.1,
            max_tokens=2500,
            top_p=0.9,
            stream=False
        )
        
        validated_response = response.choices[0].message.content.strip()
        validated_response = clean_response_formatting(validated_response)
        
        return validated_response
        
    except Exception as e:
        return clean_response_formatting(initial_response)

def generate_answer(question: str, context: str, groq_client: Groq) -> str:
    """State-of-the-art answer generation with advanced reasoning and natural language quality"""

    master_prompt = f"""You are an advanced AI assistant specializing in gift recommendations and product information. You combine the analytical precision of enterprise systems with the natural helpfulness of conversational AI.

CORE CAPABILITIES:
- Deep contextual understanding
- Multi-step reasoning
- Uncertainty acknowledgment
- Adaptive response formatting
- Source attribution

====================
RETRIEVED CONTEXT
====================

{context}

====================
USER QUESTION
====================

{question}

====================
REASONING PROTOCOL
====================

Before answering, think through these steps (internal reasoning):

1. INTENT ANALYSIS
   - What is the user really asking for?
   - What type of response would be most helpful?
   - Are there implicit needs beyond the explicit question?

2. CONTEXT ASSESSMENT
   - What relevant information is available in the context?
   - What's missing that the user might expect?
   - How confident can I be in the retrieved information?

3. INFORMATION SYNTHESIS
   - Extract all relevant data points
   - Identify patterns, relationships, or comparisons
   - Organize information for clarity and usefulness

4. RESPONSE STRATEGY
   - For rankings: Provide clear numerical ordering with context
   - For comparisons: Highlight key differentiators
   - For recommendations: Consider multiple factors (price, features, occasion)
   - For general questions: Provide comprehensive yet concise answers

====================
RESPONSE GUIDELINES
====================

TONE & STYLE:
- Be conversational yet professional
- Use natural language, not robotic patterns
- Show understanding of context and user intent
- Be helpful without being overly verbose

HANDLING UNCERTAINTY:
- If information is incomplete, acknowledge it naturally
  ❌ "The context does not contain..."
  ✅ "Based on the available information, I can tell you... though I don't have details on..."
- Offer what you know confidently while noting limitations
- Suggest related information that might be helpful

FORMATTING INTELLIGENCE:
- Adapt formatting to question type:
  • Rankings/Lists: Use numbered lists (1. 2. 3.)
  • Comparisons: Use clear sections or bullet points
  • General info: Use paragraphs with bullet points for key details
  • Quick facts: Direct, concise answers
- Keep prices consistent: $amount format
- Use plain text (no bold/italic/markdown)

CONTEXTUAL AWARENESS:
- Reference specific products/items naturally
- Draw connections between related information
- Provide context for numbers (e.g., "At $49.99, this is one of the more affordable options...")
- Consider the "why" behind data, not just the "what"

QUALITY CHECKS:
- Ensure numerical accuracy (prices, counts, rankings)
- Verify logical consistency
- Check that the answer directly addresses the question
- Confirm appropriate level of detail

====================
EXAMPLES OF EXCELLENCE
====================

INSTEAD OF:
"The most expensive item is Product A at $299.99."

PROVIDE:
"The highest-priced item in the catalog is Product A at $299.99, which positions it as a premium option in this category."

INSTEAD OF:
"1. Item A - $50
2. Item B - $45
3. Item C - $40"

PROVIDE:
"Here are the top 3 options by price:

1. Item A ($50) - The premium choice with advanced features
2. Item B ($45) - Great mid-range option balancing cost and quality
3. Item C ($40) - Most budget-friendly while maintaining solid value"

====================
YOUR TASK
====================

Now, provide a helpful, natural, and accurate answer to the user's question based on the retrieved context. Think through your reasoning, then deliver a response that would make the user feel like they're talking to a knowledgeable, helpful expert.

Remember: You're not just retrieving information—you're helping someone make decisions and find what they need."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a state-of-the-art AI assistant that combines:
- The analytical depth of enterprise RAG systems
- The conversational fluency of Claude and GPT-4
- The contextual awareness of Gemini
- The helpfulness of a knowledgeable expert

You think carefully before responding, consider context and user intent, and provide answers that are accurate, natural, and genuinely helpful. You acknowledge uncertainty gracefully and format responses intelligently based on the question type."""
                },
                {
                    "role": "user",
                    "content": master_prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000,
            top_p=0.95,
            stream=False
        )

        initial_answer = response.choices[0].message.content.strip()
        
        validated_answer = validate_and_enhance_response(
            initial_answer,
            question,
            groq_client
        )

        return validated_answer

    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."

def handle_user_input(user_question: str):
    """Handle user query with enhanced conversational capabilities"""
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and process documents first!")
        return
    
    start_time = time.time()
    
    # Add conversational context
    contextualized_question = add_conversational_memory(
        user_question, 
        st.session_state.chat_history
    )
    
    # Enhanced retrieval
    with st.spinner("🔍 Analyzing your question..."):
        context, source_docs, k_used, query_type = get_relevant_context_enhanced(
            contextualized_question, 
            st.session_state.vectorstore
        )
    
    # Generate answer with enhanced system
    with st.spinner("💭 Thinking through the best answer..."):
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
        'query_type': query_type,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# ------------------------------
# APP HEADER
# ------------------------------
st.markdown('<div class="app-title">🎁 <span class="gift-text">Gift</span><span class="xai-text">xAI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Smart Gifts, Perfectly Timed - Enhanced with State-of-the-Art AI</div>', unsafe_allow_html=True)

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
        **Enhanced RAG Architecture:**
        - Embedding: all-MiniLM-L6-v2
        - Vector Store: FAISS
        - LLM: Llama 3.3 70B
        - Chunk Size: 800 tokens
        - Retrieval: Adaptive (8-20 chunks)
        - Query Classification: 5 types
        - Conversational Memory: 3-turn
        - Validation: 2-stage enhancement
        - Temperature: 0.3 (balanced)
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
        col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
        with col1:
            st.caption(f"⏱️ {message.get('response_time', 0):.2f}s")
        with col2:
            st.caption(f"📚 {message.get('k_used', 0)} chunks")
        with col3:
            query_type = message.get('query_type', 'general')
            st.caption(f"🔍 {query_type}")
        with col4:
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
        <strong>🎯 State-of-the-Art RAG System Features</strong><br><br>
        ✅ Advanced conversational AI (Claude/GPT-4 style)<br>
        ✅ Multi-step reasoning and intent analysis<br>
        ✅ Adaptive retrieval (8-20 chunks based on query type)<br>
        ✅ Conversational memory for follow-up questions<br>
        ✅ Natural language generation with context awareness<br>
        ✅ Real-time performance metrics and query classification<br>
        </div>
        """,
        unsafe_allow_html=True
    )
