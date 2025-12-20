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
    page_title="GiftxAI - AI Gift Recommendations",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# ADVANCED CUSTOM CSS FOR PROFESSIONAL STYLING
# ------------------------------
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content Container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-top: 2rem;
    }
    
    /* App Title with Animation */
    .app-title {
        font-size: 56px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .app-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 18px;
        font-weight: 400;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] h2 {
        color: white;
        font-weight: 700;
        font-size: 22px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stFileUploader"] label {
        color: white !important;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        height: 50px;
        width: 100%;
        border-radius: 12px;
        border: none;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: white;
        color: #667eea;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f3f4f6;
        transform: translateY(-2px);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    [data-testid="stChatMessageContent"] {
        font-size: 16px;
        line-height: 1.7;
        color: #1f2937;
    }
    
    /* User Message */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stChatMessage"][data-testid*="user"] p {
        color: white !important;
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: #f9fafb;
        border-left: 4px solid #667eea;
    }
    
    /* Chat Input */
    .stChatInput {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .stChatInput:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f3f4f6;
        border-radius: 8px;
        font-weight: 600;
        color: #667eea;
    }
    
    .streamlit-expanderContent {
        background: #fafafa;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Processed Files Display */
    .processed-file {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: white;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
    
    /* Tips Section */
    .tips-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .tips-container h3 {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .tips-container ul {
        list-style: none;
        padding: 0;
    }
    
    .tips-container li {
        font-size: 16px;
        margin: 0.75rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .tips-container li:before {
        content: "✨";
        position: absolute;
        left: 0;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .app-title {
            font-size: 36px;
        }
        
        .main .block-container {
            padding: 1rem;
        }
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
        st.success("✅ Connected to Groq API successfully")
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
    """Remove markdown formatting (bold/italic) but keep bullet lists and normal spacing"""
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
    all_docs = []
    with st.spinner("🔄 Processing documents..."):
        for uploaded_file in uploaded_files:
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
                st.warning(f"⚠️ {uploaded_file.name} may not be Christmas-related. Still processing.")
            docs = create_document_chunks(text, uploaded_file.name)
            all_docs.extend(docs)
            st.session_state.processed_files.append(uploaded_file.name)
    if not all_docs:
        st.error("❌ No valid documents to process")
        return None
    with st.spinner("🧠 Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if st.session_state.vectorstore is None:
        vectorstore = FAISS.from_documents(all_docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(all_docs)
        vectorstore = st.session_state.vectorstore
    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s) into {len(all_docs)} chunks")
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
- Write naturally and clearly
- When listing items (like "top 10 most expensive"), make sure to include ALL relevant items from the context, not just a few
- Double-check that you've captured all the data points requested"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in Christmas gift recommendations. Respond clearly and naturally. You can use bullet points, numbered lists, and normal formatting to organize information. However, do NOT use bold or italic text formatting. Keep all text in regular font weight. When asked for lists or rankings, be thorough and include ALL relevant items from the context provided."
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
        st.warning("⚠️ Please upload and process documents first!")
        return
    with st.spinner("🤔 Thinking..."):
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
st.markdown('<div class="app-subtitle">AI-Powered Christmas Gift Recommendation System</div>', unsafe_allow_html=True)

# ------------------------------
# SIDEBAR: PDF UPLOAD & PROCESS
# ------------------------------
with st.sidebar:
    st.markdown("### 📄 Upload Documents")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files containing Christmas gift catalogs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files with gift information"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary"):
            st.session_state.vectorstore = process_documents(uploaded_files)
    
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("### ✅ Processed Files")
        for f in st.session_state.processed_files:
            st.markdown(f'<div class="processed-file">📎 {f}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎯 About")
    st.markdown("""
    <div style='color: white; font-size: 14px; line-height: 1.6;'>
    GiftxAI uses advanced RAG (Retrieval-Augmented Generation) technology to provide intelligent gift recommendations based on your uploaded catalogs.
    </div>
    """, unsafe_allow_html=True)

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

user_question = st.chat_input("💬 Ask a question about gifts...")
if user_question:
    handle_user_input(user_question)
    st.rerun()

# ------------------------------
# TIPS SECTION (First Time Only)
# ------------------------------
if not st.session_state.chat_history:
    st.markdown("""
    <div class="tips-container">
        <h3>💡 How to Use GiftxAI</h3>
        <ul>
            <li>Upload PDF catalogs containing Christmas gift information</li>
            <li>Ask specific questions about gifts, pricing, or comparisons</li>
            <li>Get personalized recommendations within your budget</li>
            <li>View source documents for transparency</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
