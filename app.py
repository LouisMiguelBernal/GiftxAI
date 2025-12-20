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
# CUSTOM CSS FOR STYLING
# ------------------------------
st.markdown("""
    <style>
    /* App title */
    .app-title {
        font-size: 48px;
        font-weight: 800;
        color: #4B0082;
        text-align: center;
        margin-bottom: 10px;
    }
    /* Section headers */
    h2 {
        color: #2E8B57;
        font-weight: 700;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #4B0082;
        color: white;
        font-weight: 600;
        height: 45px;
        width: 100%;
        border-radius: 10px;
    }
    /* Chat messages */
    .stMarkdown p {
        font-size: 18px;
    }
    /* Sidebar header */
    .sidebar .css-1d391kg h2 {
        font-size: 22px;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# GROQ API KEY (HIDDEN)
# ------------------------------
if 'groq_client' not in st.session_state or st.session_state.groq_client is None:
    try:
        st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        st.session_state.groq_client.models.list()  # simple test to verify connection
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
def escape_markdown(text: str) -> str:
    """Escape characters that trigger markdown formatting in Streamlit."""
    escape_chars = r"\`*_{}[]()#+-.!"
    for char in escape_chars:
        text = text.replace(char, f"\\{char}")
    return text.strip()
    
def clean_extracted_text(text: str) -> str:
    text = re.sub(r'(?<=\w)\s(?=\w)', '', text)
    text = re.sub(r'\s+', ' ', text)
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": filename, "chunk": i}) for i, chunk in enumerate(chunks)]

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

def get_relevant_context(question: str, vectorstore, k=3):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs

def generate_answer(question: str, context: str, groq_client: Groq) -> str:
    prompt = f"""Answer the question based ONLY on the context provided below about Christmas gifts.

Context:
{context}

Question: {question}

Answer:"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":"You are a helpful assistant specialized in Christmas gift recommendations."},
                      {"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def handle_user_input(user_question: str):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process documents first!")
        return
    with st.spinner("Thinking..."):
        context, source_docs = get_relevant_context(user_question, st.session_state.vectorstore)
        answer = generate_answer(user_question, context, st.session_state.groq_client)
        # Escape markdown to prevent sideways/italic text
        cleaned_answer = escape_markdown(answer)
        st.session_state.chat_history.append({
            'question': user_question,
            'answer': cleaned_answer,  # store the cleaned version
            'sources': source_docs
        })


# ------------------------------
# APP TITLE
# ------------------------------
st.markdown('<div class="app-title">🎁 GiftxAI</div>', unsafe_allow_html=True)

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
            st.text(f"📎 {f}")
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
            st.markdown(message['answer'], unsafe_allow_html=False)  # use markdown for safe plain text
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
# FOOTER: Tips (centered, one-time)
# ------------------------------
if 'tips_shown' not in st.session_state:
    st.session_state.tips_shown = True

    # Only show if there is no chat yet
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div style="text-align:center; font-size:18px; color:#4B0082; margin-top:30px;">
            <strong>💡 Tips:</strong><br>
            - Upload PDF documents with Christmas gifts, presents, or holiday shopping info<br>
            - Ask specific questions about gifts, pricing, features, or comparisons<br>
            - AI answers based <strong>ONLY</strong> on uploaded documents
            </div>
            """,
            unsafe_allow_html=True
        )


