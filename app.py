"""
GiftxAI — retrieval-augmented gift recommendation over your own PDF catalogues.

FAISS retrieval, a grounded answer pass, and optional validation passes on a
Groq-hosted Llama 3.3 backbone.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any

import streamlit as st

import theme

st.set_page_config(
    page_title="GiftxAI — Grounded Gift Recommendations",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="expanded",
)

P = theme.apply("giftxai")

MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
# The RAG stack (torch, sentence-transformers, faiss) is a large install. Import
# it lazily so the interface still renders — and explains itself — on a machine
# that only has streamlit.

@st.cache_resource(show_spinner=False)
def rag_backend() -> tuple[Any | None, str]:
    # LangChain 1.x split the monolith apart: `langchain.schema` and
    # `langchain.text_splitter` are gone. Try the current homes first and fall
    # back to the legacy ones so the app works on either generation.
    try:
        try:
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.schema import Document
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        from langchain_community.vectorstores import FAISS
        from pypdf import PdfReader
    except Exception as exc:
        return None, str(exc)

    return (
        {
            "Document": Document,
            "Splitter": RecursiveCharacterTextSplitter,
            "Embeddings": HuggingFaceEmbeddings,
            "FAISS": FAISS,
            "PdfReader": PdfReader,
        },
        "",
    )


BACKEND, BACKEND_ERROR = rag_backend()


@st.cache_resource(show_spinner="Loading the embedding model…")
def embedder():
    return BACKEND["Embeddings"](
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def groq_client(api_key: str):
    from groq import Groq

    client = Groq(api_key=api_key)
    client.models.list()  # fail fast on a bad key
    return client


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

DEFAULT_METRICS = {"queries": 0, "documents": 0, "chunks": 0, "avg_time": 0.0}

for key, default in (
    ("history", []),
    ("vectorstore", None),
    ("files", []),
    ("metrics", dict(DEFAULT_METRICS)),
    ("client", None),
    ("client_error", ""),
):
    st.session_state.setdefault(key, default)


def resolve_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


def ensure_client(api_key: str) -> None:
    """Connect once per key. Errors are stored, not raised, so a bad key
    degrades the page instead of blanking it."""
    if not api_key:
        st.session_state.client = None
        st.session_state.client_error = ""
        return
    if st.session_state.client is not None and st.session_state.get("key_used") == api_key:
        return
    try:
        st.session_state.client = groq_client(api_key)
        st.session_state.client_error = ""
    except Exception as exc:
        st.session_state.client = None
        st.session_state.client_error = str(exc)
    st.session_state["key_used"] = api_key


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------

def clean_extracted(text: str) -> str:
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\$(\d+(?:\.\d{2})?)", r"$\1 ", text)
    text = re.sub(r"([a-zA-Z])(\$\d)", r"\1 \2", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r" +", " ", text).strip()


def strip_markdown(text: str) -> str:
    """Answers are rendered as plain text — the model's markdown emphasis fights
    the page's own typographic hierarchy."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"^(\s*)[-*•]\s+", r"\1• ", text, flags=re.MULTILINE)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"(\$\d+(?:\.\d{2})?)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\$\d)", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()


DOMAIN_HINTS = (
    "christmas", "xmas", "gift", "present", "holiday", "santa", "festive",
    "celebration", "december", "winter", "toy", "decoration", "tree",
    "wrapping", "seasonal",
)


def looks_on_domain(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in DOMAIN_HINTS)


def index_documents(files) -> None:
    docs = []
    progress = st.progress(0.0, text="Reading documents…")

    for i, upload in enumerate(files, 1):
        progress.progress(i / len(files), text=f"Reading {upload.name}…")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(upload.getvalue())
            tmp_path = tmp.name
        try:
            reader = BACKEND["PdfReader"](tmp_path)
            text = clean_extracted(
                "".join(page.extract_text() or "" for page in reader.pages)
            )
        except Exception as exc:
            st.warning(f"{upload.name}: could not read — {exc}")
            continue
        finally:
            os.unlink(tmp_path)

        if not text:
            st.warning(f"{upload.name}: no extractable text (is it a scan?).")
            continue
        if not looks_on_domain(text):
            st.info(f"{upload.name}: no gift or seasonal vocabulary found — indexing anyway.")

        splitter = BACKEND["Splitter"](
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs.extend(
            BACKEND["Document"](
                page_content=chunk, metadata={"source": upload.name, "chunk": n}
            )
            for n, chunk in enumerate(splitter.split_text(text))
        )
        if upload.name not in st.session_state.files:
            st.session_state.files.append(upload.name)

    progress.empty()

    if not docs:
        st.error("Nothing indexable in those files.")
        return

    with st.spinner("Building embeddings…"):
        embed = embedder()
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = BACKEND["FAISS"].from_documents(docs, embed)
        else:
            st.session_state.vectorstore.add_documents(docs)

    st.session_state.metrics["documents"] = len(st.session_state.files)
    st.session_state.metrics["chunks"] += len(docs)
    st.success(f"Indexed {len(files)} file(s) into {len(docs)} chunks.")


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------

BROAD_QUERY_WORDS = (
    "top", "most", "best", "all", "list", "expensive", "cheapest", "compare",
    "ranking", "every", "entire", "order", "sorted", "ranked", "highest", "lowest",
)


def retrieve(question: str, store, base_k: int = 8):
    """Widen the context window for questions that need to see the whole
    catalogue — a ranking answer built from eight chunks is guesswork."""
    k = base_k
    if any(word in question.lower() for word in BROAD_QUERY_WORDS):
        k = 30
    match = re.search(r"top\s+(\d+)", question.lower())
    if match:
        k = max(40, int(match.group(1)) * 4)

    docs = store.similarity_search(question, k=k)

    seen, unique = set(), []
    for doc in docs:
        head = doc.page_content.split("\n", 1)[0][:60]
        if head not in seen:
            seen.add(head)
            unique.append(doc)

    return "\n\n".join(d.page_content for d in unique), unique, k


ANSWER_SYSTEM = (
    "You are a retrieval-grounded answer engine. You answer only from the "
    "context you are given, never from prior knowledge. You sort numerically "
    "with care and you return clean plain text."
)


def answer_prompt(question: str, context: str) -> str:
    return f"""Answer the question using ONLY the retrieved context below.

CONTEXT
{context}

QUESTION
{question}

METHOD (work through this internally, do not show it)
1. Classify the question: ranking, enumeration, comparison, lookup, or explanation.
2. Extract every relevant item, price and attribute from the context. Do not
   invent values. If the same item appears twice, keep the fullest entry.
3. Deduplicate. If the question names a count, return exactly that many items.
4. If ranking: sort by price descending for "top", "most expensive", "highest";
   ascending for "cheapest" or "lowest". Never sort alphabetically unless asked.
   Verify the ordering element by element before writing it out.
5. Confirm the count and the ordering one final time.

OUTPUT RULES
- Plain text only. No markdown, no bold, no italics, no headings, no emoji.
- Numbered lists (1. 2. 3.) for rankings, bullets (•) otherwise.
- Prices always as $amount, e.g. $649.99.
- If the context does not support an answer, say so in one sentence.
- Return the answer only, with no preamble.
"""


def validation_prompt(draft: str, n: int, total: int) -> str:
    focus = {
        1: "Extract every item and price; check nothing is missing.",
        2: "Verify the numerical ordering element by element and re-sort if wrong.",
        3: "Final sweep on counts, ordering and formatting.",
    }.get(n, "Verify accuracy, ordering and formatting.")

    return f"""You are validating a draft answer. Pass {n} of {total}. {focus}

DRAFT
{draft}

Check and correct:
- Numerical ordering is exactly right (descending for "top"/"expensive",
  ascending for "cheapest"). Re-sort the whole list if any item is out of place.
- The item count matches what was asked for, with no duplicates.
- Prices are formatted consistently as $amount.
- All markdown is removed: no **, __, *, _, #, backticks.
- Facts are unchanged. Do not add information that was not in the draft.

Return only the corrected answer. No commentary."""


def call_model(client, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.01,
        max_tokens=2500,
        top_p=0.85,
    )
    return response.choices[0].message.content.strip()


def generate(question: str, context: str, client, passes: int) -> str:
    draft = call_model(client, ANSWER_SYSTEM, answer_prompt(question, context))

    for n in range(1, passes + 1):
        try:
            draft = call_model(
                client,
                f"You are a precision validator, pass {n} of {passes}. You catch "
                "subtle sorting and counting errors and you never invent facts.",
                validation_prompt(draft, n, passes),
            )
        except Exception:
            break  # a failed validation pass should not lose the draft

    return strip_markdown(draft)


def handle_question(question: str, passes: int) -> None:
    store = st.session_state.vectorstore
    client = st.session_state.client

    if store is None:
        st.warning("Index a catalogue first — upload PDFs in the sidebar.")
        return
    if client is None:
        st.warning("Connect a Groq API key in the sidebar to generate answers.")
        return

    started = time.time()
    with st.spinner("Retrieving…"):
        context, sources, k = retrieve(question, store)
    label = "Answering…" if not passes else f"Answering, then validating ×{passes}…"
    with st.spinner(label):
        answer = generate(question, context, client, passes)
    elapsed = time.time() - started

    m = st.session_state.metrics
    m["queries"] += 1
    m["avg_time"] = (m["avg_time"] * (m["queries"] - 1) + elapsed) / m["queries"]

    st.session_state.history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
            "elapsed": elapsed,
            "k": k,
            "passes": passes,
            "at": datetime.now().strftime("%H:%M:%S"),
        }
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="tk-eyebrow">GiftxAI</div>'
        '<div style="font-size:1.05rem;font-weight:700;letter-spacing:-.02em;'
        'margin:.3rem 0 1.2rem;">Grounded Recommendations</div>',
        unsafe_allow_html=True,
    )

    env_key = resolve_api_key()
    if env_key:
        api_key = env_key
        st.markdown(
            theme.badge("Key loaded from environment", "pos", dot=True),
            unsafe_allow_html=True,
        )
    else:
        api_key = st.text_input(
            "Groq API key", type="password", placeholder="gsk_…",
            help="Or set GROQ_API_KEY in your environment / .streamlit/secrets.toml",
        )

    ensure_client(api_key)
    if st.session_state.client_error:
        st.error(f"Groq rejected the key — {st.session_state.client_error}")

    st.markdown("---")

    if BACKEND is None:
        st.markdown(theme.badge("RAG stack not installed", "warn"), unsafe_allow_html=True)
        st.caption("`pip install -r requirements.txt` to enable indexing.")
    else:
        uploads = st.file_uploader(
            "Catalogue PDFs", type=["pdf"], accept_multiple_files=True
        )
        if uploads and st.button("Index documents", type="primary"):
            index_documents(uploads)

    if st.session_state.files:
        st.markdown(
            '<div class="tk-eyebrow" style="margin:1rem 0 .5rem;">Indexed</div>'
            + "".join(
                f'<div style="font-size:.8rem;color:var(--muted);padding:.2rem 0;'
                f'font-family:var(--mono);">{theme.esc(f)}</div>'
                for f in st.session_state.files
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    m = st.session_state.metrics
    theme.kv_panel(
        "Session",
        [
            ("Documents", str(m["documents"])),
            ("Chunks", f"{m['chunks']:,}"),
            ("Queries", str(m["queries"])),
            ("Avg latency", f"{m['avg_time']:.2f}s" if m["avg_time"] else "—"),
        ],
    )

    passes = st.slider(
        "Validation passes", 0, 3, 2,
        help="Each pass is an extra model call that re-checks ordering and "
             "formatting. More passes cost latency; 0 returns the first draft.",
    )

    c1, c2 = st.columns(2)
    if c1.button("Clear chat"):
        st.session_state.history = []
        st.rerun()
    if c2.button("Reset all"):
        st.session_state.history = []
        st.session_state.vectorstore = None
        st.session_state.files = []
        st.session_state.metrics = dict(DEFAULT_METRICS)
        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

ready = st.session_state.vectorstore is not None and st.session_state.client is not None

theme.hero(
    "Gift<em>xAI</em>",
    "Point it at your product catalogues. It retrieves the passages that "
    "actually answer the question, then validates the ordering and the counts "
    "before showing you anything.",
    eyebrow="Retrieval-Augmented Recommendation",
    meta=[
        theme.badge("Llama 3.3 70B · Groq", "accent"),
        theme.badge("FAISS retrieval"),
        theme.badge(EMBED_MODEL),
        theme.badge("Ready" if ready else "Awaiting setup", "pos" if ready else "warn", dot=True),
    ],
)

if BACKEND is None:
    st.markdown(
        f'<div class="tk-panel" style="border-color:var(--accent-line);">'
        f"<h4>RAG dependencies missing</h4>"
        f"<p>The interface is running, but indexing and retrieval need the full "
        f"stack. Install it with <code>pip install -r requirements.txt</code>, "
        f"then reload.</p>"
        f'<p style="color:var(--faint);font-size:.8rem;">Import error: '
        f"{theme.esc(BACKEND_ERROR[:180])}</p></div>",
        unsafe_allow_html=True,
    )

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**{turn['question']}**")

    with st.chat_message("assistant"):
        st.text(turn["answer"])

        st.markdown(
            '<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.8rem;">'
            + theme.badge(f"{turn['elapsed']:.2f}s")
            + theme.badge(f"{turn['k']} chunks retrieved")
            + theme.badge(
                f"{turn['passes']} validation pass{'es' if turn['passes'] != 1 else ''}"
            )
            + theme.badge(turn["at"])
            + "</div>",
            unsafe_allow_html=True,
        )

        if turn["sources"]:
            with st.expander(f"Retrieved context · {len(turn['sources'])} chunks"):
                for n, doc in enumerate(turn["sources"][:5], 1):
                    st.markdown(
                        f'<div class="tk-eyebrow">Chunk {n} · '
                        f'{theme.esc(doc.metadata.get("source", "unknown"))}</div>',
                        unsafe_allow_html=True,
                    )
                    st.text(doc.page_content[:400] + "…")

if not st.session_state.history:
    steps = [
        ("01", "Connect", "Add a Groq API key in the sidebar, or set GROQ_API_KEY in your environment."),
        ("02", "Index", "Upload one or more PDF catalogues. Text is chunked at 800 characters with 150 of overlap and embedded into FAISS."),
        ("03", "Ask", "Questions widen the retrieval window automatically — a 'top 10' pulls forty chunks, a lookup pulls eight."),
        ("04", "Verify", "Every answer ships with the chunks it was built from. Open them and check."),
    ]
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));'
        'gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;'
        'overflow:hidden;margin-top:1.5rem;">'
        + "".join(
            f'<div style="background:var(--surface);padding:1.3rem 1.35rem;">'
            f'<div class="mono" style="color:var(--accent);font-size:.8rem;'
            f'font-weight:600;">{n}</div>'
            f'<div style="font-weight:650;margin:.5rem 0 .4rem;font-size:.98rem;">{t}</div>'
            f'<div style="color:var(--muted);font-size:.85rem;line-height:1.6;">{d}</div>'
            f"</div>"
            for n, t, d in steps
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    theme.section("Try asking")
    st.markdown(
        '<div style="display:flex;gap:.4rem;flex-wrap:wrap;">'
        + "".join(
            theme.badge(q)
            for q in (
                "What are the top 10 most expensive gifts?",
                "Cheapest options under $50",
                "Compare the gift sets by price",
                "What is included in the premium bundle?",
            )
        )
        + "</div>",
        unsafe_allow_html=True,
    )

question = st.chat_input("Ask about gifts, pricing, comparisons…")
if question:
    handle_question(question, passes)
    st.rerun()

theme.footer(
    "<b>GiftxAI</b> · Retrieval-augmented gift recommendation",
    "FAISS · Llama 3.3 70B on Groq · Streamlit",
)
