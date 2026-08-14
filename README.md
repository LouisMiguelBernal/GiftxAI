# GiftxAI

Retrieval-augmented gift recommendation over your own PDF catalogues. FAISS
retrieval, a grounded answer pass, and optional validation passes on a
Groq-hosted Llama 3.3 70B.

---

## Run it locally

Double-click `run.bat`, or from a terminal in the project folder:

```bash
run.bat
```

First run creates a virtual environment and installs the retrieval stack —
PyTorch and sentence-transformers make this a large download, so expect several
minutes. Then it opens at **http://localhost:8503**.

Just want to look at the interface? This installs Streamlit only and skips the
~2 GB of ML dependencies:

```bash
run.bat ui
```

The app runs either way. Without the retrieval stack it renders in full and tells
you what is missing instead of crashing.

With the full stack the **first page load takes about 40 seconds** — importing
LangChain and PyTorch is slow, and it is cached from then on. The first time you
index anything, the embedding model downloads (~90 MB).

Prefer to drive it yourself:

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && streamlit run app.py
```

---

## Configuration

Answer generation needs a [Groq API key](https://console.groq.com/keys) (free tier
is enough). Either set it before launching:

```bash
set GROQ_API_KEY=gsk_your_key_here
```

…or paste it into the sidebar at runtime. It is never written to disk. You can
also put it in `.streamlit/secrets.toml`, which is gitignored.

Retrieval and indexing work without a key — only the generation step needs one.

---

## How it works

1. **Index.** Uploaded PDFs are read with `pypdf`, cleaned, and split at 800
   characters with 150 of overlap. Chunks are embedded with `all-MiniLM-L6-v2`
   and stored in FAISS.

2. **Retrieve.** The retrieval window adapts to the question. A lookup pulls 8
   chunks; anything containing ranking language ("top", "cheapest", "compare")
   pulls 30; an explicit *top N* pulls `max(40, N × 4)`, because a ranked answer
   assembled from 8 chunks is guesswork. Near-duplicate chunks are dropped.

3. **Answer.** One grounded pass at temperature 0.01, instructed to use only the
   retrieved context and to sort numerically with care.

4. **Validate.** Up to three further passes re-check ordering, counts, and
   formatting. Each costs a round-trip, so the count is a slider — 0 returns the
   first draft, 2 is the default. A failed validation pass keeps the previous
   draft rather than losing the answer.

Every answer ships with the chunks it was built from. Open the expander and check.

---

## Layout

```
GiftxAI/
├── .streamlit/config.toml   Base theme
├── app.py                   The application
├── theme.py                 Shared design system
├── requirements.txt
└── run.bat                  One-command local launch
```

`theme.py` is shared verbatim with [QuantMaven](https://github.com/LouisMiguelBernal/QuantMaven)
and [DeepSP](https://github.com/LouisMiguelBernal/DeepSP) — one visual language,
one accent hue per project.

---

## Notes

The domain check is advisory: a catalogue with no gift or seasonal vocabulary
gets flagged and indexed anyway, rather than being rejected.

Scanned PDFs with no embedded text layer will index as empty. Run OCR over them
first.
