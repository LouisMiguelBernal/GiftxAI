# GiftxAI

Retrieval-augmented gift recommendation over your own catalogues. PDFs are parsed and
embedded in the browser, Llama 3.3 70B on Groq drafts the answer and up to three passes
validate it, and every price in the answer is checked against the retrieved text.

Version 2 is a Next.js app that deploys to Vercel as-is: no Python server, no vector
database, and the catalogues never leave the visitor's machine. Only the retrieved
passages are sent to the model.

---

## Run it locally

```bash
npm install
npm run dev        # http://localhost:3210
```

Click **Try it with a sample catalogue** to load a fictional 35-product catalogue
(`public/sample/`), or drop your own PDFs.

Tests: `npm test` · types: `npm run typecheck` · production: `npm run build && npm start`.

---

## Configuration

Generation needs a [Groq API key](https://console.groq.com/keys) (the free tier is enough).

| Option | How |
|---|---|
| Server key (demo for every visitor) | Set `GROQ_API_KEY` in `.env.local` or in Vercel project settings. Template: `config/env.template`. Limited to 20 questions per 10 minutes per IP. |
| Visitor's own key | Paste it under **Generation**. Kept in that browser tab only (sessionStorage) and forwarded to Groq by the API route; not limited. |
| No key | Retrieval still works and shows the passages it found. |

---

## How it works

1. **Index.** pdf.js extracts text and rebuilds lines by position so names stay next to
   their prices. Text is cleaned and split at 800 characters with 150 of overlap.
   Chunks are embedded with `all-MiniLM-L6-v2` through transformers.js, in the browser.
   Until the model has downloaded (or if it cannot load), retrieval falls back to BM25.

2. **Retrieve.** The window adapts to the question: a lookup pulls 8 chunks, ranking
   language ("top", "cheapest", "compare") pulls 30, and an explicit *top N* pulls
   `max(40, N × 4)`. Near-duplicate chunks are dropped.

3. **Answer.** One grounded pass at temperature 0.01, told to use only the context and to
   sort numerically with care.

4. **Validate.** Up to three more passes, each given the context again, remove anything
   not in it and re-check ordering and counts. A failed pass keeps the previous draft.

5. **Verify.** A deterministic check matches every `$` amount in the answer against the
   retrieved text and flags any that are missing. The chunks behind each answer are one
   click away.

---

## Layout

```
GiftxAI/
├── public/sample/                     Fictional sample catalogue (PDF)
├── scripts/make_sample_catalogue.py   Regenerates it (reportlab)
├── src/app/api/answer/route.ts        Groq draft + validation passes + grounding
├── src/components/App.tsx             Catalogue panel and chat
├── src/lib/browser.ts                 pdf.js and transformers.js loaders
├── src/lib/retrieve.ts                Window sizing, cosine search, BM25
├── src/lib/text.ts                    Cleaning, splitter, markdown strip, price grounding
└── src/lib/prompts.ts                 Answer and validator prompts
```

---

## Notes

- pdf.js 4.10.38 and transformers.js 3.8.1 load from jsDelivr at runtime, pinned.
- Scanned PDFs with no text layer index as empty. Run OCR first.
- The gift-vocabulary check is advisory: other catalogues are flagged and indexed anyway.
