'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { cleanExtracted, splitText, looksOnDomain, type Chunk, type Grounding } from '@/lib/text'
import { chooseK, vectorSearch, bm25, type Hit } from '@/lib/retrieve'
import { pdfText, embed, loadEmbedder, EMBED_MODEL } from '@/lib/browser'

type Turn = {
  q: string
  answer?: string
  error?: string
  hits: Hit[]
  k: number
  mode: 'semantic' | 'keyword'
  ms?: number
  passes?: number
  steps?: { label: string; ms: number }[]
  grounding?: Grounding
  at: string
  pending?: boolean
}

type Embed = { status: 'idle' | 'loading' | 'ready' | 'failed'; pct: number }

const SAMPLE = '/sample/evergreen-holiday-catalogue.pdf'
const SUGGESTIONS = [
  'What are the top 10 most expensive gifts?',
  'Cheapest options under $50',
  'What is included in the Premium Holiday Hamper?',
  'Compare the tech gadgets by price',
]

const now = () => new Date().toLocaleTimeString('en-GB', { hour12: false })
const ss = {
  get: (k: string) => { try { return sessionStorage.getItem(k) ?? '' } catch { return '' } },
  set: (k: string, v: string) => { try { if (v) sessionStorage.setItem(k, v); else sessionStorage.removeItem(k) } catch {} },
}

export default function App() {
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [vecs, setVecs] = useState<Float32Array[]>([])
  const [files, setFiles] = useState<{ name: string; chunks: number; warn?: string }[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [emb, setEmb] = useState<Embed>({ status: 'idle', pct: 0 })
  const [serverKey, setServerKey] = useState<boolean | null>(null)
  const [ownKey, setOwnKey] = useState('')
  const [passes, setPasses] = useState(2)
  const [history, setHistory] = useState<Turn[]>([])
  const [q, setQ] = useState('')
  const [panel, setPanel] = useState(false)
  const [drag, setDrag] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const loadedSample = useRef(false)

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- read once from session storage on mount
    setOwnKey(ss.get('groq-key'))
    fetch('/api/answer').then(r => r.json()).then(j => setServerKey(!!j.serverKey)).catch(() => setServerKey(false))
  }, [])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [history])

  const canGenerate = !!ownKey || !!serverKey
  const semantic = emb.status === 'ready' && vecs.length === chunks.length && chunks.length > 0

  const ensureEmbedder = useCallback(async () => {
    if (emb.status === 'ready') return true
    setEmb(e => ({ ...e, status: 'loading' }))
    try {
      await loadEmbedder(pct => setEmb({ status: 'loading', pct }))
      setEmb({ status: 'ready', pct: 100 })
      return true
    } catch {
      setEmb({ status: 'failed', pct: 0 })
      return false
    }
  }, [emb.status])

  const index = useCallback(async (list: File[]) => {
    if (!list.length) return
    const added: Chunk[] = []
    const meta: { name: string; chunks: number; warn?: string }[] = []
    let next = chunks.length
    for (const f of list) {
      setBusy(`Reading ${f.name}…`)
      let text = ''
      try {
        text = f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf') ? await pdfText(f) : await f.text()
      } catch {
        meta.push({ name: f.name, chunks: 0, warn: 'Could not read this file.' })
        continue
      }
      text = cleanExtracted(text)
      if (!text) { meta.push({ name: f.name, chunks: 0, warn: 'No extractable text. Is it a scan?' }); continue }
      const parts = splitText(text, 800, 150)
      parts.forEach((t, n) => added.push({ id: next++, source: f.name, n, text: t }))
      meta.push({ name: f.name, chunks: parts.length, warn: looksOnDomain(text) ? undefined : 'No gift vocabulary found. Indexed anyway.' })
    }
    setChunks(c => [...c, ...added])
    setFiles(fs => [...fs.filter(x => !meta.some(m => m.name === x.name)), ...meta])
    if (added.length) {
      setBusy('Loading the embedding model…')
      if (await ensureEmbedder()) {
        setBusy(`Embedding ${added.length} chunks…`)
        try {
          const v = await embed(added.map(a => a.text))
          setVecs(old => [...old, ...v])
        } catch {
          setEmb({ status: 'failed', pct: 0 })
        }
      }
    }
    setBusy(null)
  }, [chunks.length, ensureEmbedder])

  const loadSample = useCallback(async () => {
    if (loadedSample.current) return
    loadedSample.current = true
    setBusy('Fetching the sample catalogue…')
    const r = await fetch(SAMPLE)
    const blob = await r.blob()
    await index([new File([blob], 'evergreen-holiday-catalogue.pdf', { type: 'application/pdf' })])
  }, [index])

  const ask = useCallback(async (question: string) => {
    question = question.trim()
    if (!question || !chunks.length || busy) return
    setQ('')
    const k = chooseK(question)
    let hits: Hit[]
    let mode: Turn['mode'] = 'keyword'
    if (semantic) {
      const [qv] = await embed([question])
      hits = vectorSearch(qv, chunks, vecs, k)
      mode = 'semantic'
    } else {
      hits = bm25(question, chunks, k)
    }
    const turn: Turn = { q: question, hits, k, mode, at: now(), pending: canGenerate }
    if (!canGenerate) {
      setHistory(h => [...h, turn])
      return
    }
    setHistory(h => [...h, turn])
    const context = hits.map(h => h.chunk.text).join('\n\n')
    try {
      const r = await fetch('/api/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...(ownKey ? { 'x-groq-key': ownKey } : {}) },
        body: JSON.stringify({ question, context, passes }),
      })
      const j = await r.json()
      setHistory(h => h.map(t => (t === turn ? { ...t, pending: false, ...(r.ok ? j : { error: j.error ?? `Error ${r.status}` }) } : t)))
    } catch {
      setHistory(h => h.map(t => (t === turn ? { ...t, pending: false, error: 'Network error. Try again.' } : t)))
    }
  }, [chunks, vecs, semantic, canGenerate, ownKey, passes, busy])

  const reset = () => { setChunks([]); setVecs([]); setFiles([]); setHistory([]); loadedSample.current = false }

  const answered = history.filter(h => h.ms)
  const avg = answered.length ? answered.reduce((s, h) => s + (h.ms ?? 0), 0) / answered.length / 1000 : 0
  const genLabel = ownKey ? 'Your Groq key' : serverKey ? 'Demo key' : serverKey === null ? 'Checking…' : 'Retrieval only'

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden>
            <Mark size={30} />
          </span>
          <span className="wordmark">Gift<em>x</em>AI</span>
        </div>
        <div className="status">
          <span className={'pill ' + (chunks.length ? 'ok' : '')}><i />{chunks.length ? `${chunks.length} chunks` : 'No catalogue'}</span>
          <span className={'pill hide-sm ' + (semantic ? 'ok' : emb.status === 'loading' ? 'warn' : '')}><i />{semantic ? 'Semantic' : emb.status === 'loading' ? `Model ${Math.round(emb.pct)}%` : 'Keyword'}</span>
          <span className={'pill hide-sm ' + (canGenerate ? 'ok' : 'warn')}><i />{genLabel}</span>
          <button className="panel-btn" onClick={() => setPanel(p => !p)} aria-expanded={panel}>Catalogue</button>
        </div>
      </header>

      <div className="layout">
        <aside className={'side ' + (panel ? 'open' : '')} aria-label="Catalogue and settings">
          <section>
            <h3>Catalogue</h3>
            <div
              className={'drop ' + (drag ? 'drag' : '')}
              onDragOver={e => { e.preventDefault(); setDrag(true) }}
              onDragLeave={() => setDrag(false)}
              onDrop={e => { e.preventDefault(); setDrag(false); index([...e.dataTransfer.files]) }}
              onClick={() => fileRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && fileRef.current?.click()}
            >
              <b>Drop PDF catalogues</b>
              <span>or click to browse · PDF, TXT, MD</span>
              <input ref={fileRef} type="file" accept=".pdf,.txt,.md,application/pdf,text/plain" multiple hidden onChange={e => { index([...(e.target.files ?? [])]); e.target.value = '' }} />
            </div>
            <button className="btn ghost full" onClick={loadSample} disabled={!!busy || loadedSample.current}>
              {loadedSample.current ? 'Sample catalogue loaded' : 'Load the sample catalogue'}
            </button>
            <p className="hint">Files are parsed and embedded in your browser. Only the retrieved passages are sent to the model.</p>
            {busy && <div className="busy"><span className="spin" />{busy}</div>}
            {files.length > 0 && (
              <ul className="files">
                {files.map(f => (
                  <li key={f.name}>
                    <span className="fname">{f.name}</span>
                    <span className="fmeta">{f.chunks} chunks</span>
                    {f.warn && <span className="fwarn">{f.warn}</span>}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section>
            <h3>Session</h3>
            <dl className="kv">
              <div><dt>Documents</dt><dd>{files.filter(f => f.chunks).length}</dd></div>
              <div><dt>Chunks</dt><dd>{chunks.length.toLocaleString()}</dd></div>
              <div><dt>Queries</dt><dd>{history.length}</dd></div>
              <div><dt>Avg latency</dt><dd>{avg ? avg.toFixed(2) + 's' : '—'}</dd></div>
              <div><dt>Embeddings</dt><dd className="mono">{EMBED_MODEL.split('/')[1]}</dd></div>
            </dl>
          </section>

          <section>
            <h3>Generation</h3>
            <label className="field">
              <span>Validation passes <b>{passes}</b></span>
              <input type="range" min={0} max={3} value={passes} onChange={e => setPasses(+e.target.value)} />
              <small>Each pass is an extra model call that re-checks facts, ordering and counts.</small>
            </label>
            <label className="field">
              <span>Your Groq API key {serverKey ? '(optional)' : ''}</span>
              <input
                type="password"
                placeholder="gsk_…"
                value={ownKey}
                autoComplete="off"
                onChange={e => { setOwnKey(e.target.value.trim()); ss.set('groq-key', e.target.value.trim()) }}
              />
              <small>Kept in this tab only and sent straight to Groq through this site&apos;s API route.</small>
            </label>
            <div className="row">
              <button className="btn ghost" onClick={() => setHistory([])}>Clear chat</button>
              <button className="btn ghost" onClick={reset}>Reset all</button>
            </div>
          </section>
        </aside>

        <main className="chat">
          {history.length === 0 ? (
            <div className="intro">
              <div className="eyebrow">Retrieval-augmented recommendation</div>
              <h1>Gift<em>x</em>AI</h1>
              <p className="lede">
                Point it at your product catalogues. It retrieves the passages that actually answer the question,
                drafts with Llama 3.3 70B, then validates facts, ordering and counts before showing you anything.
              </p>
              {!chunks.length && (
                <button className="btn" onClick={loadSample} disabled={!!busy}>Try it with a sample catalogue</button>
              )}
              <div className="steps">
                {[
                  ['01', 'Index', 'PDFs are read, chunked at 800 characters with 150 overlap, and embedded in your browser.'],
                  ['02', 'Retrieve', 'A lookup pulls 8 chunks; a ranking pulls 30; "top 10" pulls 40.'],
                  ['03', 'Validate', 'Up to three passes re-check every item against the context and re-sort.'],
                  ['04', 'Verify', 'Every price in the answer is matched against the sources, and the chunks are one click away.'],
                ].map(([n, t, d]) => (
                  <div key={n} className="step"><span>{n}</span><b>{t}</b><p>{d}</p></div>
                ))}
              </div>
            </div>
          ) : (
            <div className="thread">
              {history.map((t, i) => <TurnView key={i} t={t} />)}
              <div ref={endRef} />
            </div>
          )}

          <div className="composer">
            {chunks.length > 0 && history.length === 0 && (
              <div className="sugg">
                {SUGGESTIONS.map(s => <button key={s} onClick={() => ask(s)} disabled={!!busy}>{s}</button>)}
              </div>
            )}
            <form onSubmit={e => { e.preventDefault(); ask(q) }}>
              <input
                value={q}
                onChange={e => setQ(e.target.value)}
                placeholder={chunks.length ? 'Ask about gifts, pricing, comparisons…' : 'Load a catalogue first'}
                disabled={!chunks.length}
                maxLength={600}
                aria-label="Question"
              />
              <button className="send" disabled={!q.trim() || !chunks.length || !!busy} aria-label="Ask">
                <svg viewBox="0 0 24 24" width="18" height="18"><path d="M4 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
            </form>
          </div>
        </main>
      </div>
    </div>
  )
}

/** The GiftxAI mark from 1nonly.me: a gift box from above, ribbons crossing as an "x", knot at the center. */
function Mark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
      <defs><clipPath id="gx-box"><rect x="7" y="7" width="34" height="34" rx="9" /></clipPath></defs>
      <rect x="7" y="7" width="34" height="34" rx="9" stroke="currentColor" strokeWidth="3" />
      <g clipPath="url(#gx-box)" stroke="var(--accent)" strokeWidth="5" strokeLinecap="round">
        <path d="M7 7 L41 41" />
        <path d="M41 7 L7 41" />
      </g>
      <rect x="19.5" y="19.5" width="9" height="9" rx="2.5" transform="rotate(45 24 24)" fill="var(--bg)" stroke="var(--accent)" strokeWidth="2.6" />
    </svg>
  )
}

function TurnView({ t }: { t: Turn }) {
  const [open, setOpen] = useState(!t.pending && !t.answer && !t.error)
  const g = t.grounding
  return (
    <article className="turn">
      <div className="q">{t.q}</div>
      <div className="a">
        {t.pending && <div className="typing"><span /><span /><span /> Drafting and validating…</div>}
        {t.error && <div className="err">{t.error}</div>}
        {t.answer && <div className="answer">{t.answer}</div>}
        {!t.pending && !t.answer && !t.error && (
          <div className="answer muted">
            No Groq key is connected, so here are the passages retrieval found. Add a key under Generation to get a written answer.
          </div>
        )}
        <div className="badges">
          {t.ms !== undefined && <span>{(t.ms / 1000).toFixed(2)}s</span>}
          <span>{t.hits.length} of {t.k} chunks · {t.mode}</span>
          {t.passes !== undefined && <span>{t.passes} validation pass{t.passes === 1 ? '' : 'es'}</span>}
          {g && g.checked > 0 && (
            <span className={g.missing.length ? 'bad' : 'good'}>
              {g.grounded}/{g.checked} prices found in sources
            </span>
          )}
          <span>{t.at}</span>
        </div>
        {g && g.missing.length > 0 && (
          <div className="err small">Not found in the retrieved text: {g.missing.map(m => '$' + m.toFixed(2)).join(', ')}. Treat these with care.</div>
        )}
        <button className="src-toggle" onClick={() => setOpen(o => !o)} aria-expanded={open}>
          {open ? 'Hide' : 'Show'} retrieved context · {Math.min(5, t.hits.length)} of {t.hits.length}
        </button>
        {open && (
          <div className="sources">
            {t.hits.slice(0, 5).map(h => (
              <div key={h.chunk.id} className="src">
                <div className="src-h">{h.chunk.source} · chunk {h.chunk.n + 1} · score {h.score.toFixed(3)}</div>
                <pre>{h.chunk.text.length > 420 ? h.chunk.text.slice(0, 420) + '…' : h.chunk.text}</pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </article>
  )
}
