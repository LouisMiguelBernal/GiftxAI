import type { Chunk } from './text.ts'

const BROAD = ['top', 'most', 'best', 'all', 'list', 'expensive', 'cheapest', 'compare', 'ranking', 'every', 'entire', 'order', 'sorted', 'ranked', 'highest', 'lowest']

/** Questions that need to see the whole catalogue get a wider window: a ranking built from eight chunks is guesswork. */
export function chooseK(question: string, base = 8): number {
  const q = question.toLowerCase()
  const words = q.split(/[^a-z0-9]+/)
  let k = BROAD.some(w => words.includes(w)) ? 30 : base
  const m = q.match(/top\s+(\d+)/)
  if (m) k = Math.max(40, Number(m[1]) * 4)
  return k
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export type Hit = { chunk: Chunk; score: number }

/** Cosine search over normalised vectors, then drop near-duplicate chunks by their first line. */
export function vectorSearch(q: Float32Array, chunks: Chunk[], vecs: Float32Array[], k: number): Hit[] {
  const scored = chunks.map((c, i) => ({ chunk: c, score: dot(q, vecs[i]) }))
  scored.sort((a, b) => b.score - a.score)
  return dedupe(scored.slice(0, k))
}

export function dedupe(hits: Hit[]): Hit[] {
  const seen = new Set<string>()
  return hits.filter(h => {
    const head = h.chunk.text.split('\n', 1)[0].slice(0, 60)
    if (seen.has(head)) return false
    seen.add(head)
    return true
  })
}

const tok = (s: string) => s.toLowerCase().match(/[a-z0-9$.]+/g) ?? []

/** BM25 fallback while the embedding model downloads, or if it cannot load at all. */
export function bm25(question: string, chunks: Chunk[], k: number): Hit[] {
  const docs = chunks.map(c => tok(c.text))
  const N = docs.length
  if (!N) return []
  const avg = docs.reduce((s, d) => s + d.length, 0) / N
  const df = new Map<string, number>()
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1)
  const q = [...new Set(tok(question))]
  const k1 = 1.4, b = 0.75
  const scored = docs.map((d, i) => {
    const tf = new Map<string, number>()
    for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1)
    let s = 0
    for (const t of q) {
      const f = tf.get(t)
      if (!f) continue
      const idf = Math.log(1 + (N - (df.get(t) ?? 0) + 0.5) / ((df.get(t) ?? 0) + 0.5))
      s += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * d.length) / avg)))
    }
    return { chunk: chunks[i], score: s }
  })
  scored.sort((a, b2) => b2.score - a.score)
  return dedupe(scored.slice(0, k))
}
