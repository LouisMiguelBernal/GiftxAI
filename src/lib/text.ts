/** Text cleanup, chunking and answer post-processing. Pure functions, shared by client and tests. */

export type Chunk = { id: number; source: string; n: number; text: string }

export function cleanExtracted(text: string): string {
  return text
    .replace(/ +/g, ' ')
    .replace(/\$(\d+(?:,\d{3})*(?:\.\d{2})?)(?=[A-Za-z])/g, '$$$1 ')
    .replace(/([a-zA-Z])(\$\d)/g, '$1 $2')
    .replace(/\n\s*\n+/g, '\n\n')
    .split('\n').map(l => l.trim()).join('\n')
    .replace(/ +/g, ' ')
    .trim()
}

/**
 * Recursive character splitter with the same behaviour as LangChain's
 * RecursiveCharacterTextSplitter: try the coarsest separator first and only
 * fall back to finer ones for pieces that are still too long.
 */
export function splitText(text: string, size = 800, overlap = 150, seps = ['\n\n', '\n', '. ', ' ', '']): string[] {
  const pieces = (t: string, si: number): string[] => {
    if (t.length <= size) return [t]
    const sep = seps[si]
    if (sep === undefined || sep === '') {
      const out: string[] = []
      for (let i = 0; i < t.length; i += size - overlap) out.push(t.slice(i, i + size))
      return out
    }
    const parts = t.split(sep)
    const out: string[] = []
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i] + (i < parts.length - 1 ? sep : '')
      if (p.length > size) out.push(...pieces(p, si + 1))
      else out.push(p)
    }
    return out
  }

  const small = pieces(text, 0)
  const chunks: string[] = []
  let cur: string[] = []
  let len = 0
  for (const p of small) {
    if (len + p.length > size && cur.length) {
      chunks.push(cur.join('').trim())
      // carry the tail of the previous chunk forward as overlap
      while (len > overlap && cur.length) { len -= cur[0].length; cur.shift() }
    }
    cur.push(p)
    len += p.length
  }
  if (cur.length) chunks.push(cur.join('').trim())
  return chunks.filter(c => c.length > 0)
}

const DOMAIN_HINTS = ['christmas', 'xmas', 'gift', 'present', 'holiday', 'santa', 'festive', 'celebration', 'december', 'winter', 'toy', 'decoration', 'tree', 'wrapping', 'seasonal']

export const looksOnDomain = (t: string) => {
  const s = t.toLowerCase()
  return DOMAIN_HINTS.some(w => s.includes(w))
}

/** Answers render as plain text: the model's markdown fights the page's own hierarchy. */
export function stripMarkdown(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/^(\s*)[-*•]\s+/gm, '$1• ')
    .replace(/(^|[^\w])\*(.+?)\*(?!\w)/g, '$1$2')
    .replace(/^#+\s+/gm, '')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/`(.+?)`/g, '$1')
    .replace(/(\$\d+(?:\.\d{2})?)([a-zA-Z])/g, '$1 $2')
    .replace(/([a-zA-Z])(\$\d)/g, '$1 $2')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+/g, ' ')
    .trim()
}

const PRICE = /\$\s?(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?/g

export function prices(text: string): number[] {
  const out: number[] = []
  for (const m of text.matchAll(PRICE)) out.push(Number(m[1].replace(/,/g, '') + '.' + (m[2] ?? '00')))
  return out
}

export type Grounding = { checked: number; grounded: number; missing: number[] }

/**
 * Deterministic grounding check: every price in the answer must appear in the
 * retrieved context. A model can still be wrong, but it cannot quietly invent a number.
 */
export function groundPrices(answer: string, context: string): Grounding {
  const have = new Set(prices(context).map(p => p.toFixed(2)))
  const asked = [...new Set(prices(answer).map(p => p.toFixed(2)))]
  const missing = asked.filter(p => !have.has(p)).map(Number)
  return { checked: asked.length, grounded: asked.length - missing.length, missing }
}
