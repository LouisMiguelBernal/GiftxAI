'use client'

/**
 * Browser-only loaders. pdf.js and transformers.js are pulled from jsDelivr at
 * runtime (pinned versions) so the bundler never touches their workers or WASM,
 * and the PDFs and embeddings never leave the visitor's machine.
 */

const PDFJS = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build'
const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js'
export const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2'

// eslint-disable-next-line @typescript-eslint/no-implied-eval
const load = new Function('u', 'return import(u)') as (u: string) => Promise<any> // eslint-disable-line @typescript-eslint/no-explicit-any

let pdfjs: Promise<any> | null = null // eslint-disable-line @typescript-eslint/no-explicit-any

export async function pdfText(file: File): Promise<string> {
  pdfjs ??= load(`${PDFJS}/pdf.min.mjs`).then(m => {
    m.GlobalWorkerOptions.workerSrc = `${PDFJS}/pdf.worker.min.mjs`
    return m
  })
  const lib = await pdfjs
  const doc = await lib.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise
  const pages: string[] = []
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i)
    const tc = await page.getTextContent()
    // Rebuild lines from the y-coordinate so name and price stay on one line.
    let lastY: number | null = null, line = '', out = ''
    for (const it of tc.items as { str: string; transform: number[]; hasEOL?: boolean }[]) {
      const y = Math.round(it.transform[5])
      if (lastY !== null && Math.abs(y - lastY) > 2) { out += line.trim() + '\n'; line = '' }
      line += (line && !line.endsWith(' ') ? ' ' : '') + it.str
      lastY = y
    }
    out += line.trim()
    pages.push(out)
  }
  return pages.join('\n\n')
}

type Extractor = (texts: string | string[], opts: { pooling: 'mean'; normalize: boolean }) => Promise<{ data: Float32Array; dims: number[] }>
let extractor: Promise<Extractor> | null = null

export function loadEmbedder(onProgress?: (pct: number) => void): Promise<Extractor> {
  extractor ??= load(TRANSFORMERS).then(async m => {
    const files = new Map<string, number>()
    return m.pipeline('feature-extraction', EMBED_MODEL, {
      dtype: 'q8',
      progress_callback: (p: { status: string; file?: string; progress?: number }) => {
        if (p.status === 'progress' && p.file) {
          files.set(p.file, p.progress ?? 0)
          const v = [...files.values()]
          onProgress?.(v.reduce((a, b) => a + b, 0) / v.length)
        }
      },
    }) as Promise<Extractor>
  })
  extractor.catch(() => { extractor = null })
  return extractor
}

export async function embed(texts: string[], batch = 16): Promise<Float32Array[]> {
  const ex = await loadEmbedder()
  const out: Float32Array[] = []
  for (let i = 0; i < texts.length; i += batch) {
    const slice = texts.slice(i, i + batch)
    const r = await ex(slice, { pooling: 'mean', normalize: true })
    const d = r.dims[r.dims.length - 1]
    for (let j = 0; j < slice.length; j++) out.push(r.data.slice(j * d, (j + 1) * d))
  }
  return out
}
