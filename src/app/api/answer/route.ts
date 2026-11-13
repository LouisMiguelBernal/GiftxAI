import { NextResponse } from 'next/server'
import { MODEL, ANSWER_SYSTEM, answerPrompt, validatorSystem, validationPrompt } from '@/lib/prompts'
import { stripMarkdown, groundPrices } from '@/lib/text'

export const runtime = 'nodejs'
export const maxDuration = 60

const GROQ = (process.env.GROQ_BASE_URL || 'https://api.groq.com/openai/v1') + '/chat/completions'
const MAX_CONTEXT = 30_000
const MAX_QUESTION = 600

/** Light per-instance limiter for the shared server key. A visitor's own key is not limited here. */
const hits = new Map<string, number[]>()
function limited(ip: string): boolean {
  const now = Date.now(), win = 10 * 60_000, max = 20
  const list = (hits.get(ip) ?? []).filter(t => now - t < win)
  list.push(now)
  hits.set(ip, list)
  return list.length > max
}

async function call(key: string, system: string, user: string): Promise<string> {
  const r = await fetch(GROQ, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      temperature: 0.01,
      top_p: 0.85,
      max_tokens: 2500,
    }),
  })
  if (!r.ok) {
    const body = await r.text().catch(() => '')
    const msg = r.status === 401 || r.status === 403 ? 'Groq rejected the API key.' : r.status === 429 ? 'Groq rate limit reached. Try again in a minute.' : `Groq error ${r.status}.`
    throw Object.assign(new Error(msg), { status: r.status, detail: body.slice(0, 200) })
  }
  const j = await r.json()
  return String(j.choices?.[0]?.message?.content ?? '').trim()
}

export async function GET() {
  return NextResponse.json({ serverKey: Boolean(process.env.GROQ_API_KEY), model: MODEL })
}

export async function POST(req: Request) {
  let body: { question?: unknown; context?: unknown; passes?: unknown }
  try { body = await req.json() } catch { return NextResponse.json({ error: 'Invalid JSON.' }, { status: 400 }) }

  const question = typeof body.question === 'string' ? body.question.trim().slice(0, MAX_QUESTION) : ''
  const context = typeof body.context === 'string' ? body.context.slice(0, MAX_CONTEXT) : ''
  const passes = Math.max(0, Math.min(3, Math.trunc(Number(body.passes) || 0)))
  if (!question || !context) return NextResponse.json({ error: 'A question and retrieved context are required.' }, { status: 400 })

  const own = req.headers.get('x-groq-key')?.trim()
  const key = own || process.env.GROQ_API_KEY
  if (!key) return NextResponse.json({ error: 'No Groq key is configured. Add your own key under Generation.' }, { status: 401 })
  if (!own) {
    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 'local'
    if (limited(ip)) return NextResponse.json({ error: 'Demo limit reached (20 questions per 10 minutes). Add your own Groq key to continue.' }, { status: 429 })
  }

  const t0 = Date.now()
  const steps: { label: string; ms: number }[] = []
  try {
    let draft = await call(key, ANSWER_SYSTEM, answerPrompt(question, context))
    steps.push({ label: 'Draft', ms: Date.now() - t0 })
    let done = 0
    for (let n = 1; n <= passes; n++) {
      const s = Date.now()
      try {
        draft = await call(key, validatorSystem(n, passes), validationPrompt(draft, context, n, passes))
        done++
        steps.push({ label: `Validation ${n}`, ms: Date.now() - s })
      } catch {
        break // a failed validation pass should not lose the draft
      }
    }
    const answer = stripMarkdown(draft)
    return NextResponse.json({ answer, passes: done, steps, ms: Date.now() - t0, model: MODEL, grounding: groundPrices(answer, context) })
  } catch (e) {
    const err = e as Error & { status?: number }
    return NextResponse.json({ error: err.message }, { status: err.status === 401 || err.status === 403 ? 401 : err.status === 429 ? 429 : 502 })
  }
}
