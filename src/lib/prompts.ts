export const MODEL = 'llama-3.3-70b-versatile'

export const ANSWER_SYSTEM =
  'You are a retrieval-grounded answer engine. You answer only from the context you are given, never from prior knowledge. You sort numerically with care and you return clean plain text.'

export function answerPrompt(question: string, context: string): string {
  return `Answer the question using ONLY the retrieved context below.

CONTEXT
${context}

QUESTION
${question}

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
- Return the answer only, with no preamble.`
}

const FOCUS: Record<number, string> = {
  1: 'Extract every item and price; check nothing is missing.',
  2: 'Verify the numerical ordering element by element and re-sort if wrong.',
  3: 'Final sweep on counts, ordering and formatting.',
}

export function validatorSystem(n: number, total: number): string {
  return `You are a precision validator, pass ${n} of ${total}. You catch subtle sorting and counting errors and you never invent facts.`
}

export function validationPrompt(draft: string, context: string, n: number, total: number): string {
  return `You are validating a draft answer. Pass ${n} of ${total}. ${FOCUS[n] ?? 'Verify accuracy, ordering and formatting.'}

CONTEXT (the only allowed source of facts)
${context}

DRAFT
${draft}

Check and correct:
- Every item and price in the draft appears in the context. Remove anything that does not.
- Numerical ordering is exactly right (descending for "top"/"expensive",
  ascending for "cheapest"). Re-sort the whole list if any item is out of place.
- The item count matches what was asked for, with no duplicates.
- Prices are formatted consistently as $amount.
- All markdown is removed: no **, __, *, _, #, backticks.

Return only the corrected answer. No commentary.`
}
