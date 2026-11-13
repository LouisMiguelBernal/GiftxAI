import { test } from 'node:test'
import assert from 'node:assert/strict'
import { splitText, cleanExtracted, stripMarkdown, groundPrices, prices, type Chunk } from './text.ts'
import { chooseK, bm25, vectorSearch } from './retrieve.ts'

test('splitter respects size and overlaps neighbours', () => {
  const para = (i: number) => `Item ${i} costs $${i}.00 and is a lovely holiday gift for the family.`
  const text = Array.from({ length: 80 }, (_, i) => para(i)).join('\n')
  const chunks = splitText(text, 800, 150)
  assert.ok(chunks.length > 4)
  for (const c of chunks) assert.ok(c.length <= 800, `chunk of ${c.length}`)
  const lastLine = chunks[0].split('\n').pop()!
  assert.ok(chunks[1].includes(lastLine), 'overlap carries the tail forward')
})

test('splitter hard-cuts a single huge word', () => {
  const chunks = splitText('x'.repeat(2000), 800, 150)
  assert.ok(chunks.every(c => c.length <= 800))
  assert.ok(chunks.length >= 3)
})

test('cleanExtracted separates glued prices', () => {
  assert.equal(cleanExtracted('Kettle$129.50Smart'), 'Kettle $129.50 Smart')
})

test('stripMarkdown leaves plain text', () => {
  assert.equal(stripMarkdown('**Top** gifts:\n- *Kettle* `x`\n# Head'), 'Top gifts:\n• Kettle x\nHead')
})

test('price grounding flags invented numbers', () => {
  const ctx = 'Copper Espresso Machine $649.99\nCashmere Throw $295.00\nWallet $45'
  const g = groundPrices('1. Copper Espresso Machine $649.99\n2. Throw $295\n3. Mystery $1,234.00\n4. Wallet $45.00', ctx)
  assert.equal(g.checked, 4)
  assert.equal(g.grounded, 3)
  assert.deepEqual(g.missing, [1234])
  assert.deepEqual(prices('$1,299.00 and $5'), [1299, 5])
})

test('chooseK widens for rankings', () => {
  assert.equal(chooseK('what is in the hamper?'), 8)
  assert.equal(chooseK('cheapest gifts'), 30)
  assert.equal(chooseK('top 15 most expensive'), 60)
  assert.equal(chooseK('topaz necklace'), 8)
})

const C = (text: string, id: number): Chunk => ({ id, source: 's.pdf', n: id, text })

test('bm25 ranks the matching chunk first', () => {
  const chunks = [C('Ceramic mug set of four', 0), C('Noise cancelling headphones with long battery', 1), C('Wooden train set', 2)]
  assert.equal(bm25('headphones battery', chunks, 3)[0].chunk.id, 1)
})

test('vector search orders by cosine and dedupes', () => {
  const chunks = [C('a\nx', 0), C('a\ny', 1), C('b', 2)]
  const v = (x: number, y: number) => Float32Array.from([x, y])
  const hits = vectorSearch(v(1, 0), chunks, [v(0.9, 0.1), v(0.8, 0.2), v(0, 1)], 3)
  assert.deepEqual(hits.map(h => h.chunk.id), [0, 2])
})
