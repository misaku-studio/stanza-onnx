/** Model-dependent English adapter checks; run under Node.js after building tests. */
import assert from 'node:assert/strict';
import { parseArgs } from 'node:util';
import { createNativePipeline } from '../adapters/node/native.js';

const { values } = parseArgs({
  options: {
    binary: { type: 'string' },
    'ort-library': { type: 'string' },
    bundle: { type: 'string' },
  },
});
if (!values.binary || !values['ort-library'] || !values.bundle) {
  throw new Error('Usage: integration.js --binary PATH --ort-library PATH --bundle ENGLISH_BUNDLE');
}

const pipeline = await createNativePipeline({
  executable: values.binary,
  modelDir: values.bundle,
  ortLibrary: values['ort-library'],
  offsetUnit: 'utf16',
});
try {
  assert.equal(pipeline.language, 'en');
  assert.equal(pipeline.info.unicodeVersion, '18.0.0');
  assert.equal(pipeline.capabilities.lemma, 'trained');
  const text = '😊 The children read books.';
  const first = await pipeline.analyze(text);
  assert.equal(first[0].tokens[0].text, '😊');
  assert.equal(first[0].tokens[0].end, 2);
  for (const sentence of first) {
    for (const token of sentence.tokens) {
      assert.equal(text.slice(token.start, token.end), token.text);
    }
  }
  // Public argument rejection must leave the persistent child usable.
  await assert.rejects(pipeline.analyze(false as never), /expects a string/);
  await assert.rejects(pipeline.analyze('\ud800'), /unpaired UTF-16 surrogate/);
  const second = await pipeline.analyze('The children were reading books.');
  const lemmas = second.flatMap((sentence) =>
    sentence.tokens.flatMap((token) => token.words.map((word) => word.lemma)),
  );
  for (const lemma of ['child', 'be', 'read', 'book']) {
    assert.ok(lemmas.includes(lemma));
  }
} finally {
  await pipeline.close();
}
await pipeline.close();
await assert.rejects(pipeline.analyze('closed'), /Pipeline is closed/);
console.log(
  JSON.stringify({
    analyses: 2,
    nativeProcess: true,
    utf16Offsets: true,
    argumentErrorRecovery: true,
    lemmas: true,
    gracefulClose: true,
  }),
);
