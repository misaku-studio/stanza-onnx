/** Real native Bengali adapter checks. No ONNX Runtime library or model download. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import {
  createNativePipeline,
  type NativeOffsetUnit,
  type Sentence,
} from '../adapters/node/native.js';

interface Fixture {
  id: string;
  text: string;
  expectedTokens: string[][];
}

const { values } = parseArgs({
  options: {
    binary: { type: 'string' },
    bundle: { type: 'string' },
    fixtures: { type: 'string', default: 'fixtures/bn.json' },
  },
});
if (!values.binary || !values.bundle) {
  throw new Error('Usage: bengali.js --binary PATH --bundle BENGALI_BUNDLE [--fixtures JSON]');
}
const document = JSON.parse(await readFile(values.fixtures, 'utf8')) as { texts: Fixture[] };
assert.ok(Array.isArray(document.texts) && document.texts.length > 0);
const fixtures = document.texts;
const capabilities = {
  tokenize: true,
  sentenceSplit: true,
  multiwordExpansion: false,
  pos: false,
  morphology: false,
  lemma: 'none',
};
// An attempted native-library load must fail, rather than silently using a local installation.
process.env.ORT_DYLIB_PATH = resolve(values.bundle, 'missing-onnxruntime-library');

function sourceOffset(prefix: string, unit: NativeOffsetUnit): number {
  if (unit === 'utf16') {
    return prefix.length;
  }
  return unit === 'utf8' ? Buffer.byteLength(prefix, 'utf8') : Array.from(prefix).length;
}

function sourceSlice(text: string, start: number, end: number, unit: NativeOffsetUnit): string {
  if (unit === 'utf16') {
    return text.slice(start, end);
  }
  if (unit === 'utf8') {
    return new TextDecoder('utf-8', { fatal: true }).decode(
      Buffer.from(text, 'utf8').subarray(start, end),
    );
  }
  return Array.from(text).slice(start, end).join('');
}

function check(entry: Fixture, sentences: Sentence[], unit: NativeOffsetUnit): void {
  const label = `${entry.id} (${unit})`;
  assert.deepEqual(
    sentences.map((sentence) => sentence.tokens.map((token) => token.text)),
    entry.expectedTokens,
    `${label}: authored token/sentence expectations`,
  );
  let cursor = 0;
  for (const sentence of sentences) {
    for (const token of sentence.tokens) {
      const start = entry.text.indexOf(token.text, cursor);
      assert.ok(start >= cursor, `${label}: token must occur in original source`);
      const end = start + token.text.length;
      assert.equal(token.start, sourceOffset(entry.text.slice(0, start), unit));
      assert.equal(token.end, sourceOffset(entry.text.slice(0, end), unit));
      assert.equal(sourceSlice(entry.text, token.start, token.end, unit), token.text);
      assert.deepEqual(token.words, [
        { text: token.text, upos: null, xpos: null, feats: null, lemma: null },
      ]);
      cursor = end;
    }
  }
}

const units: NativeOffsetUnit[] = ['unicode-code-point', 'utf16', 'utf8'];
for (const offsetUnit of units) {
  // ortLibrary is intentionally omitted: rule-only bundles do not need it.
  const pipeline = await createNativePipeline({
    executable: resolve(values.binary),
    modelDir: resolve(values.bundle),
    offsetUnit,
    timeoutMs: 30_000,
  });
  try {
    assert.equal(pipeline.language, 'bn');
    assert.equal(pipeline.info.unicodeVersion, '18.0.0');
    assert.equal(pipeline.info.tokenizerKind, 'bengali-rule');
    assert.deepEqual(pipeline.capabilities, capabilities);
    // Queue real requests together; one persistent native process must preserve order.
    const analyses = await Promise.all(fixtures.map((entry) => pipeline.analyze(entry.text)));
    fixtures.forEach((entry, index) => check(entry, analyses[index], offsetUnit));
    await assert.rejects(pipeline.analyze(false as never), /expects a string/);
    await assert.rejects(pipeline.analyze('\ud800'), /unpaired UTF-16 surrogate/);
    check(fixtures[0], await pipeline.analyze(fixtures[0].text), offsetUnit);
    assert.deepEqual(await pipeline.analyze(''), []);
  } finally {
    await pipeline.close();
  }
  await pipeline.close();
  await assert.rejects(pipeline.analyze('বন্ধ'), /Pipeline is closed/);
}
console.log(
  JSON.stringify({
    language: 'bn',
    fixtureCases: fixtures.length,
    offsetUnits: units,
    nativeProcess: true,
    onnxRuntimeRequired: false,
    ortLibraryOmitted: true,
    argumentErrorRecovery: true,
    queuedReuse: true,
    gracefulClose: true,
  }),
);
