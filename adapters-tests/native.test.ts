import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  createNativePipeline,
  NativeRuntimeError,
  type NativePipelineOptions,
} from '../adapters/node/native.js';

// This includes real child startup, which can exceed 500 ms on busy CI runners.
// Hanging-request tests still exercise the same bounded timeout and cleanup.
const FIXTURE_TIMEOUT_MS = 5_000;

interface FixtureOptions {
  ignoreShutdown?: boolean;
  closeStdinAfterInfo?: boolean;
  startup?: 'invalid' | 'unversioned' | 'hang' | 'exit';
  pidFile?: string;
}

// Real subprocesses exercise pipe transport and process cleanup.
// No models, ONNX Runtime or runtime mocks are needed.
function fakeRuntime(settings: FixtureOptions = {}): string {
  return `
const { createInterface } = require('node:readline');
const { writeFileSync, closeSync } = require('node:fs');
const settings = ${JSON.stringify(settings)};
if (settings.pidFile) {
  writeFileSync(settings.pidFile, String(process.pid));
}

const capabilities = {
  tokenize: true,
  sentenceSplit: true,
  multiwordExpansion: false,
  pos: false,
  morphology: false,
  lemma: 'none',
};
const info = {
  language: 'fake',
  capabilities,
  runtimeVersion: '0.2.0',
  stanzaVersion: '1.11.0',
  resourceVersion: '1.11.0',
  unicodeVersion: '18.0.0',
};

const write = (stream, value) => new Promise((resolve, reject) => {
  stream.write(value, error => {
    if (error) {
      reject(error);
    } else {
      resolve();
    }
  });
});
const send = value => write(process.stdout, JSON.stringify(value) + '\\n');

let active = 0;
createInterface({ input: process.stdin, crlfDelay: Infinity }).on('line', async line => {
  try {
    const { id, method, params } = JSON.parse(line);

    if (method === 'info') {
      if (settings.startup === 'hang') {
        return;
      }
      if (settings.startup === 'exit') {
        await write(process.stderr, 'failed to load model');
        process.exit(4);
      }
      if (settings.startup === 'unversioned') {
        delete info.unicodeVersion;
      }

      await send({
        id,
        result: settings.startup === 'invalid' ? { ...info, language: 123 } : info,
      });

      if (settings.closeStdinAfterInfo) {
        process.stdin.on('error', () => {});
        process.stdin.destroy();
        try {
          closeSync(0);
        } catch (error) {
          if (error.code !== 'EBADF') {
            throw error;
          }
        }
        // Stay alive with a broken input pipe so cleanup can be verified.
        setInterval(() => {}, 1000);
      }
      return;
    }

    if (method === 'shutdown') {
      await send({ id, result: { closed: true } });
      if (settings.ignoreShutdown) {
        setInterval(() => {}, 1000);
        return;
      }
      process.exit(0);
    }

    const text = params.text;
    if (text === 'hang') {
      return;
    }
    if (text === 'error') {
      await send({
        id,
        error: { code: 'example_failure', message: 'Example request failed' },
      });
      return;
    }
    if (text === 'exit') {
      await write(process.stderr, 'x'.repeat(65536) + ' native died');
      process.exit(7);
    }
    if (text === 'invalid') {
      await write(process.stdout, 'invalid json\\n');
      return;
    }
    if (text === 'invalid_utf8') {
      await write(process.stdout, Buffer.from([0xff, 0x0a]));
      return;
    }
    if (text === 'unexpected_id') {
      await send({ id: id + 100, result: {} });
      return;
    }
    if (text === 'ambiguous') {
      await send({ id, result: {}, error: { message: 'both' } });
      return;
    }

    active++;
    if (active > 1) {
      await send({
        id,
        error: { code: 'overlap', message: 'Concurrent requests overlap' },
      });
      active--;
      return;
    }
    if (text === 'slow') {
      await new Promise(resolve => setTimeout(resolve, 25));
    }

    const written = text === 'pid' ? String(process.pid) : text;
    const endOffset = params.offsetUnit === 'utf16'
      ? written.length
      : params.offsetUnit === 'utf8'
        ? Buffer.byteLength(written)
        : [...written].length;
    const token = {
      text: written,
      start: 0,
      end: endOffset,
      words: [{ text: written, upos: null, xpos: null, feats: null, lemma: null }],
    };
    const bytes = Buffer.from(JSON.stringify({
      id,
      result: {
        language: 'fake',
        offsetUnit: params.offsetUnit,
        sentences: [{ tokens: [token] }],
      },
    }) + '\\n');

    if (text.includes('😀')) {
      // Split both a UTF-8 character and the final JSON bytes across writes.
      const index = bytes.indexOf(0xf0);
      await write(process.stdout, bytes.subarray(0, index + 2));
      await new Promise(resolve => setTimeout(resolve, 2));
      await write(process.stdout, bytes.subarray(index + 2, bytes.length - 3));
      await write(process.stdout, bytes.subarray(bytes.length - 3));
    } else {
      await write(process.stdout, bytes);
    }
    active--;
  } catch (error) {
    console.error(error);
    process.exit(9);
  }
});
`;
}

function options(
  overrides: Partial<NativePipelineOptions> = {},
  fixture: FixtureOptions = {},
): NativePipelineOptions {
  return {
    executable: process.env.STANZA_ONNX_TEST_NODE ?? 'node',
    executableArgs: ['--eval', fakeRuntime(fixture), '--'],
    modelDir: 'model-free',
    ortLibrary: 'unused',
    timeoutMs: FIXTURE_TIMEOUT_MS,
    ...overrides,
  };
}

async function assertStopped(pid: number): Promise<void> {
  for (let attempt = 0; attempt < 30; attempt++) {
    try {
      process.kill(pid, 0);
    } catch (error) {
      assert.equal((error as NodeJS.ErrnoException).code, 'ESRCH');
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.fail(`Child ${pid} is still alive`);
}

describe('persistent native JavaScript adapter', () => {
  test('keeps one child alive across analyses and releases it on close', async () => {
    const pipeline = await createNativePipeline(options());
    const pid = Number((await pipeline.analyze('pid'))[0].tokens[0].text);
    try {
      assert.equal(pipeline.language, 'fake');
      assert.equal(pipeline.capabilities.lemma, 'none');
      assert.equal(pipeline.info.unicodeVersion, '18.0.0');
      assert.equal(pipeline.info.tokenizerKind, 'stanza');
      assert.ok(Object.isFrozen(pipeline.capabilities));
      assert.equal(Number((await pipeline.analyze('pid'))[0].tokens[0].text), pid);
      assert.equal((await pipeline.analyze('second'))[0].tokens[0].text, 'second');
    } finally {
      await pipeline.close();
    }
    await assertStopped(pid);
  });

  for (const [offsetUnit, expectedEnd] of [
    ['unicode-code-point', 3],
    ['utf16', 4],
    ['utf8', 7],
  ] as const) {
    test(`decodes split UTF-8 JSON and ${offsetUnit} offsets`, async () => {
      const pipeline = await createNativePipeline(options({ offsetUnit }));
      try {
        const token = (await pipeline.analyze('a😀é'))[0].tokens[0];
        assert.equal(token.text, 'a😀é');
        assert.equal(token.end, expectedEnd);
      } finally {
        await pipeline.close();
      }
    });
  }

  test('serializes accepted requests; close drains them and rejects new calls', async () => {
    const pipeline = await createNativePipeline(options());
    const first = pipeline.analyze('slow');
    const second = pipeline.analyze('second');
    const closing = pipeline.close();
    assert.equal(pipeline.close(), closing);
    await assert.rejects(pipeline.analyze('too late'), /Pipeline is closed/);
    assert.equal((await first)[0].tokens[0].text, 'slow');
    assert.equal((await second)[0].tokens[0].text, 'second');
    await closing;
    await pipeline.close();
  });

  test('a structured request error leaves the child usable', async () => {
    const pipeline = await createNativePipeline(options());
    try {
      await assert.rejects(pipeline.analyze('error'), (error) => {
        assert.ok(error instanceof NativeRuntimeError);
        assert.equal(error.code, 'example_failure');
        return true;
      });
      assert.equal((await pipeline.analyze('recovered'))[0].tokens[0].text, 'recovered');
    } finally {
      await pipeline.close();
    }
  });

  test('child exit rejects queued work with bounded stderr diagnostics', async () => {
    const pipeline = await createNativePipeline(options());
    const first = pipeline.analyze('exit');
    const second = pipeline.analyze('pending');
    // Attach both rejection handlers immediately, including on Node strict mode.
    const firstCheck = assert.rejects(first, (error) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /exited with code 7/);
      assert.match(error.message, /native died/);
      assert.ok(error.message.length < 17_000);
      return true;
    });
    const secondCheck = assert.rejects(second, /exited with code 7/);
    await Promise.all([firstCheck, secondCheck]);
    await assert.rejects(pipeline.close(), /exited with code 7/);
  });

  test('request timeout kills the child and rejects queued work', async () => {
    const pipeline = await createNativePipeline(options());
    const pid = Number((await pipeline.analyze('pid'))[0].tokens[0].text);
    const checks = [
      assert.rejects(pipeline.analyze('hang'), /timed out/),
      assert.rejects(pipeline.analyze('pending'), /timed out/),
    ];
    await Promise.all(checks);
    await assert.rejects(pipeline.close(), /timed out/);
    await assertStopped(pid);
  });

  for (const [text, message] of [
    ['invalid', /invalid JSON/],
    ['invalid_utf8', /encoded data|UTF-8/i],
    ['unexpected_id', /unexpected response id/],
    ['ambiguous', /either result or error/],
  ] as const) {
    test(`rejects ${text} protocol output and cleans up`, async () => {
      const pipeline = await createNativePipeline(options());
      await assert.rejects(pipeline.analyze(text), message);
      await assert.rejects(pipeline.close(), message);
    });
  }

  test('shutdown timeout kills a child that acknowledged but never exited', async () => {
    const pipeline = await createNativePipeline(options({}, { ignoreShutdown: true }));
    const pid = Number((await pipeline.analyze('pid'))[0].tokens[0].text);
    await assert.rejects(pipeline.close(), /shutdown timed out/);
    await assertStopped(pid);
  });

  for (const [startup, message] of [
    ['invalid', /invalid pipeline information/],
    ['unversioned', /invalid pipeline information/],
    ['hang', /info request timed out/],
    ['exit', /failed to load model/],
  ] as const) {
    test(`rejects ${startup} startup and terminates its child`, async () => {
      const directory = await mkdtemp(join(tmpdir(), 'stanza-adapter-'));
      const pidFile = join(directory, 'pid');
      try {
        await assert.rejects(createNativePipeline(options({}, { startup, pidFile })), message);
        await assertStopped(Number(await readFile(pidFile, 'utf8')));
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    });
  }

  test('reports a missing executable as a startup error', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'stanza-adapter-missing-'));
    try {
      await assert.rejects(
        createNativePipeline(options({ executable: join(directory, 'missing') })),
        /Cannot start native runtime|ENOENT/,
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test('broken stdin rejects pending requests and cleans up without an uncaught EPIPE', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'stanza-adapter-input-'));
    const pidFile = join(directory, 'pid');
    try {
      const pipeline = await createNativePipeline(
        options({}, { closeStdinAfterInfo: true, pidFile }),
      );
      const pid = Number(await readFile(pidFile, 'utf8'));
      await new Promise((resolve) => setTimeout(resolve, 30));
      // Node reports EPIPE immediately. Some compatible stream implementations
      // defer detecting a closed input pipe; the deadline must still bound it.
      const failure = /input failed|timed out/;
      const checks = [
        assert.rejects(pipeline.analyze('first'), failure),
        assert.rejects(pipeline.analyze('queued'), failure),
      ];
      await Promise.all(checks);
      await assert.rejects(pipeline.close(), failure);
      await assertStopped(pid);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  test('validates configuration before spawning', async () => {
    await assert.rejects(createNativePipeline(options({ threads: 0 })), /threads/);
    await assert.rejects(createNativePipeline(options({ timeoutMs: 0 })), /timeoutMs/);
    await assert.rejects(createNativePipeline(options({ executable: '' })), /executable/);
    await assert.rejects(createNativePipeline(options({ ortLibrary: '' })), /ortLibrary/);
    await assert.rejects(createNativePipeline(options({ ortLibrary: 0 as never })), /ortLibrary/);
    await assert.rejects(
      createNativePipeline(options({ offsetUnit: 'invalid' as never })),
      /offsetUnit/,
    );
    await assert.rejects(
      createNativePipeline(options({ provider: 'invalid' as never })),
      /provider/,
    );
    await assert.rejects(
      createNativePipeline(options({ executableArgs: [0] as never })),
      /executableArgs/,
    );
  });

  test('rejects invalid arguments and unpaired surrogates without interrupting the child', async () => {
    const pipeline = await createNativePipeline(options());
    try {
      await assert.rejects(pipeline.analyze(123 as never), /expects a string/);
      for (const text of ['\ud800', 'a\udfff']) {
        await assert.rejects(pipeline.analyze(text), /unpaired UTF-16 surrogate/);
      }
      assert.equal((await pipeline.analyze('valid😀'))[0].tokens[0].text, 'valid😀');
    } finally {
      await pipeline.close();
    }
  });
});
