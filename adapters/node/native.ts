import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import type { Sentence, Capabilities } from './types.js';
export type { Word, Token, Sentence, Capabilities } from './types.js';

export type NativeOffsetUnit = 'unicode-code-point' | 'utf16' | 'utf8';
export type NativeProvider = 'cpu' | 'coreml' | 'directml';

export interface NativePipelineOptions {
  executable: string;
  /** Arguments placed before the native runtime's flags, for executable wrappers. */
  executableArgs?: readonly string[];
  modelDir: string;
  /** Optional for rule-only bundles; neural bundles need this or ORT_DYLIB_PATH. */
  ortLibrary?: string;
  threads?: number;
  provider?: NativeProvider;
  offsetUnit?: NativeOffsetUnit;
  /** Deadline for startup, each request, and graceful shutdown. */
  timeoutMs?: number;
}

export interface NativePipelineInfo {
  language: string;
  capabilities: Readonly<Capabilities>;
  runtimeVersion: string;
  stanzaVersion: string;
  resourceVersion: string;
  unicodeVersion: string;
  tokenizerKind: 'stanza' | 'bengali-rule';
}

export interface NativePipeline {
  readonly language: string;
  readonly capabilities: Readonly<Capabilities>;
  readonly info: Readonly<NativePipelineInfo>;
  analyze(text: string): Promise<Sentence[]>;
  close(): Promise<void>;
}

export class NativeRuntimeError extends Error {
  readonly code?: string;
  constructor(message: string, code?: string) {
    super(message);
    this.code = code;
    this.name = 'NativeRuntimeError';
  }
}

type Child = ChildProcessWithoutNullStreams;
interface Pending {
  resolve(value: unknown): void;
  reject(error: Error): void;
  timer: ReturnType<typeof setTimeout>;
}
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const STDERR_TAIL = 16_384;

// Rust strings contain Unicode scalar values; reject unpaired JavaScript
// surrogates before sending JSON, rather than replacing source characters.
function wellFormedUnicode(text: string): boolean {
  for (let index = 0; index < text.length; index++) {
    const code = text.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = text.charCodeAt(++index);
      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        return false;
      }
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return false;
    }
  }
  return true;
}

/** A single child owns its loaded sessions; requests use JSON lines, without a shell. */
class NativeTransport {
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private failure?: Error;
  private stderrTail = '';
  private shutdownAcknowledged = false;
  private stdoutTask: Promise<void>;
  private stderrTask: Promise<void>;
  private exitTask: Promise<number>;

  private child: Child;
  private timeoutMs: number;

  constructor(child: Child, timeoutMs: number) {
    this.child = child;
    this.timeoutMs = timeoutMs;
    child.on('error', (error) =>
      this.fail(this.diagnostic(`Cannot start native runtime: ${error.message}`)),
    );
    child.stdin.on('error', (error) =>
      this.fail(this.diagnostic(`Native runtime input failed: ${error.message}`)),
    );
    this.stdoutTask = this.readStdout().catch((error) => this.fail(error));
    this.stderrTask = this.readStderr().catch((error) => this.fail(error));
    const exited = new Promise<{
      code: number | null;
      signal: NodeJS.Signals | null;
    }>((resolve) => {
      child.once('close', (code, signal) => resolve({ code, signal }));
    });
    this.exitTask = exited
      .then(async ({ code, signal }) => {
        // close occurs after stdio closes; finish decoding buffered responses first.
        await Promise.all([this.stdoutTask, this.stderrTask]);
        if (code !== 0 || this.pending.size || !this.shutdownAcknowledged) {
          this.fail(
            this.diagnostic(
              `Native runtime exited with code ${code}${signal ? ` (${signal})` : ''}`,
            ),
            false,
          );
        }
        return code ?? -1;
      })
      .catch((error) => {
        this.fail(error);
        return -1;
      });
  }

  private diagnostic(message: string): Error {
    return new NativeRuntimeError(message + (this.stderrTail ? `\n${this.stderrTail}` : ''));
  }

  private fail(value: unknown, kill = true): void {
    // Transport failures are terminal. Every queued request sees the first error.
    if (this.failure) {
      return;
    }
    const error = value instanceof Error ? value : new NativeRuntimeError(String(value));
    this.failure = error;
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
    if (kill) {
      this.terminateChild();
    }
  }

  private terminateChild(): void {
    try {
      this.child.kill('SIGKILL');
    } catch {
      // The child may already have exited.
    }
    try {
      this.child.stdin.end();
    } catch {
      // Its input pipe may already be broken or closed.
    }
  }

  private receive(line: string): void {
    if (!line.trim() || this.failure) {
      return;
    }
    let response: unknown;
    try {
      response = JSON.parse(line);
    } catch {
      throw this.diagnostic('Native runtime returned invalid JSON');
    }
    if (!record(response) || !Number.isSafeInteger(response.id)) {
      throw this.diagnostic('Native runtime returned an invalid response id');
    }
    const id = response.id as number;
    const pending = this.pending.get(id);
    if (!pending) {
      throw this.diagnostic(`Native runtime returned an unexpected response id ${id}`);
    }
    if (Object.hasOwn(response, 'error') === Object.hasOwn(response, 'result')) {
      throw this.diagnostic('Native response must contain either result or error');
    }
    if (Object.hasOwn(response, 'error')) {
      if (!record(response.error) || typeof response.error.message !== 'string') {
        throw this.diagnostic('Native runtime returned an invalid error response');
      }
      clearTimeout(pending.timer);
      this.pending.delete(id);
      // A valid request error rejects only this call; the process stays usable.
      pending.reject(
        new NativeRuntimeError(
          response.error.message,
          typeof response.error.code === 'string' ? response.error.code : undefined,
        ),
      );
    } else {
      clearTimeout(pending.timer);
      this.pending.delete(id);
      pending.resolve(response.result);
    }
  }

  private async readStdout(): Promise<void> {
    const decoder = new TextDecoder('utf-8', { fatal: true });
    let buffered = '';
    for await (const value of this.child.stdout) {
      buffered += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffered.indexOf('\n')) >= 0) {
        this.receive(buffered.slice(0, end));
        buffered = buffered.slice(end + 1);
      }
    }
    buffered += decoder.decode();
    if (buffered.trim()) {
      this.receive(buffered);
    }
  }

  private async readStderr(): Promise<void> {
    const decoder = new TextDecoder();
    for await (const value of this.child.stderr) {
      this.stderrTail = (this.stderrTail + decoder.decode(value, { stream: true })).slice(
        -STDERR_TAIL,
      );
    }
    this.stderrTail = (this.stderrTail + decoder.decode()).slice(-STDERR_TAIL);
  }

  request(method: string, params: unknown = {}): Promise<unknown> {
    if (this.failure) {
      return Promise.reject(this.failure);
    }
    const id = this.nextId++;
    const response = new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(
        () =>
          this.fail(
            this.diagnostic(
              `Native runtime ${method} request timed out after ${this.timeoutMs} ms`,
            ),
          ),
        this.timeoutMs,
      );
      this.pending.set(id, { resolve, reject, timer });
    });
    // Register the response and deadline before writing, including synchronous errors.
    try {
      this.child.stdin.write(JSON.stringify({ id, method, params }) + '\n', (error) => {
        if (error) {
          this.fail(this.diagnostic(`Native runtime input failed: ${error.message}`));
        }
      });
    } catch (error) {
      this.fail(
        this.diagnostic(
          `Native runtime input failed: ${error instanceof Error ? error.message : String(error)}`,
        ),
      );
    }
    return response;
  }

  async shutdown(): Promise<void> {
    const result = await this.request('shutdown');
    if (!record(result) || result.closed !== true) {
      throw this.diagnostic('Native runtime did not acknowledge shutdown');
    }
    this.shutdownAcknowledged = true;
    this.child.stdin.end();
    await this.waitForExit();
    if (this.failure) {
      throw this.failure;
    }
  }

  private async waitForExit(): Promise<void> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        this.exitTask,
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => {
            const error = this.diagnostic(
              `Native runtime shutdown timed out after ${this.timeoutMs} ms`,
            );
            this.fail(error);
            reject(error);
          }, this.timeoutMs);
        }),
      ]);
    } finally {
      if (timer) {
        clearTimeout(timer);
      }
    }
  }

  async dispose(): Promise<void> {
    this.terminateChild();
    await this.waitForExit();
  }
}

function parseInfo(value: unknown): NativePipelineInfo {
  if (
    !record(value) ||
    typeof value.language !== 'string' ||
    !value.language ||
    typeof value.runtimeVersion !== 'string' ||
    typeof value.stanzaVersion !== 'string' ||
    typeof value.resourceVersion !== 'string' ||
    typeof value.unicodeVersion !== 'string' ||
    !value.unicodeVersion ||
    !record(value.capabilities)
  ) {
    throw new NativeRuntimeError('Native runtime returned invalid pipeline information');
  }
  const capabilities = value.capabilities;
  for (const name of ['tokenize', 'sentenceSplit', 'multiwordExpansion', 'pos', 'morphology']) {
    if (typeof capabilities[name] !== 'boolean') {
      throw new NativeRuntimeError(`Invalid native capability ${name}`);
    }
  }
  if (!['trained', 'identity', 'none'].includes(capabilities.lemma as string)) {
    throw new NativeRuntimeError('Invalid native lemma capability');
  }
  // Older native runtimes predate the explicit kind and use Stanza tokenizers.
  const tokenizerKind = value.tokenizerKind === undefined ? 'stanza' : value.tokenizerKind;
  if (tokenizerKind !== 'stanza' && tokenizerKind !== 'bengali-rule') {
    throw new NativeRuntimeError('Invalid native tokenizer kind');
  }
  return Object.freeze({
    language: value.language,
    capabilities: Object.freeze({
      ...capabilities,
    }) as unknown as Readonly<Capabilities>,
    runtimeVersion: value.runtimeVersion,
    stanzaVersion: value.stanzaVersion,
    resourceVersion: value.resourceVersion,
    unicodeVersion: value.unicodeVersion,
    tokenizerKind,
  });
}

class ProcessPipeline implements NativePipeline {
  readonly language: string;
  readonly capabilities: Readonly<Capabilities>;
  private closed = false;
  private pending: Promise<unknown> = Promise.resolve();
  private closing?: Promise<void>;

  readonly info: Readonly<NativePipelineInfo>;
  private transport: NativeTransport;
  private offsetUnit: NativeOffsetUnit;

  constructor(
    info: Readonly<NativePipelineInfo>,
    transport: NativeTransport,
    offsetUnit: NativeOffsetUnit,
  ) {
    this.info = info;
    this.transport = transport;
    this.offsetUnit = offsetUnit;
    this.language = info.language;
    this.capabilities = info.capabilities;
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.pending.then(operation);
    // Keep the queue usable after a request-level error without masking that call.
    this.pending = result.catch(() => undefined);
    return result;
  }

  analyze(text: string): Promise<Sentence[]> {
    if (this.closed) {
      return Promise.reject(new NativeRuntimeError('Pipeline is closed'));
    }
    if (typeof text !== 'string') {
      return Promise.reject(new TypeError('analyze expects a string'));
    }
    if (!wellFormedUnicode(text)) {
      return Promise.reject(
        new TypeError('analyze expects well-formed Unicode text; unpaired UTF-16 surrogate'),
      );
    }
    return this.enqueue(async () => {
      const result = await this.transport.request('analyze', {
        text,
        offsetUnit: this.offsetUnit,
      });
      if (
        !record(result) ||
        result.language !== this.language ||
        result.offsetUnit !== this.offsetUnit ||
        !Array.isArray(result.sentences)
      ) {
        throw new NativeRuntimeError('Native runtime returned invalid analysis results');
      }
      return result.sentences as Sentence[];
    });
  }

  close(): Promise<void> {
    if (this.closing) {
      return this.closing;
    }
    // Reject new calls immediately, then drain accepted analyses before shutdown.
    this.closed = true;
    this.closing = this.enqueue(async () => {
      try {
        await this.transport.shutdown();
      } finally {
        await this.transport.dispose();
      }
    });
    return this.closing;
  }
}

/** Start one persistent native pipeline, keeping models loaded across analyses. */
export async function createNativePipeline(
  options: NativePipelineOptions,
): Promise<NativePipeline> {
  if (!options) {
    throw new TypeError('Native pipeline options are required');
  }
  for (const name of ['executable', 'modelDir'] as const) {
    if (typeof options[name] !== 'string' || !options[name].trim()) {
      throw new TypeError(`${name} is required`);
    }
  }
  if (
    options.ortLibrary !== undefined &&
    (typeof options.ortLibrary !== 'string' || !options.ortLibrary.trim())
  ) {
    throw new TypeError('ortLibrary must be a nonempty string when supplied');
  }
  const {
    threads = 1,
    provider = 'cpu',
    offsetUnit = 'unicode-code-point',
    timeoutMs = 120_000,
    executableArgs = [],
  } = options;
  if (!Number.isSafeInteger(threads) || threads < 1) {
    throw new TypeError('threads must be a positive integer');
  }
  if (!['cpu', 'coreml', 'directml'].includes(provider)) {
    throw new TypeError('Unsupported provider');
  }
  if (!['unicode-code-point', 'utf16', 'utf8'].includes(offsetUnit)) {
    throw new TypeError('Unsupported offsetUnit');
  }
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647) {
    throw new TypeError('timeoutMs must be a positive integer within the timer range');
  }
  if (!Array.isArray(executableArgs) || executableArgs.some((value) => typeof value !== 'string')) {
    throw new TypeError('executableArgs must be an array of strings');
  }
  const child = spawn(
    options.executable,
    [
      ...executableArgs,
      '--model-dir',
      options.modelDir,
      ...(options.ortLibrary === undefined ? [] : ['--ort-library', options.ortLibrary]),
      '--threads',
      String(threads),
      '--provider',
      provider,
      'serve',
    ],
    { stdio: ['pipe', 'pipe', 'pipe'], shell: false, windowsHide: true },
  );
  const transport = new NativeTransport(child, timeoutMs);
  try {
    const info = parseInfo(await transport.request('info'));
    return new ProcessPipeline(info, transport, offsetUnit);
  } catch (error) {
    await transport.dispose();
    throw error;
  }
}
