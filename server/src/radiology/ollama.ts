// Local-only LLM classifier for radiology span routing.
// It supports both Ollama and the llama.cpp OpenAI-compatible server used by
// docker-compose. The response is still validated by arranger.ts as IDs only.

export type LLMCall = (system: string, user: string) => Promise<string>;
export type RadiologyLocalLLMProvider = 'ollama' | 'llama';

const DEFAULT_URL =
  process.env.RADIOLOGY_LLM_URL
  || process.env.LLM_SERVER_URL
  || 'http://127.0.0.1:11434';
const DEFAULT_MODEL =
  process.env.RADIOLOGY_LLM_MODEL
  || process.env.LLM_MODEL
  || 'qwen3.5:9b';
const DEFAULT_SEED = 42;
const DEFAULT_NUM_CTX = 8192;
const DEFAULT_NUM_PREDICT = 2048;
const DEFAULT_TIMEOUT_MS = 120_000;

const ASSIGNMENT_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['assignments'],
  properties: {
    assignments: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['atomId', 'sectionId'],
        properties: {
          atomId: { type: 'string' },
          sectionId: {
            anyOf: [
              { type: 'string' },
              { type: 'null' },
            ],
          },
        },
      },
    },
  },
} as const;

function boundedInteger(
  value: string | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const parsed = Number.parseInt(value ?? '', 10);
  if (!Number.isSafeInteger(parsed)) return fallback;
  return Math.min(maximum, Math.max(minimum, parsed));
}

function configuredProvider(): RadiologyLocalLLMProvider {
  const value = (
    process.env.RADIOLOGY_LLM_PROVIDER
    || process.env.LLM_PROVIDER
    || 'ollama'
  ).trim().toLowerCase();
  if (value === 'ollama' || value === 'llama') return value;
  throw new Error(
    `Radiology routing requires an on-prem ollama or llama provider, got: ${value}`,
  );
}

function configuredChecksum(): string | undefined {
  const value = (
    process.env.RADIOLOGY_LLM_CHECKSUM
    || process.env.LLM_MODEL_CHECKSUM
    || ''
  ).trim().toLowerCase();
  if (!value) return undefined;
  if (!/^sha256:[a-f0-9]{64}$/u.test(value)) {
    throw new Error('RADIOLOGY_LLM_CHECKSUM must use sha256:<64 lowercase hex>');
  }
  return value;
}

function strictIdentityRequired(): boolean {
  return (
    process.env.RADIOLOGY_LLM_STRICT_IDENTITY
    || ''
  ).trim().toLowerCase() === 'true';
}

function configuredRuntimeIdentity(): string | undefined {
  const value = (
    process.env.RADIOLOGY_LLM_RUNTIME_IDENTITY
    || process.env.RADIOLOGY_LLM_RUNTIME_IMAGE
    || ''
  ).trim().toLowerCase();
  if (!value) return undefined;
  if (
    !/^sha256:[a-f0-9]{64}$/u.test(value)
    && !/@sha256:[a-f0-9]{64}$/u.test(value)
  ) {
    throw new Error(
      'RADIOLOGY_LLM_RUNTIME_IDENTITY must be sha256:<hex> or a content-addressed image',
    );
  }
  return value;
}

export interface RadiologyLLMIdentity {
  provider: RadiologyLocalLLMProvider;
  url: string;
  model: string;
  modelChecksum?: string;
  runtimeIdentity?: string;
  temperature: 0;
  seed: number;
  think: false;
  numCtx: number;
  numPredict: number;
  timeoutMs: number;
}

export function radiologyLLMIdentity(): RadiologyLLMIdentity {
  const modelChecksum = configuredChecksum();
  const runtimeIdentity = configuredRuntimeIdentity();
  if (strictIdentityRequired() && (!modelChecksum || !runtimeIdentity)) {
    throw new Error(
      'Strict radiology LLM identity requires model SHA-256 and runtime identity',
    );
  }
  return {
    provider: configuredProvider(),
    url: DEFAULT_URL.replace(/\/+$/u, ''),
    model: DEFAULT_MODEL,
    ...(modelChecksum ? { modelChecksum } : {}),
    ...(runtimeIdentity ? { runtimeIdentity } : {}),
    temperature: 0,
    seed: DEFAULT_SEED,
    think: false,
    numCtx: boundedInteger(
      process.env.RADIOLOGY_LLM_NUM_CTX,
      DEFAULT_NUM_CTX,
      1024,
      131_072,
    ),
    numPredict: boundedInteger(
      process.env.RADIOLOGY_LLM_NUM_PREDICT,
      DEFAULT_NUM_PREDICT,
      64,
      8192,
    ),
    timeoutMs: boundedInteger(
      process.env.RADIOLOGY_LLM_TIMEOUT_MS || process.env.LLM_TIMEOUT_MS,
      DEFAULT_TIMEOUT_MS,
      1000,
      600_000,
    ),
  };
}

function llamaResponseContent(value: unknown): string {
  if (!value || typeof value !== 'object') return '';
  const data = value as {
    content?: unknown;
    choices?: Array<{
      message?: { content?: unknown };
      text?: unknown;
    }>;
  };
  if (typeof data.content === 'string') return data.content;
  const first = data.choices?.[0];
  if (typeof first?.message?.content === 'string') return first.message.content;
  if (typeof first?.text === 'string') return first.text;
  return '';
}

export function ollamaLLM(
  opts: {
    provider?: RadiologyLocalLLMProvider;
    url?: string;
    model?: string;
    timeoutMs?: number;
    seed?: number;
    numCtx?: number;
    numPredict?: number;
  } = {},
): LLMCall {
  const defaults = radiologyLLMIdentity();
  const provider = opts.provider ?? defaults.provider;
  const url = (opts.url || defaults.url).replace(/\/+$/u, '');
  const model = opts.model || defaults.model;
  const timeoutMs = opts.timeoutMs ?? defaults.timeoutMs;
  const seed = opts.seed ?? defaults.seed;
  const numCtx = opts.numCtx ?? defaults.numCtx;
  const numPredict = opts.numPredict ?? defaults.numPredict;

  return async (system: string, user: string): Promise<string> => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const endpoint = provider === 'ollama'
        ? `${url}/api/chat`
        : `${url}/v1/chat/completions`;
      const body = provider === 'ollama'
        ? {
            model,
            stream: false,
            think: false,
            format: ASSIGNMENT_JSON_SCHEMA,
            options: {
              temperature: 0,
              seed,
              num_ctx: numCtx,
              num_predict: numPredict,
            },
            messages: [
              { role: 'system', content: system },
              { role: 'user', content: user },
            ],
          }
        : {
            model,
            stream: false,
            temperature: 0,
            seed,
            max_tokens: numPredict,
            messages: [
              { role: 'system', content: system },
              { role: 'user', content: user },
            ],
            response_format: {
              type: 'json_schema',
              json_schema: {
                name: 'radiology_span_assignments',
                strict: true,
                schema: ASSIGNMENT_JSON_SCHEMA,
              },
            },
          };
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(
          `${provider} ${response.status}: ${(await response.text()).slice(0, 200)}`,
        );
      }
      const data = await response.json() as unknown;
      if (provider === 'ollama') {
        const content = (
          data
          && typeof data === 'object'
          && 'message' in data
          && typeof (data as { message?: unknown }).message === 'object'
          && (data as { message?: { content?: unknown } }).message
          && typeof (data as { message?: { content?: unknown } }).message?.content === 'string'
        )
          ? (data as { message: { content: string } }).message.content
          : '';
        return content;
      }
      return llamaResponseContent(data);
    } finally {
      clearTimeout(timer);
    }
  };
}
