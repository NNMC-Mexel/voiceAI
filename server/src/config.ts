import { ServerConfig, LLMProviderKind, defaultConfig } from './types.js';

function parseIntSafe(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseFloatSafe(value: string | undefined, fallback: number): number {
  const parsed = Number.parseFloat(value ?? '');
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (value == null) return fallback;
  return value.trim().toLowerCase() === 'true';
}

function parseCorsOrigins(raw: string | undefined, fallback: string[]): string[] {
  if (!raw?.trim()) return fallback;
  return raw
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);
}

function resolveLlmProvider(): LLMProviderKind {
  const raw = process.env.LLM_PROVIDER?.trim().toLowerCase();
  if (raw === 'anthropic') return 'anthropic';
  return 'llama';
}

function resolveAnthropicConfig() {
  const apiKey = process.env.ANTHROPIC_API_KEY?.trim();
  if (!apiKey) return undefined;
  return {
    apiKey,
    model: process.env.ANTHROPIC_MODEL?.trim() || 'claude-haiku-4-5',
    maxTokens: parseIntSafe(process.env.ANTHROPIC_MAX_TOKENS, 4096),
    maxRetries: parseIntSafe(process.env.ANTHROPIC_MAX_RETRIES, 3),
  };
}

function resolveLlmServerUrl(): string {
  const direct = process.env.LLM_SERVER_URL?.trim();
  if (direct) return direct;

  const isProduction = process.env.NODE_ENV === 'production';
  const productionUrl = process.env.LLM_SERVER_URL_PROD?.trim();
  if (isProduction && productionUrl) return productionUrl;

  return defaultConfig.llm.serverUrl;
}

export function loadConfig(): ServerConfig {
  return {
    port: parseIntSafe(process.env.PORT, defaultConfig.port),
    host: process.env.HOST || defaultConfig.host,
    uploadDir: process.env.UPLOAD_DIR || defaultConfig.uploadDir,
    whisper: {
      modelPath: process.env.WHISPER_MODEL_PATH || defaultConfig.whisper.modelPath,
      language: process.env.WHISPER_LANGUAGE || defaultConfig.whisper.language,
      device: (process.env.WHISPER_DEVICE as 'cuda' | 'cpu') || defaultConfig.whisper.device,
      serverUrl: process.env.WHISPER_SERVER_URL?.trim() || undefined,
    },
    llm: {
      provider: resolveLlmProvider(),
      serverUrl: resolveLlmServerUrl(),
      model: process.env.LLM_MODEL || defaultConfig.llm.model,
      maxTokens: parseIntSafe(process.env.LLM_MAX_TOKENS, defaultConfig.llm.maxTokens),
      temperature: parseFloatSafe(process.env.LLM_TEMPERATURE, defaultConfig.llm.temperature),
      parallelSlots: parseIntSafe(process.env.LLM_PARALLEL_SLOTS, defaultConfig.llm.parallelSlots),
      requestTimeoutMs: parseIntSafe(process.env.LLM_TIMEOUT_MS, defaultConfig.llm.requestTimeoutMs),
      allowMockOnFailure: parseBoolean(process.env.ALLOW_MOCK_LLM, defaultConfig.llm.allowMockOnFailure),
      anthropic: resolveAnthropicConfig(),
    },
    tts: {
      serverUrl: process.env.TTS_SERVER_URL?.trim() || '',
      enabled: parseBoolean(process.env.TTS_ENABLED, false),
    },
    security: {
      corsOrigins: parseCorsOrigins(process.env.CORS_ORIGINS, defaultConfig.security.corsOrigins),
      apiKey: process.env.API_KEY?.trim() || undefined,
      authPassword: process.env.AUTH_PASSWORD?.trim() || undefined,
      rateLimitWindowMs: parseIntSafe(process.env.RATE_LIMIT_WINDOW_MS, defaultConfig.security.rateLimitWindowMs),
      rateLimitMaxRequests: parseIntSafe(process.env.RATE_LIMIT_MAX_REQUESTS, defaultConfig.security.rateLimitMaxRequests),
    },
  };
}
