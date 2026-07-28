import assert from 'node:assert/strict';
import test from 'node:test';
import { ollamaLLM, radiologyLLMIdentity } from './radiology/ollama.js';

interface CapturedRequest {
  url: string;
  body: Record<string, any>;
}

async function withCapturedFetch(
  responseBody: unknown,
  run: (captured: CapturedRequest[]) => Promise<void>,
): Promise<void> {
  const originalFetch = globalThis.fetch;
  const captured: CapturedRequest[] = [];
  globalThis.fetch = (async (
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    captured.push({
      url: String(input),
      body: JSON.parse(String(init?.body ?? '{}')) as Record<string, any>,
    });
    return new Response(JSON.stringify(responseBody), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }) as typeof fetch;
  try {
    await run(captured);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test('llama.cpp radiology provider uses the OpenAI-compatible IDs-only contract', async () => {
  await withCapturedFetch(
    { choices: [{ message: { content: '{"assignments":[]}' } }] },
    async (captured) => {
      const call = ollamaLLM({
        provider: 'llama',
        url: 'http://llm:8080/',
        model: 'deepseek-14b',
        seed: 42,
        numPredict: 256,
      });
      assert.equal(await call('system', '{"atoms":[]}'), '{"assignments":[]}');
      assert.equal(captured.length, 1);
      assert.equal(captured[0].url, 'http://llm:8080/v1/chat/completions');
      assert.equal(captured[0].body.model, 'deepseek-14b');
      assert.equal(captured[0].body.temperature, 0);
      assert.equal(captured[0].body.seed, 42);
      assert.equal(captured[0].body.max_tokens, 256);
      assert.equal(captured[0].body.response_format.type, 'json_schema');
      assert.deepEqual(
        captured[0].body.messages.map((message: { role: string }) => message.role),
        ['system', 'user'],
      );
    },
  );
});

test('Ollama radiology provider remains local, deterministic, and JSON-constrained', async () => {
  await withCapturedFetch(
    { message: { content: '{"assignments":[{"atomId":"a1","sectionId":null}]}' } },
    async (captured) => {
      const call = ollamaLLM({
        provider: 'ollama',
        url: 'http://ollama:11434',
        model: 'deepseek-r1:14b',
        seed: 42,
        numCtx: 4096,
        numPredict: 128,
      });
      assert.equal(
        await call('system', '{"atoms":[{"atomId":"a1"}]}'),
        '{"assignments":[{"atomId":"a1","sectionId":null}]}',
      );
      assert.equal(captured.length, 1);
      assert.equal(captured[0].url, 'http://ollama:11434/api/chat');
      assert.equal(captured[0].body.stream, false);
      assert.equal(captured[0].body.think, false);
      assert.equal(captured[0].body.options.temperature, 0);
      assert.equal(captured[0].body.options.seed, 42);
      assert.equal(captured[0].body.options.num_ctx, 4096);
      assert.equal(captured[0].body.options.num_predict, 128);
      assert.equal(captured[0].body.format.additionalProperties, false);
    },
  );
});

test('strict production identity rejects floating runtime or missing model SHA', () => {
  const previous = {
    strict: process.env.RADIOLOGY_LLM_STRICT_IDENTITY,
    model: process.env.RADIOLOGY_LLM_CHECKSUM,
    runtime: process.env.RADIOLOGY_LLM_RUNTIME_IDENTITY,
    image: process.env.RADIOLOGY_LLM_RUNTIME_IMAGE,
  };
  try {
    process.env.RADIOLOGY_LLM_STRICT_IDENTITY = 'true';
    delete process.env.RADIOLOGY_LLM_CHECKSUM;
    delete process.env.RADIOLOGY_LLM_RUNTIME_IDENTITY;
    process.env.RADIOLOGY_LLM_RUNTIME_IMAGE = 'ghcr.io/example/llm:latest';
    assert.throws(
      () => radiologyLLMIdentity(),
      /content-addressed image/u,
    );

    process.env.RADIOLOGY_LLM_CHECKSUM = `sha256:${'a'.repeat(64)}`;
    process.env.RADIOLOGY_LLM_RUNTIME_IMAGE =
      `ghcr.io/example/llm@sha256:${'b'.repeat(64)}`;
    const identity = radiologyLLMIdentity();
    assert.equal(identity.modelChecksum, `sha256:${'a'.repeat(64)}`);
    assert.equal(
      identity.runtimeIdentity,
      `ghcr.io/example/llm@sha256:${'b'.repeat(64)}`,
    );
  } finally {
    const restore = (name: string, value: string | undefined) => {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    };
    restore('RADIOLOGY_LLM_STRICT_IDENTITY', previous.strict);
    restore('RADIOLOGY_LLM_CHECKSUM', previous.model);
    restore('RADIOLOGY_LLM_RUNTIME_IDENTITY', previous.runtime);
    restore('RADIOLOGY_LLM_RUNTIME_IMAGE', previous.image);
  }
});
