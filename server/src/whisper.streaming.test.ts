import { createHash } from 'node:crypto';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  REMOTE_SINGLE_PAYLOAD_MAX_BYTES,
  WhisperService,
} from './services/whisper.js';

function service(): WhisperService {
  return new WhisperService({
    modelPath: '',
    language: 'ru',
    device: 'cpu',
    serverUrl: 'http://asr.test',
    beamSize: 1,
  });
}

function sha256(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

test('radiology ASR sends only the approved context and preserves GigaAM provenance', async () => {
  const originalFetch = globalThis.fetch;
  const originalScope = process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE;
  const rawText = ' найха без очаговых изменений \n';
  let requestBody: Record<string, unknown> | undefined;
  process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE = 'ct_abdomen_contrast';
  globalThis.fetch = async (_input, init) => {
    requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
    return new Response(JSON.stringify({
      schema_version: 'gigaam.transcription.v2',
      source: 'gigaam',
      text: rawText,
      raw_text: rawText,
      language: 'ru',
      runtime_id: 'b'.repeat(64),
      context_bias: {
        scope: 'ct_abdomen_contrast',
        active: true,
        terms: 9,
      },
      hashes: {
        audio_sha256: 'audio-sha',
        normalized_audio_sha256: 'wav-sha',
        raw_text_sha256: sha256(rawText),
      },
      words: [
        { text: 'найха', start: 0.1, end: 0.5, confidence: 0.91 },
      ],
      model: {
        name: 'v3_ctc',
        acousticDecoder: 'CTCDecoder',
        ctcDecoder: {
          mode: 'beam',
          active: true,
          implementation: 'medical_decoder:create',
          beamWidth: 32,
          languageModel: {
            active: true,
            file: 'ct-abdomen.arpa',
            sha256: 'lm-sha',
            alpha: 0.7,
            beta: 1.2,
          },
          contexts: {
            configured: true,
            file: 'contexts.json',
            sha256: 'contexts-sha',
            scopes: { ct_abdomen_contrast: 9 },
          },
        },
        checkpoint: {
          verified: true,
          hashes: { sha256: 'a'.repeat(64) },
        },
      },
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await service().transcribeBase64Detailed(
      'ZmFrZS1hdWRpbw==',
      { templateId: 'CT_ABDOMEN_MIKHAILOV' },
    );

    assert.equal(requestBody?.context_scope, 'ct_abdomen_contrast');
    assert.equal('hotwords' in (requestBody ?? {}), false);
    assert.equal(result.rawText, rawText, 'raw runtime output must not be trimmed');
    assert.equal(result.rawAvailable, true);
    assert.equal(result.normalizedText, 'найха без очаговых изменений');
    assert.equal(result.model.dictionary.name, 'medical-dictionary-disabled');
    assert.equal(result.model.asr.checksum, 'a'.repeat(64));
    assert.equal(result.checkpointVerified, true);
    assert.equal(result.provenance.checkpointVerified, true);
    assert.equal(result.model.decoder.name, 'medical_decoder:create');
    assert.match(result.model.decoder.version, /acoustic=CTCDecoder/);
    assert.match(result.model.decoder.version, /context=ct_abdomen_contrast/);
    assert.equal(result.model.decoder.checksum, 'contexts-sha');
    assert.deepEqual(result.model.languageModel, {
      name: 'ct-abdomen.arpa',
      version: `${'b'.repeat(64)};alpha=0.7;beta=1.2`,
      checksum: 'lm-sha',
    });
    assert.equal(result.contextBias.scope, 'ct_abdomen_contrast');
    assert.equal(result.hashes.audioSha256, 'audio-sha');
    assert.equal(result.hashes.rawTextSha256, sha256(rawText));
    assert.equal(result.schemaVersion, 'gigaam.transcription.v2');
    assert.equal(result.provenance.ctcDecoder?.beamWidth, 32);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalScope === undefined) {
      delete process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE;
    } else {
      process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE = originalScope;
    }
  }
});

test('legacy text fallback is normalized but never marked as raw ASR', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    source: 'whisper',
    text: 'найха',
    language: 'ru',
    model: { name: 'legacy-whisper', decoder: 'legacy-greedy' },
  }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

  try {
    const result = await service().transcribeBase64Detailed('ZmFrZQ==');
    assert.equal(result.rawText, 'найха');
    assert.equal(result.rawAvailable, false);
    assert.equal(result.checkpointVerified, false);
    assert.equal(result.provenance.checkpointVerified, false);
    assert.equal(result.normalizedText, 'NYHA');
    assert.equal(result.model.dictionary.name, 'medical-dictionary');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('unapproved templates cannot select a configured decoder scope', async () => {
  const originalFetch = globalThis.fetch;
  const originalScope = process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE;
  let requestBody: Record<string, unknown> | undefined;
  process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE = 'ct_abdomen_contrast';
  globalThis.fetch = async (_input, init) => {
    requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
    return new Response(JSON.stringify({
      source: 'gigaam',
      text: 'текст',
      raw_text: 'текст',
      language: 'ru',
      context_bias: { scope: null, active: false, terms: 0 },
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    await service().transcribeBase64Detailed(
      'ZmFrZQ==',
      { templateId: 'CT_BRAIN_MIKHAILOV' },
    );
    assert.equal('context_scope' in (requestBody ?? {}), false);
    assert.equal('hotwords' in (requestBody ?? {}), false);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalScope === undefined) {
      delete process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE;
    } else {
      process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE = originalScope;
    }
  }
});

test('remote single-payload cap stays safely below forty MiB', () => {
  assert.ok(REMOTE_SINGLE_PAYLOAD_MAX_BYTES <= 32 * 1024 * 1024);
  assert.ok(REMOTE_SINGLE_PAYLOAD_MAX_BYTES <= 40 * 1024 * 1024);
});
