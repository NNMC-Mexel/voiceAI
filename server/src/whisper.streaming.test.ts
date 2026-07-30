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

const AUDIO_BASE64 = 'ZmFrZS1hdWRpbw==';
const AUDIO_SHA256 = sha256('fake-audio');
const NORMALIZED_AUDIO_SHA256 = 'c'.repeat(64);
const RUNTIME_ID = 'b'.repeat(64);
const CHECKPOINT_SHA256 = 'a'.repeat(64);

function validModel() {
  return {
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
        sha256: 'd'.repeat(64),
        alpha: 0.7,
        beta: 1.2,
      },
      contexts: {
        configured: true,
        file: 'contexts.json',
        sha256: 'e'.repeat(64),
        scopes: { ct_abdomen_contrast: 9 },
      },
    },
    checkpoint: {
      expectedChecksum: `sha256:${CHECKPOINT_SHA256}`,
      verified: true,
      hashes: { sha256: CHECKPOINT_SHA256 },
    },
  };
}

function validRuntimeMetadata(model = validModel()) {
  return {
    schema_version: 'gigaam.runtime.v1',
    runtime_id: RUNTIME_ID,
    model,
    configuration: {
      ctc_decoder: model.ctcDecoder,
    },
  };
}

function validTranscription(rawText: string, model = validModel()) {
  return {
    schema_version: 'gigaam.transcription.v2',
    source: 'gigaam',
    text: rawText,
    raw_text: rawText,
    language: 'ru',
    runtime_id: RUNTIME_ID,
    context_bias: {
      scope: 'ct_abdomen_contrast',
      active: true,
      terms: 9,
    },
    hashes: {
      audio_sha256: AUDIO_SHA256,
      normalized_audio_sha256: NORMALIZED_AUDIO_SHA256,
      raw_text_sha256: sha256(rawText),
    },
    words: [
      {
        text: rawText.trim().split(/\s+/u)[0],
        start: 0.1,
        end: 0.5,
        confidence: 0.91,
        avg_logprob: -0.094,
        score_type: 'ctc_greedy_token_peak_geomean',
      },
    ],
    model,
  };
}

async function withApprovedScope<T>(run: () => Promise<T>): Promise<T> {
  const original = process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE;
  process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE = 'ct_abdomen_contrast';
  try {
    return await run();
  } finally {
    if (original === undefined) {
      delete process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE;
    } else {
      process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE = original;
    }
  }
}

test('radiology ASR sends only the approved context and preserves GigaAM provenance', async () => {
  const originalFetch = globalThis.fetch;
  const originalScope = process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE;
  const rawText = ' найха без очаговых изменений \n';
  let requestBody: Record<string, unknown> | undefined;
  process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE = 'ct_abdomen_contrast';
  const model = validModel();
  globalThis.fetch = async (input, init) => {
    if (String(input).endsWith('/metadata')) {
      return new Response(JSON.stringify(validRuntimeMetadata(model)), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
    return new Response(JSON.stringify(validTranscription(rawText, model)), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  try {
    const result = await service().transcribeBase64Detailed(
      AUDIO_BASE64,
      { templateId: 'CT_ABDOMEN_MIKHAILOV' },
    );

    assert.equal(requestBody?.context_scope, 'ct_abdomen_contrast');
    assert.equal('hotwords' in (requestBody ?? {}), false);
    assert.equal(result.rawText, rawText, 'raw runtime output must not be trimmed');
    assert.equal(result.rawAvailable, true);
    assert.equal(result.normalizedText, 'найха без очаговых изменений');
    assert.equal(result.model.dictionary.name, 'medical-dictionary-disabled');
    assert.equal(result.model.asr.checksum, CHECKPOINT_SHA256);
    assert.equal(result.checkpointVerified, true);
    assert.equal(result.provenance.checkpointVerified, true);
    assert.equal(result.model.decoder.name, 'medical_decoder:create');
    assert.match(result.model.decoder.version, /acoustic=CTCDecoder/);
    assert.match(result.model.decoder.version, /context=ct_abdomen_contrast/);
    assert.equal(result.model.decoder.checksum, 'e'.repeat(64));
    assert.deepEqual(result.model.languageModel, {
      name: 'ct-abdomen.arpa',
      version: `${RUNTIME_ID};alpha=0.7;beta=1.2`,
      checksum: 'd'.repeat(64),
    });
    assert.equal(result.contextBias.scope, 'ct_abdomen_contrast');
    assert.equal(result.hashes.audioSha256, AUDIO_SHA256);
    assert.equal(result.hashes.rawTextSha256, sha256(rawText));
    assert.equal(result.schemaVersion, 'gigaam.transcription.v2');
    assert.equal(result.provenance.ctcDecoder?.beamWidth, 32);
    assert.deepEqual(result.verification, {
      metadataAvailable: true,
      metadataSchema: true,
      transcriptionSchema: true,
      runtimeIdentity: true,
      checkpoint: true,
      decoder: true,
      hashes: true,
      wordEvidence: true,
      productionReady: true,
    });
    assert.deepEqual(result.provenance.verification, result.verification);
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
  globalThis.fetch = async (input, init) => {
    if (String(input).endsWith('/metadata')) {
      return new Response('{}', {
        status: 404,
        headers: { 'Content-Type': 'application/json' },
      });
    }
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
    const result = await service().transcribeBase64Detailed(
      'ZmFrZQ==',
      { templateId: 'CT_BRAIN_MIKHAILOV' },
    );
    assert.equal('context_scope' in (requestBody ?? {}), false);
    assert.equal('hotwords' in (requestBody ?? {}), false);
    assert.equal(result.verification.metadataAvailable, false);
    assert.equal(result.verification.productionReady, false);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalScope === undefined) {
      delete process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE;
    } else {
      process.env.GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE = originalScope;
    }
  }
});

test('a locally computed raw hash is not presented as remote provenance', async () => {
  const originalFetch = globalThis.fetch;
  const rawText = 'печень КВР 150';
  const model = validModel();
  const transcription = validTranscription(rawText, model);
  const { raw_text_sha256: _omitted, ...remoteHashes } = transcription.hashes;
  globalThis.fetch = async (input) => new Response(JSON.stringify(
    String(input).endsWith('/metadata')
      ? validRuntimeMetadata(model)
      : { ...transcription, hashes: remoteHashes },
  ), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

  try {
    await withApprovedScope(async () => {
      const result = await service().transcribeBase64Detailed(
        AUDIO_BASE64,
        { templateId: 'CT_ABDOMEN_MIKHAILOV' },
      );
      assert.equal(result.rawAvailable, true);
      assert.equal(result.hashes.rawTextSha256, undefined);
      assert.equal(result.verification.checkpoint, true);
      assert.equal(result.verification.decoder, true);
      assert.equal(result.verification.hashes, false);
      assert.equal(result.verification.wordEvidence, true);
      assert.equal(result.verification.productionReady, false);
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('finite confidence without acoustic score provenance is not training evidence', async () => {
  const originalFetch = globalThis.fetch;
  const rawText = 'печень КВР 150';
  const model = validModel();
  const transcription = validTranscription(rawText, model);
  transcription.words = transcription.words.map(({ avg_logprob: _omitted, ...word }) => word);
  globalThis.fetch = async (input) => new Response(JSON.stringify(
    String(input).endsWith('/metadata')
      ? validRuntimeMetadata(model)
      : transcription,
  ), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

  try {
    await withApprovedScope(async () => {
      const result = await service().transcribeBase64Detailed(
        AUDIO_BASE64,
        { templateId: 'CT_ABDOMEN_MIKHAILOV' },
      );
      assert.equal(result.words[0]?.confidence, 0.91);
      assert.equal(result.words[0]?.avgLogprob, null);
      assert.equal(result.verification.hashes, true);
      assert.equal(result.verification.wordEvidence, false);
      assert.equal(result.verification.productionReady, false);
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('runtime identity mismatch between metadata and transcription is rejected', async () => {
  const originalFetch = globalThis.fetch;
  const model = validModel();
  const transcription = {
    ...validTranscription('печень КВР 150', model),
    runtime_id: 'f'.repeat(64),
  };
  globalThis.fetch = async (input) => new Response(JSON.stringify(
    String(input).endsWith('/metadata')
      ? validRuntimeMetadata(model)
      : transcription,
  ), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

  try {
    await withApprovedScope(async () => {
      await assert.rejects(
        service().transcribeBase64Detailed(
          AUDIO_BASE64,
          { templateId: 'CT_ABDOMEN_MIKHAILOV' },
        ),
        /runtime_id does not match/u,
      );
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('decoder mismatch between metadata and transcription is rejected', async () => {
  const originalFetch = globalThis.fetch;
  const responseModel = validModel();
  const metadataModel = validModel();
  metadataModel.ctcDecoder.beamWidth = 16;
  const transcription = validTranscription('печень КВР 150', responseModel);
  globalThis.fetch = async (input) => new Response(JSON.stringify(
    String(input).endsWith('/metadata')
      ? validRuntimeMetadata(metadataModel)
      : transcription,
  ), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

  try {
    await withApprovedScope(async () => {
      await assert.rejects(
        service().transcribeBase64Detailed(
          AUDIO_BASE64,
          { templateId: 'CT_ABDOMEN_MIKHAILOV' },
        ),
        /decoder metadata differs/u,
      );
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('checkpoint mismatch between metadata and transcription is rejected', async () => {
  const originalFetch = globalThis.fetch;
  const responseModel = validModel();
  const metadataModel = validModel();
  metadataModel.checkpoint.hashes.sha256 = '9'.repeat(64);
  metadataModel.checkpoint.expectedChecksum = `sha256:${'9'.repeat(64)}`;
  const transcription = validTranscription('печень КВР 150', responseModel);
  globalThis.fetch = async (input) => new Response(JSON.stringify(
    String(input).endsWith('/metadata')
      ? validRuntimeMetadata(metadataModel)
      : transcription,
  ), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

  try {
    await withApprovedScope(async () => {
      await assert.rejects(
        service().transcribeBase64Detailed(
          AUDIO_BASE64,
          { templateId: 'CT_ABDOMEN_MIKHAILOV' },
        ),
        /checkpoint metadata differs/u,
      );
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('checkpoint without a SHA-256 artifact remains explicitly unverified', async () => {
  const originalFetch = globalThis.fetch;
  const model = {
    ...validModel(),
    checkpoint: {
      expectedChecksum: `md5:${'1'.repeat(32)}`,
      verified: true,
      hashes: { md5: '1'.repeat(32) },
    },
  };
  const transcription = validTranscription('печень КВР 150', model);
  globalThis.fetch = async (input) => new Response(JSON.stringify(
    String(input).endsWith('/metadata')
      ? validRuntimeMetadata(model)
      : transcription,
  ), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

  try {
    await withApprovedScope(async () => {
      const result = await service().transcribeBase64Detailed(
        AUDIO_BASE64,
        { templateId: 'CT_ABDOMEN_MIKHAILOV' },
      );
      assert.equal(result.verification.checkpoint, false);
      assert.equal(result.checkpointVerified, false);
      assert.equal(result.verification.productionReady, false);
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('remote single-payload cap stays safely below forty MiB', () => {
  assert.ok(REMOTE_SINGLE_PAYLOAD_MAX_BYTES <= 32 * 1024 * 1024);
  assert.ok(REMOTE_SINGLE_PAYLOAD_MAX_BYTES <= 40 * 1024 * 1024);
});
