import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readdir, rm, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import Fastify from 'fastify';
import { registerRadiologyRoutes } from './routes-radiology.js';
import {
  RadiologyArtifactStore,
  RadiologySessionService,
} from './radiology-session.js';
import type {
  RadiologyChunkTranscriber,
  RadiologyModelMetadata,
  RadiologyTranscriptStructurer,
} from './radiology-session.js';
import type { DictationReport } from './radiology/dictation.js';
import { structureDictation } from './radiology/dictation.js';
import { verifyNumbers } from './radiology/number-check.js';
import { verifyRadiologySafety } from './radiology/safety.js';

const model: RadiologyModelMetadata = {
  asr: { name: 'GigaAM', version: 'v3_ctc', checksum: 'a'.repeat(64) },
  vad: { name: 'pyannote/segmentation-3.0', version: 'e66f3d3b9eb0873085418a7b813d3b369bf160bb' },
  decoder: { name: 'ctc-prefix-beam', version: '1' },
  languageModel: { name: 'ct-abdomen-4gram', version: '2026-07-28' },
  contextVocabulary: { name: 'ct-abdomen-context', version: '1' },
  dictionary: { name: 'radiology-normalizer', version: '1' },
  normalizer: { name: 'gigaam-denormalizer-strict', version: '2' },
  template: { name: 'CT_ABDOMEN_MIKHAILOV', version: '1' },
  router: { name: 'radiology-span-router', version: '2' },
  prompt: { name: 'radiology-span-router-prompt', version: 'radiology-span-router-v2' },
  structurer: { name: 'radiology-structurer-guardrails', version: '1' },
  llm: { name: 'test-router', version: 'fixed-seed-42' },
  safety: { name: 'radiology-safety-verifier', version: '1' },
};

function reportFor(transcript: string, output = transcript): DictationReport {
  const numberCheck = verifyNumbers(transcript, output);
  const safety = verifyRadiologySafety(transcript, output);
  return {
    title: 'CT abdomen',
    blocks: [{
      id: 'liver',
      label: 'Liver',
      text: output,
      source: 'dictated',
      evidence: [{ start: 0, end: transcript.length, text: transcript, source: 'transcript' }],
      provenanceStatus: 'linked',
      origin: 'transcript',
    }],
    fullText: `CT abdomen\nLiver: ${output}`,
    evidenceBackedText: output,
    templateDefaults: [],
    routing: {
      atoms: [],
      assignments: [],
      unmatchedAtomIds: [],
    },
    structuringRun: {
      routerVersion: 'radiology-span-router-v2',
      promptVersion: 'test-prompt-v2',
      llmAllowed: false,
      llmCalled: false,
      llmValid: true,
      llmInputSha256: null,
      llmResponseSha256: null,
      issues: [],
    },
    unmatched: '',
    unmatchedSpans: [],
    generateConclusion: false,
    numberCheck,
    safety,
    provenance: {
      sections: {
        liver: [{ start: 0, end: transcript.length, text: transcript, source: 'transcript' }],
      },
      unmatched: [],
    },
  };
}

function verifiedTranscription(
  audioBase64: string,
  rawText: string,
  normalizedText = rawText,
) {
  const audio = Buffer.from(audioBase64, 'base64');
  const hashes = {
    audioSha256: createHash('sha256').update(audio).digest('hex'),
    rawTextSha256: createHash('sha256').update(rawText).digest('hex'),
    normalizedTextSha256: createHash('sha256').update(normalizedText).digest('hex'),
  };
  const verification = {
    metadataAvailable: true,
    metadataSchema: true,
    transcriptionSchema: true,
    runtimeIdentity: true,
    checkpoint: true,
    decoder: true,
    hashes: true,
    wordEvidence: true,
    productionReady: true,
  };
  return {
    schemaVersion: 'gigaam.transcription.v2',
    runtimeId: 'b'.repeat(64),
    checkpointVerified: true,
    rawText,
    normalizedText,
    rawAvailable: true,
    source: 'gigaam' as const,
    words: [{
      text: rawText.trim().split(/\s+/u)[0] ?? rawText,
      startMs: 0,
      endMs: 100,
      confidence: 0.97,
      avgLogprob: -0.03,
      scoreType: 'ctc_acoustic_logprob',
    }],
    model,
    hashes,
    verification,
    provenance: {
      schemaVersion: 'gigaam.transcription.v2',
      runtimeId: 'b'.repeat(64),
      checkpointVerified: true,
      acousticDecoder: 'ctc-greedy',
      ctcDecoder: null,
      contextBias: { scope: null, active: false, terms: 0 },
      hashes,
      verification,
    },
  };
}

async function fixture(
  transcribeChunk: RadiologyChunkTranscriber,
  structureOutput?: (transcript: string) => string,
  structureTranscript?: RadiologyTranscriptStructurer,
) {
  const dataDir = await mkdtemp(path.join(os.tmpdir(), 'voicemed-radiology-session-'));
  const store = new RadiologyArtifactStore(dataDir);
  const service = new RadiologySessionService({
    store,
    transcribeChunk,
    structureTranscript: structureTranscript ?? (async (_templateId, transcript) =>
      reportFor(transcript, structureOutput?.(transcript) ?? transcript)),
    model,
    allowUnownedSessions: true,
    allowAudioPersistence: true,
  });
  const app = Fastify();
  app.addHook('onRequest', async (request) => {
    const doctorId = request.headers['x-test-doctor-id'];
    if (typeof doctorId !== 'string') return;
    const numericDoctorId = Number(doctorId);
    (request as unknown as {
      user: { doctorId: number | string; role: 'admin' | 'doctor' };
    }).user = {
      doctorId: Number.isSafeInteger(numericDoctorId) ? numericDoctorId : doctorId,
      role: request.headers['x-test-role'] === 'admin' ? 'admin' : 'doctor',
    };
  });
  registerRadiologyRoutes(app, { sessionService: service });
  return { app, dataDir, store };
}

test('canonical radiology session returns and persists a provenance-rich artifact', async (t) => {
  const transcriber: RadiologyChunkTranscriber = async (audioBase64, context) => {
    const audio = Buffer.from(audioBase64, 'base64');
    const marker = audio.toString('utf8');
    const rawText = context.chunkIndex === 0 ? ' raw zero\n' : 'raw one ';
    const normalizedText = context.chunkIndex === 0 ? 'печень 15 мм' : 'без особенностей';
    const contextBias = { scope: 'ct-abdomen', active: true, terms: 42 };
    const hashes = {
      audioSha256: createHash('sha256').update(audio).digest('hex'),
      rawTextSha256: createHash('sha256').update(rawText).digest('hex'),
      normalizedTextSha256: createHash('sha256').update(normalizedText).digest('hex'),
    };
    return {
      schemaVersion: 'gigaam.transcription.v2',
      runtimeId: 'b'.repeat(64),
      checkpointVerified: true,
      rawText,
      normalizedText,
      rawAvailable: true,
      language: 'ru',
      source: 'gigaam',
      words: [{
        text: marker,
        startMs: 0,
        endMs: 100,
        confidence: 0.97,
        avgLogprob: -0.03,
        scoreType: 'ctc_acoustic_logprob',
      }],
      model,
      contextBias,
      hashes,
      verification: {
        metadataAvailable: true,
        metadataSchema: true,
        transcriptionSchema: true,
        runtimeIdentity: true,
        checkpoint: true,
        decoder: true,
        hashes: true,
        wordEvidence: true,
        productionReady: true,
      },
      provenance: {
        schemaVersion: 'gigaam.transcription.v2',
        runtimeId: 'b'.repeat(64),
        checkpointVerified: true,
        acousticDecoder: 'ctc-greedy',
        ctcDecoder: { mode: 'beam', beamWidth: 32 },
        contextBias,
        hashes,
        verification: {
          metadataAvailable: true,
          metadataSchema: true,
          transcriptionSchema: true,
          runtimeIdentity: true,
          checkpoint: true,
          decoder: true,
          hashes: true,
          wordEvidence: true,
          productionReady: true,
        },
      },
    };
  };
  const { app, dataDir } = await fixture(transcriber);
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      mimeType: 'audio/webm',
      retainAudio: true,
    },
  });
  assert.equal(created.statusCode, 201);
  const session = created.json();
  assert.equal(session.mode, 'radiology');
  assert.equal(session.retainAudio, true);

  const chunkOne = Buffer.from('second-webm-segment').toString('base64');
  const chunkZero = Buffer.from('first-webm-segment').toString('base64');
  assert.equal((await app.inject({
    method: 'POST',
    url: `/api/sessions/${session.sessionId}/chunks`,
    payload: { audio_base64: chunkOne, chunk_index: 1 },
  })).statusCode, 200);
  assert.equal((await app.inject({
    method: 'POST',
    url: `/api/sessions/${session.sessionId}/chunks`,
    payload: { audio_base64: chunkZero, chunk_index: 0 },
  })).statusCode, 200);

  const duplicate = await app.inject({
    method: 'POST',
    url: `/api/sessions/${session.sessionId}/chunks`,
    payload: { audio_base64: chunkZero, chunk_index: 0 },
  });
  assert.equal(duplicate.statusCode, 200);
  assert.equal(duplicate.json().duplicate, true);

  const conflict = await app.inject({
    method: 'POST',
    url: `/api/sessions/${session.sessionId}/chunks`,
    payload: { audio_base64: Buffer.from('different').toString('base64'), chunk_index: 0 },
  });
  assert.equal(conflict.statusCode, 409);
  assert.equal(conflict.json().error, 'chunk_index_conflict');

  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${session.sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200);
  const artifact = finished.json().artifact;
  assert.equal(artifact.schemaVersion, 2);
  assert.equal(artifact.kind, 'radiology-transcription');
  assert.equal(artifact.source.type, 'gigaam');
  assert.equal(artifact.rawTranscript.text, 'raw zero raw one');
  assert.equal(artifact.rawTranscript.rawAvailable, true);
  assert.equal(artifact.normalizedTranscript.text, 'raw zero raw one');
  assert.equal(artifact.normalization.text, artifact.normalizedTranscript.text);
  assert.equal(artifact.normalization.version, 'gigaam-denormalizer-strict-v2.1');
  assert.equal(artifact.audio.hashKind, 'sha256-index-length-prefixed-chunks-v1');
  assert.match(artifact.audio.sha256, /^[a-f0-9]{64}$/u);
  assert.deepEqual(artifact.audio.chunks.map((chunk: { index: number }) => chunk.index), [0, 1]);
  assert.ok(artifact.audio.chunks.every((chunk: { stored: boolean }) => chunk.stored));
  assert.equal(artifact.audio.stored, true);
  assert.equal(artifact.model.asr.version, 'v3_ctc');
  assert.equal(artifact.model.structurer.version, '1');
  assert.equal(artifact.model.safety.version, '1');
  assert.match(artifact.model.template.checksum, /^[a-f0-9]{64}$/u);
  assert.equal(artifact.components.prompt.version, 'radiology-span-router-v2');
  for (const component of Object.values(artifact.components) as Array<
    { configSha256?: string } | null
  >) {
    if (component) assert.match(component.configSha256, /^[a-f0-9]{64}$/u);
  }
  assert.equal(artifact.asrChunks.length, 2);
  assert.equal(artifact.asrChunks[0].rawText, ' raw zero\n');
  assert.equal(
    artifact.asrChunks[0].rawTextSha256,
    createHash('sha256').update(' raw zero\n').digest('hex'),
  );
  assert.equal(artifact.asrChunks[0].provenance.schemaVersion, 'gigaam.transcription.v2');
  assert.equal(artifact.asrChunks[0].provenance.runtimeId, 'b'.repeat(64));
  assert.deepEqual(artifact.asrChunks[0].provenance.verification, {
    schema: true,
    runtime: true,
    checkpoint: true,
    hashes: true,
    metadata: true,
    decoder: true,
    wordEvidence: true,
    productionContract: true,
  });
  assert.equal(artifact.asrChunks[0].provenance.contextBias.scope, 'ct-abdomen');
  assert.equal(
    artifact.asrChunks[0].provenance.hashes.audioSha256,
    artifact.audio.chunks[0].sha256,
  );
  assert.equal(artifact.safety.status, 'passed');
  assert.equal(artifact.safety.approvalBlocked, false);
  assert.equal(artifact.sections[0].evidence[0].transcript, 'normalized');
  assert.equal(artifact.report.fullText, 'CT abdomen\nLiver: raw zero raw one');
  assert.equal(
    artifact.reportSha256,
    createHash('sha256').update(artifact.report.fullText).digest('hex'),
  );
  assert.equal(artifact.training.eligible, true);

  const audioFiles = await readdir(path.join(dataDir, 'schema-v2', 'audio', session.sessionId));
  assert.deepEqual(audioFiles, ['00000000.chunk', '00000001.chunk']);

  const persisted = await app.inject({
    method: 'GET',
    url: `/api/radiology/sessions/${session.sessionId}/artifact`,
  });
  assert.equal(persisted.statusCode, 200);
  assert.equal(persisted.json().artifact.audio.sha256, artifact.audio.sha256);

  const retriedFinish = await app.inject({
    method: 'POST',
    url: `/api/sessions/${session.sessionId}/finish`,
    payload: {},
  });
  assert.equal(retriedFinish.statusCode, 200);
  assert.equal(retriedFinish.json().artifact.audio.sha256, artifact.audio.sha256);
});

test('word-number field evidence maps to the exact immutable raw phrase end to end', async (t) => {
  const rawText = 'печень КВР сто пятьдесят миллиметров';
  const { app, dataDir } = await fixture(
    async (audioBase64) => verifiedTranscription(audioBase64, rawText),
    undefined,
    async (templateId, transcript, context) => structureDictation(
      templateId,
      transcript,
      async () => {
        throw new Error('The deterministic liver anchor must not invoke the LLM');
      },
      context
        ? {
            allowLLM: context.allowLLM,
            rawTranscript: context.rawTranscript,
            normalizationAlignment: context.normalizationAlignment,
          }
        : {},
    ),
  );
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  assert.equal(created.statusCode, 201);
  const { sessionId } = created.json();
  assert.equal((await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('word-number-evidence').toString('base64'),
      chunk_index: 0,
    },
  })).statusCode, 200);

  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200);
  const artifact = finished.json().artifact;
  assert.equal(artifact.rawTranscript.text, rawText);
  assert.equal(artifact.normalization.text, 'печень КВР 150 миллиметров');
  const assignment = artifact.report.fieldAssignments.find(
    (item: { fieldId: string }) => item.fieldId === 'liver.kvr',
  );
  assert.equal(assignment.status, 'applied');
  assert.equal(assignment.value, 150);
  assert.match(artifact.report.reviewDraft.fullText, /КВР 150 мм/iu);

  const valueEvidence = assignment.evidence.find(
    (span: { normalized: { text: string } }) => span.normalized.text === '150',
  );
  assert.deepEqual(valueEvidence.raw, {
    start: rawText.indexOf('сто'),
    end: rawText.indexOf('сто') + 'сто пятьдесят'.length,
    text: 'сто пятьдесят',
  });
  const unitEvidence = assignment.evidence.find(
    (span: { normalized: { text: string } }) => /миллиметр/iu.test(span.normalized.text),
  );
  assert.equal(unitEvidence.raw.text, 'миллиметров');
  assert.ok(artifact.normalization.transformations.some(
    (transformation: { kind: string; source: { text: string } }) => (
      transformation.kind === 'cardinal'
      && transformation.source.text === 'сто пятьдесят'
    ),
  ));
});

test('mixed ASR runtime metadata fails closed before artifact persistence', async (t) => {
  const { app, dataDir } = await fixture(async (_audioBase64, context) => ({
    schemaVersion: 'gigaam.transcription.v2',
    runtimeId: context.chunkIndex === 0 ? 'runtime-a' : 'runtime-b',
    rawText: `raw ${context.chunkIndex}`,
    normalizedText: `text ${context.chunkIndex}`,
    rawAvailable: true,
    language: 'ru',
    source: 'gigaam',
    contextBias: { scope: 'ct-abdomen', active: true, terms: 42 },
    model,
  }));
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
    },
  });
  const sessionId = created.json().sessionId;
  for (const chunkIndex of [0, 1]) {
    assert.equal((await app.inject({
      method: 'POST',
      url: `/api/sessions/${sessionId}/chunks`,
      payload: {
        audio_base64: Buffer.from(`audio-${chunkIndex}`).toString('base64'),
        chunk_index: chunkIndex,
      },
    })).statusCode, 200);
  }

  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 409);
  assert.equal(finished.json().error, 'mixed_asr_provenance');
});

test('ASR-supplied hashes are verified against exact session payloads', async (t) => {
  const { app, dataDir } = await fixture(async () => ({
    schemaVersion: 'gigaam.transcription.v2',
    runtimeId: 'runtime-fixed',
    rawText: 'exact raw',
    normalizedText: 'exact normalized',
    rawAvailable: true,
    source: 'gigaam',
    model,
    hashes: {
      rawTextSha256: '0'.repeat(64),
    },
  }));
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });
  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
    },
  });
  const sessionId = created.json().sessionId;
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('audio').toString('base64'),
      chunk_index: 0,
    },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 422);
  assert.equal(finished.json().error, 'asr_provenance_hash_mismatch');
});

test('rejected ASR chunk can be retried and single-chunk audio uses byte SHA-256', async (t) => {
  let attempts = 0;
  const audio = Buffer.from('one-valid-webm-payload');
  const { app, dataDir } = await fixture(async () => {
    attempts += 1;
    if (attempts === 1) throw new Error('temporary ASR failure');
    return {
      schemaVersion: 'gigaam.transcription.v2',
      runtimeId: 'runtime-fixed',
      rawText: '  печень без особенностей\n',
      normalizedText: 'печень без особенностей',
      rawAvailable: true,
      language: 'ru',
      source: 'gigaam',
      contextBias: { scope: null, active: false, terms: 0 },
      model,
    };
  });
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const sessionId = created.json().sessionId;
  const payload = {
    audio_base64: audio.toString('base64'),
    chunk_index: 0,
  };
  assert.equal((await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload,
  })).statusCode, 200);
  await new Promise<void>((resolve) => setImmediate(resolve));

  const retry = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload,
  });
  assert.equal(retry.statusCode, 200);
  assert.equal(retry.json().duplicate, true);
  assert.equal(retry.json().retried, true);

  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200);
  assert.equal(attempts, 2);
  assert.equal(finished.json().artifact.rawTranscript.text, '  печень без особенностей\n');
  assert.equal(
    finished.json().artifact.rawTranscript.sha256,
    createHash('sha256').update('  печень без особенностей\n').digest('hex'),
  );
  assert.equal(
    finished.json().artifact.rawTranscript.sha256,
    finished.json().artifact.asrChunks[0].rawTextSha256,
  );
  assert.equal(finished.json().artifact.audio.hashKind, 'sha256-bytes');
  assert.equal(
    finished.json().artifact.audio.sha256,
    createHash('sha256').update(audio).digest('hex'),
  );
});

test('generic remote ASR rejection is exposed as a retryable 503 at finish', async (t) => {
  const { app, dataDir } = await fixture(async () => {
    throw new Error('remote transport closed');
  });
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });
  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
    },
  });
  const sessionId = created.json().sessionId;
  assert.equal((await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('remote-failure').toString('base64'),
      chunk_index: 0,
    },
  })).statusCode, 200);

  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 503, finished.body);
  assert.equal(finished.json().error, 'asr_transcription_failed');
});

test('radiology chunk route accepts payloads above Fastify default body limit', async (t) => {
  const { app, dataDir } = await fixture(async () => ({
    rawText: 'text',
    normalizedText: 'text',
    rawAvailable: true,
    source: 'gigaam',
    model,
  }));
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });
  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
    },
  });
  const largeAudio = Buffer.alloc(1_100_000, 0x61);
  const response = await app.inject({
    method: 'POST',
    url: `/api/sessions/${created.json().sessionId}/chunks`,
    payload: {
      audio_base64: largeAudio.toString('base64'),
      chunk_index: 0,
    },
  });
  assert.equal(response.statusCode, 200);
});

test('global ASR backpressure rejects new chunks without reserving their index', async (t) => {
  const dataDir = await mkdtemp(path.join(os.tmpdir(), 'voicemed-radiology-backpressure-'));
  t.after(async () => {
    await rm(dataDir, { recursive: true, force: true });
  });
  let release: (() => void) | undefined;
  const service = new RadiologySessionService({
    store: new RadiologyArtifactStore(dataDir),
    transcribeChunk: () => new Promise((resolve) => {
      release = () => resolve({
        rawText: 'text',
        normalizedText: 'text',
        rawAvailable: true,
        source: 'gigaam',
      });
    }),
    allowUnownedSessions: true,
    maxPendingTranscriptions: 1,
  });
  const input = {
    templateId: 'CT_ABDOMEN_MIKHAILOV',
    source: 'gigaam' as const,
  };
  const first = service.create(input);
  const second = service.create(input);
  await service.addChunk(first.sessionId, {
    audioBase64: Buffer.from('first').toString('base64'),
    chunkIndex: 0,
  });
  await assert.rejects(
    service.addChunk(second.sessionId, {
      audioBase64: Buffer.from('second').toString('base64'),
      chunkIndex: 0,
    }),
    (error: unknown) =>
      error instanceof Error
      && 'code' in error
      && error.code === 'asr_backpressure',
  );

  release?.();
  await service.finish(first.sessionId);
  const accepted = await service.addChunk(second.sessionId, {
    audioBase64: Buffer.from('second').toString('base64'),
    chunkIndex: 0,
  });
  assert.equal(accepted.duplicate, false);
  release?.();
  await service.finish(second.sessionId);
});

test('feedback is immutable, versioned, span-checked, and training eligible only with real raw audio', async (t) => {
  const { app, dataDir, store } = await fixture(async (audioBase64) => {
    const rawText = 'печень пятнадцать миллиметров';
    const normalizedText = 'печень 15 мм';
    return verifiedTranscription(audioBase64, rawText, normalizedText);
  });
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const { sessionId } = created.json();
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: { audio_base64: Buffer.from('webm').toString('base64'), chunk_index: 0 },
  });
  assert.equal((await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  })).statusCode, 200);

  const feedbackPayload = {
    idempotencyKey: 'feedback-test-00000001',
    verbatimTranscript: 'печень 15 миллиметров',
    finalReport: 'Печень: размер 15 мм.',
    author: 'doctor-1',
    approved: true,
    spanCorrections: [{
      start: 7,
      end: 17,
      originalText: 'пятнадцать',
      correctedText: '15',
      entityType: 'number_unit',
      confidence: 0.61,
      modality: 'CT',
    }],
  };
  const feedback = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: feedbackPayload,
  });
  assert.equal(feedback.statusCode, 201);
  assert.equal(feedback.json().revision, 1);
  assert.equal(feedback.json().datasetVersion, 'radiology-feedback/v2');
  assert.equal(feedback.json().training.eligible, true);
  assert.equal(feedback.json().idempotentReplay, false);

  const replay = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: feedbackPayload,
  });
  assert.equal(replay.statusCode, 200);
  assert.equal(replay.json().idempotentReplay, true);
  assert.equal(replay.json().feedbackId, feedback.json().feedbackId);
  assert.equal(replay.json().revision, feedback.json().revision);

  const { idempotencyKey: _missingKey, ...withoutIdempotencyKey } = feedbackPayload;
  const missingIdempotencyKey = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: withoutIdempotencyKey,
  });
  assert.equal(missingIdempotencyKey.statusCode, 400);

  const idempotencyConflict = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: { ...feedbackPayload, approved: false },
  });
  assert.equal(idempotencyConflict.statusCode, 409);
  assert.equal(idempotencyConflict.json().error, 'feedback_idempotency_conflict');

  const second = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...feedbackPayload,
      idempotencyKey: 'feedback-test-00000002',
      approved: false,
    },
  });
  assert.equal(second.statusCode, 201);
  assert.equal(second.json().revision, 2);
  assert.equal(second.json().training.eligible, false);

  const stored = await store.listFeedback(sessionId);
  assert.equal(stored.length, 2);
  assert.equal(stored[0].author, 'doctor-1');
  assert.equal(stored[0].spanCorrections[0].author, 'doctor-1');
  assert.equal(stored[0].safety.ok, true);
  assert.equal(
    stored[0].verbatimTranscriptSha256,
    createHash('sha256').update(feedbackPayload.verbatimTranscript).digest('hex'),
  );
  assert.equal(
    stored[0].finalReportSha256,
    createHash('sha256').update(feedbackPayload.finalReport).digest('hex'),
  );

  const mismatched = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...feedbackPayload,
      idempotencyKey: 'feedback-test-00000003',
      approved: false,
      spanCorrections: [{ ...feedbackPayload.spanCorrections[0], originalText: 'wrong' }],
    },
  });
  assert.equal(mismatched.statusCode, 409);
  assert.equal(mismatched.json().error, 'correction_span_mismatch');

  const wrongModality = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...feedbackPayload,
      idempotencyKey: 'feedback-test-00000004',
      approved: false,
      spanCorrections: [{ ...feedbackPayload.spanCorrections[0], modality: 'MRI' }],
    },
  });
  assert.equal(wrongModality.statusCode, 409);
  assert.equal(wrongModality.json().error, 'correction_modality_mismatch');

  const undocumentedEdit = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...feedbackPayload,
      idempotencyKey: 'feedback-test-00000005',
      approved: false,
      verbatimTranscript: 'печень 25 миллиметров',
      spanCorrections: [],
    },
  });
  assert.equal(undocumentedEdit.statusCode, 409);
  assert.equal(undocumentedEdit.json().error, 'verbatim_corrections_mismatch');

  const overlapping = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...feedbackPayload,
      idempotencyKey: 'feedback-test-00000006',
      approved: false,
      spanCorrections: [
        feedbackPayload.spanCorrections[0],
        {
          ...feedbackPayload.spanCorrections[0],
          start: 10,
          originalText: 'надцать',
        },
      ],
    },
  });
  assert.equal(overlapping.statusCode, 409);
  assert.equal(overlapping.json().error, 'correction_spans_overlap');
});

test('legacy, browser, and manual transcripts are explicitly excluded from training', async (t) => {
  const { app, dataDir } = await fixture(async () => 'postprocessed legacy text');
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const legacy = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: { mode: 'radiology', templateId: 'CT_ABDOMEN_MIKHAILOV', source: 'gigaam' },
  });
  const legacyId = legacy.json().sessionId;
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${legacyId}/chunks`,
    payload: { audio_base64: Buffer.from('legacy').toString('base64'), chunk_index: 0 },
  });
  const legacyFinish = await app.inject({
    method: 'POST',
    url: `/api/sessions/${legacyId}/finish`,
    payload: {},
  });
  const legacyArtifact = legacyFinish.json().artifact;
  assert.equal(legacyArtifact.rawTranscript.rawAvailable, false);
  assert.equal(legacyArtifact.training.eligible, false);
  assert.ok(legacyArtifact.training.exclusionReasons.includes('raw_asr_unavailable'));
  assert.ok(legacyArtifact.training.exclusionReasons.includes('audio_not_retained'));

  const browser = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: { mode: 'radiology', templateId: 'CT_ABDOMEN_MIKHAILOV', source: 'browser' },
  });
  const browserFinish = await app.inject({
    method: 'POST',
    url: `/api/sessions/${browser.json().sessionId}/finish`,
    payload: { browserTranscript: 'печень без особенностей' },
  });
  assert.equal(browserFinish.statusCode, 200);
  const browserArtifact = browserFinish.json().artifact;
  assert.equal(browserArtifact.source.type, 'browser');
  assert.equal(browserArtifact.audio.sha256, null);
  assert.equal(browserArtifact.audio.hashKind, 'none');
  assert.equal(browserArtifact.training.eligible, false);
  assert.ok(browserArtifact.training.exclusionReasons.includes('browser_asr_source'));

  const manual = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: { mode: 'radiology', templateId: 'CT_ABDOMEN_MIKHAILOV', source: 'manual' },
  });
  const manualFinish = await app.inject({
    method: 'POST',
    url: `/api/sessions/${manual.json().sessionId}/finish`,
    payload: { browserTranscript: 'печень без особенностей' },
  });
  assert.equal(manualFinish.statusCode, 200);
  const manualArtifact = manualFinish.json().artifact;
  assert.equal(manualArtifact.source.type, 'manual');
  assert.equal(manualArtifact.rawTranscript.rawAvailable, false);
  assert.equal(manualArtifact.audio.sha256, null);
  assert.equal(manualArtifact.training.eligible, false);
  assert.ok(manualArtifact.training.exclusionReasons.includes('manual_transcript_source'));
});

test('Whisper challenger artifacts never enter the governed training dataset', async (t) => {
  const rawText = 'печень без особенностей';
  const { app, dataDir } = await fixture(async (audioBase64) => {
    const transcription = verifiedTranscription(audioBase64, rawText);
    return {
      ...transcription,
      schemaVersion: 'whisper.transcription.v1',
      source: 'whisper' as const,
      provenance: {
        ...transcription.provenance,
        schemaVersion: 'whisper.transcription.v1',
      },
    };
  });
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'whisper',
      retainAudio: true,
    },
  });
  const sessionId = created.json().sessionId;
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('whisper-challenger').toString('base64'),
      chunk_index: 0,
    },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200);
  const artifact = finished.json().artifact;
  assert.equal(artifact.source.type, 'whisper');
  assert.equal(artifact.training.eligible, false);
  assert.ok(
    artifact.training.exclusionReasons.includes('whisper_challenger_source'),
  );
});

test('schema-v1 artifacts are adapted as incomplete and cannot be approved', async (t) => {
  const { app, dataDir } = await fixture(async () => 'unused');
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const sessionId = 'legacy-v1-artifact';
  const text = 'печень без особенностей';
  const sha256 = createHash('sha256').update(text).digest('hex');
  const legacyDir = path.join(dataDir, 'schema-v1', 'artifacts');
  await mkdir(legacyDir, { recursive: true });
  await writeFile(path.join(legacyDir, `${sessionId}.json`), JSON.stringify({
    schemaVersion: 1,
    kind: 'radiology-transcription',
    sessionId,
    ownerDoctorId: null,
    templateId: 'CT_ABDOMEN_MIKHAILOV',
    createdAt: new Date().toISOString(),
    completedAt: new Date().toISOString(),
    source: 'gigaam',
    audio: {
      sha256: null,
      hashKind: 'none',
      bytes: 0,
      stored: false,
      chunks: [],
    },
    rawTranscript: {
      text,
      sha256,
      language: 'ru',
      rawAvailable: true,
      words: [],
    },
    normalizedTranscript: { text, sha256 },
    asrChunks: [],
    sections: [],
    unmatchedText: '',
    report: reportFor(text),
    safety: {},
    model,
    training: { eligible: false, exclusionReasons: [] },
  }), 'utf8');

  const loaded = await app.inject({
    method: 'GET',
    url: `/api/radiology/sessions/${sessionId}/artifact`,
  });
  assert.equal(loaded.statusCode, 200);
  const adapted = loaded.json().artifact;
  assert.equal(adapted.schemaVersion, 2);
  assert.equal(adapted.legacySchemaVersion, 1);
  assert.equal(adapted.safety.status, 'incomplete');
  assert.equal(adapted.safety.approvalBlocked, true);
  assert.equal(adapted.training.eligible, false);
  assert.ok(adapted.training.exclusionReasons.includes('legacy_artifact_schema'));

  const approval = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'legacy-v1-approval-000001',
      verbatimTranscript: text,
      finalReport: reportFor(text).fullText,
      spanCorrections: [],
      approved: true,
      author: 'doctor-1',
    },
  });
  assert.equal(approval.statusCode, 422);
  assert.equal(approval.json().error, 'legacy_artifact_approval_blocked');
});

test('unverified ASR provenance remains reviewable but is fail-closed for training', async (t) => {
  const { app, dataDir } = await fixture(async () => ({
    rawText: 'печень без особенностей',
    normalizedText: 'печень без особенностей',
    rawAvailable: true,
    source: 'gigaam',
    model,
  }));
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const sessionId = created.json().sessionId;
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('unverified-runtime').toString('base64'),
      chunk_index: 0,
    },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200);
  const artifact = finished.json().artifact;
  assert.equal(artifact.normalizedTranscript.text, 'печень без особенностей');
  assert.equal(artifact.training.eligible, false);
  assert.deepEqual(artifact.asrChunks[0].provenance.verification, {
    schema: false,
    runtime: false,
    checkpoint: false,
    hashes: false,
    metadata: false,
    decoder: false,
    wordEvidence: false,
    productionContract: false,
  });
  for (const reason of [
    'asr_schema_unverified',
    'asr_runtime_unverified',
    'asr_checkpoint_unverified',
    'asr_hashes_unverified',
    'asr_metadata_unverified',
    'asr_decoder_unverified',
    'asr_word_evidence_unverified',
    'asr_contract_unverified',
  ]) {
    assert.ok(artifact.training.exclusionReasons.includes(reason), reason);
  }
});

test('feedback revision cap is bounded while idempotent replays remain available', async (t) => {
  const dataDir = await mkdtemp(path.join(os.tmpdir(), 'voicemed-radiology-feedback-cap-'));
  const store = new RadiologyArtifactStore(dataDir, {
    maxFeedbackRevisionsPerSession: 2,
  });
  const service = new RadiologySessionService({
    store,
    transcribeChunk: async (audioBase64) =>
      verifiedTranscription(audioBase64, 'печень без особенностей'),
    structureTranscript: async (_templateId, transcript) => reportFor(transcript),
    model,
    allowUnownedSessions: true,
    allowAudioPersistence: true,
  });
  const app = Fastify();
  registerRadiologyRoutes(app, { sessionService: service });
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const sessionId = created.json().sessionId;
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('feedback-cap').toString('base64'),
      chunk_index: 0,
    },
  });
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });

  const payload = {
    idempotencyKey: 'feedback-cap-000000001',
    verbatimTranscript: 'печень без особенностей',
    finalReport: 'печень без особенностей',
    spanCorrections: [],
    approved: false,
    author: 'doctor-1',
  };
  const first = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload,
  });
  assert.equal(first.statusCode, 201);
  const second = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: { ...payload, idempotencyKey: 'feedback-cap-000000002' },
  });
  assert.equal(second.statusCode, 201);
  const capped = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: { ...payload, idempotencyKey: 'feedback-cap-000000003' },
  });
  assert.equal(capped.statusCode, 409);
  assert.equal(capped.json().error, 'feedback_revision_limit');

  const replay = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload,
  });
  assert.equal(replay.statusCode, 200);
  assert.equal(replay.json().revision, 1);
  assert.equal(replay.json().idempotentReplay, true);
  assert.equal((await store.listFeedback(sessionId)).length, 2);
});

test('expired persisted artifacts remove associated audio and feedback', async (t) => {
  const dataDir = await mkdtemp(path.join(os.tmpdir(), 'voicemed-radiology-retention-'));
  t.after(async () => {
    await rm(dataDir, { recursive: true, force: true });
  });
  let nowMs = 0;
  const now = () => new Date(nowMs);
  const store = new RadiologyArtifactStore(dataDir, {
    storageRetentionMs: 1_000,
    cleanupIntervalMs: 0,
    now,
  });
  const service = new RadiologySessionService({
    store,
    transcribeChunk: async (audioBase64) =>
      verifiedTranscription(audioBase64, 'печень без особенностей'),
    structureTranscript: async (_templateId, transcript) => reportFor(transcript),
    model,
    allowUnownedSessions: true,
    allowAudioPersistence: true,
    now,
  });
  const created = service.create({
    templateId: 'CT_ABDOMEN_MIKHAILOV',
    source: 'gigaam',
    retainAudio: true,
  });
  await service.addChunk(created.sessionId, {
    audioBase64: Buffer.from('retained-audio').toString('base64'),
    chunkIndex: 0,
  });
  await service.finish(created.sessionId);
  await service.saveFeedback(created.sessionId, {
    idempotencyKey: 'retention-feedback-00001',
    verbatimTranscript: 'печень без особенностей',
    finalReport: 'печень без особенностей',
    spanCorrections: [],
    approved: false,
    author: 'doctor-1',
  }, 'doctor-1');

  nowMs = 1_001;
  assert.equal(await store.getArtifact(created.sessionId), null);
  await assert.rejects(
    readdir(path.join(dataDir, 'schema-v2', 'audio', created.sessionId)),
    (error: unknown) => error instanceof Error && 'code' in error && error.code === 'ENOENT',
  );
  await assert.rejects(
    readdir(path.join(dataDir, 'schema-v2', 'feedback', created.sessionId)),
    (error: unknown) => error instanceof Error && 'code' in error && error.code === 'ENOENT',
  );
});

test('critical safety failure returns an artifact but blocks approved feedback', async (t) => {
  const { app, dataDir } = await fixture(
    async () => ({
      rawText: 'очаг 15 мм',
      normalizedText: 'очаг 15 мм',
      rawAvailable: true,
      source: 'gigaam',
      model,
    }),
    () => 'очаг 50 мм',
  );
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const { sessionId } = created.json();
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: { audio_base64: Buffer.from('safety').toString('base64'), chunk_index: 0 },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200);
  assert.equal(finished.json().artifact.safety.status, 'failed');
  assert.equal(finished.json().artifact.safety.approvalBlocked, true);

  const approved = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'safety-approved-00000001',
      verbatimTranscript: 'очаг 15 мм',
      finalReport: 'очаг 50 мм',
      spanCorrections: [],
      approved: true,
      author: 'doctor-1',
    },
  });
  assert.equal(approved.statusCode, 422);
  assert.equal(approved.json().error, 'safety_approval_blocked');

  const rejected = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'safety-rejected-00000001',
      verbatimTranscript: 'очаг 15 мм',
      finalReport: 'очаг 50 мм',
      spanCorrections: [],
      approved: false,
      author: 'doctor-1',
    },
  });
  assert.equal(rejected.statusCode, 201);
});

test('ambiguous number sequences produce a deterministic blocked draft until explicitly resolved', async (t) => {
  const rawText = 'плотность пятьдесят пятьдесят три HU';
  let structurerContext: Parameters<RadiologyTranscriptStructurer>[2];
  const { app, dataDir } = await fixture(
    async (audioBase64) => verifiedTranscription(audioBase64, rawText),
    undefined,
    async (_templateId, transcript, context) => {
      structurerContext = context;
      return reportFor(transcript);
    },
  );
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const { sessionId } = created.json();
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('ambiguous-number').toString('base64'),
      chunk_index: 0,
    },
  });

  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200);
  const artifact = finished.json().artifact;
  assert.equal(artifact.normalization.text, 'плотность 50 53 HU');
  assert.doesNotMatch(artifact.normalization.text, /\b103\b/u);
  assert.equal(artifact.normalization.issues.length, 1);
  assert.equal(artifact.normalization.issues[0].code, 'ambiguous_number_sequence');
  assert.equal(structurerContext?.allowLLM, false);
  assert.equal(structurerContext?.normalizationAmbiguous, true);
  assert.equal(structurerContext?.rawTranscript, rawText);
  assert.ok((structurerContext?.normalizationAlignment.length ?? 0) > 0);
  assert.notEqual(artifact.report, null);
  assert.equal(artifact.safety.status, 'incomplete');
  assert.equal(artifact.safety.approvalBlocked, true);

  const unresolvedApproval = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'ambiguous-unresolved-0001',
      verbatimTranscript: rawText,
      finalReport: artifact.report.fullText,
      spanCorrections: [],
      approved: true,
      author: 'doctor-1',
    },
  });
  assert.equal(unresolvedApproval.statusCode, 422);
  assert.equal(unresolvedApproval.json().error, 'normalization_resolution_required');

  const originalText = 'пятьдесят пятьдесят три';
  const correctedText = 'пятьдесят — пятьдесят три';
  const start = rawText.indexOf(originalText);
  const verbatimTranscript = rawText.replace(originalText, correctedText);
  const approved = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'ambiguous-resolved-000001',
      verbatimTranscript,
      finalReport: 'CT abdomen\nLiver: плотность 50 — 53 HU',
      spanCorrections: [{
        start,
        end: start + originalText.length,
        originalText,
        correctedText,
        entityType: 'number_unit',
        modality: 'CT',
      }],
      normalizationResolutions: [{
        issueId: artifact.normalization.issues[0].id,
        replacementText: correctedText,
        resolution: 'confirmed_range',
      }],
      approved: true,
      author: 'doctor-1',
    },
  });
  assert.equal(approved.statusCode, 201);
  assert.equal(approved.json().success, true);
  assert.equal(approved.json().training.eligible, true);
});

test('critical long-form seam conflicts remain approval-blocking after text review', async (t) => {
  const text = 'печень без особенностей';
  const { app, dataDir } = await fixture(async (audioBase64) => ({
    ...verifiedTranscription(audioBase64, text),
    longform: {
      mode: 'emission_stitch' as const,
      degraded: false,
      vad: model.vad,
      seams: [{
        startMs: 18_000,
        endMs: 20_000,
        conflict: true,
        critical: true,
        leftText: '15 мм',
        rightText: '50 мм',
      }],
    },
  }));
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const { sessionId } = created.json();
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('critical-seam').toString('base64'),
      chunk_index: 0,
    },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  const artifact = finished.json().artifact;
  assert.equal(artifact.safety.status, 'failed');
  assert.equal(artifact.safety.approvalBlocked, true);
  assert.ok(artifact.safety.issues.some(
    (issue: { code: string }) => issue.code === 'overlap_seam_conflict',
  ));
  assert.equal(artifact.training.eligible, false);

  const approval = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'critical-seam-approval-001',
      verbatimTranscript: text,
      finalReport: artifact.report.fullText,
      spanCorrections: [],
      approved: true,
      author: 'doctor-1',
    },
  });
  assert.equal(approval.statusCode, 422);
  assert.equal(approval.json().error, 'longform_integrity_approval_blocked');
});

test('feedback safety is recalculated after edits and blocks newly unsupported facts', async (t) => {
  const { app, dataDir } = await fixture(async () => ({
    rawText: 'печень без особенностей',
    normalizedText: 'печень без особенностей',
    rawAvailable: true,
    source: 'gigaam',
    model,
  }));
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const sessionId = created.json().sessionId;
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: { audio_base64: Buffer.from('safe-edit').toString('base64'), chunk_index: 0 },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.json().artifact.safety.status, 'passed');

  const feedback = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'unsafe-edit-000000000001',
      verbatimTranscript: 'печень без особенностей',
      finalReport: `${finished.json().artifact.report.fullText}\nПочка слева: образование 50 мм.`,
      spanCorrections: [],
      approved: true,
      author: 'doctor-1',
    },
  });
  assert.equal(feedback.statusCode, 422);
  assert.equal(feedback.json().error, 'safety_approval_blocked');
});

test('doctor can approve a previously unsafe draft after exact reviewed corrections', async (t) => {
  const { app, dataDir, store } = await fixture(
    async (audioBase64) => verifiedTranscription(audioBase64, 'очаг 50 мм'),
    () => 'очаг 15 мм',
  );
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const sessionId = created.json().sessionId;
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: { audio_base64: Buffer.from('corrected-safety').toString('base64'), chunk_index: 0 },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.json().artifact.safety.status, 'failed');

  const approved = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'corrected-safe-000000001',
      verbatimTranscript: 'очаг 15 мм',
      finalReport: finished.json().artifact.report.fullText,
      spanCorrections: [{
        start: 5,
        end: 7,
        originalText: '50',
        correctedText: '15',
        entityType: 'number_unit',
        modality: 'CT',
      }],
      approved: true,
      author: 'doctor-1',
    },
  });
  assert.equal(approved.statusCode, 201);
  const stored = await store.listFeedback(sessionId);
  assert.equal(stored[0].safety.ok, true);
  assert.equal(stored[0].training.eligible, true);
});

test('canonical create validates mode and template without changing legacy routes', async (t) => {
  const { app, dataDir } = await fixture(async () => 'text');
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const mode = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: { mode: 'consultation', templateId: 'CT_ABDOMEN_MIKHAILOV' },
  });
  assert.equal(mode.statusCode, 400);
  assert.equal(mode.json().error, 'unsupported_mode');

  const template = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: { mode: 'radiology', templateId: 'UNKNOWN' },
  });
  assert.equal(template.statusCode, 404);
  assert.equal(template.json().error, 'template_not_found');
});

test('canonical PHI sessions are owner-scoped, accept numeric JWT ids, and disable caching', async (t) => {
  const { app, dataDir, store } = await fixture(async () => ({
    rawText: 'очаг 15 мм',
    normalizedText: 'очаг 15 мм',
    rawAvailable: true,
    source: 'gigaam',
    model,
  }));
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const ownerHeaders = { 'x-test-doctor-id': '101' };
  const otherHeaders = { 'x-test-doctor-id': '202' };
  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    headers: ownerHeaders,
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  assert.equal(created.statusCode, 201);
  assert.equal(created.headers['cache-control'], 'no-store');
  const { sessionId } = created.json();

  const otherChunk = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    headers: otherHeaders,
    payload: { audio_base64: Buffer.from('webm').toString('base64'), chunk_index: 0 },
  });
  assert.equal(otherChunk.statusCode, 403);
  assert.equal(otherChunk.json().error, 'radiology_session_forbidden');
  assert.equal(otherChunk.headers['cache-control'], 'no-store');

  const ownerChunk = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    headers: ownerHeaders,
    payload: { audio_base64: Buffer.from('webm').toString('base64'), chunk_index: 0 },
  });
  assert.equal(ownerChunk.statusCode, 200);
  assert.equal(ownerChunk.headers['cache-control'], 'no-store');

  const otherFinish = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    headers: otherHeaders,
    payload: {},
  });
  assert.equal(otherFinish.statusCode, 403);

  const ownerFinish = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    headers: ownerHeaders,
    payload: {},
  });
  assert.equal(ownerFinish.statusCode, 200);
  assert.equal(ownerFinish.headers['cache-control'], 'no-store');
  assert.equal(ownerFinish.json().artifact.ownerDoctorId, '101');

  const unauthenticatedRead = await app.inject({
    method: 'GET',
    url: `/api/radiology/sessions/${sessionId}/artifact`,
  });
  assert.equal(unauthenticatedRead.statusCode, 403);

  const otherRead = await app.inject({
    method: 'GET',
    url: `/api/radiology/sessions/${sessionId}/artifact`,
    headers: otherHeaders,
  });
  assert.equal(otherRead.statusCode, 403);
  assert.equal(otherRead.headers['cache-control'], 'no-store');

  const adminRead = await app.inject({
    method: 'GET',
    url: `/api/radiology/sessions/${sessionId}/artifact`,
    headers: { 'x-test-doctor-id': '999', 'x-test-role': 'admin' },
  });
  assert.equal(adminRead.statusCode, 200);
  assert.equal(adminRead.headers['cache-control'], 'no-store');

  const feedbackPayload = {
    idempotencyKey: 'owner-feedback-00000001',
    verbatimTranscript: 'очаг 15 мм',
    finalReport: 'очаг 15 мм',
    spanCorrections: [],
    approved: true,
  };
  const otherFeedback = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    headers: otherHeaders,
    payload: feedbackPayload,
  });
  assert.equal(otherFeedback.statusCode, 403);
  assert.equal(otherFeedback.headers['cache-control'], 'no-store');

  const ownerFeedback = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    headers: ownerHeaders,
    payload: feedbackPayload,
  });
  assert.equal(ownerFeedback.statusCode, 201);
  assert.equal(ownerFeedback.headers['cache-control'], 'no-store');
  const feedback = await store.listFeedback(sessionId);
  assert.equal(feedback[0]?.author, '101');
});

test('radiology sessions enforce active/chunk/total-byte limits and refresh TTL on activity', async (t) => {
  const dataDir = await mkdtemp(path.join(os.tmpdir(), 'voicemed-radiology-limits-'));
  t.after(async () => {
    await rm(dataDir, { recursive: true, force: true });
  });
  let nowMs = 0;
  let nextId = 0;
  const service = new RadiologySessionService({
    store: new RadiologyArtifactStore(dataDir),
    transcribeChunk: async () => ({
      rawText: 'text',
      normalizedText: 'text',
      rawAvailable: true,
      source: 'gigaam',
    }),
    allowUnownedSessions: true,
    sessionTtlMs: 1_000,
    maxActiveSessions: 1,
    maxChunkBytes: 10,
    maxChunksPerSession: 2,
    maxTotalAudioBytes: 4,
    now: () => new Date(nowMs),
    idFactory: () => `session-${++nextId}`,
  });
  const input = {
    templateId: 'CT_ABDOMEN_MIKHAILOV',
    source: 'gigaam' as const,
  };

  const first = service.create(input);
  assert.throws(
    () => service.create(input),
    (error: unknown) =>
      error instanceof Error
      && 'code' in error
      && error.code === 'active_session_limit',
  );

  nowMs = 900;
  await service.addChunk(first.sessionId, {
    audioBase64: Buffer.from('aa').toString('base64'),
    chunkIndex: 0,
  });
  await new Promise<void>((resolve) => setImmediate(resolve));
  nowMs = 1_500;
  assert.throws(
    () => service.create(input),
    (error: unknown) =>
      error instanceof Error
      && 'code' in error
      && error.code === 'active_session_limit',
  );

  nowMs = 1_901;
  const second = service.create(input);
  const duplicate = await service.addChunk(second.sessionId, {
    audioBase64: Buffer.from('aa').toString('base64'),
    chunkIndex: 0,
  });
  assert.equal(duplicate.duplicate, false);
  assert.equal((await service.addChunk(second.sessionId, {
    audioBase64: Buffer.from('aa').toString('base64'),
    chunkIndex: 0,
  })).duplicate, true);
  await service.addChunk(second.sessionId, {
    audioBase64: Buffer.from('bb').toString('base64'),
    chunkIndex: 1,
  });
  await assert.rejects(
    service.addChunk(second.sessionId, {
      audioBase64: Buffer.from('c').toString('base64'),
      chunkIndex: 2,
    }),
    (error: unknown) =>
      error instanceof Error
      && 'code' in error
      && error.code === 'session_chunk_limit',
  );
  await new Promise<void>((resolve) => setImmediate(resolve));

  nowMs = 3_000;
  const third = service.create(input);
  await service.addChunk(third.sessionId, {
    audioBase64: Buffer.from('aaa').toString('base64'),
    chunkIndex: 0,
  });
  await assert.rejects(
    service.addChunk(third.sessionId, {
      audioBase64: Buffer.from('bb').toString('base64'),
      chunkIndex: 1,
    }),
    (error: unknown) =>
      error instanceof Error
      && 'code' in error
      && error.code === 'session_audio_too_large',
  );
});

test('radiology sessions enforce per-owner and global active audio memory quotas', async (t) => {
  const dataDir = await mkdtemp(path.join(os.tmpdir(), 'voicemed-radiology-owner-limits-'));
  t.after(async () => {
    await rm(dataDir, { recursive: true, force: true });
  });
  let nextId = 0;
  const service = new RadiologySessionService({
    store: new RadiologyArtifactStore(dataDir),
    transcribeChunk: async () => ({
      rawText: 'text',
      normalizedText: 'text',
      rawAvailable: true,
      source: 'gigaam',
    }),
    maxActiveSessions: 10,
    maxOwnerActiveSessions: 1,
    maxActiveAudioBytes: 5,
    maxOwnerActiveAudioBytes: 3,
    maxChunkBytes: 10,
    maxTotalAudioBytes: 10,
    idFactory: () => `owner-session-${++nextId}`,
  });
  const input = {
    templateId: 'CT_ABDOMEN_MIKHAILOV',
    source: 'gigaam' as const,
  };
  const ownerOne = { authenticated: true as const, doctorId: 'doctor-1', role: 'doctor' };
  const ownerTwo = { authenticated: true as const, doctorId: 'doctor-2', role: 'doctor' };

  const first = service.create(input, ownerOne);
  assert.throws(
    () => service.create(input, ownerOne),
    (error: unknown) =>
      error instanceof Error
      && 'code' in error
      && error.code === 'owner_active_session_limit',
  );
  const second = service.create(input, ownerTwo);

  await service.addChunk(first.sessionId, {
    audioBase64: Buffer.from('aa').toString('base64'),
    chunkIndex: 0,
  }, ownerOne);
  await service.addChunk(first.sessionId, {
    audioBase64: Buffer.from('b').toString('base64'),
    chunkIndex: 1,
  }, ownerOne);
  await assert.rejects(
    service.addChunk(first.sessionId, {
      audioBase64: Buffer.from('c').toString('base64'),
      chunkIndex: 2,
    }, ownerOne),
    (error: unknown) =>
      error instanceof Error
      && 'code' in error
      && error.code === 'owner_active_audio_bytes_limit',
  );

  await service.addChunk(second.sessionId, {
    audioBase64: Buffer.from('dd').toString('base64'),
    chunkIndex: 0,
  }, ownerTwo);
  await assert.rejects(
    service.addChunk(second.sessionId, {
      audioBase64: Buffer.from('e').toString('base64'),
      chunkIndex: 1,
    }, ownerTwo),
    (error: unknown) =>
      error instanceof Error
      && 'code' in error
      && error.code === 'active_audio_bytes_limit',
  );
});

test('template review feedback is bound to immutable draft SHA, segment ids, and residual review', async (t) => {
  const rawText = 'печень КВР 150 плотность 60. селезёнка 12 на 6 на 5';
  const { app, dataDir, store } = await fixture(
    async (audioBase64) => verifiedTranscription(audioBase64, rawText),
    undefined,
    async (templateId, transcript) => structureDictation(
      templateId,
      transcript,
      async (_system, user) => {
        const atoms = (JSON.parse(user) as { atoms: Array<{ atomId: string }> }).atoms;
        return JSON.stringify({
          assignments: atoms.map((atom) => ({
            atomId: atom.atomId,
            sectionId: null,
          })),
        });
      },
    ),
  );
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const { sessionId } = created.json();
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('template-review-feedback').toString('base64'),
      chunk_index: 0,
    },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200);
  const artifact = finished.json().artifact;
  const draft = artifact.report.reviewDraft;
  assert.ok(draft);
  assert.equal(
    artifact.report.blocks.find((block: { id: string }) => block.id === 'liver').text,
    'печень КВР 150 плотность 60.',
  );
  assert.match(draft.fullText, /КВР 150 мм/iu);
  assert.match(draft.fullText, /\+60 HU/iu);
  assert.equal(artifact.components.composer.version, draft.version);
  assert.equal(draft.composerVersion, draft.version);
  assert.equal(draft.templateSha256, artifact.components.template.checksum);

  const placeholderSegments = draft.segments.filter(
    (segment: { defaultKind?: string }) => segment.defaultKind === 'placeholder',
  );
  const finalReportWithoutPlaceholders = [...placeholderSegments]
    .sort((left, right) => right.start - left.start)
    .reduce(
      (text, segment) => `${text.slice(0, segment.start)}${text.slice(segment.end)}`,
      draft.fullText,
    );
  const acceptedTemplateSegmentIds = draft.segments
    .filter((segment: { confirmationRequired: boolean; defaultKind?: string }) => (
      segment.confirmationRequired && segment.defaultKind !== 'placeholder'
    ))
    .map((segment: { id: string }) => segment.id);
  assert.ok(acceptedTemplateSegmentIds.length > 0);
  const reviewedResidualAtomIds = [...draft.residualAtomIds];
  const basePayload = {
    verbatimTranscript: rawText,
    finalReport: finalReportWithoutPlaceholders,
    spanCorrections: [],
    normalizationResolutions: [],
    acceptedTemplateSegmentIds,
    reviewedResidualAtomIds,
    approved: true,
    author: 'doctor-1',
  };

  const staleDraft = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-draft-stale-000001',
      baseDraftSha256: '0'.repeat(64),
    },
  });
  assert.equal(staleDraft.statusCode, 409);
  assert.equal(staleDraft.json().error, 'review_draft_sha_mismatch');

  const foreignSegment = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-draft-segment-0001',
      baseDraftSha256: draft.sha256,
      acceptedTemplateSegmentIds: [...acceptedTemplateSegmentIds, 'foreign-segment'],
    },
  });
  assert.equal(foreignSegment.statusCode, 422, foreignSegment.body);
  assert.equal(foreignSegment.json().error, 'template_segment_mismatch');

  const nonConfirmableSegment = draft.segments.find(
    (segment: { confirmationRequired: boolean }) => !segment.confirmationRequired,
  );
  assert.ok(nonConfirmableSegment);
  const nonConfirmableDecision = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-draft-not-confirmable-0001',
      baseDraftSha256: draft.sha256,
      acceptedTemplateSegmentIds: [
        ...acceptedTemplateSegmentIds,
        nonConfirmableSegment.id,
      ],
    },
  });
  assert.equal(nonConfirmableDecision.statusCode, 422, nonConfirmableDecision.body);
  assert.equal(
    nonConfirmableDecision.json().error,
    'template_segment_not_confirmable',
  );

  const unacceptedButPresent = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-draft-missing-0001',
      baseDraftSha256: draft.sha256,
      acceptedTemplateSegmentIds: acceptedTemplateSegmentIds.slice(1),
    },
  });
  assert.equal(unacceptedButPresent.statusCode, 422);
  assert.equal(
    unacceptedButPresent.json().error,
    'unaccepted_template_segment_present',
  );

  const transcriptSegment = draft.segments.find(
    (segment: { kind: string; sectionId: string }) => (
      segment.kind === 'transcript_value' && segment.sectionId === 'liver'
    ),
  );
  assert.ok(transcriptSegment);
  const movedTranscriptValue = finalReportWithoutPlaceholders
    .replace(transcriptSegment.text, '')
    .replace(
      'Желчный пузырь:',
      `Желчный пузырь: ${transcriptSegment.text} `,
    );
  const movedValue = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-draft-moved-value-0001',
      baseDraftSha256: draft.sha256,
      finalReport: movedTranscriptValue,
    },
  });
  assert.equal(movedValue.statusCode, 409, movedValue.body);
  assert.equal(
    movedValue.json().error,
    'transcript_template_segment_mismatch',
  );

  const kvrSegment = draft.segments.find(
    (segment: { fieldId?: string }) => segment.fieldId === 'liver.kvr',
  );
  const densitySegment = draft.segments.find(
    (segment: { fieldId?: string }) => segment.fieldId === 'liver.density',
  );
  assert.ok(kvrSegment);
  assert.ok(densitySegment);
  const swappedLiverValues = finalReportWithoutPlaceholders
    .replace(kvrSegment.text, 'SWAPVALUE')
    .replace(densitySegment.text, kvrSegment.text)
    .replace('SWAPVALUE', densitySegment.text);
  const swappedValues = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-draft-swapped-values-001',
      baseDraftSha256: draft.sha256,
      finalReport: swappedLiverValues,
    },
  });
  assert.equal(swappedValues.statusCode, 409, swappedValues.body);
  assert.ok(
    [
      'accepted_template_segment_mismatch',
      'transcript_template_segment_mismatch',
    ].includes(swappedValues.json().error),
    swappedValues.body,
  );

  const unsupportedEdit = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-draft-unsafe-edit-001',
      baseDraftSha256: draft.sha256,
      finalReport: `${finalReportWithoutPlaceholders}\nПочка слева: образование 50 мм.`,
    },
  });
  assert.equal(unsupportedEdit.statusCode, 422);
  assert.equal(
    unsupportedEdit.json().error,
    'safety_approval_blocked',
    unsupportedEdit.body,
  );

  const approved = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-draft-approved-0001',
      baseDraftSha256: draft.sha256,
    },
  });
  assert.equal(approved.statusCode, 201, approved.body);
  const [stored] = await store.listFeedback(sessionId);
  assert.equal(stored.baseDraftSha256, draft.sha256);
  assert.deepEqual(stored.acceptedTemplateSegmentIds, acceptedTemplateSegmentIds);
  assert.deepEqual(stored.reviewedResidualAtomIds, reviewedResidualAtomIds);
  assert.equal(stored.safetyStage.status, 'passed');
});

test('corrected verbatim recomposes an immutable template revision used by approved feedback', async (t) => {
  const rawText = 'печень КВР 105. селезёнка 12 на 6 на 5';
  const correctedText = 'печень КВР 150. селезёнка 12 на 6 на 5';
  const { app, dataDir, store } = await fixture(
    async (audioBase64) => verifiedTranscription(audioBase64, rawText),
    undefined,
    async (templateId, transcript, context) => structureDictation(
      templateId,
      transcript,
      async () => JSON.stringify({ assignments: [] }),
      {
        allowLLM: context?.allowLLM,
        rawTranscript: context?.rawTranscript,
        normalizationAlignment: context?.normalizationAlignment,
      },
    ),
  );
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const sessionId = created.json().sessionId;
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('recompose-kvr').toString('base64'),
      chunk_index: 0,
    },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  assert.equal(finished.statusCode, 200, finished.body);
  const originalArtifact = finished.json().artifact;
  const originalDraft = originalArtifact.report.reviewDraft;
  assert.equal(
    originalDraft.fieldAssignments.find(
      (assignment: { fieldId: string }) => assignment.fieldId === 'liver.kvr',
    ).value,
    105,
  );

  const correction = {
    start: rawText.indexOf('105'),
    end: rawText.indexOf('105') + 3,
    originalText: '105',
    correctedText: '150',
    entityType: 'measurement',
    confidence: 1,
    modality: 'CT',
    author: 'doctor-1',
  };
  const recomposed = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/recompose`,
    payload: {
      verbatimTranscript: correctedText,
      spanCorrections: [correction],
    },
  });
  assert.equal(recomposed.statusCode, 200, recomposed.body);
  const revision = recomposed.json().revision;
  const revisedDraft = revision.report.reviewDraft;
  assert.match(revision.sourceArtifactSha256, /^[a-f0-9]{64}$/u);
  assert.notEqual(revisedDraft.sha256, originalDraft.sha256);
  assert.equal(
    revisedDraft.fieldAssignments.find(
      (assignment: { fieldId: string }) => assignment.fieldId === 'liver.kvr',
    ).value,
    150,
  );

  const persistedAfterRecompose = await store.getArtifact(sessionId);
  assert.equal(persistedAfterRecompose?.rawTranscript.text, rawText);
  assert.equal(persistedAfterRecompose?.report?.reviewDraft?.sha256, originalDraft.sha256);

  const finalReport = [...revisedDraft.segments]
    .filter((segment: { defaultKind?: string }) => segment.defaultKind === 'placeholder')
    .sort((left, right) => right.start - left.start)
    .reduce(
      (text, segment) => `${text.slice(0, segment.start)}${text.slice(segment.end)}`,
      revisedDraft.fullText,
    );
  const acceptedTemplateSegmentIds = revisedDraft.segments
    .filter((segment: { confirmationRequired: boolean; defaultKind?: string }) => (
      segment.confirmationRequired && segment.defaultKind !== 'placeholder'
    ))
    .map((segment: { id: string }) => segment.id);
  const approved = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      idempotencyKey: 'recompose-feedback-approved-0001',
      verbatimTranscript: correctedText,
      finalReport,
      spanCorrections: [correction],
      normalizationResolutions: [],
      baseDraftSha256: revisedDraft.sha256,
      acceptedTemplateSegmentIds,
      reviewedResidualAtomIds: revisedDraft.residualAtomIds,
      approved: true,
      author: 'doctor-1',
    },
  });
  assert.equal(approved.statusCode, 201, approved.body);
  const [storedFeedback] = await store.listFeedback(sessionId);
  assert.equal(storedFeedback.baseDraftSha256, revisedDraft.sha256);
  assert.equal(storedFeedback.recomposeRevision?.report?.reviewDraft?.sha256, revisedDraft.sha256);
  assert.equal(storedFeedback.recomposeRevision?.sourceArtifactSha256, revision.sourceArtifactSha256);
  assert.equal((await store.getArtifact(sessionId))?.rawTranscript.text, rawText);
});

test('approved template feedback requires every composer residual atom to be reviewed', async (t) => {
  const rawText = 'печень КВР 150 дополнительный текст. селезёнка 12 на 6 на 5';
  const { app, dataDir } = await fixture(
    async (audioBase64) => verifiedTranscription(audioBase64, rawText),
    undefined,
    async (templateId, transcript) => structureDictation(
      templateId,
      transcript,
      async () => JSON.stringify({ assignments: [] }),
    ),
  );
  t.after(async () => {
    await app.close();
    await rm(dataDir, { recursive: true, force: true });
  });

  const created = await app.inject({
    method: 'POST',
    url: '/api/sessions',
    payload: {
      mode: 'radiology',
      templateId: 'CT_ABDOMEN_MIKHAILOV',
      source: 'gigaam',
      retainAudio: true,
    },
  });
  const { sessionId } = created.json();
  await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/chunks`,
    payload: {
      audio_base64: Buffer.from('template-review-residual').toString('base64'),
      chunk_index: 0,
    },
  });
  const finished = await app.inject({
    method: 'POST',
    url: `/api/sessions/${sessionId}/finish`,
    payload: {},
  });
  const draft = finished.json().artifact.report.reviewDraft;
  assert.equal(draft.status, 'partial');
  assert.deepEqual(draft.residualAtomIds, ['a0001']);
  const acceptedTemplateSegmentIds = draft.segments
    .filter((segment: { confirmationRequired: boolean; defaultKind?: string }) => (
      segment.confirmationRequired && segment.defaultKind !== 'placeholder'
    ))
    .map((segment: { id: string }) => segment.id);
  const finalReportWithoutPlaceholders = [...draft.segments]
    .filter((segment: { defaultKind?: string }) => segment.defaultKind === 'placeholder')
    .sort((left, right) => right.start - left.start)
    .reduce(
      (text, segment) => `${text.slice(0, segment.start)}${text.slice(segment.end)}`,
      draft.fullText,
    );
  const basePayload = {
    verbatimTranscript: rawText,
    finalReport: finalReportWithoutPlaceholders,
    spanCorrections: [],
    normalizationResolutions: [],
    baseDraftSha256: draft.sha256,
    acceptedTemplateSegmentIds,
    approved: true,
    author: 'doctor-1',
  };

  const foreignResidual = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-residual-foreign-0001',
      reviewedResidualAtomIds: ['foreign-atom'],
    },
  });
  assert.equal(foreignResidual.statusCode, 422);
  assert.equal(foreignResidual.json().error, 'residual_atom_mismatch');

  const missingResidual = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-residual-missing-0001',
      reviewedResidualAtomIds: [],
    },
  });
  assert.equal(missingResidual.statusCode, 422);
  assert.equal(missingResidual.json().error, 'residual_atom_review_required');

  const reviewed = await app.inject({
    method: 'POST',
    url: `/api/radiology/sessions/${sessionId}/feedback`,
    payload: {
      ...basePayload,
      idempotencyKey: 'review-residual-approved-001',
      reviewedResidualAtomIds: ['a0001'],
    },
  });
  assert.equal(reviewed.statusCode, 201, reviewed.body);
});
