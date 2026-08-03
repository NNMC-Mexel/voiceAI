// HTTP-маршруты движка лучевой диагностики (structured reporting).
// Stateless: клиент присылает шаблон + список голосовых команд сессии,
// сервер прогоняет их через свежий движок и возвращает собранный протокол.
// Такой replay-контракт совпадает с потоком чанков whisper: накопили команды —
// переслали — получили актуальный протокол; идемпотентно и переживает reconnect.

import { createHash } from 'node:crypto';
import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { RadiologyEngine, getTemplate, templates } from './radiology/index.js';
import { DocEngine } from './radiology/doc-engine.js';
import { docTemplates, getDocTemplate } from './radiology/doc-registry.js';
import { buildHints } from './radiology/doc-hints.js';
import { structureDictation } from './radiology/dictation.js';
import { ollamaLLM, radiologyLLMIdentity } from './radiology/ollama.js';
import {
  defaultRadiologyDataDir,
  RadiologyArtifactStore,
  RadiologySessionError,
  RadiologySessionService,
} from './radiology-session.js';
import type {
  RadiologyChunkTranscriber,
  RadiologyFeedbackInput,
  RadiologyModelMetadata,
  NormalizationResolutionInput,
  RadiologyRecomposeInput,
  RadiologySessionActor,
  RadiologyTranscriptStructurer,
  RadiologyTranscriptionSource,
  SpanCorrectionInput,
} from './radiology-session.js';

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function sha256Json(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value), 'utf8').digest('hex');
}

const RADIOLOGY_SOURCES = new Set<RadiologyTranscriptionSource>([
  'gigaam',
  'whisper',
  'browser',
  'manual',
  'unknown',
]);

const RADIOLOGY_CHUNK_BODY_LIMIT_BYTES = Math.min(
  64 * 1024 * 1024,
  Math.max(
    1024 * 1024,
    Number.parseInt(process.env.RADIOLOGY_CHUNK_BODY_LIMIT_BYTES || '', 10)
      || 48 * 1024 * 1024,
  ),
);

export interface RadiologyRouteOptions {
  sessionService?: RadiologySessionService;
  dataDir?: string;
  transcribeChunk?: RadiologyChunkTranscriber;
  structureTranscript?: RadiologyTranscriptStructurer;
  model?: Partial<RadiologyModelMetadata>;
  allowAudioPersistence?: boolean;
  allowUnownedSessions?: boolean;
  sessionTtlMs?: number;
  maxChunkBytes?: number;
  maxChunksPerSession?: number;
  maxTotalAudioBytes?: number;
  maxActiveSessions?: number;
  maxActiveAudioBytes?: number;
  maxOwnerActiveSessions?: number;
  maxOwnerActiveAudioBytes?: number;
  maxPendingTranscriptions?: number;
  maxFeedbackRevisionsPerSession?: number;
  storageRetentionMs?: number;
  orphanGraceMs?: number;
  storageCleanupIntervalMs?: number;
  enableLegacyPipelines?: boolean;
}

function boundedEnvInteger(
  name: string,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const parsed = Number.parseInt(process.env[name] || '', 10);
  if (!Number.isSafeInteger(parsed)) return fallback;
  return Math.min(maximum, Math.max(minimum, parsed));
}

function requestActor(request: FastifyRequest): RadiologySessionActor | undefined {
  const holder = request as FastifyRequest & {
    user?: { doctorId?: unknown; role?: unknown };
  };
  if (holder.user === undefined) return undefined;
  const user = holder.user;
  const doctorId = typeof user.doctorId === 'string' || typeof user.doctorId === 'number'
    ? String(user.doctorId)
    : undefined;
  const role = typeof user.role === 'string' ? user.role : undefined;
  return {
    authenticated: true,
    ...(doctorId ? { doctorId } : {}),
    ...(role ? { role } : {}),
  };
}

function requestAuthor(request: FastifyRequest): string | undefined {
  const user = requestActor(request);
  return user?.doctorId;
}

function markPhiResponse(reply: FastifyReply): void {
  reply.header('Cache-Control', 'no-store');
  reply.header('Pragma', 'no-cache');
}

function sessionError(
  request: FastifyRequest,
  reply: FastifyReply,
  error: unknown,
): FastifyReply {
  if (error instanceof RadiologySessionError) {
    return reply.status(error.statusCode).send({ error: error.code, message: error.message });
  }
  request.log.error({ err: error }, 'radiology session request failed');
  return reply.status(500).send({
    error: 'radiology_session_failed',
    message: error instanceof Error ? error.message : 'Unknown error',
  });
}

function requiredString(
  value: unknown,
  field: string,
  options: { allowEmpty?: boolean; maxLength?: number } = {},
): string {
  if (typeof value !== 'string') {
    throw new RadiologySessionError(400, 'invalid_feedback', `${field} must be a string`);
  }
  if (!options.allowEmpty && !value.trim()) {
    throw new RadiologySessionError(400, 'invalid_feedback', `${field} must not be empty`);
  }
  if (value.length > (options.maxLength ?? 1_000_000)) {
    throw new RadiologySessionError(413, 'feedback_too_large', `${field} is too large`);
  }
  return value;
}

function parseSpanCorrection(value: unknown, index: number): SpanCorrectionInput {
  if (!isRecord(value)) {
    throw new RadiologySessionError(400, 'invalid_feedback', `spanCorrections[${index}] must be an object`);
  }
  const start = value.start;
  const end = value.end;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || Number(start) < 0 || Number(end) < Number(start)) {
    throw new RadiologySessionError(
      400,
      'invalid_feedback',
      `spanCorrections[${index}] has invalid start/end`,
    );
  }
  const confidence = value.confidence;
  if (
    confidence !== undefined
    && confidence !== null
    && (typeof confidence !== 'number' || !Number.isFinite(confidence) || confidence < 0 || confidence > 1)
  ) {
    throw new RadiologySessionError(
      400,
      'invalid_feedback',
      `spanCorrections[${index}].confidence must be between 0 and 1`,
    );
  }
  return {
    start: Number(start),
    end: Number(end),
    originalText: requiredString(
      value.originalText,
      `spanCorrections[${index}].originalText`,
      { allowEmpty: true, maxLength: 10_000 },
    ),
    correctedText: requiredString(
      value.correctedText,
      `spanCorrections[${index}].correctedText`,
      { allowEmpty: true, maxLength: 10_000 },
    ),
    entityType: requiredString(value.entityType, `spanCorrections[${index}].entityType`, { maxLength: 100 }),
    confidence: typeof confidence === 'number' ? confidence : null,
    modality: requiredString(value.modality, `spanCorrections[${index}].modality`, { maxLength: 100 }),
    ...(typeof value.author === 'string'
      ? { author: requiredString(value.author, `spanCorrections[${index}].author`, { maxLength: 200 }) }
      : {}),
  };
}

function parseNormalizationResolution(
  value: unknown,
  index: number,
): NormalizationResolutionInput {
  if (!isRecord(value)) {
    throw new RadiologySessionError(
      400,
      'invalid_feedback',
      `normalizationResolutions[${index}] must be an object`,
    );
  }
  const resolution = value.resolution;
  if (
    resolution !== 'confirmed_single'
    && resolution !== 'confirmed_range'
    && resolution !== 'confirmed_verbatim'
  ) {
    throw new RadiologySessionError(
      400,
      'invalid_feedback',
      `normalizationResolutions[${index}].resolution is invalid`,
    );
  }
  return {
    issueId: requiredString(
      value.issueId,
      `normalizationResolutions[${index}].issueId`,
      { maxLength: 200 },
    ),
    replacementText: requiredString(
      value.replacementText,
      `normalizationResolutions[${index}].replacementText`,
      { maxLength: 10_000 },
    ),
    resolution,
  };
}

function parseOptionalUniqueIdArray(
  value: unknown,
  field: string,
): string[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length > 2_000) {
    throw new RadiologySessionError(
      Array.isArray(value) ? 413 : 400,
      'invalid_feedback',
      `${field} must be an array with at most 2000 entries`,
    );
  }
  const ids = value.map((entry, index) => requiredString(
    entry,
    `${field}[${index}]`,
    { maxLength: 200 },
  ));
  if (new Set(ids).size !== ids.length) {
    throw new RadiologySessionError(
      400,
      'invalid_feedback',
      `${field} must not contain duplicate ids`,
    );
  }
  return ids;
}

function parseFeedback(body: unknown): RadiologyFeedbackInput {
  if (!isRecord(body)) {
    throw new RadiologySessionError(400, 'invalid_feedback', 'Request body must be an object');
  }
  if (!Array.isArray(body.spanCorrections) || body.spanCorrections.length > 1_000) {
    throw new RadiologySessionError(
      body.spanCorrections instanceof Array ? 413 : 400,
      'invalid_feedback',
      'spanCorrections must be an array with at most 1000 entries',
    );
  }
  if (typeof body.approved !== 'boolean') {
    throw new RadiologySessionError(400, 'invalid_feedback', 'approved must be a boolean');
  }
  const normalizationResolutions = body.normalizationResolutions ?? [];
  if (
    !Array.isArray(normalizationResolutions)
    || normalizationResolutions.length > 1_000
  ) {
    throw new RadiologySessionError(
      Array.isArray(normalizationResolutions) ? 413 : 400,
      'invalid_feedback',
      'normalizationResolutions must be an array with at most 1000 entries',
    );
  }
  const idempotencyKey = requiredString(
    body.idempotencyKey,
    'idempotencyKey',
    { maxLength: 128 },
  );
  if (
    idempotencyKey.length < 16
    || !/^[A-Za-z0-9._:-]+$/u.test(idempotencyKey)
  ) {
    throw new RadiologySessionError(
      400,
      'invalid_idempotency_key',
      'idempotencyKey must be 16-128 characters using letters, digits, dot, underscore, colon, or hyphen',
    );
  }
  const baseDraftSha256 = body.baseDraftSha256 === undefined
    ? undefined
    : requiredString(body.baseDraftSha256, 'baseDraftSha256', { maxLength: 64 });
  if (
    baseDraftSha256 !== undefined
    && !/^[a-f0-9]{64}$/u.test(baseDraftSha256)
  ) {
    throw new RadiologySessionError(
      400,
      'invalid_feedback',
      'baseDraftSha256 must be a lowercase SHA-256 hex digest',
    );
  }
  const acceptedTemplateSegmentIds = parseOptionalUniqueIdArray(
    body.acceptedTemplateSegmentIds,
    'acceptedTemplateSegmentIds',
  );
  const reviewedResidualAtomIds = parseOptionalUniqueIdArray(
    body.reviewedResidualAtomIds,
    'reviewedResidualAtomIds',
  );
  return {
    idempotencyKey,
    verbatimTranscript: requiredString(
      body.verbatimTranscript,
      'verbatimTranscript',
      { allowEmpty: true },
    ),
    finalReport: requiredString(body.finalReport, 'finalReport', { allowEmpty: true }),
    spanCorrections: body.spanCorrections.map(parseSpanCorrection),
    normalizationResolutions: normalizationResolutions.map(parseNormalizationResolution),
    ...(baseDraftSha256 !== undefined ? { baseDraftSha256 } : {}),
    ...(acceptedTemplateSegmentIds !== undefined ? { acceptedTemplateSegmentIds } : {}),
    ...(reviewedResidualAtomIds !== undefined ? { reviewedResidualAtomIds } : {}),
    approved: body.approved,
    ...(typeof body.author === 'string'
      ? { author: requiredString(body.author, 'author', { maxLength: 200 }) }
      : {}),
  };
}

function parseRecompose(body: unknown): RadiologyRecomposeInput {
  if (!isRecord(body)) {
    throw new RadiologySessionError(400, 'invalid_recompose', 'Request body must be an object');
  }
  if (!Array.isArray(body.spanCorrections) || body.spanCorrections.length > 1_000) {
    throw new RadiologySessionError(
      Array.isArray(body.spanCorrections) ? 413 : 400,
      'invalid_recompose',
      'spanCorrections must be an array with at most 1000 entries',
    );
  }
  return {
    verbatimTranscript: requiredString(
      body.verbatimTranscript,
      'verbatimTranscript',
      { allowEmpty: true },
    ),
    spanCorrections: body.spanCorrections.map(parseSpanCorrection),
  };
}

// Разбить строку диктовки на отдельные команды («жидкости нет. газа нет.»)
function splitCommands(line: string): string[] {
  return line.split(/[.\n;]+/).map((s) => s.trim()).filter(Boolean);
}

export function registerRadiologyRoutes(
  fastify: FastifyInstance,
  options: RadiologyRouteOptions = {},
): void {
  const llmIdentity = radiologyLLMIdentity();
  const configuredLegacyFlag = process.env.RADIOLOGY_ENABLE_LEGACY_PIPELINES;
  const legacyPipelinesEnabled = options.enableLegacyPipelines
    ?? (
      configuredLegacyFlag === undefined
        ? process.env.NODE_ENV !== 'production'
        : configuredLegacyFlag.trim().toLowerCase() === 'true'
    );
  const rejectLegacyPipeline = (reply: FastifyReply): FastifyReply | null => {
    if (legacyPipelinesEnabled) return null;
    markPhiResponse(reply);
    return reply.status(410).send({
      error: 'legacy_radiology_pipeline_disabled',
      message: 'Use canonical POST /api/sessions with mode=radiology',
    });
  };
  const sessionService = options.sessionService ?? new RadiologySessionService({
    store: new RadiologyArtifactStore(
      options.dataDir ?? defaultRadiologyDataDir(),
      {
        maxFeedbackRevisionsPerSession:
          options.maxFeedbackRevisionsPerSession
          ?? boundedEnvInteger('RADIOLOGY_MAX_FEEDBACK_REVISIONS', 50, 1, 1_000),
        storageRetentionMs:
          options.storageRetentionMs
          ?? boundedEnvInteger(
            'RADIOLOGY_STORAGE_RETENTION_DAYS',
            30,
            1,
            3_650,
          ) * 24 * 60 * 60 * 1000,
        orphanGraceMs:
          options.orphanGraceMs
          ?? boundedEnvInteger(
            'RADIOLOGY_ORPHAN_GRACE_HOURS',
            24,
            1,
            720,
          ) * 60 * 60 * 1000,
        cleanupIntervalMs:
          options.storageCleanupIntervalMs
          ?? boundedEnvInteger(
            'RADIOLOGY_STORAGE_CLEANUP_INTERVAL_MS',
            10 * 60 * 1000,
            60_000,
            24 * 60 * 60 * 1000,
          ),
      },
    ),
    transcribeChunk: options.transcribeChunk,
    structureTranscript: options.structureTranscript
      ?? ((templateId, transcript, context) => structureDictation(
        templateId,
        transcript,
        ollamaLLM(),
        {
          allowLLM: context?.allowLLM,
          rawTranscript: context?.rawTranscript,
          normalizationAlignment: context?.normalizationAlignment,
        },
      )),
    model: {
      llm: {
        name: `${llmIdentity.provider}:${llmIdentity.model}`,
        version: `temperature=0;seed=${llmIdentity.seed}`,
        ...(llmIdentity.modelChecksum ? { checksum: llmIdentity.modelChecksum } : {}),
        configSha256: sha256Json(llmIdentity),
      },
      ...options.model,
    },
    sessionTtlMs:
      options.sessionTtlMs
      ?? boundedEnvInteger('RADIOLOGY_SESSION_TTL_MS', 30 * 60 * 1000, 60_000, 24 * 60 * 60 * 1000),
    maxChunkBytes:
      options.maxChunkBytes
      ?? boundedEnvInteger('RADIOLOGY_MAX_CHUNK_BYTES', 32 * 1024 * 1024, 1024, 32 * 1024 * 1024),
    maxChunksPerSession:
      options.maxChunksPerSession
      ?? boundedEnvInteger('RADIOLOGY_MAX_CHUNKS_PER_SESSION', 64, 1, 1_024),
    maxTotalAudioBytes:
      options.maxTotalAudioBytes
      ?? boundedEnvInteger('RADIOLOGY_MAX_SESSION_AUDIO_BYTES', 64 * 1024 * 1024, 1024, 512 * 1024 * 1024),
    maxActiveSessions:
      options.maxActiveSessions
      ?? boundedEnvInteger('RADIOLOGY_MAX_ACTIVE_SESSIONS', 32, 1, 1_000),
    maxActiveAudioBytes:
      options.maxActiveAudioBytes
      ?? boundedEnvInteger('RADIOLOGY_MAX_ACTIVE_AUDIO_BYTES', 256 * 1024 * 1024, 1024, 4 * 1024 * 1024 * 1024),
    maxOwnerActiveSessions:
      options.maxOwnerActiveSessions
      ?? boundedEnvInteger('RADIOLOGY_MAX_OWNER_ACTIVE_SESSIONS', 4, 1, 100),
    maxOwnerActiveAudioBytes:
      options.maxOwnerActiveAudioBytes
      ?? boundedEnvInteger('RADIOLOGY_MAX_OWNER_ACTIVE_AUDIO_BYTES', 64 * 1024 * 1024, 1024, 1024 * 1024 * 1024),
    maxPendingTranscriptions:
      options.maxPendingTranscriptions
      ?? boundedEnvInteger('RADIOLOGY_MAX_PENDING_TRANSCRIPTIONS', 8, 1, 128),
    allowUnownedSessions: options.allowUnownedSessions,
    allowAudioPersistence: options.allowAudioPersistence,
  });

  // Canonical, versioned radiology sessions. The existing singular
  // /api/session/* streaming endpoints remain untouched for older clients.
  fastify.post('/api/sessions', async (request: FastifyRequest, reply: FastifyReply) => {
    markPhiResponse(reply);
    try {
      const body = request.body;
      if (!isRecord(body)) {
        throw new RadiologySessionError(400, 'invalid_request', 'Request body must be an object');
      }
      if (body.mode !== 'radiology') {
        throw new RadiologySessionError(400, 'unsupported_mode', 'mode must be radiology');
      }
      const templateId = typeof body.templateId === 'string' ? body.templateId.trim() : '';
      if (!templateId) {
        throw new RadiologySessionError(400, 'template_required', 'templateId is required');
      }
      if (!getDocTemplate(templateId)) {
        throw new RadiologySessionError(404, 'template_not_found', `Unknown template: ${templateId}`);
      }
      const rawSource = typeof body.source === 'string' ? body.source : 'unknown';
      if (!RADIOLOGY_SOURCES.has(rawSource as RadiologyTranscriptionSource)) {
        throw new RadiologySessionError(400, 'invalid_source', 'Unsupported transcription source');
      }
      const mimeType = typeof body.mimeType === 'string' ? body.mimeType.trim().slice(0, 200) : undefined;
      if (body.retainAudio !== undefined && typeof body.retainAudio !== 'boolean') {
        throw new RadiologySessionError(400, 'invalid_retain_audio', 'retainAudio must be a boolean');
      }
      return reply.status(201).send(sessionService.create({
        templateId,
        source: rawSource as RadiologyTranscriptionSource,
        ...(mimeType ? { mimeType } : {}),
        retainAudio: body.retainAudio === true,
      }, requestActor(request)));
    } catch (error) {
      return sessionError(request, reply, error);
    }
  });

  fastify.post(
    '/api/sessions/:id/chunks',
    { bodyLimit: RADIOLOGY_CHUNK_BODY_LIMIT_BYTES },
    async (request: FastifyRequest, reply: FastifyReply) => {
    markPhiResponse(reply);
    try {
      const { id } = request.params as { id: string };
      const body = request.body;
      if (!isRecord(body) || typeof body.audio_base64 !== 'string') {
        throw new RadiologySessionError(400, 'audio_required', 'audio_base64 is required');
      }
      const chunkIndex = body.chunk_index === undefined ? undefined : body.chunk_index;
      if (chunkIndex !== undefined && typeof chunkIndex !== 'number') {
        throw new RadiologySessionError(400, 'invalid_chunk_index', 'chunk_index must be a number');
      }
      if (body.mime_type !== undefined && typeof body.mime_type !== 'string') {
        throw new RadiologySessionError(400, 'invalid_mime_type', 'mime_type must be a string');
      }
      return await sessionService.addChunk(id, {
        audioBase64: body.audio_base64,
        ...(typeof chunkIndex === 'number' ? { chunkIndex } : {}),
        ...(typeof body.mime_type === 'string' ? { mimeType: body.mime_type } : {}),
      }, requestActor(request));
    } catch (error) {
      return sessionError(request, reply, error);
    }
    },
  );

  fastify.post('/api/sessions/:id/finish', async (request: FastifyRequest, reply: FastifyReply) => {
    markPhiResponse(reply);
    try {
      const { id } = request.params as { id: string };
      const body = request.body;
      if (body !== undefined && body !== null && !isRecord(body)) {
        throw new RadiologySessionError(400, 'invalid_request', 'Request body must be an object');
      }
      const browserTranscript = isRecord(body) && typeof body.browserTranscript === 'string'
        ? body.browserTranscript
        : undefined;
      const artifact = await sessionService.finish(id, {
        ...(browserTranscript !== undefined ? { browserTranscript } : {}),
      }, requestActor(request));
      return { success: true, artifact };
    } catch (error) {
      return sessionError(request, reply, error);
    }
  });

  fastify.get('/api/radiology/sessions/:id/artifact', async (request: FastifyRequest, reply: FastifyReply) => {
    markPhiResponse(reply);
    try {
      const { id } = request.params as { id: string };
      const artifact = await sessionService.getArtifact(id, requestActor(request));
      if (!artifact) {
        throw new RadiologySessionError(404, 'artifact_not_found', 'Radiology session artifact not found');
      }
      return { artifact };
    } catch (error) {
      return sessionError(request, reply, error);
    }
  });

  fastify.get('/api/radiology/sessions/:id/approved-report', async (request: FastifyRequest, reply: FastifyReply) => {
    markPhiResponse(reply);
    try {
      const { id } = request.params as { id: string };
      const approvedReport = await sessionService.getApprovedReport(
        id,
        requestActor(request),
      );
      return { approvedReport };
    } catch (error) {
      return sessionError(request, reply, error);
    }
  });

  fastify.post('/api/radiology/sessions/:id/recompose', async (request: FastifyRequest, reply: FastifyReply) => {
    markPhiResponse(reply);
    try {
      const { id } = request.params as { id: string };
      const revision = await sessionService.recompose(
        id,
        parseRecompose(request.body),
        requestActor(request),
      );
      return { success: true, revision };
    } catch (error) {
      return sessionError(request, reply, error);
    }
  });

  fastify.post('/api/radiology/sessions/:id/feedback', async (request: FastifyRequest, reply: FastifyReply) => {
    markPhiResponse(reply);
    try {
      const { id } = request.params as { id: string };
      const input = parseFeedback(request.body);
      const result = await sessionService.saveFeedback(
        id,
        input,
        requestAuthor(request),
        requestActor(request),
      );
      const feedback = result.feedback;
      return reply.status(result.idempotentReplay ? 200 : 201).send({
        success: true,
        idempotentReplay: result.idempotentReplay,
        feedbackId: feedback.feedbackId,
        revision: feedback.revision,
        datasetVersion: feedback.datasetVersion,
        training: feedback.training,
      });
    } catch (error) {
      return sessionError(request, reply, error);
    }
  });

  // ─── Fill-in движок (рабочие шаблоны Михайлова) ───────────────────────────
  fastify.get('/api/radiology/doc-templates', async () => ({
    templates: docTemplates.map((t) => ({ id: t.id, name: t.name, modality: t.modality, title: t.title })),
  }));

  // Template defaults are exposed only as a read-only preview. They are UI
  // guidance and never become transcript evidence through this endpoint.
  fastify.get('/api/radiology/doc-templates/:id/preview', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const tpl = getDocTemplate(id);
    if (!tpl) return reply.status(404).send({ error: 'Шаблон не найден' });
    const report = new DocEngine(tpl).build();
    return {
      preview: {
        templateId: tpl.id,
        title: report.title,
        blocks: report.blocks.map((block) => ({
          ...block,
          origin: 'template_default' as const,
        })),
        text: report.text,
      },
    };
  });

  // Подсказки «что можно диктовать» по шаблону (примеры команд из конфига).
  fastify.get('/api/radiology/doc-templates/:id/hints', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const tpl = getDocTemplate(id);
    if (!tpl) return reply.status(404).send({ error: 'Шаблон не найден' });
    return { hints: buildHints(tpl) };
  });

  fastify.post('/api/radiology/doc-build', async (request: FastifyRequest, reply: FastifyReply) => {
    const rejected = rejectLegacyPipeline(reply);
    if (rejected) return rejected;
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Некорректное тело запроса' });
    const templateId = typeof body.templateId === 'string' ? body.templateId : '';
    const commands = Array.isArray(body.commands)
      ? body.commands.filter((c): c is string => typeof c === 'string') : [];

    const tpl = getDocTemplate(templateId);
    if (!tpl) return reply.status(404).send({ error: `Неизвестный шаблон: ${templateId}` });

    const engine = new DocEngine(tpl);
    const applied: { command: string; ok: boolean; action: string; blockId?: string; detail?: string }[] = [];
    for (const line of commands) {
      for (const cmd of splitCommands(line)) {
        const r = engine.apply(cmd);
        applied.push({ command: cmd, ok: r.ok, action: r.action, blockId: r.blockId, detail: r.detail });
      }
    }
    return { report: engine.build(), applied };
  });

  // Список доступных шаблонов лучевой диагностики (для экрана выбора).
  fastify.get('/api/radiology/templates', async () => ({
    templates: templates.map((t) => ({
      id: t.id, name: t.name, modality: t.modality, aliases: t.aliases,
    })),
  }));

  // Структура секций шаблона (для превью/каркаса фронта).
  fastify.get('/api/radiology/templates/:id', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const tpl = getTemplate(id);
    if (!tpl) return reply.status(404).send({ error: 'Шаблон не найден' });
    return {
      id: tpl.id, name: tpl.name, modality: tpl.modality,
      sections: tpl.sections.map((s) => ({ id: s.id, organ: s.organ })),
    };
  });

  // Прогнать команды сессии и собрать протокол.
  fastify.post('/api/radiology/build', async (request: FastifyRequest, reply: FastifyReply) => {
    const rejected = rejectLegacyPipeline(reply);
    if (rejected) return rejected;
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Некорректное тело запроса' });

    const templateId = typeof body.templateId === 'string' ? body.templateId : '';
    const commands = Array.isArray(body.commands)
      ? body.commands.filter((c): c is string => typeof c === 'string')
      : [];

    const tpl = getTemplate(templateId);
    if (!tpl) return reply.status(404).send({ error: `Неизвестный шаблон: ${templateId}` });

    const engine = new RadiologyEngine(tpl);
    // Разбиваем каждую строку на отдельные фразы-команды (по точкам/переносам),
    // т.к. врач диктует потоком: «жидкости нет. газа нет.»
    const applied: { command: string; ok: boolean; handled: string; warnings?: string[] }[] = [];
    for (const line of commands) {
      for (const cmd of line.split(/[.\n;]+/).map((s) => s.trim()).filter(Boolean)) {
        const r = engine.apply(cmd);
        applied.push({ command: cmd, ok: r.ok, handled: r.handled, warnings: r.warnings });
      }
    }

    const report = engine.build();
    return { report, applied };
  });

  // Свободная диктовка → LLM-укладчик (локальный Ollama) → документ + сверка чисел.
  fastify.post('/api/radiology/structure', async (request: FastifyRequest, reply: FastifyReply) => {
    const rejected = rejectLegacyPipeline(reply);
    if (rejected) return rejected;
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Некорректное тело запроса' });
    const templateId = typeof body.templateId === 'string' ? body.templateId : '';
    const transcript = typeof body.transcript === 'string' ? body.transcript.trim() : '';
    if (!getDocTemplate(templateId)) return reply.status(404).send({ error: `Неизвестный шаблон: ${templateId}` });
    if (!transcript) return reply.status(400).send({ error: 'Пустой транскрипт' });

    try {
      const report = await structureDictation(templateId, transcript, ollamaLLM());
      return { report };
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Ошибка структурирования';
      request.log.error({ err }, 'radiology structure failed');
      // Недоступность локального LLM — 503, чтобы фронт показал «сервис недоступен»
      return reply.status(/ollama|fetch|abort|ECONN|timeout/i.test(msg) ? 503 : 500).send({ error: msg });
    }
  });
}
