import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { pipeline } from 'stream/promises';
import { createWriteStream } from 'fs';
import { mkdir, unlink } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { WhisperService } from './services/whisper.js';
import { LLMService } from './services/llm.js';
import { TtsService } from './services/tts.js';
import {
  getUserCorrections,
  addUserCorrection,
  deleteUserCorrection,
} from './services/medical-dictionary.js';
import type { ServerConfig, MedicalDocument } from './types.js';

interface RateState {
  count: number;
  windowStartedAt: number;
}

const rewriteableFields = [
  'complaints',
  'anamnesis',
  'clinicalCourse',
  'allergyHistory',
  'objectiveStatus',
  'neurologicalStatus',
  'diagnosis',
  'conclusion',
  'recommendations',
  'doctorNotes',
] as const;

type RewriteableField = (typeof rewriteableFields)[number];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

export function toSafeUploadFilename(originalName: string): string {
  const base = path.basename(originalName);
  const sanitized = base.replace(/[^A-Za-z0-9._-]/g, '_');
  return sanitized || 'audio.webm';
}

export function resolveUploadPath(uploadDir: string, filename: string): string {
  const uploadRoot = path.resolve(uploadDir);
  const candidate = path.resolve(uploadRoot, filename);
  const normalizedRoot = uploadRoot.endsWith(path.sep) ? uploadRoot : `${uploadRoot}${path.sep}`;

  if (!candidate.startsWith(normalizedRoot)) {
    throw new Error('Invalid file path');
  }

  return candidate;
}

export function isValidMedicalDocument(doc: unknown): doc is MedicalDocument {
  if (!isRecord(doc)) return false;
  if (!isRecord(doc.patient)) return false;

  const patient = doc.patient as Record<string, unknown>;
  const patientKeys = ['fullName', 'age', 'gender', 'complaintDate'];
  for (const key of patientKeys) {
    if (typeof patient[key] !== 'string') return false;
  }

  const textKeys = [
    'complaints',
    'anamnesis',
    'clinicalCourse',
    'allergyHistory',
    'objectiveStatus',
    'neurologicalStatus',
    'diagnosis',
    'conclusion',
    'recommendations',
    'doctorNotes',
  ];

  for (const key of textKeys) {
    if (typeof doc[key] !== 'string') return false;
  }

  return true;
}

export async function registerRoutes(
  fastify: FastifyInstance,
  config: ServerConfig,
  whisperService: WhisperService,
  llmService: LLMService,
  ttsService: TtsService
): Promise<void> {
  if (!existsSync(config.uploadDir)) {
    await mkdir(config.uploadDir, { recursive: true });
  }

  const rateMap = new Map<string, RateState>();
  const authTokens = new Set<string>();

  fastify.addHook('onRequest', async (request, reply) => {
    const url = request.url;
    if (url.startsWith('/api/health')) return;
    if (url.startsWith('/api/auth/')) return;

    // Auth by password (session token)
    if (config.security.authPassword) {
      const auth = request.headers.authorization;
      const token = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';

      if (!token || !authTokens.has(token)) {
        return reply.status(401).send({ error: 'Unauthorized' });
      }
    }

    // Legacy API key auth
    if (config.security.apiKey) {
      const rawApiKey = request.headers['x-api-key'];
      const headerApiKey = typeof rawApiKey === 'string' ? rawApiKey : '';
      const auth = request.headers.authorization;
      const bearerApiKey = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
      const provided = headerApiKey || bearerApiKey;

      if (!provided || provided !== config.security.apiKey) {
        return reply.status(401).send({ error: 'Unauthorized' });
      }
    }

    const now = Date.now();
    const key = request.ip || 'unknown';
    const state = rateMap.get(key);

    if (!state || now - state.windowStartedAt >= config.security.rateLimitWindowMs) {
      rateMap.set(key, { count: 1, windowStartedAt: now });
      return;
    }

    state.count += 1;
    if (state.count > config.security.rateLimitMaxRequests) {
      return reply.status(429).send({ error: 'Too many requests' });
    }
  });

  // --- Auth endpoint ---
  fastify.post('/api/auth/login', async (request, reply) => {
    if (!config.security.authPassword) {
      return { success: true, token: 'no-auth' };
    }

    const body = request.body as Record<string, unknown> | null;
    const password = typeof body?.password === 'string' ? body.password : '';

    if (password !== config.security.authPassword) {
      return reply.status(401).send({ error: 'Неверный пароль' });
    }

    const token = randomUUID();
    authTokens.add(token);
    return { success: true, token };
  });

  fastify.post('/api/auth/logout', async (request) => {
    const auth = request.headers.authorization;
    const token = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
    if (token) authTokens.delete(token);
    return { success: true };
  });

  fastify.get('/api/auth/check', async (request, reply) => {
    if (!config.security.authPassword) {
      return { authenticated: true };
    }
    const auth = request.headers.authorization;
    const token = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
    if (!token || !authTokens.has(token)) {
      return reply.status(401).send({ authenticated: false });
    }
    return { authenticated: true };
  });

  fastify.get('/api/health', async () => {
    const [llmReady, whisperReady] = await Promise.all([
      llmService.healthCheck(),
      whisperService.healthCheck(),
    ]);

    return {
      status: llmReady && whisperReady ? 'ok' : 'degraded',
      timestamp: new Date().toISOString(),
      services: {
        whisper: whisperReady ? 'ready' : 'unavailable',
        llm: llmReady ? 'ready' : 'unavailable',
      },
    };
  });

  fastify.post('/api/upload', async (request: FastifyRequest, reply: FastifyReply) => {
    const data = await request.file();

    if (!data) {
      return reply.status(400).send({ error: 'No file uploaded' });
    }

    const sourceName = toSafeUploadFilename(data.filename);
    const filename = `${Date.now()}_${sourceName}`;
    const filepath = resolveUploadPath(config.uploadDir, filename);

    await pipeline(data.file, createWriteStream(filepath));

    return {
      success: true,
      filename,
      mimetype: data.mimetype,
    };
  });

  fastify.post('/api/transcribe', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    const filename = isRecord(body) && typeof body.filename === 'string' ? body.filename : undefined;

    if (!filename) {
      return reply.status(400).send({ error: 'Filename is required' });
    }

    if (filename !== path.basename(filename)) {
      return reply.status(400).send({ error: 'Invalid filename' });
    }

    const filepath = resolveUploadPath(config.uploadDir, filename);

    if (!existsSync(filepath)) {
      return reply.status(404).send({ error: 'File not found' });
    }

    try {
      const result = await whisperService.transcribeFile(filepath, filename);

      try {
        await unlink(filepath);
      } catch {
        // Ignore cleanup errors
      }

      return {
        success: true,
        ...result,
      };
    } catch (error) {
      console.error('Transcription error:', error);
      return reply.status(500).send({
        error: 'Transcription failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/structure', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    const text = isRecord(body) && typeof body.text === 'string' ? body.text.trim() : '';

    if (!text) {
      return reply.status(400).send({ error: 'Text is required' });
    }

    try {
      const result = await llmService.structureText(text);
      return {
        success: true,
        ...result,
      };
    } catch (error) {
      console.error('Structuring error:', error);
      return reply.status(500).send({
        error: 'Text structuring failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/rewrite-field', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const fieldRaw = typeof body.field === 'string' ? body.field : '';
    const text = typeof body.text === 'string' ? body.text.trim() : '';
    const field = rewriteableFields.find((x) => x === fieldRaw) as RewriteableField | undefined;

    if (!field) {
      return reply.status(400).send({ error: 'Valid field is required' });
    }

    if (!text) {
      return reply.status(400).send({ error: 'Text is required' });
    }

    try {
      const rewrittenText = await llmService.rewriteDocumentField(field, text);
      return {
        success: true,
        field,
        text: rewrittenText || text,
      };
    } catch (error) {
      console.error('Rewrite field error:', error);
      return reply.status(500).send({
        error: 'Field rewrite failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/recommendations', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    const docCandidate = isRecord(body) ? body.document : undefined;

    if (!isValidMedicalDocument(docCandidate)) {
      return reply.status(400).send({ error: 'Valid document is required' });
    }

    try {
      const recommendations = await llmService.generateRecommendations(docCandidate);
      return { success: true, recommendations };
    } catch (error) {
      console.error('Recommendations error:', error);
      return reply.status(500).send({
        error: 'Recommendations failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/chat', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const question = typeof body.question === 'string' ? body.question.trim() : '';
    const historyRaw = Array.isArray(body.history) ? body.history : [];
    const history = historyRaw
      .filter((m) => isRecord(m) && (m.role === 'user' || m.role === 'assistant') && typeof m.text === 'string')
      .map((m) => ({ role: m.role as 'user' | 'assistant', text: String(m.text) }))
      .slice(-12);

    const document = isValidMedicalDocument(body.document) ? body.document : undefined;

    if (!question) {
      return reply.status(400).send({ error: 'Question is required' });
    }

    try {
      const answer = await llmService.chat(question, history, document);
      return { success: true, answer };
    } catch (error) {
      console.error('Chat error:', error);
      return reply.status(500).send({
        error: 'Chat failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/augment', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const text = typeof body.text === 'string' ? body.text.trim() : '';
    const document = isValidMedicalDocument(body.document) ? body.document : undefined;

    if (!document) {
      return reply.status(400).send({ error: 'Valid document is required' });
    }

    if (!text) {
      return reply.status(400).send({ error: 'Text is required' });
    }

    try {
      const updated = await llmService.applyAddendum(document, text);

      return {
        success: true,
        document: updated,
      };
    } catch (error) {
      console.error('Augment error:', error);
      return reply.status(500).send({
        error: 'Document augmentation failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/instruct', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const instruction = typeof body.instruction === 'string' ? body.instruction.trim() : '';
    const document = isValidMedicalDocument(body.document) ? body.document : undefined;

    if (!document) {
      return reply.status(400).send({ error: 'Valid document is required' });
    }

    if (!instruction) {
      return reply.status(400).send({ error: 'Instruction is required' });
    }

    try {
      const updated = await llmService.applyInstruction(document, instruction);
      const changedFields = rewriteableFields.filter((key) => updated[key] !== document[key]);
      const patientChanged =
        updated.patient.fullName !== document.patient.fullName ||
        updated.patient.age !== document.patient.age ||
        updated.patient.gender !== document.patient.gender ||
        updated.patient.complaintDate !== document.patient.complaintDate;

      return {
        success: true,
        document: updated,
        changedFields,
        patientChanged,
      };
    } catch (error) {
      console.error('Instruction apply error:', error);
      return reply.status(500).send({
        error: 'Instruction apply failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/process', async (request: FastifyRequest, reply: FastifyReply) => {
    const data = await request.file();

    if (!data) {
      return reply.status(400).send({ error: 'No file uploaded' });
    }

    const sourceName = toSafeUploadFilename(data.filename);
    const filename = `${Date.now()}_${sourceName}`;
    const filepath = resolveUploadPath(config.uploadDir, filename);

    try {
      const t0 = Date.now();
      await pipeline(data.file, createWriteStream(filepath));
      const t1 = Date.now();

      const transcription = await whisperService.transcribeFile(filepath, filename);
      const t2 = Date.now();

      const structured = await llmService.structureText(transcription.text);
      const t3 = Date.now();

      fastify.log.info(
        {
          timings_ms: {
            save_file: t1 - t0,
            whisper: t2 - t1,
            llm: t3 - t2,
            total: t3 - t0,
          },
        },
        'process timings'
      );

      try {
        await unlink(filepath);
      } catch {
        // Ignore cleanup errors
      }

      return {
        success: true,
        transcription: {
          text: transcription.text,
          duration: transcription.duration,
          language: transcription.language,
        },
        document: structured.document,
        processingTime: transcription.duration + structured.processingTime,
      };
    } catch (error) {
      console.error('Processing error:', error);

      try {
        await unlink(filepath);
      } catch {
        // Ignore cleanup errors
      }

      return reply.status(500).send({
        error: 'Processing failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/documents', async (request: FastifyRequest, reply: FastifyReply) => {
    const document = request.body;

    if (!isValidMedicalDocument(document)) {
      return reply.status(400).send({ error: 'Valid document is required' });
    }

    return {
      success: true,
      document,
      id: `doc_${Date.now()}`,
      savedAt: new Date().toISOString(),
    };
  });

  fastify.get('/api/config', async () => {
    return {
      maxRecordingDuration: 30 * 60,
      supportedAudioFormats: ['audio/webm', 'audio/wav', 'audio/mp3', 'audio/ogg'],
      language: config.whisper.language,
      llmModel: config.llm.model,
      ttsEnabled: ttsService.isEnabled,
    };
  });

  fastify.post('/api/tts', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    const text = isRecord(body) && typeof body.text === 'string' ? body.text.trim() : '';

    if (!text) {
      return reply.status(400).send({ error: 'text is required' });
    }

    if (!ttsService.isEnabled) {
      return reply.status(503).send({ error: 'TTS is not enabled on this server' });
    }

    try {
      const audioBase64 = await ttsService.synthesize(text);
      return { success: true, audio_base64: audioBase64, format: 'wav' };
    } catch (error) {
      console.error('TTS error:', error);
      return reply.status(500).send({
        error: 'TTS synthesis failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  // ─── Corrections API (пользовательские замены медицинского словаря) ────────

  fastify.get('/api/corrections', async () => {
    const corrections = getUserCorrections();
    return { corrections, total: corrections.length };
  });

  fastify.post('/api/corrections', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const wrong = typeof body.wrong === 'string' ? body.wrong.trim() : '';
    const correct = typeof body.correct === 'string' ? body.correct.trim() : '';

    if (!wrong || !correct) {
      return reply.status(400).send({ error: "Поля 'wrong' и 'correct' обязательны" });
    }

    if (wrong === correct) {
      return reply.status(400).send({ error: 'Значения должны отличаться' });
    }

    try {
      const correction = await addUserCorrection(wrong, correct);
      const all = getUserCorrections();
      return { success: true, id: correction.id, totalCorrections: all.length };
    } catch (error) {
      console.error('Add correction error:', error);
      return reply.status(500).send({
        error: 'Failed to add correction',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.delete('/api/corrections/:id', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };

    if (!id) {
      return reply.status(400).send({ error: 'ID is required' });
    }

    const deleted = await deleteUserCorrection(id);
    if (!deleted) {
      return reply.status(404).send({ error: 'Замена не найдена' });
    }

    return { success: true };
  });
}

