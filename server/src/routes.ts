import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { pipeline } from 'stream/promises';
import { createWriteStream } from 'fs';
import { mkdir, unlink, appendFile } from 'fs/promises';
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
import type { UserCorrectionScope } from './services/medical-dictionary.js';
import type { ServerConfig, MedicalDocument, QualityWarning } from './types.js';

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
  'finalDiagnosis',
  'conclusion',
  'recommendations',
  'doctorNotes',
  'outpatientExams',
] as const;

type RewriteableField = (typeof rewriteableFields)[number];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function normalizeCorrectionScope(scope: string | undefined): UserCorrectionScope {
  return scope === 'medications' || scope === 'exams' || scope === 'neurological'
    ? scope
    : 'global';
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

// Заголовки разделов, которые LLM/Whisper иногда оставляют как «контент» поля
// (например `complaints = "Жалобы"`). Такие фрагменты должны считаться пустыми
// при оценке полезности документа: иначе фронт получает success:true с
// псевдо-заполненным протоколом.
const SECTION_HEADER_RE =
  /^(?:жалобы|анамнез(?:\s+(?:заболевани[а-яёА-ЯЁ\w]*|жизни))?|объективн[а-яёА-ЯЁ\w]*\s+статус|объективно|перенесённые\s+заболевани[а-яёА-ЯЁ\w]*|аллерголог[а-яёА-ЯЁ\w]*\s+анамнез|неврологическ[а-яёА-ЯЁ\w]*\s+статус|диагноз(?:\s+(?:предварительн[а-яёА-ЯЁ\w]*|заключительн[а-яёА-ЯЁ\w]*|основн[а-яёА-ЯЁ\w]*))?|план(?:\s+(?:обследовани[а-яёА-ЯЁ\w]*|лечени[а-яёА-ЯЁ\w]*))?|рекомендации|рекомендация|заключение|сопутствующ[а-яёА-ЯЁ\w]*\s+диагноз|амбулаторн[а-яёА-ЯЁ\w]*\s+терапи[а-яёА-ЯЁ\w]*|анамнез|данные|статус)\.?$/iu;

function stripSectionHeaders(s: string): string {
  if (!s || !s.trim()) return '';
  return s
    .split(/[.!?\n]+/u)
    .map((f) => f.trim())
    .filter((f) => f && !SECTION_HEADER_RE.test(f))
    .join('. ')
    .trim();
}

// Поле считается «содержательным», если после удаления одиночных заголовков
// остаётся хотя бы 10 символов И ≥2 слов (≥3 букв каждое). Это режет случаи
// «Диагноз: АГ» (длина после strip 4) и «Жалобы» (после strip 0).
export function isFieldMeaningful(s: string): boolean {
  const stripped = stripSectionHeaders(s);
  if (stripped.length < 10) return false;
  const words = stripped.match(/[а-яёА-ЯЁa-zA-Z]{3,}/gu) || [];
  return words.length >= 2;
}

export type DocumentUsefulness = {
  status: 'ok' | 'labs_only' | 'empty';
  emptyFields: string[];
  placeholderFields: string[];
  meaningfulFields: string[];
  reason?: string;
};

const NON_EXAMS_CLINICAL_FIELDS = [
  'complaints',
  'anamnesis',
  'clinicalCourse',
  'allergyHistory',
  'objectiveStatus',
  'neurologicalStatus',
  'diagnosis',
  'finalDiagnosis',
  'conclusion',
  'doctorNotes',
  'recommendations',
] as const;

const SUSPICIOUS_UNIT_GARBAGE_RE =
  /(?:СЛЧЭЛЬ|СЛЦАЛЬ|ЛЩЕ|ММС\s+ЛЩЕ|ГЭСЛЧЭЛЬ|ГЕСЛШЕЛЬ|КОЭСЛЧ|МОЛЬСЛЧ|МГСЛЧЭЛЬ)/iu;

function documentClinicalText(doc: MedicalDocument): string {
  return [
    doc.patient.fullName,
    doc.patient.age,
    doc.patient.gender,
    doc.patient.complaintDate,
    doc.complaints,
    doc.anamnesis,
    doc.outpatientExams,
    doc.clinicalCourse,
    doc.allergyHistory,
    doc.objectiveStatus,
    doc.neurologicalStatus,
    doc.diagnosis,
    doc.finalDiagnosis,
    doc.conclusion,
    doc.doctorNotes,
    doc.recommendations,
    doc.manualCheck || '',
  ].join('\n');
}

export function collectDocumentQualityWarnings(
  rawText: string,
  doc: MedicalDocument,
  usefulness: DocumentUsefulness,
): string[] {
  return [...new Set(collectDocumentQualityWarningDetails(rawText, doc, usefulness)
    .map((warning) => warning.code === 'important_number_missing'
      ? `important_number_missing:${warning.evidence || 'value'}`
      : warning.code))];
}

export function collectDocumentQualityWarningDetails(
  rawText: string,
  doc: MedicalDocument,
  usefulness: DocumentUsefulness,
): QualityWarning[] {
  const warnings: QualityWarning[] = [];
  const seen = new Set<string>();
  const add = (warning: QualityWarning) => {
    const key = `${warning.code}:${warning.field || ''}:${warning.evidence || ''}`;
    if (seen.has(key)) return;
    seen.add(key);
    warnings.push(warning);
  };
  const docText = documentClinicalText(doc);

  if (usefulness.status === 'labs_only') {
    add({
      code: 'document_labs_only',
      severity: 'warning',
      field: 'outpatientExams',
      message: 'В документе распознаны в основном обследования, клинические разделы почти пустые.',
    });
  }

  if (usefulness.status === 'ok') {
    const meaningfulNonExams = usefulness.meaningfulFields.filter((f) => f !== 'outpatientExams');
    if (meaningfulNonExams.length < 3) {
      add({
        code: 'suspiciously_few_clinical_fields',
        severity: 'warning',
        field: 'document',
        message: 'Заполнено мало клинических разделов; проверьте, не потерялись ли жалобы, анамнез или диагноз.',
      });
    }
  }

  if (SUSPICIOUS_UNIT_GARBAGE_RE.test(docText)) {
    add({
      code: 'suspicious_unit_garbage_in_document',
      severity: 'warning',
      field: 'document',
      message: 'В документе остались подозрительные единицы измерения после распознавания.',
    });
  }

  const rawBpValues = rawText.match(/\b\d{2,3}\s*\/\s*\d{2,3}\b/gu) || [];
  for (const value of rawBpValues) {
    const normalized = value.replace(/\s+/g, '');
    if (!docText.replace(/\s+/g, '').includes(normalized)) {
      add({
        code: 'important_number_missing',
        severity: 'critical',
        field: 'document',
        message: `В raw есть значение АД ${normalized}, но оно не найдено в итоговых полях.`,
        evidence: `bp_${normalized}`,
      });
    }
  }

  collectAdvancedQualityWarnings(rawText, doc, add);

  return warnings;
}

function normalizeForCoverage(value: string): string {
  return value
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^a-zа-я0-9\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function collectAdvancedQualityWarnings(
  rawText: string,
  doc: MedicalDocument,
  add: (warning: QualityWarning) => void,
): void {
  const rawNorm = normalizeForCoverage(rawText);
  const docText = documentClinicalText(doc);

  const rawLabish = /(?:оак|оам|биохим|анализ|гемоглобин|креатинин|глюкоз|холестерин|лпнп|лпвп|триглицерид|лейкоцит|эритроцит|тромбоцит|соэ|hba1c|гликирован)/iu.test(rawText);
  if (rawLabish) {
    const values = rawText.match(/\d+(?:[,.]\d+)?/gu) || [];
    for (const value of values) {
      const compact = value.replace(',', '.');
      if (/^\d{1,2}$/.test(compact)) continue;
      if (/^(?:19|20)\d{2}$/.test(compact)) continue;
      const docHas = docText.includes(value) || docText.includes(value.replace(',', '.')) || docText.includes(value.replace('.', ','));
      if (!docHas) {
        add({
          code: 'possibleLostLabValue',
          severity: 'warning',
          field: 'outpatientExams',
          message: `В raw есть лабораторное число ${value}, но оно не найдено в итоговых обследованиях.`,
          evidence: value,
        });
      }
    }
  }

  const examLines = (doc.outpatientExams || '').split(/\n+/).map((line) => line.trim()).filter(Boolean);
  for (const line of examLines) {
    const label = line.match(/(?:^|\s)(КТ|МРТ|МСКТ|ЭКГ|ЭхоКГ|УЗИ|УЗДГ|Холтер|СМАД)(?:\s|$)/iu)?.[1];
    if (!label) continue;
    const labelRe = new RegExp(`(^|[^а-яёa-z])${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}($|[^а-яёa-z])`, 'iu');
    if (!labelRe.test(rawText)) {
      add({
        code: 'suspiciousExamRescue',
        severity: 'warning',
        field: 'outpatientExams',
        message: `В итоговых обследованиях есть ${label}, но в raw нет такого отдельного маркера.`,
        evidence: line.slice(0, 160),
      });
    }
  }

  const lifeHistoryInRecommendations = (doc.recommendations || '')
    .split(/\n+/)
    .find((line) => /(?:алкогол[а-яё]*\s+употреб|наследственност|туберкул|операци|травм|профессиональн[а-яё]*\s+вредност|физическ[а-яё]*\s+активность\s+низк|курит\s+\d)/iu.test(line));
  if (lifeHistoryInRecommendations) {
    add({
      code: 'sectionRoutingIssue',
      severity: 'warning',
      field: 'recommendations',
      message: 'В рекомендациях обнаружена фраза, похожая на анамнез жизни.',
      evidence: lifeHistoryInRecommendations.slice(0, 160),
    });
  }

  const conclusionLifeTail = (doc.conclusion || '')
    .split(/\n+/)
    .find((line) => /(?:туберкул|вирусн[а-яё]*\s+гепатит|вич|операци|травм|наследственност|алкогол|курение|профессиональн[а-яё]*\s+вредност)/iu.test(line));
  if (conclusionLifeTail) {
    add({
      code: 'sectionRoutingIssue',
      severity: 'warning',
      field: 'conclusion',
      message: 'В амбулаторной терапии обнаружена фраза, похожая на анамнез жизни.',
      evidence: conclusionLifeTail.slice(0, 160),
    });
  }

  for (const field of ['conclusion', 'recommendations'] as const) {
    const line = (doc[field] || '')
      .split(/\n+/)
      .find((item) => (item.match(/\d+(?:[,.]\d+)?\s*(?:мг|мкг|г|мл)/giu) || []).length >= 2);
    if (line) {
      add({
        code: 'drugListMayBeMerged',
        severity: 'warning',
        field,
        message: 'Пункт содержит несколько дозировок; возможно, несколько препаратов склеились в один пункт.',
        evidence: line.slice(0, 180),
      });
    }
  }

  const docSentences = docText
    .split(/(?<=[.!?])\s+|\n+/u)
    .map((s) => s.trim())
    .filter((s) => s.length >= 35);
  for (const sentence of docSentences) {
    if (!/(?:кт|мрт|мскт|тредмил|коронар|ацетилсалицил|нитроглицерин|эмпаглифлозин|розувастатин|диагноз|стенокард|инфаркт)/iu.test(sentence)) {
      continue;
    }
    const ns = normalizeForCoverage(sentence);
    if (!ns || rawNorm.includes(ns.slice(0, Math.min(35, ns.length)))) continue;
    const words = ns.split(/\s+/).filter((w) => w.length >= 5);
    const overlap = words.filter((w) => rawNorm.includes(w)).length;
    if (words.length >= 4 && overlap / words.length < 0.45) {
      add({
        code: 'possibleAddedFact',
        severity: 'warning',
        field: 'document',
        message: 'Итоговый документ содержит медицинский факт, который плохо подтверждается raw-текстом.',
        evidence: sentence.slice(0, 180),
      });
      break;
    }
  }
}

/**
 * Оценивает «полезность» структурированного документа. Не учитывает patient —
 * один заполненный fullName не должен спасать документ без клинического
 * контента.
 *
 *   ok          — есть ≥1 содержательное клиническое поле (не лаборатория).
 *   labs_only   — содержательно только outpatientExams (валидный частичный
 *                 документ, но фронт должен показать баннер «только анализы»).
 *   empty       — ни одного содержательного поля; placeholder-заголовки и/или
 *                 одиночное ФИО не считаются.
 */
export function assessDocumentUsefulness(doc: MedicalDocument): DocumentUsefulness {
  const empty: string[] = [];
  const placeholder: string[] = [];
  const meaningful: string[] = [];

  for (const f of [...NON_EXAMS_CLINICAL_FIELDS, 'outpatientExams'] as const) {
    const v = doc[f];
    if (!v || !v.trim()) {
      empty.push(f);
    } else if (!isFieldMeaningful(v)) {
      placeholder.push(f);
    } else {
      meaningful.push(f);
    }
  }

  const meaningfulNonExams = meaningful.filter((f) => f !== 'outpatientExams');
  const examsMeaningful = meaningful.includes('outpatientExams');

  if (meaningfulNonExams.length === 0 && !examsMeaningful) {
    return {
      status: 'empty',
      emptyFields: empty,
      placeholderFields: placeholder,
      meaningfulFields: [],
      reason: 'document_appears_empty',
    };
  }
  if (meaningfulNonExams.length === 0 && examsMeaningful) {
    return {
      status: 'labs_only',
      emptyFields: empty,
      placeholderFields: placeholder,
      meaningfulFields: meaningful,
    };
  }
  return {
    status: 'ok',
    emptyFields: empty,
    placeholderFields: placeholder,
    meaningfulFields: meaningful,
  };
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
    'outpatientExams',
    'clinicalCourse',
    'allergyHistory',
    'objectiveStatus',
    'neurologicalStatus',
    'diagnosis',
    'finalDiagnosis',
    'conclusion',
    'doctorNotes',
    'recommendations',
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

  // Сессионные токены с TTL. Без TTL — утечка памяти от мёртвых сессий.
  const AUTH_TOKEN_TTL_MS = 24 * 60 * 60 * 1000; // 24 часа
  const authTokens = new Map<string, number>(); // token → expiresAt
  const isTokenValid = (token: string): boolean => {
    const exp = authTokens.get(token);
    if (!exp) return false;
    if (Date.now() > exp) {
      authTokens.delete(token);
      return false;
    }
    return true;
  };
  // Периодическая очистка истёкших записей (раз в час).
  // Без неё authTokens/loginAttempts/rateMap утекают в памяти при долгом аптайме.
  const cleanupInterval = setInterval(() => {
    const now = Date.now();
    for (const [token, exp] of authTokens) {
      if (now > exp) authTokens.delete(token);
    }
    for (const [ip, st] of loginAttempts) {
      // Удаляем записи, у которых lockout истёк И счётчик не растёт > 1ч
      if (st.lockedUntil > 0 && now > st.lockedUntil) loginAttempts.delete(ip);
    }
    for (const [ip, st] of rateMap) {
      if (now - st.windowStartedAt > config.security.rateLimitWindowMs * 2) {
        rateMap.delete(ip);
      }
    }
  }, 60 * 60 * 1000);
  cleanupInterval.unref?.();

  // Brute-force защита на /api/auth/login: lockout по IP после N неудачных попыток.
  const LOGIN_MAX_ATTEMPTS = 5;
  const LOGIN_LOCKOUT_MS = 15 * 60 * 1000; // 15 минут
  const LOGIN_FAIL_DELAY_MS = 500; // задержка ответа на неверный пароль
  const loginAttempts = new Map<string, { count: number; lockedUntil: number }>();

  fastify.addHook('onRequest', async (request, reply) => {
    const url = request.url;
    // Only protect API routes; let static files (frontend) pass through
    if (!url.startsWith('/api/')) return;
    if (url.startsWith('/api/health')) return;
    if (url.startsWith('/api/auth/')) return;

    // Auth by password (session token)
    if (config.security.authPassword) {
      const auth = request.headers.authorization;
      const token = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';

      if (!token || !isTokenValid(token)) {
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

    const ip = request.ip || 'unknown';
    const now = Date.now();
    const attempt = loginAttempts.get(ip);

    // Lockout по IP
    if (attempt && attempt.lockedUntil > now) {
      const minutesLeft = Math.ceil((attempt.lockedUntil - now) / 60000);
      return reply.status(429).send({
        error: `Слишком много попыток. Попробуйте через ${minutesLeft} мин.`,
      });
    }

    const body = request.body as Record<string, unknown> | null;
    const password = typeof body?.password === 'string' ? body.password : '';

    if (password !== config.security.authPassword) {
      // Задержка ответа — замедляем перебор
      await new Promise((resolve) => setTimeout(resolve, LOGIN_FAIL_DELAY_MS));

      const next = attempt && attempt.lockedUntil <= now
        ? { count: attempt.count + 1, lockedUntil: 0 }
        : { count: (attempt?.count ?? 0) + 1, lockedUntil: 0 };

      if (next.count >= LOGIN_MAX_ATTEMPTS) {
        next.lockedUntil = now + LOGIN_LOCKOUT_MS;
        next.count = 0;
        loginAttempts.set(ip, next);
        request.log.warn({ ip }, 'Login lockout triggered');
        return reply.status(429).send({
          error: 'Слишком много неверных попыток. Заблокировано на 15 минут.',
        });
      }

      loginAttempts.set(ip, next);
      return reply.status(401).send({ error: 'Неверный пароль' });
    }

    // Успешный логин — сбрасываем счётчик
    loginAttempts.delete(ip);

    const token = randomUUID();
    authTokens.set(token, now + AUTH_TOKEN_TTL_MS);
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
    if (!token || !isTokenValid(token)) {
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
      // PHI-дамп (raw + JSON документа) пишется только при STRUCTURE_LOG_DUMP=true.
      // По умолчанию выключен: содержимое жалоб/анамнеза/диагноза — медданные.
      if (process.env.STRUCTURE_LOG_DUMP === 'true') {
        try {
          const ts = new Date().toISOString().replace(/[:.]/g, '-');
          const logDir = path.join(config.uploadDir, '..', 'temp', 'structure-logs');
          await mkdir(logDir, { recursive: true });
          const logPath = path.join(logDir, `${ts}.log`);
          const dump = [
            `=== ${ts} ===`,
            `--- RAW WHISPER TEXT (${text.length} chars) ---`,
            text,
            '',
            `--- LLM STRUCTURED RESULT ---`,
            JSON.stringify(result, null, 2),
            '',
          ].join('\n');
          await appendFile(logPath, dump, 'utf-8');
          console.log(`[structure-log] wrote ${logPath}`);
        } catch (logErr) {
          console.warn('[structure-log] failed:', logErr);
        }
      }

      // Жёсткая проверка «полезности» — не отдаём success:true на пустой документ.
      const usefulness = assessDocumentUsefulness(result.document);
      if (usefulness.status === 'empty') {
        request.log.warn(
          { emptyFields: usefulness.emptyFields, placeholderFields: usefulness.placeholderFields },
          'structure: document_appears_empty',
        );
        return reply.status(422).send({
          success: false,
          error: 'document_appears_empty',
          reason: usefulness.reason,
          emptyFields: usefulness.emptyFields,
          placeholderFields: usefulness.placeholderFields,
        });
      }

      const qualityWarnings = collectDocumentQualityWarningDetails(text, result.document, usefulness);

      return {
        success: true,
        ...result,
        warnings: [...new Set(qualityWarnings.map((warning) => warning.code === 'important_number_missing'
          ? `important_number_missing:${warning.evidence || 'value'}`
          : warning.code))],
        qualityWarnings,
      };
    } catch (error) {
      console.error('Structuring error:', error);
      const message = error instanceof Error ? error.message : 'Unknown error';
      const isTimeout = /timeout|таймаут|aborted|ECONNABORTED/i.test(message);
      return reply.status(isTimeout ? 408 : 500).send({
        error: isTimeout ? 'LLM request timeout' : 'Text structuring failed',
        message,
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

      if (process.env.STRUCTURE_LOG_DUMP === 'true') {
        try {
          const ts = new Date().toISOString().replace(/[:.]/g, '-');
          const logDir = path.join(config.uploadDir, '..', 'temp', 'structure-logs');
          await mkdir(logDir, { recursive: true });
          const logPath = path.join(logDir, `${ts}_${sourceName}.log`);
          const dump = [
            `=== ${ts} | source=${sourceName} ===`,
            `--- RAW WHISPER TEXT (${transcription.text.length} chars) ---`,
            transcription.text,
            '',
            `--- LLM STRUCTURED RESULT ---`,
            JSON.stringify(structured.document, null, 2),
            '',
          ].join('\n');
          await appendFile(logPath, dump, 'utf-8');
          console.log(`[structure-log] wrote ${logPath}`);
        } catch (logErr) {
          console.warn('[structure-log] failed:', logErr);
        }
      }

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

      // Жёсткая проверка «полезности» — синхронно с /api/structure.
      const usefulness = assessDocumentUsefulness(structured.document);
      if (usefulness.status === 'empty') {
        request.log.warn(
          {
            emptyFields: usefulness.emptyFields,
            placeholderFields: usefulness.placeholderFields,
            transcriptionChars: transcription.text.length,
          },
          'process: document_appears_empty',
        );
        return reply.status(422).send({
          success: false,
          error: 'document_appears_empty',
          reason: usefulness.reason,
          emptyFields: usefulness.emptyFields,
          placeholderFields: usefulness.placeholderFields,
          transcription: {
            text: transcription.text,
            duration: transcription.duration,
            language: transcription.language,
          },
        });
      }

      const qualityWarnings = collectDocumentQualityWarningDetails(transcription.text, structured.document, usefulness);

      return {
        success: true,
        transcription: {
          text: transcription.text,
          duration: transcription.duration,
          language: transcription.language,
        },
        document: structured.document,
        processingTime: transcription.duration + structured.processingTime,
        warnings: [...new Set(qualityWarnings.map((warning) => warning.code === 'important_number_missing'
          ? `important_number_missing:${warning.evidence || 'value'}`
          : warning.code))],
        qualityWarnings,
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
    const scope = normalizeCorrectionScope(typeof body.scope === 'string' ? body.scope : undefined);
    const requireDose = body.requireDose === true;

    if (!wrong || !correct) {
      return reply.status(400).send({ error: "Поля 'wrong' и 'correct' обязательны" });
    }

    if (wrong === correct) {
      return reply.status(400).send({ error: 'Значения должны отличаться' });
    }

    try {
      const correction = await addUserCorrection(wrong, correct, { scope, requireDose });
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

