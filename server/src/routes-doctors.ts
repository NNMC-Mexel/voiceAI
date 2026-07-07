/**
 * routes-doctors.ts — Auth, Patient cards, Visit history.
 *
 * Регистрируется отдельно от routes.ts чтобы не трогать существующую логику.
 * Все маршруты здесь требуют JWT кроме /api/auth/register и /api/auth/login.
 */

import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import bcrypt from 'bcryptjs';
import { eq, like, desc, and, lt } from 'drizzle-orm';
import { randomUUID } from 'crypto';
import { existsSync } from 'fs';
import { readdir, stat } from 'fs/promises';
import path from 'path';
import mammoth from 'mammoth';
import type { AppDb } from './db/index.js';
import { doctors, patients, visits, syncSessions, specialties, protocolTemplates } from './db/schema.js';
import { DocumentExtractorService } from './services/document-extractor.js';
import { LLMService } from './services/llm.js';
import { documentFromConsultationProtocolText, documentFromExactSourceText, toSafeUploadFilename } from './routes.js';
import type { MedicalDocument } from './types.js';

const BCRYPT_ROUNDS = 12;
const MAX_PATIENTS_PER_PAGE = 50;
type DoctorRole = 'admin' | 'doctor';

function now(): string {
  return new Date().toISOString();
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function parseRole(v: unknown): DoctorRole | null {
  return v === 'admin' || v === 'doctor' ? v : null;
}

const SYNC_TTL_HOURS = 2;
const DEFAULT_TEMPLATE_SOURCE_DIR = 'C:\\Users\\AI\\Downloads\\ШАБЛОН ПРОТОКОЛ СТАЦИОНАР (1)\\ПРОТОКОЛ СТАЦИОНАР';

function syncExpiresAt(): string {
  const d = new Date();
  d.setHours(d.getHours() + SYNC_TTL_HOURS);
  return d.toISOString();
}

function normalizeTemplateName(filename: string): string {
  return path.basename(filename, path.extname(filename))
    .replace(/\s+—\s+копия(?:\s+\(\d+\))?/giu, '')
    .replace(/\s+-\s+копия(?:\s+\(\d+\))?/giu, '')
    .replace(/\s*\(\d+\)\s*$/u, '')
    .replace(/\.+$/u, '')
    .replace(/\s+/gu, ' ')
    .trim();
}

function inferTemplateModality(filename: string, contentText: string): string {
  const text = `${filename}\n${contentText}`.toLowerCase();
  if (/(?:\bкт\b|мскт|компьютерн\S+\s+томограф)/iu.test(text)) return 'КТ';
  if (/(?:\bмрт\b|магнитно-резонансн\S+\s+томограф)/iu.test(text)) return 'МРТ';
  if (/(?:эхокг|эхо\s*кг|эхокардиограф|узи сердца)/iu.test(text)) return 'ЭхоКГ';
  if (/(?:узи|ультразвуков)/iu.test(text)) return 'УЗИ';
  return '';
}

function inferTemplateBodyPart(filename: string): string {
  const name = normalizeTemplateName(filename).toLowerCase();
  const known: Array<[RegExp, string]> = [
    [/обп|брюшн/iu, 'Органы брюшной полости'],
    [/поч/iu, 'Почки'],
    [/щитов/iu, 'Щитовидная железа'],
    [/молоч|грудн/iu, 'Молочные железы'],
    [/вен|сосуд|портальн|аорт/iu, 'Сосуды'],
    [/плеч|коленн|сустав/iu, 'Суставы'],
    [/мочев|трузи|прост/iu, 'Мочеполовая система'],
    [/мошон/iu, 'Мошонка'],
    [/плеврал/iu, 'Плевральная полость'],
    [/лимф/iu, 'Лимфоузлы'],
    [/эхокг|сердц/iu, 'Сердце'],
    [/грыж/iu, 'Грыжа'],
  ];
  return known.find(([pattern]) => pattern.test(name))?.[1] || normalizeTemplateName(filename);
}

function makeTemplateAliases(filename: string, name: string, modality: string, bodyPart: string): string[] {
  const values = new Set<string>();
  for (const value of [name, bodyPart, filename, `${modality} ${bodyPart}`]) {
    const clean = value.toLowerCase().replace(/\.[a-z0-9]+$/iu, '').replace(/\s+/gu, ' ').trim();
    if (clean) values.add(clean);
  }
  return Array.from(values);
}

function publicTemplate(template: typeof protocolTemplates.$inferSelect) {
  let aliases: string[] = [];
  try {
    const parsed = JSON.parse(template.aliasesJson || '[]');
    aliases = Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : [];
  } catch { /* ignore */ }
  return {
    id: template.id,
    specialtyId: template.specialtyId,
    name: template.name,
    modality: template.modality,
    bodyPart: template.bodyPart,
    sourceFilename: template.sourceFilename,
    sourcePath: template.sourcePath,
    contentText: template.contentText,
    aliases,
    isActive: template.isActive,
    createdAt: template.createdAt,
    updatedAt: template.updatedAt,
  };
}

export async function registerDoctorRoutes(
  fastify: FastifyInstance,
  db: AppDb,
  documentExtractor?: DocumentExtractorService,
  llmService?: LLMService,
): Promise<void> {
  // Периодическая очистка просроченных sync-сессий
  const cleanupSyncs = () => {
    try {
      db.delete(syncSessions).where(lt(syncSessions.expiresAt, new Date().toISOString())).run();
    } catch { /* ignore */ }
  };
  const cleanupTimer = setInterval(cleanupSyncs, 15 * 60 * 1000);
  cleanupTimer.unref?.();
  cleanupSyncs();

  const requireActiveDoctor = async (request: FastifyRequest, reply: FastifyReply) => {
    await fastify.authenticate(request, reply);
    if (reply.sent) return;

    const doctor = db.select({
      id: doctors.id,
      name: doctors.name,
      email: doctors.email,
      role: doctors.role,
      isActive: doctors.isActive,
      departmentId: doctors.departmentId,
    }).from(doctors).where(eq(doctors.id, request.user.doctorId)).get();

    if (!doctor) {
      return reply.status(401).send({ error: 'Unauthorized' });
    }
    if (!doctor.isActive) {
      return reply.status(403).send({ error: 'Аккаунт деактивирован' });
    }

    request.user = {
      doctorId: doctor.id,
      email: doctor.email,
      name: doctor.name,
      role: doctor.role,
    };
  };

  const hasOtherActiveAdmin = (doctorId: number): boolean => {
    return db.select({ id: doctors.id }).from(doctors)
      .where(and(eq(doctors.role, 'admin'), eq(doctors.isActive, true)))
      .all()
      .some((doctor) => doctor.id !== doctorId);
  };
  // ─── Auth ─────────────────────────────────────────────────────────────────

  /**
   * POST /api/auth/register
   * Первый зарегистрированный врач становится admin. Дальше врачей создаёт admin.
   */
  fastify.get('/api/auth/setup-status', async () => {
    const existing = db.select({ id: doctors.id }).from(doctors).limit(1).all();
    return { setupRequired: existing.length === 0 };
  });

  fastify.post('/api/auth/register', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

    const name     = typeof body.name     === 'string' ? body.name.trim()     : '';
    const email    = typeof body.email    === 'string' ? body.email.trim().toLowerCase() : '';
    const password = typeof body.password === 'string' ? body.password        : '';
    const specialty = typeof body.specialty === 'string' ? body.specialty.trim() : '';

    if (!name || !email || !password) {
      return reply.status(400).send({ error: 'name, email и password обязательны' });
    }

    if (password.length < 8) {
      return reply.status(400).send({ error: 'Пароль должен быть не менее 8 символов' });
    }

    const existing = db.select({ id: doctors.id }).from(doctors).limit(1).all();
    if (existing.length > 0) {
      return reply.status(403).send({ error: 'Регистрация закрыта. Новых врачей добавляет администратор в настройках.' });
    }
    const role: DoctorRole = 'admin';

    const emailExists = db.select({ id: doctors.id }).from(doctors).where(eq(doctors.email, email)).get();
    if (emailExists) {
      return reply.status(409).send({ error: 'Email уже зарегистрирован' });
    }

    const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);
    const [doctor] = db.insert(doctors).values({
      name,
      email,
      passwordHash,
      specialty,
      role,
      isActive: true,
      createdAt: now(),
    }).returning().all();

    const token = await reply.jwtSign({ doctorId: doctor.id, email: doctor.email, name: doctor.name, role: doctor.role });

    return {
      success: true,
      token,
      doctor: { id: doctor.id, name: doctor.name, email: doctor.email, specialty: doctor.specialty, departmentId: doctor.departmentId, role: doctor.role },
    };
  });

  /**
   * POST /api/auth/login
   * { email, password } → { token, doctor }
   */
  fastify.post('/api/auth/login', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

    const email    = typeof body.email    === 'string' ? body.email.trim().toLowerCase() : '';
    const password = typeof body.password === 'string' ? body.password : '';

    if (!email || !password) {
      return reply.status(400).send({ error: 'email и password обязательны' });
    }

    const doctor = db.select().from(doctors).where(eq(doctors.email, email)).get();

    // Constant-time comparison to prevent timing attacks
    const dummyHash = '$2a$12$aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const valid = doctor
      ? await bcrypt.compare(password, doctor.passwordHash)
      : await bcrypt.compare(password, dummyHash).then(() => false);

    if (!valid || !doctor) {
      await new Promise((r) => setTimeout(r, 400));
      return reply.status(401).send({ error: 'Неверный email или пароль' });
    }

    if (!doctor.isActive) {
      return reply.status(403).send({ error: 'Аккаунт деактивирован' });
    }

    const token = await reply.jwtSign({
      doctorId: doctor.id,
      email: doctor.email,
      name: doctor.name,
      role: doctor.role,
    });

    return {
      success: true,
      token,
      doctor: { id: doctor.id, name: doctor.name, email: doctor.email, specialty: doctor.specialty, departmentId: doctor.departmentId, role: doctor.role },
    };
  });

  /** GET /api/auth/me — текущий доктор */
  fastify.get('/api/auth/me', { preValidation: [requireActiveDoctor] }, async (request) => {
    const { doctorId } = request.user;
    const doctor = db.select({
      id: doctors.id,
      name: doctors.name,
      email: doctors.email,
      specialty: doctors.specialty,
      departmentId: doctors.departmentId,
      role: doctors.role,
      isActive: doctors.isActive,
    }).from(doctors).where(eq(doctors.id, doctorId)).get();

    if (!doctor) return { error: 'Doctor not found' };
    if (!doctor.isActive) return { error: 'Doctor inactive' };
    return { doctor };
  });

  /** GET /api/auth/check — проверка токена (JWT-совместимый алиас) */
  fastify.get('/api/auth/check', async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      await request.jwtVerify();
      const doctor = db.select({ isActive: doctors.isActive })
        .from(doctors)
        .where(eq(doctors.id, request.user.doctorId))
        .get();
      if (!doctor?.isActive) {
        return reply.status(401).send({ authenticated: false });
      }
      return { authenticated: true };
    } catch {
      return reply.status(401).send({ authenticated: false });
    }
  });

  /** POST /api/auth/logout — клиент удаляет токен, сервер просто подтверждает */
  fastify.post('/api/auth/logout', async () => ({ success: true }));

  // ─── Admin / Settings ─────────────────────────────────────────────────────

  fastify.get(
    '/api/admin/doctors',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }

      const list = db.select({
        id: doctors.id,
        name: doctors.name,
        email: doctors.email,
        specialty: doctors.specialty,
        departmentId: doctors.departmentId,
        role: doctors.role,
        isActive: doctors.isActive,
        createdAt: doctors.createdAt,
      }).from(doctors).orderBy(desc(doctors.createdAt)).all();

      return { doctors: list };
    },
  );

  fastify.post(
    '/api/admin/doctors',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }

      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const name = typeof body.name === 'string' ? body.name.trim() : '';
      const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : '';
      const password = typeof body.password === 'string' ? body.password : '';
      const specialty = typeof body.specialty === 'string' ? body.specialty.trim() : '';
      const departmentId = typeof body.departmentId === 'number' ? body.departmentId : null;
      const role = parseRole(body.role) ?? 'doctor';

      if (!name || !email || !password) {
        return reply.status(400).send({ error: 'name, email и password обязательны' });
      }
      if (password.length < 8) {
        return reply.status(400).send({ error: 'Пароль должен быть не менее 8 символов' });
      }

      const emailExists = db.select({ id: doctors.id }).from(doctors).where(eq(doctors.email, email)).get();
      if (emailExists) {
        return reply.status(409).send({ error: 'Email уже зарегистрирован' });
      }

      const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);
      const [doctor] = db.insert(doctors).values({
        name,
        email,
        passwordHash,
        specialty,
        departmentId,
        role,
        isActive: true,
        createdAt: now(),
      }).returning({
        id: doctors.id,
        name: doctors.name,
        email: doctors.email,
        specialty: doctors.specialty,
        departmentId: doctors.departmentId,
        role: doctors.role,
        isActive: doctors.isActive,
        createdAt: doctors.createdAt,
      }).all();

      return { success: true, doctor };
    },
  );

  fastify.patch(
    '/api/admin/doctors/:id',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }

      const { id } = request.params as { id: string };
      const doctorId = parseInt(id, 10);
      if (!Number.isInteger(doctorId)) return reply.status(400).send({ error: 'Invalid doctor id' });

      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const updates: Partial<typeof doctors.$inferInsert> = {};
      if (typeof body.name === 'string') updates.name = body.name.trim();
      if (typeof body.specialty === 'string') updates.specialty = body.specialty.trim();
      if (body.departmentId !== undefined) {
        if (body.departmentId === null) {
          updates.departmentId = null;
        } else if (typeof body.departmentId === 'number' && Number.isInteger(body.departmentId)) {
          const department = db.select({ id: specialties.id }).from(specialties).where(eq(specialties.id, body.departmentId)).get();
          if (!department) return reply.status(400).send({ error: 'Отдел не найден' });
          updates.departmentId = body.departmentId;
        } else {
          return reply.status(400).send({ error: 'Invalid departmentId' });
        }
      }
      if (body.role !== undefined) {
        const role = parseRole(body.role);
        if (!role) return reply.status(400).send({ error: 'Invalid role' });
        if (doctorId === request.user.doctorId && role !== 'admin') {
          return reply.status(400).send({ error: 'Нельзя снять роль администратора у себя' });
        }
        updates.role = role;
      }
      if (body.isActive !== undefined) {
        if (typeof body.isActive !== 'boolean') return reply.status(400).send({ error: 'Invalid isActive' });
        if (doctorId === request.user.doctorId && !body.isActive) {
          return reply.status(400).send({ error: 'Нельзя деактивировать себя' });
        }
        updates.isActive = body.isActive;
      }

      const existing = db.select({
        id: doctors.id,
        role: doctors.role,
        isActive: doctors.isActive,
      }).from(doctors).where(eq(doctors.id, doctorId)).get();
      if (!existing) return reply.status(404).send({ error: 'Врач не найден' });

      const nextRole = updates.role ?? existing.role;
      const nextIsActive = updates.isActive ?? existing.isActive;
      if (existing.role === 'admin' && existing.isActive && (nextRole !== 'admin' || !nextIsActive) && !hasOtherActiveAdmin(doctorId)) {
        return reply.status(400).send({ error: 'Нельзя оставить систему без активного администратора' });
      }

      const [doctor] = Object.keys(updates).length > 0
        ? db.update(doctors).set(updates).where(eq(doctors.id, doctorId)).returning({
            id: doctors.id,
            name: doctors.name,
            email: doctors.email,
            specialty: doctors.specialty,
            departmentId: doctors.departmentId,
            role: doctors.role,
            isActive: doctors.isActive,
            createdAt: doctors.createdAt,
          }).all()
        : db.select({
            id: doctors.id,
            name: doctors.name,
            email: doctors.email,
            specialty: doctors.specialty,
            departmentId: doctors.departmentId,
            role: doctors.role,
            isActive: doctors.isActive,
            createdAt: doctors.createdAt,
          }).from(doctors).where(eq(doctors.id, doctorId)).all();

      return { success: true, doctor };
    },
  );

  fastify.put(
    '/api/admin/doctors/:id/password',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }

      const { id } = request.params as { id: string };
      const doctorId = parseInt(id, 10);
      if (!Number.isInteger(doctorId)) return reply.status(400).send({ error: 'Invalid doctor id' });

      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const newPassword = typeof body.newPassword === 'string' ? body.newPassword : '';
      if (!newPassword) {
        return reply.status(400).send({ error: 'newPassword обязателен' });
      }
      if (newPassword.length < 8) {
        return reply.status(400).send({ error: 'Пароль должен быть не менее 8 символов' });
      }

      const existing = db.select({ id: doctors.id }).from(doctors).where(eq(doctors.id, doctorId)).get();
      if (!existing) return reply.status(404).send({ error: 'Врач не найден' });

      const passwordHash = await bcrypt.hash(newPassword, BCRYPT_ROUNDS);
      db.update(doctors).set({ passwordHash }).where(eq(doctors.id, doctorId)).run();
      return { success: true };
    },
  );

  fastify.get(
    '/api/specialties',
    { preValidation: [requireActiveDoctor] },
    async () => {
      const list = db.select().from(specialties)
        .where(eq(specialties.isActive, true))
        .orderBy(specialties.name)
        .all();
      return { specialties: list };
    },
  );

  fastify.post(
    '/api/admin/specialties',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const name = typeof body.name === 'string' ? body.name.trim() : '';
      const code = typeof body.code === 'string' ? body.code.trim() : '';
      if (!name) return reply.status(400).send({ error: 'name обязателен' });

      const existing = db.select().from(specialties).where(eq(specialties.name, name)).get();
      if (existing) return { success: true, specialty: existing };

      const [specialty] = db.insert(specialties).values({
        name,
        code,
        isActive: true,
        createdAt: now(),
      }).returning().all();
      return { success: true, specialty };
    },
  );

  fastify.get(
    '/api/protocol-templates',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest) => {
      const query = request.query as Record<string, string | undefined>;
      const specialtyId = Number.parseInt(query.specialtyId || '', 10);
      const currentDoctor = db.select({
        role: doctors.role,
        departmentId: doctors.departmentId,
      }).from(doctors).where(eq(doctors.id, request.user.doctorId)).get();
      const allTemplates = db.select().from(protocolTemplates)
        .where(eq(protocolTemplates.isActive, true))
        .orderBy(protocolTemplates.name)
        .all();
      const allowedDepartmentId = currentDoctor?.role === 'admin'
        ? (Number.isInteger(specialtyId) ? specialtyId : null)
        : currentDoctor?.departmentId;
      const filtered = allowedDepartmentId
        ? allTemplates.filter((template) => template.specialtyId === allowedDepartmentId)
        : currentDoctor?.role === 'admin'
          ? allTemplates
          : [];
      return { templates: filtered.map(publicTemplate) };
    },
  );

  fastify.get(
    '/api/admin/protocol-templates',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }
      const list = db.select().from(protocolTemplates).orderBy(desc(protocolTemplates.updatedAt)).all();
      return { templates: list.map(publicTemplate) };
    },
  );

  fastify.post(
    '/api/admin/protocol-templates/import-folder',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const folderPath = typeof body.folderPath === 'string' && body.folderPath.trim()
        ? body.folderPath.trim()
        : process.env.PROTOCOL_TEMPLATE_SOURCE_DIR || DEFAULT_TEMPLATE_SOURCE_DIR;
      if (!existsSync(folderPath)) {
        return reply.status(400).send({ error: `Папка не найдена: ${folderPath}` });
      }

      const specialtyName = typeof body.specialtyName === 'string' && body.specialtyName.trim()
        ? body.specialtyName.trim()
        : 'Лучевая диагностика';
      let specialty = db.select().from(specialties).where(eq(specialties.name, specialtyName)).get();
      if (!specialty) {
        [specialty] = db.insert(specialties).values({
          name: specialtyName,
          code: specialtyName.toLowerCase().replace(/\s+/gu, '_'),
          isActive: true,
          createdAt: now(),
        }).returning().all();
      }

      const entries = await readdir(folderPath);
      const imported: Array<{ id: number; name: string; filename: string }> = [];
      const skipped: Array<{ filename: string; reason: string }> = [];

      for (const filename of entries) {
        if (filename.startsWith('~$')) {
          skipped.push({ filename, reason: 'временный файл Word' });
          continue;
        }
        const ext = path.extname(filename).toLowerCase();
        if (ext !== '.docx') {
          skipped.push({ filename, reason: ext === '.doc' ? 'старый .doc, нужен .docx' : 'неподдерживаемый формат' });
          continue;
        }

        const filePath = path.join(folderPath, filename);
        const info = await stat(filePath);
        if (!info.isFile()) continue;

        try {
          const extracted = await mammoth.extractRawText({ path: filePath });
          const contentText = extracted.value.trim();
          if (!contentText) {
            skipped.push({ filename, reason: 'не удалось извлечь текст' });
            continue;
          }

          const name = normalizeTemplateName(filename);
          const modality = inferTemplateModality(filename, contentText);
          const bodyPart = inferTemplateBodyPart(filename);
          const aliasesJson = JSON.stringify(makeTemplateAliases(filename, name, modality, bodyPart));
          const ts = now();
          const existing = db.select().from(protocolTemplates).where(eq(protocolTemplates.sourcePath, filePath)).get();

          const values = {
            specialtyId: specialty.id,
            name,
            modality,
            bodyPart,
            sourceFilename: filename,
            sourcePath: filePath,
            contentText,
            aliasesJson,
            isActive: true,
            updatedAt: ts,
          };

          const [template] = existing
            ? db.update(protocolTemplates).set(values).where(eq(protocolTemplates.id, existing.id)).returning().all()
            : db.insert(protocolTemplates).values({ ...values, createdAt: ts }).returning().all();

          imported.push({ id: template.id, name: template.name, filename });
        } catch (err) {
          skipped.push({ filename, reason: err instanceof Error ? err.message : 'ошибка чтения файла' });
        }
      }

      return {
        success: true,
        folderPath,
        specialty,
        imported,
        skipped,
      };
    },
  );

  fastify.patch(
    '/api/admin/protocol-templates/:id',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }
      const { id } = request.params as { id: string };
      const templateId = Number.parseInt(id, 10);
      if (!Number.isInteger(templateId)) return reply.status(400).send({ error: 'Invalid template id' });
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const updates: Partial<typeof protocolTemplates.$inferInsert> = { updatedAt: now() };
      if (typeof body.name === 'string') updates.name = body.name.trim();
      if (typeof body.modality === 'string') updates.modality = body.modality.trim();
      if (typeof body.bodyPart === 'string') updates.bodyPart = body.bodyPart.trim();
      if (typeof body.contentText === 'string') updates.contentText = body.contentText.trim();
      if (typeof body.isActive === 'boolean') updates.isActive = body.isActive;
      if (typeof body.specialtyId === 'number') updates.specialtyId = body.specialtyId;
      if (Array.isArray(body.aliases)) {
        updates.aliasesJson = JSON.stringify(body.aliases.filter((x): x is string => typeof x === 'string'));
      }

      const [template] = db.update(protocolTemplates).set(updates)
        .where(eq(protocolTemplates.id, templateId))
        .returning()
        .all();
      if (!template) return reply.status(404).send({ error: 'Шаблон не найден' });
      return { success: true, template: publicTemplate(template) };
    },
  );

  fastify.post(
    '/api/protocols/fill',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });
      const templateId = typeof body.templateId === 'number' ? body.templateId : Number.parseInt(String(body.templateId || ''), 10);
      const text = typeof body.text === 'string' ? body.text.trim() : '';
      if (!Number.isInteger(templateId)) return reply.status(400).send({ error: 'templateId обязателен' });
      if (!text) return reply.status(400).send({ error: 'text обязателен' });

      const template = db.select().from(protocolTemplates)
        .where(and(eq(protocolTemplates.id, templateId), eq(protocolTemplates.isActive, true)))
        .get();
      if (!template) return reply.status(404).send({ error: 'Шаблон не найден' });
      if (request.user.role !== 'admin') {
        const currentDoctor = db.select({ departmentId: doctors.departmentId })
          .from(doctors)
          .where(eq(doctors.id, request.user.doctorId))
          .get();
        if (!currentDoctor?.departmentId || template.specialtyId !== currentDoctor.departmentId) {
          return reply.status(403).send({ error: 'Шаблон недоступен для отдела врача' });
        }
      }

      try {
        const filledText = llmService
          ? await llmService.fillProtocolTemplate(template.contentText, text)
          : `${template.contentText}\n\nДиктовка:\n${text}`;
        return {
          success: true,
          template: publicTemplate(template),
          rawText: text,
          filledText,
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Ошибка заполнения протокола';
        return reply.status(500).send({ error: message });
      }
    },
  );

  fastify.delete(
    '/api/admin/doctors/:id',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'Требуются права администратора' });
      }

      const { id } = request.params as { id: string };
      const doctorId = parseInt(id, 10);
      if (!Number.isInteger(doctorId)) return reply.status(400).send({ error: 'Invalid doctor id' });
      if (doctorId === request.user.doctorId) {
        return reply.status(400).send({ error: 'Нельзя удалить себя' });
      }

      const existing = db.select({
        id: doctors.id,
        role: doctors.role,
        isActive: doctors.isActive,
      }).from(doctors).where(eq(doctors.id, doctorId)).get();
      if (!existing) return reply.status(404).send({ error: 'Врач не найден' });
      if (existing.role === 'admin' && existing.isActive && !hasOtherActiveAdmin(doctorId)) {
        return reply.status(400).send({ error: 'Нельзя оставить систему без активного администратора' });
      }

      const updated = db.update(doctors).set({ isActive: false }).where(eq(doctors.id, doctorId)).run();
      if (!updated.changes) return reply.status(404).send({ error: 'Врач не найден' });
      return { success: true };
    },
  );

  fastify.put(
    '/api/settings/profile',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const updates: Partial<typeof doctors.$inferInsert> = {};
      if (typeof body.name === 'string') updates.name = body.name.trim();
      if (typeof body.specialty === 'string') updates.specialty = body.specialty.trim();

      if (updates.name === '') return reply.status(400).send({ error: 'Имя не может быть пустым' });

      const [doctor] = Object.keys(updates).length > 0
        ? db.update(doctors).set(updates).where(eq(doctors.id, request.user.doctorId)).returning({
            id: doctors.id,
            name: doctors.name,
            email: doctors.email,
            specialty: doctors.specialty,
            departmentId: doctors.departmentId,
            role: doctors.role,
          }).all()
        : db.select({
            id: doctors.id,
            name: doctors.name,
            email: doctors.email,
            specialty: doctors.specialty,
            departmentId: doctors.departmentId,
            role: doctors.role,
          }).from(doctors).where(eq(doctors.id, request.user.doctorId)).all();

      if (!doctor) return reply.status(404).send({ error: 'Doctor not found' });
      return { success: true, doctor };
    },
  );

  fastify.put(
    '/api/settings/password',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const currentPassword = typeof body.currentPassword === 'string' ? body.currentPassword : '';
      const newPassword = typeof body.newPassword === 'string' ? body.newPassword : '';
      if (!currentPassword || !newPassword) {
        return reply.status(400).send({ error: 'currentPassword и newPassword обязательны' });
      }
      if (newPassword.length < 8) {
        return reply.status(400).send({ error: 'Новый пароль должен быть не менее 8 символов' });
      }

      const doctor = db.select().from(doctors).where(eq(doctors.id, request.user.doctorId)).get();
      if (!doctor) return reply.status(404).send({ error: 'Doctor not found' });
      const valid = await bcrypt.compare(currentPassword, doctor.passwordHash);
      if (!valid) return reply.status(400).send({ error: 'Текущий пароль неверный' });

      const passwordHash = await bcrypt.hash(newPassword, BCRYPT_ROUNDS);
      db.update(doctors).set({ passwordHash }).where(eq(doctors.id, doctor.id)).run();
      return { success: true };
    },
  );

  // ─── Patients ─────────────────────────────────────────────────────────────

  /**
   * GET /api/patients?q=...&page=1
   * Возвращает список пациентов текущего доктора.
   */
  fastify.get(
    '/api/patients',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest) => {
      const { doctorId } = request.user;
      const query = request.query as Record<string, string>;
      const search = (query.q || '').trim();
      const page = Math.max(1, parseInt(query.page || '1', 10));
      const offset = (page - 1) * MAX_PATIENTS_PER_PAGE;

      const baseCondition = eq(patients.doctorId, doctorId);
      const condition = search
        ? and(baseCondition, like(patients.fullName, `%${search}%`))
        : baseCondition;

      const list = db
        .select({
          id:        patients.id,
          fullName:  patients.fullName,
          birthDate: patients.birthDate,
          gender:    patients.gender,
          phone:     patients.phone,
          iin:       patients.iin,
          updatedAt: patients.updatedAt,
        })
        .from(patients)
        .where(condition)
        .orderBy(desc(patients.updatedAt))
        .limit(MAX_PATIENTS_PER_PAGE)
        .offset(offset)
        .all();

      return { patients: list, page, hasMore: list.length === MAX_PATIENTS_PER_PAGE };
    },
  );

  /**
   * POST /api/patients — создать пациента
   */
  fastify.post(
    '/api/patients',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const fullName = typeof body.fullName === 'string' ? body.fullName.trim() : '';
      if (!fullName) return reply.status(400).send({ error: 'fullName обязателен' });

      const ts = now();
      const [patient] = db.insert(patients).values({
        doctorId,
        fullName,
        birthDate:  typeof body.birthDate  === 'string' ? body.birthDate  : '',
        gender:     typeof body.gender     === 'string' ? body.gender     : '',
        phone:      typeof body.phone      === 'string' ? body.phone      : '',
        iin:        typeof body.iin        === 'string' ? body.iin        : '',
        notes:      typeof body.notes      === 'string' ? body.notes      : '',
        createdAt:  ts,
        updatedAt:  ts,
      }).returning().all();

      return { success: true, patient };
    },
  );

  /**
   * GET /api/patients/:id — карточка пациента + последние 20 визитов
   */
  fastify.get(
    '/api/patients/:id',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const { id } = request.params as { id: string };
      const patientId = parseInt(id, 10);

      const patient = db
        .select()
        .from(patients)
        .where(and(eq(patients.id, patientId), eq(patients.doctorId, doctorId)))
        .get();

      if (!patient) return reply.status(404).send({ error: 'Пациент не найден' });

      const visitList = db
        .select({
          id:        visits.id,
          visitDate: visits.visitDate,
          createdAt: visits.createdAt,
          // Краткое превью: только диагноз (первые 120 символов)
          diagnosisPreview: visits.documentJson,
        })
        .from(visits)
        .where(eq(visits.patientId, patientId))
        .orderBy(desc(visits.visitDate))
        .limit(20)
        .all()
        .map((v) => {
          let diagnosisPreview = '';
          try {
            const doc = JSON.parse(v.diagnosisPreview) as Partial<MedicalDocument>;
            diagnosisPreview = (doc.diagnosis || doc.finalDiagnosis || '').slice(0, 120);
          } catch { /* skip */ }
          return { id: v.id, visitDate: v.visitDate, createdAt: v.createdAt, diagnosisPreview };
        });

      return { patient, visits: visitList };
    },
  );

  /**
   * PUT /api/patients/:id — обновить карточку
   */
  fastify.put(
    '/api/patients/:id',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const { id } = request.params as { id: string };
      const patientId = parseInt(id, 10);
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const existing = db
        .select({ id: patients.id })
        .from(patients)
        .where(and(eq(patients.id, patientId), eq(patients.doctorId, doctorId)))
        .get();

      if (!existing) return reply.status(404).send({ error: 'Пациент не найден' });

      const updates: Partial<typeof patients.$inferInsert> = { updatedAt: now() };
      if (typeof body.fullName  === 'string') updates.fullName  = body.fullName.trim();
      if (typeof body.birthDate === 'string') updates.birthDate = body.birthDate;
      if (typeof body.gender    === 'string') updates.gender    = body.gender;
      if (typeof body.phone     === 'string') updates.phone     = body.phone;
      if (typeof body.iin       === 'string') updates.iin       = body.iin;
      if (typeof body.notes     === 'string') updates.notes     = body.notes;

      const [updated] = db
        .update(patients)
        .set(updates)
        .where(eq(patients.id, patientId))
        .returning()
        .all();

      return { success: true, patient: updated };
    },
  );

  // ─── Visits ───────────────────────────────────────────────────────────────

  /**
   * POST /api/patients/:id/visits — сохранить осмотр к карточке пациента
   * Body: { document: MedicalDocument, rawTranscription?: string, visitDate?: string }
   */
  fastify.post(
    '/api/patients/:id/visits',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const { id } = request.params as { id: string };
      const patientId = parseInt(id, 10);
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      // Проверяем, что пациент принадлежит этому доктору
      const patient = db
        .select({ id: patients.id })
        .from(patients)
        .where(and(eq(patients.id, patientId), eq(patients.doctorId, doctorId)))
        .get();

      if (!patient) return reply.status(404).send({ error: 'Пациент не найден' });

      const document = body.document;
      if (!isRecord(document)) return reply.status(400).send({ error: 'document обязателен' });

      const ts = now();
      const visitDate = typeof body.visitDate === 'string' && body.visitDate
        ? body.visitDate
        : ts.slice(0, 10);

      const [visit] = db.insert(visits).values({
        patientId,
        doctorId,
        documentJson:     JSON.stringify(document),
        rawTranscription: typeof body.rawTranscription === 'string' ? body.rawTranscription : '',
        visitDate,
        createdAt: ts,
      }).returning().all();

      // Обновляем updatedAt у пациента
      db.update(patients).set({ updatedAt: ts }).where(eq(patients.id, patientId)).run();

      return { success: true, visitId: visit.id, visitDate: visit.visitDate };
    },
  );

  /**
   * GET /api/visits/:id — полный документ конкретного визита
   */
  fastify.get(
    '/api/visits/:id',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const { id } = request.params as { id: string };
      const visitId = parseInt(id, 10);

      const visit = db
        .select()
        .from(visits)
        .where(and(eq(visits.id, visitId), eq(visits.doctorId, doctorId)))
        .get();

      if (!visit) return reply.status(404).send({ error: 'Осмотр не найден' });

      let document: MedicalDocument | null = null;
      try { document = JSON.parse(visit.documentJson); } catch { /* corrupt */ }

      return {
        visit: {
          id:               visit.id,
          patientId:        visit.patientId,
          visitDate:        visit.visitDate,
          createdAt:        visit.createdAt,
          rawTranscription: visit.rawTranscription,
          document,
        },
      };
    },
  );

  // ─── Mobile ↔ Desktop Sync ────────────────────────────────────────────────

  /**
   * POST /api/sync/upload — врач загружает PDF/Word/фото с телефона.
   * Обрабатывается асинхронно. Возвращает { syncId } немедленно.
   * Статус документа можно запросить через GET /api/sync/:id/status.
   */
  fastify.post(
    '/api/sync/upload',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (!documentExtractor || !llmService) {
        return reply.status(503).send({ error: 'Document processing not configured' });
      }

      const data = await request.file();
      if (!data) return reply.status(400).send({ error: 'Файл не загружен' });

      const MAX = 20 * 1024 * 1024;
      const chunks: Buffer[] = [];
      let total = 0;
      for await (const chunk of data.file) {
        total += chunk.length;
        if (total > MAX) return reply.status(413).send({ error: 'Файл > 20 МБ' });
        chunks.push(chunk);
      }
      const buffer = Buffer.concat(chunks);
      if (!buffer.length) return reply.status(400).send({ error: 'Пустой файл' });

      const { doctorId } = request.user;
      const syncId = randomUUID();
      const filename = toSafeUploadFilename(data.filename || 'document');
      const ts = now();

      // Создаём сессию со статусом 'processing' — сразу возвращаем syncId
      db.insert(syncSessions).values({
        id: syncId,
        doctorId,
        status: 'processing',
        filename,
        createdAt: ts,
        expiresAt: syncExpiresAt(),
      }).run();

      // Обрабатываем в фоне — не блокируем ответ
      (async () => {
        try {
          const extraction = await documentExtractor.extract(buffer, data.mimetype, filename);
          const protocolDocument = documentFromConsultationProtocolText(extraction.text);
          const document = protocolDocument ||
            (extraction.extractionMethod === 'vision'
              ? documentFromExactSourceText(extraction.text)
              : (await llmService.structureText(extraction.text)).document);
          db.update(syncSessions).set({
            status: 'ready',
            documentJson: JSON.stringify(document),
            rawTranscription: extraction.text,
          }).where(eq(syncSessions.id, syncId)).run();
        } catch (err) {
          const msg = err instanceof Error ? err.message : 'Processing error';
          db.update(syncSessions).set({ status: 'error', errorMessage: msg })
            .where(eq(syncSessions.id, syncId)).run();
          fastify.log.error({ syncId }, `sync processing error: ${msg}`);
        }
      })();

      return { success: true, syncId };
    },
  );

  /**
   * GET /api/sync/:id/status — телефон опрашивает статус обработки
   */
  fastify.get(
    '/api/sync/:id/status',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const { id } = request.params as { id: string };

      const session = db.select({
        id: syncSessions.id,
        status: syncSessions.status,
        filename: syncSessions.filename,
        errorMessage: syncSessions.errorMessage,
        expiresAt: syncSessions.expiresAt,
      }).from(syncSessions)
        .where(and(eq(syncSessions.id, id), eq(syncSessions.doctorId, doctorId)))
        .get();

      if (!session) return reply.status(404).send({ error: 'Sync session not found' });
      return { session };
    },
  );

  /**
   * GET /api/sync/pending — десктоп получает список готовых документов
   */
  fastify.get(
    '/api/sync/pending',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest) => {
      const { doctorId } = request.user;
      const list = db.select({
        id: syncSessions.id,
        filename: syncSessions.filename,
        createdAt: syncSessions.createdAt,
        expiresAt: syncSessions.expiresAt,
      }).from(syncSessions)
        .where(and(
          eq(syncSessions.doctorId, doctorId),
          eq(syncSessions.status, 'ready'),
        ))
        .orderBy(desc(syncSessions.createdAt))
        .all();

      return { sessions: list };
    },
  );

  /**
   * POST /api/sync/:id/claim — десктоп забирает документ и удаляет сессию
   */
  fastify.post(
    '/api/sync/:id/claim',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const { id } = request.params as { id: string };

      const session = db.select().from(syncSessions)
        .where(and(
          eq(syncSessions.id, id),
          eq(syncSessions.doctorId, doctorId),
          eq(syncSessions.status, 'ready'),
        ))
        .get();

      if (!session) return reply.status(404).send({ error: 'Session not found or not ready' });

      let document = null;
      try { document = JSON.parse(session.documentJson ?? ''); } catch { /* corrupt */ }

      // Удаляем после claim — одноразовый
      db.delete(syncSessions).where(eq(syncSessions.id, id)).run();

      return {
        success: true,
        document,
        rawTranscription: session.rawTranscription || '',
        filename: session.filename,
      };
    },
  );

  /**
   * DELETE /api/sync/:id — отменить/удалить сессию
   */
  fastify.delete(
    '/api/sync/:id',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const { id } = request.params as { id: string };
      const deleted = db.delete(syncSessions)
        .where(and(eq(syncSessions.id, id), eq(syncSessions.doctorId, doctorId)))
        .run();
      if (!deleted.changes) return reply.status(404).send({ error: 'Not found' });
      return { success: true };
    },
  );
}
