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
import type { AppDb } from './db/index.js';
import { doctors, patients, visits, syncSessions } from './db/schema.js';
import { DocumentExtractorService } from './services/document-extractor.js';
import { LLMService } from './services/llm.js';
import { documentFromExactSourceText, toSafeUploadFilename } from './routes.js';
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

function syncExpiresAt(): string {
  const d = new Date();
  d.setHours(d.getHours() + SYNC_TTL_HOURS);
  return d.toISOString();
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
   * Первый зарегистрированный врач становится admin, следующие — doctor.
   */
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
    const role: DoctorRole = existing.length === 0 ? 'admin' : 'doctor';

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
      doctor: { id: doctor.id, name: doctor.name, email: doctor.email, specialty: doctor.specialty, role: doctor.role },
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
      doctor: { id: doctor.id, name: doctor.name, email: doctor.email, specialty: doctor.specialty, role: doctor.role },
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
        role,
        isActive: true,
        createdAt: now(),
      }).returning({
        id: doctors.id,
        name: doctors.name,
        email: doctors.email,
        specialty: doctors.specialty,
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
            role: doctors.role,
            isActive: doctors.isActive,
            createdAt: doctors.createdAt,
          }).all()
        : db.select({
            id: doctors.id,
            name: doctors.name,
            email: doctors.email,
            specialty: doctors.specialty,
            role: doctors.role,
            isActive: doctors.isActive,
            createdAt: doctors.createdAt,
          }).from(doctors).where(eq(doctors.id, doctorId)).all();

      return { success: true, doctor };
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
            role: doctors.role,
          }).all()
        : db.select({
            id: doctors.id,
            name: doctors.name,
            email: doctors.email,
            specialty: doctors.specialty,
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
          const document = extraction.extractionMethod === 'vision'
            ? documentFromExactSourceText(extraction.text)
            : (await llmService.structureText(extraction.text)).document;
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
