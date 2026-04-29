/**
 * routes-doctors.ts вЂ” Auth, Patient cards, Visit history.
 *
 * Р РµРіРёСЃС‚СЂРёСЂСѓРµС‚СЃСЏ РѕС‚РґРµР»СЊРЅРѕ РѕС‚ routes.ts С‡С‚РѕР±С‹ РЅРµ С‚СЂРѕРіР°С‚СЊ СЃСѓС‰РµСЃС‚РІСѓСЋС‰СѓСЋ Р»РѕРіРёРєСѓ.
 * Р’СЃРµ РјР°СЂС€СЂСѓС‚С‹ Р·РґРµСЃСЊ С‚СЂРµР±СѓСЋС‚ JWT РєСЂРѕРјРµ /api/auth/register Рё /api/auth/login.
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
  // РџРµСЂРёРѕРґРёС‡РµСЃРєР°СЏ РѕС‡РёСЃС‚РєР° РїСЂРѕСЃСЂРѕС‡РµРЅРЅС‹С… sync-СЃРµСЃСЃРёР№
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
      return reply.status(403).send({ error: 'РђРєРєР°СѓРЅС‚ РґРµР°РєС‚РёРІРёСЂРѕРІР°РЅ' });
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
  // в”Ђв”Ђв”Ђ Auth в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

  /**
   * POST /api/auth/register
   * Р Р°Р±РѕС‚Р°РµС‚ РўРћР›Р¬РљРћ РµСЃР»Рё РІ Р‘Р” РЅРµС‚ РЅРё РѕРґРЅРѕРіРѕ РґРѕРєС‚РѕСЂР° (РїРµСЂРІРёС‡РЅР°СЏ РЅР°СЃС‚СЂРѕР№РєР°).
   * РџРѕСЃР»Рµ СЃРѕР·РґР°РЅРёСЏ РїРµСЂРІРѕРіРѕ Р°РєРєР°СѓРЅС‚Р° РІРѕР·РІСЂР°С‰Р°РµС‚ 409 РґР»СЏ СЃР»РµРґСѓСЋС‰РёС… РїРѕРїС‹С‚РѕРє
   * (РЅРѕРІС‹С… РІСЂР°С‡РµР№ РґРѕР±Р°РІР»СЏРµС‚ С‚РѕР»СЊРєРѕ admin вЂ” С„СѓРЅРєС†РёРѕРЅР°Р» TODO Step 3).
   */
  fastify.post('/api/auth/register', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

    const name     = typeof body.name     === 'string' ? body.name.trim()     : '';
    const email    = typeof body.email    === 'string' ? body.email.trim().toLowerCase() : '';
    const password = typeof body.password === 'string' ? body.password        : '';
    const specialty = typeof body.specialty === 'string' ? body.specialty.trim() : '';

    if (!name || !email || !password) {
      return reply.status(400).send({ error: 'name, email Рё password РѕР±СЏР·Р°С‚РµР»СЊРЅС‹' });
    }

    if (password.length < 8) {
      return reply.status(400).send({ error: 'РџР°СЂРѕР»СЊ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РЅРµ РјРµРЅРµРµ 8 СЃРёРјРІРѕР»РѕРІ' });
    }

    // Р Р°Р·СЂРµС€Р°РµРј СЂРµРіРёСЃС‚СЂР°С†РёСЋ С‚РѕР»СЊРєРѕ РµСЃР»Рё РґРѕРєС‚РѕСЂРѕРІ РµС‰С‘ РЅРµС‚
    const existing = db.select({ id: doctors.id }).from(doctors).limit(1).all();
    if (existing.length > 0) {
      return reply.status(409).send({
        error: 'Р РµРіРёСЃС‚СЂР°С†РёСЏ Р·Р°РєСЂС‹С‚Р°. РћР±СЂР°С‚РёС‚РµСЃСЊ Рє Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂСѓ.',
      });
    }

    const emailExists = db.select({ id: doctors.id }).from(doctors).where(eq(doctors.email, email)).get();
    if (emailExists) {
      return reply.status(409).send({ error: 'Email СѓР¶Рµ Р·Р°СЂРµРіРёСЃС‚СЂРёСЂРѕРІР°РЅ' });
    }

    const passwordHash = await bcrypt.hash(password, BCRYPT_ROUNDS);
    const [doctor] = db.insert(doctors).values({
      name,
      email,
      passwordHash,
      specialty,
      role: 'admin',
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
   * { email, password } в†’ { token, doctor }
   */
  fastify.post('/api/auth/login', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

    const email    = typeof body.email    === 'string' ? body.email.trim().toLowerCase() : '';
    const password = typeof body.password === 'string' ? body.password : '';

    if (!email || !password) {
      return reply.status(400).send({ error: 'email Рё password РѕР±СЏР·Р°С‚РµР»СЊРЅС‹' });
    }

    const doctor = db.select().from(doctors).where(eq(doctors.email, email)).get();

    // Constant-time comparison to prevent timing attacks
    const dummyHash = '$2a$12$aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const valid = doctor
      ? await bcrypt.compare(password, doctor.passwordHash)
      : await bcrypt.compare(password, dummyHash).then(() => false);

    if (!valid || !doctor) {
      await new Promise((r) => setTimeout(r, 400));
      return reply.status(401).send({ error: 'РќРµРІРµСЂРЅС‹Р№ email РёР»Рё РїР°СЂРѕР»СЊ' });
    }

    if (!doctor.isActive) {
      return reply.status(403).send({ error: 'РђРєРєР°СѓРЅС‚ РґРµР°РєС‚РёРІРёСЂРѕРІР°РЅ' });
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

  /** GET /api/auth/me вЂ” С‚РµРєСѓС‰РёР№ РґРѕРєС‚РѕСЂ */
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

  /** GET /api/auth/check вЂ” РїСЂРѕРІРµСЂРєР° С‚РѕРєРµРЅР° (JWT-СЃРѕРІРјРµСЃС‚РёРјС‹Р№ Р°Р»РёР°СЃ) */
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

  /** POST /api/auth/logout вЂ” РєР»РёРµРЅС‚ СѓРґР°Р»СЏРµС‚ С‚РѕРєРµРЅ, СЃРµСЂРІРµСЂ РїСЂРѕСЃС‚Рѕ РїРѕРґС‚РІРµСЂР¶РґР°РµС‚ */
  fastify.post('/api/auth/logout', async () => ({ success: true }));

  // в”Ђв”Ђв”Ђ Admin / Settings в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

  fastify.get(
    '/api/admin/doctors',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (request.user.role !== 'admin') {
        return reply.status(403).send({ error: 'РўСЂРµР±СѓСЋС‚СЃСЏ РїСЂР°РІР° Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂР°' });
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
        return reply.status(403).send({ error: 'РўСЂРµР±СѓСЋС‚СЃСЏ РїСЂР°РІР° Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂР°' });
      }

      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const name = typeof body.name === 'string' ? body.name.trim() : '';
      const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : '';
      const password = typeof body.password === 'string' ? body.password : '';
      const specialty = typeof body.specialty === 'string' ? body.specialty.trim() : '';
      const role = parseRole(body.role) ?? 'doctor';

      if (!name || !email || !password) {
        return reply.status(400).send({ error: 'name, email Рё password РѕР±СЏР·Р°С‚РµР»СЊРЅС‹' });
      }
      if (password.length < 8) {
        return reply.status(400).send({ error: 'РџР°СЂРѕР»СЊ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РЅРµ РјРµРЅРµРµ 8 СЃРёРјРІРѕР»РѕРІ' });
      }

      const emailExists = db.select({ id: doctors.id }).from(doctors).where(eq(doctors.email, email)).get();
      if (emailExists) {
        return reply.status(409).send({ error: 'Email СѓР¶Рµ Р·Р°СЂРµРіРёСЃС‚СЂРёСЂРѕРІР°РЅ' });
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
        return reply.status(403).send({ error: 'РўСЂРµР±СѓСЋС‚СЃСЏ РїСЂР°РІР° Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂР°' });
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
          return reply.status(400).send({ error: 'РќРµР»СЊР·СЏ СЃРЅСЏС‚СЊ СЂРѕР»СЊ Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂР° Сѓ СЃРµР±СЏ' });
        }
        updates.role = role;
      }
      if (body.isActive !== undefined) {
        if (typeof body.isActive !== 'boolean') return reply.status(400).send({ error: 'Invalid isActive' });
        if (doctorId === request.user.doctorId && !body.isActive) {
          return reply.status(400).send({ error: 'РќРµР»СЊР·СЏ РґРµР°РєС‚РёРІРёСЂРѕРІР°С‚СЊ СЃРµР±СЏ' });
        }
        updates.isActive = body.isActive;
      }

      const existing = db.select({
        id: doctors.id,
        role: doctors.role,
        isActive: doctors.isActive,
      }).from(doctors).where(eq(doctors.id, doctorId)).get();
      if (!existing) return reply.status(404).send({ error: 'Р’СЂР°С‡ РЅРµ РЅР°Р№РґРµРЅ' });

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
        return reply.status(403).send({ error: 'РўСЂРµР±СѓСЋС‚СЃСЏ РїСЂР°РІР° Р°РґРјРёРЅРёСЃС‚СЂР°С‚РѕСЂР°' });
      }

      const { id } = request.params as { id: string };
      const doctorId = parseInt(id, 10);
      if (!Number.isInteger(doctorId)) return reply.status(400).send({ error: 'Invalid doctor id' });
      if (doctorId === request.user.doctorId) {
        return reply.status(400).send({ error: 'РќРµР»СЊР·СЏ СѓРґР°Р»РёС‚СЊ СЃРµР±СЏ' });
      }

      const existing = db.select({
        id: doctors.id,
        role: doctors.role,
        isActive: doctors.isActive,
      }).from(doctors).where(eq(doctors.id, doctorId)).get();
      if (!existing) return reply.status(404).send({ error: 'Р’СЂР°С‡ РЅРµ РЅР°Р№РґРµРЅ' });
      if (existing.role === 'admin' && existing.isActive && !hasOtherActiveAdmin(doctorId)) {
        return reply.status(400).send({ error: 'Нельзя оставить систему без активного администратора' });
      }

      const updated = db.update(doctors).set({ isActive: false }).where(eq(doctors.id, doctorId)).run();
      if (!updated.changes) return reply.status(404).send({ error: 'Р’СЂР°С‡ РЅРµ РЅР°Р№РґРµРЅ' });
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

      if (updates.name === '') return reply.status(400).send({ error: 'РРјСЏ РЅРµ РјРѕР¶РµС‚ Р±С‹С‚СЊ РїСѓСЃС‚С‹Рј' });

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
        return reply.status(400).send({ error: 'currentPassword Рё newPassword РѕР±СЏР·Р°С‚РµР»СЊРЅС‹' });
      }
      if (newPassword.length < 8) {
        return reply.status(400).send({ error: 'РќРѕРІС‹Р№ РїР°СЂРѕР»СЊ РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РЅРµ РјРµРЅРµРµ 8 СЃРёРјРІРѕР»РѕРІ' });
      }

      const doctor = db.select().from(doctors).where(eq(doctors.id, request.user.doctorId)).get();
      if (!doctor) return reply.status(404).send({ error: 'Doctor not found' });
      const valid = await bcrypt.compare(currentPassword, doctor.passwordHash);
      if (!valid) return reply.status(400).send({ error: 'РўРµРєСѓС‰РёР№ РїР°СЂРѕР»СЊ РЅРµРІРµСЂРЅС‹Р№' });

      const passwordHash = await bcrypt.hash(newPassword, BCRYPT_ROUNDS);
      db.update(doctors).set({ passwordHash }).where(eq(doctors.id, doctor.id)).run();
      return { success: true };
    },
  );

  // в”Ђв”Ђв”Ђ Patients в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

  /**
   * GET /api/patients?q=...&page=1
   * Р’РѕР·РІСЂР°С‰Р°РµС‚ СЃРїРёСЃРѕРє РїР°С†РёРµРЅС‚РѕРІ С‚РµРєСѓС‰РµРіРѕ РґРѕРєС‚РѕСЂР°.
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
   * POST /api/patients вЂ” СЃРѕР·РґР°С‚СЊ РїР°С†РёРµРЅС‚Р°
   */
  fastify.post(
    '/api/patients',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const { doctorId } = request.user;
      const body = request.body;
      if (!isRecord(body)) return reply.status(400).send({ error: 'Invalid body' });

      const fullName = typeof body.fullName === 'string' ? body.fullName.trim() : '';
      if (!fullName) return reply.status(400).send({ error: 'fullName РѕР±СЏР·Р°С‚РµР»РµРЅ' });

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
   * GET /api/patients/:id вЂ” РєР°СЂС‚РѕС‡РєР° РїР°С†РёРµРЅС‚Р° + РїРѕСЃР»РµРґРЅРёРµ 20 РІРёР·РёС‚РѕРІ
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

      if (!patient) return reply.status(404).send({ error: 'РџР°С†РёРµРЅС‚ РЅРµ РЅР°Р№РґРµРЅ' });

      const visitList = db
        .select({
          id:        visits.id,
          visitDate: visits.visitDate,
          createdAt: visits.createdAt,
          // РљСЂР°С‚РєРѕРµ РїСЂРµРІСЊСЋ: С‚РѕР»СЊРєРѕ РґРёР°РіРЅРѕР· (РїРµСЂРІС‹Рµ 120 СЃРёРјРІРѕР»РѕРІ)
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
   * PUT /api/patients/:id вЂ” РѕР±РЅРѕРІРёС‚СЊ РєР°СЂС‚РѕС‡РєСѓ
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

      if (!existing) return reply.status(404).send({ error: 'РџР°С†РёРµРЅС‚ РЅРµ РЅР°Р№РґРµРЅ' });

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

  // в”Ђв”Ђв”Ђ Visits в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

  /**
   * POST /api/patients/:id/visits вЂ” СЃРѕС…СЂР°РЅРёС‚СЊ РѕСЃРјРѕС‚СЂ Рє РєР°СЂС‚РѕС‡РєРµ РїР°С†РёРµРЅС‚Р°
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

      // РџСЂРѕРІРµСЂСЏРµРј, С‡С‚Рѕ РїР°С†РёРµРЅС‚ РїСЂРёРЅР°РґР»РµР¶РёС‚ СЌС‚РѕРјСѓ РґРѕРєС‚РѕСЂСѓ
      const patient = db
        .select({ id: patients.id })
        .from(patients)
        .where(and(eq(patients.id, patientId), eq(patients.doctorId, doctorId)))
        .get();

      if (!patient) return reply.status(404).send({ error: 'РџР°С†РёРµРЅС‚ РЅРµ РЅР°Р№РґРµРЅ' });

      const document = body.document;
      if (!isRecord(document)) return reply.status(400).send({ error: 'document РѕР±СЏР·Р°С‚РµР»РµРЅ' });

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

      // РћР±РЅРѕРІР»СЏРµРј updatedAt Сѓ РїР°С†РёРµРЅС‚Р°
      db.update(patients).set({ updatedAt: ts }).where(eq(patients.id, patientId)).run();

      return { success: true, visitId: visit.id, visitDate: visit.visitDate };
    },
  );

  /**
   * GET /api/visits/:id вЂ” РїРѕР»РЅС‹Р№ РґРѕРєСѓРјРµРЅС‚ РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ РІРёР·РёС‚Р°
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

      if (!visit) return reply.status(404).send({ error: 'РћСЃРјРѕС‚СЂ РЅРµ РЅР°Р№РґРµРЅ' });

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

  // в”Ђв”Ђв”Ђ Mobile в†” Desktop Sync в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

  /**
   * POST /api/sync/upload вЂ” РІСЂР°С‡ Р·Р°РіСЂСѓР¶Р°РµС‚ PDF/Word/С„РѕС‚Рѕ СЃ С‚РµР»РµС„РѕРЅР°.
   * РћР±СЂР°Р±Р°С‚С‹РІР°РµС‚СЃСЏ Р°СЃРёРЅС…СЂРѕРЅРЅРѕ. Р’РѕР·РІСЂР°С‰Р°РµС‚ { syncId } РЅРµРјРµРґР»РµРЅРЅРѕ.
   * РЎС‚Р°С‚СѓСЃ РґРѕРєСѓРјРµРЅС‚Р° РјРѕР¶РЅРѕ Р·Р°РїСЂРѕСЃРёС‚СЊ С‡РµСЂРµР· GET /api/sync/:id/status.
   */
  fastify.post(
    '/api/sync/upload',
    { preValidation: [requireActiveDoctor] },
    async (request: FastifyRequest, reply: FastifyReply) => {
      if (!documentExtractor || !llmService) {
        return reply.status(503).send({ error: 'Document processing not configured' });
      }

      const data = await request.file();
      if (!data) return reply.status(400).send({ error: 'Р¤Р°Р№Р» РЅРµ Р·Р°РіСЂСѓР¶РµРЅ' });

      const MAX = 20 * 1024 * 1024;
      const chunks: Buffer[] = [];
      let total = 0;
      for await (const chunk of data.file) {
        total += chunk.length;
        if (total > MAX) return reply.status(413).send({ error: 'Р¤Р°Р№Р» > 20 РњР‘' });
        chunks.push(chunk);
      }
      const buffer = Buffer.concat(chunks);
      if (!buffer.length) return reply.status(400).send({ error: 'РџСѓСЃС‚РѕР№ С„Р°Р№Р»' });

      const { doctorId } = request.user;
      const syncId = randomUUID();
      const filename = toSafeUploadFilename(data.filename || 'document');
      const ts = now();

      // РЎРѕР·РґР°С‘Рј СЃРµСЃСЃРёСЋ СЃРѕ СЃС‚Р°С‚СѓСЃРѕРј 'processing' вЂ” СЃСЂР°Р·Сѓ РІРѕР·РІСЂР°С‰Р°РµРј syncId
      db.insert(syncSessions).values({
        id: syncId,
        doctorId,
        status: 'processing',
        filename,
        createdAt: ts,
        expiresAt: syncExpiresAt(),
      }).run();

      // РћР±СЂР°Р±Р°С‚С‹РІР°РµРј РІ С„РѕРЅРµ вЂ” РЅРµ Р±Р»РѕРєРёСЂСѓРµРј РѕС‚РІРµС‚
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
   * GET /api/sync/:id/status вЂ” С‚РµР»РµС„РѕРЅ РѕРїСЂР°С€РёРІР°РµС‚ СЃС‚Р°С‚СѓСЃ РѕР±СЂР°Р±РѕС‚РєРё
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
   * GET /api/sync/pending вЂ” РґРµСЃРєС‚РѕРї РїРѕР»СѓС‡Р°РµС‚ СЃРїРёСЃРѕРє РіРѕС‚РѕРІС‹С… РґРѕРєСѓРјРµРЅС‚РѕРІ
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
   * POST /api/sync/:id/claim вЂ” РґРµСЃРєС‚РѕРї Р·Р°Р±РёСЂР°РµС‚ РґРѕРєСѓРјРµРЅС‚ Рё СѓРґР°Р»СЏРµС‚ СЃРµСЃСЃРёСЋ
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

      // РЈРґР°Р»СЏРµРј РїРѕСЃР»Рµ claim вЂ” РѕРґРЅРѕСЂР°Р·РѕРІС‹Р№
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
   * DELETE /api/sync/:id вЂ” РѕС‚РјРµРЅРёС‚СЊ/СѓРґР°Р»РёС‚СЊ СЃРµСЃСЃРёСЋ
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
