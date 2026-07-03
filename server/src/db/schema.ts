import { sqliteTable, integer, text } from 'drizzle-orm/sqlite-core';

export const specialties = sqliteTable('specialties', {
  id:        integer('id').primaryKey({ autoIncrement: true }),
  name:      text('name').notNull().unique(),
  code:      text('code').notNull().default(''),
  isActive:  integer('is_active', { mode: 'boolean' }).notNull().default(true),
  createdAt: text('created_at').notNull(),
});

export const doctors = sqliteTable('doctors', {
  id:           integer('id').primaryKey({ autoIncrement: true }),
  name:         text('name').notNull(),
  email:        text('email').notNull().unique(),
  passwordHash: text('password_hash').notNull(),
  specialty:    text('specialty').notNull().default(''),
  departmentId: integer('department_id').references(() => specialties.id),
  role:         text('role', { enum: ['admin', 'doctor'] }).notNull().default('doctor'),
  isActive:     integer('is_active', { mode: 'boolean' }).notNull().default(true),
  createdAt:    text('created_at').notNull(),
});

export const protocolTemplates = sqliteTable('protocol_templates', {
  id:             integer('id').primaryKey({ autoIncrement: true }),
  specialtyId:    integer('specialty_id').references(() => specialties.id),
  name:           text('name').notNull(),
  modality:       text('modality').notNull().default(''),
  bodyPart:       text('body_part').notNull().default(''),
  sourceFilename: text('source_filename').notNull().default(''),
  sourcePath:     text('source_path').notNull().default(''),
  contentText:    text('content_text').notNull(),
  aliasesJson:    text('aliases_json').notNull().default('[]'),
  isActive:       integer('is_active', { mode: 'boolean' }).notNull().default(true),
  createdAt:      text('created_at').notNull(),
  updatedAt:      text('updated_at').notNull(),
});

export const patients = sqliteTable('patients', {
  id:        integer('id').primaryKey({ autoIncrement: true }),
  doctorId:  integer('doctor_id').notNull().references(() => doctors.id),
  fullName:  text('full_name').notNull(),
  birthDate: text('birth_date').default(''),
  gender:    text('gender').default(''),
  phone:     text('phone').default(''),
  iin:       text('iin').default(''),
  notes:     text('notes').default(''),
  createdAt: text('created_at').notNull(),
  updatedAt: text('updated_at').notNull(),
});

export const visits = sqliteTable('visits', {
  id:               integer('id').primaryKey({ autoIncrement: true }),
  patientId:        integer('patient_id').notNull().references(() => patients.id),
  doctorId:         integer('doctor_id').notNull().references(() => doctors.id),
  documentJson:     text('document_json').notNull(),
  rawTranscription: text('raw_transcription').default(''),
  visitDate:        text('visit_date').notNull(),
  createdAt:        text('created_at').notNull(),
});

// Сессии кросс-устройственной синхронизации.
// Врач загружает фото/документ с телефона → результат появляется на десктопе.
export const syncSessions = sqliteTable('sync_sessions', {
  id:               text('id').primaryKey(),             // UUID
  doctorId:         integer('doctor_id').notNull().references(() => doctors.id),
  status:           text('status').notNull().default('processing'), // processing|ready|error
  documentJson:     text('document_json'),                // null пока обрабатывается
  rawTranscription: text('raw_transcription').default(''),
  filename:         text('filename').default(''),
  errorMessage:     text('error_message').default(''),
  createdAt:        text('created_at').notNull(),
  expiresAt:        text('expires_at').notNull(),
});

export type SyncSession = typeof syncSessions.$inferSelect;

export type Doctor           = typeof doctors.$inferSelect;
export type Specialty        = typeof specialties.$inferSelect;
export type ProtocolTemplate = typeof protocolTemplates.$inferSelect;
export type Patient          = typeof patients.$inferSelect;
export type Visit            = typeof visits.$inferSelect;

export type NewDoctor           = typeof doctors.$inferInsert;
export type NewSpecialty        = typeof specialties.$inferInsert;
export type NewProtocolTemplate = typeof protocolTemplates.$inferInsert;
export type NewPatient          = typeof patients.$inferInsert;
export type NewVisit            = typeof visits.$inferInsert;
