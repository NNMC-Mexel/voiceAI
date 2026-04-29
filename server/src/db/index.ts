import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import { mkdirSync } from 'fs';
import path from 'path';
import * as schema from './schema.js';

export type AppDb = ReturnType<typeof drizzle<typeof schema>>;

let _db: AppDb | null = null;

export function initDb(dbPath: string): AppDb {
  mkdirSync(path.dirname(path.resolve(dbPath)), { recursive: true });

  const sqlite = new Database(dbPath);

  // Performance + integrity pragmas
  sqlite.pragma('journal_mode = WAL');
  sqlite.pragma('synchronous = NORMAL');
  sqlite.pragma('foreign_keys = ON');
  sqlite.pragma('busy_timeout = 5000');

  // Auto-migration: CREATE TABLE IF NOT EXISTS is idempotent
  sqlite.exec(`
    CREATE TABLE IF NOT EXISTS doctors (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      name          TEXT    NOT NULL,
      email         TEXT    NOT NULL UNIQUE,
      password_hash TEXT    NOT NULL,
      specialty     TEXT    NOT NULL DEFAULT '',
      role          TEXT    NOT NULL DEFAULT 'doctor',
      is_active     INTEGER NOT NULL DEFAULT 1,
      created_at    TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS patients (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      doctor_id   INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
      full_name   TEXT    NOT NULL,
      birth_date  TEXT    NOT NULL DEFAULT '',
      gender      TEXT    NOT NULL DEFAULT '',
      phone       TEXT    NOT NULL DEFAULT '',
      iin         TEXT    NOT NULL DEFAULT '',
      notes       TEXT    NOT NULL DEFAULT '',
      created_at  TEXT    NOT NULL,
      updated_at  TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS visits (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id        INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      doctor_id         INTEGER NOT NULL REFERENCES doctors(id),
      document_json     TEXT    NOT NULL,
      raw_transcription TEXT    NOT NULL DEFAULT '',
      visit_date        TEXT    NOT NULL,
      created_at        TEXT    NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_patients_doctor ON patients(doctor_id);
    CREATE INDEX IF NOT EXISTS idx_visits_patient  ON visits(patient_id);
    CREATE INDEX IF NOT EXISTS idx_visits_doctor   ON visits(doctor_id);
    CREATE INDEX IF NOT EXISTS idx_patients_name   ON patients(full_name COLLATE NOCASE);

    CREATE TABLE IF NOT EXISTS sync_sessions (
      id                TEXT    PRIMARY KEY,
      doctor_id         INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
      status            TEXT    NOT NULL DEFAULT 'processing',
      document_json     TEXT,
      raw_transcription TEXT    NOT NULL DEFAULT '',
      filename          TEXT    NOT NULL DEFAULT '',
      error_message     TEXT    NOT NULL DEFAULT '',
      created_at        TEXT    NOT NULL,
      expires_at        TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sync_doctor ON sync_sessions(doctor_id, status);
  `);

  try { sqlite.exec("ALTER TABLE doctors ADD COLUMN role TEXT NOT NULL DEFAULT 'doctor'"); } catch { /* already exists */ }
  try { sqlite.exec('ALTER TABLE doctors ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1'); } catch { /* already exists */ }
  sqlite.exec(`
    UPDATE doctors
    SET role = 'admin'
    WHERE id = (SELECT MIN(id) FROM doctors)
      AND NOT EXISTS (SELECT 1 FROM doctors WHERE role = 'admin');
  `);

  _db = drizzle(sqlite, { schema });
  return _db;
}

export function getDb(): AppDb {
  if (!_db) throw new Error('DB not initialised — call initDb() first');
  return _db;
}
