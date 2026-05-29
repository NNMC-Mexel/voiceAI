require('dotenv').config();

const path = require('path');
const fs = require('fs');
const Database = require('better-sqlite3');
const { Pool } = require('pg');

const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  console.error('DATABASE_URL is required');
  process.exit(1);
}

const sqlitePath = process.env.SQLITE_DB_PATH || process.env.DB_PATH || path.join(__dirname, '..', 'data', 'meddok.db');
if (!fs.existsSync(sqlitePath)) {
  console.error(`SQLite database not found: ${sqlitePath}`);
  process.exit(1);
}

const pool = new Pool({
  connectionString: databaseUrl,
  ssl: process.env.DATABASE_SSL === 'true' ? { rejectUnauthorized: false } : undefined,
});

async function createSchema(client) {
  await client.query(`
    CREATE TABLE IF NOT EXISTS doctors (
      id            SERIAL PRIMARY KEY,
      name          TEXT    NOT NULL,
      email         TEXT    NOT NULL UNIQUE,
      password_hash TEXT    NOT NULL,
      specialty     TEXT    NOT NULL DEFAULT '',
      role          TEXT    NOT NULL DEFAULT 'doctor',
      is_active     BOOLEAN NOT NULL DEFAULT TRUE,
      created_at    TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS patients (
      id          SERIAL PRIMARY KEY,
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
      id                SERIAL PRIMARY KEY,
      patient_id        INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
      doctor_id         INTEGER NOT NULL REFERENCES doctors(id),
      document_json     TEXT    NOT NULL,
      raw_transcription TEXT    NOT NULL DEFAULT '',
      visit_date        TEXT    NOT NULL,
      created_at        TEXT    NOT NULL
    );

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

    CREATE INDEX IF NOT EXISTS idx_patients_doctor ON patients(doctor_id);
    CREATE INDEX IF NOT EXISTS idx_visits_patient  ON visits(patient_id);
    CREATE INDEX IF NOT EXISTS idx_visits_doctor   ON visits(doctor_id);
    CREATE INDEX IF NOT EXISTS idx_patients_name   ON patients(full_name);
    CREATE INDEX IF NOT EXISTS idx_sync_doctor     ON sync_sessions(doctor_id, status);
  `);
}

async function resetSequences(client) {
  for (const table of ['doctors', 'patients', 'visits']) {
    await client.query(`
      SELECT setval(
        pg_get_serial_sequence('${table}', 'id'),
        COALESCE((SELECT MAX(id) FROM ${table}), 1),
        (SELECT COALESCE(MAX(id), 0) > 0 FROM ${table})
      )
    `);
  }
}

async function migrate() {
  const sqlite = new Database(sqlitePath, { readonly: true });
  const client = await pool.connect();

  try {
    const doctors = sqlite.prepare('SELECT * FROM doctors ORDER BY id').all();
    const patients = sqlite.prepare('SELECT * FROM patients ORDER BY id').all();
    const visits = sqlite.prepare('SELECT * FROM visits ORDER BY id').all();
    const syncSessions = sqlite.prepare('SELECT * FROM sync_sessions ORDER BY created_at').all();

    await client.query('BEGIN');
    await createSchema(client);

    for (const row of doctors) {
      await client.query(
        `INSERT INTO doctors (id, name, email, password_hash, specialty, role, is_active, created_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         ON CONFLICT (id) DO UPDATE SET
           name = EXCLUDED.name,
           email = EXCLUDED.email,
           password_hash = EXCLUDED.password_hash,
           specialty = EXCLUDED.specialty,
           role = EXCLUDED.role,
           is_active = EXCLUDED.is_active,
           created_at = EXCLUDED.created_at`,
        [row.id, row.name, row.email, row.password_hash, row.specialty || '', row.role || 'doctor', Boolean(row.is_active), row.created_at],
      );
    }

    for (const row of patients) {
      await client.query(
        `INSERT INTO patients (id, doctor_id, full_name, birth_date, gender, phone, iin, notes, created_at, updated_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
         ON CONFLICT (id) DO UPDATE SET
           doctor_id = EXCLUDED.doctor_id,
           full_name = EXCLUDED.full_name,
           birth_date = EXCLUDED.birth_date,
           gender = EXCLUDED.gender,
           phone = EXCLUDED.phone,
           iin = EXCLUDED.iin,
           notes = EXCLUDED.notes,
           created_at = EXCLUDED.created_at,
           updated_at = EXCLUDED.updated_at`,
        [
          row.id,
          row.doctor_id,
          row.full_name,
          row.birth_date || '',
          row.gender || '',
          row.phone || '',
          row.iin || '',
          row.notes || '',
          row.created_at,
          row.updated_at,
        ],
      );
    }

    for (const row of visits) {
      await client.query(
        `INSERT INTO visits (id, patient_id, doctor_id, document_json, raw_transcription, visit_date, created_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (id) DO UPDATE SET
           patient_id = EXCLUDED.patient_id,
           doctor_id = EXCLUDED.doctor_id,
           document_json = EXCLUDED.document_json,
           raw_transcription = EXCLUDED.raw_transcription,
           visit_date = EXCLUDED.visit_date,
           created_at = EXCLUDED.created_at`,
        [
          row.id,
          row.patient_id,
          row.doctor_id,
          row.document_json,
          row.raw_transcription || '',
          row.visit_date,
          row.created_at,
        ],
      );
    }

    for (const row of syncSessions) {
      await client.query(
        `INSERT INTO sync_sessions (id, doctor_id, status, document_json, raw_transcription, filename, error_message, created_at, expires_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
         ON CONFLICT (id) DO UPDATE SET
           doctor_id = EXCLUDED.doctor_id,
           status = EXCLUDED.status,
           document_json = EXCLUDED.document_json,
           raw_transcription = EXCLUDED.raw_transcription,
           filename = EXCLUDED.filename,
           error_message = EXCLUDED.error_message,
           created_at = EXCLUDED.created_at,
           expires_at = EXCLUDED.expires_at`,
        [
          row.id,
          row.doctor_id,
          row.status || 'processing',
          row.document_json ?? null,
          row.raw_transcription || '',
          row.filename || '',
          row.error_message || '',
          row.created_at,
          row.expires_at,
        ],
      );
    }

    await resetSequences(client);
    await client.query('COMMIT');

    console.log(`Migrated doctors=${doctors.length}, patients=${patients.length}, visits=${visits.length}, sync_sessions=${syncSessions.length}`);
  } catch (err) {
    await client.query('ROLLBACK');
    throw err;
  } finally {
    client.release();
    sqlite.close();
    await pool.end();
  }
}

migrate().catch((err) => {
  console.error(err);
  process.exit(1);
});
