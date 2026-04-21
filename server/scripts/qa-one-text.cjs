#!/usr/bin/env node
/**
 * Send a single raw-text file to /api/structure and print all fields.
 * Usage: node scripts/qa-one-text.cjs <file.txt>
 */
const fs = require('fs');
const path = require('path');

const SERVER = process.env.SERVER_URL || 'http://localhost:3001';
const AUTH_PASS = process.env.AUTH_PASS || 'meddok2026';

async function main() {
  const file = process.argv[2];
  if (!file) throw new Error('Usage: qa-one-text.cjs <file.txt>');
  const text = fs.readFileSync(path.resolve(file), 'utf-8').trim();
  console.log(`Input: ${file} (${text.length} chars)`);

  const loginResp = await fetch(`${SERVER}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password: AUTH_PASS }),
  });
  const { token } = await loginResp.json();
  if (!token) throw new Error('Login failed');

  const start = Date.now();
  const resp = await fetch(`${SERVER}/api/structure`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  if (!resp.ok) {
    const t = await resp.text();
    console.error(`ERROR ${resp.status}: ${t}`);
    process.exit(1);
  }
  const data = await resp.json();
  console.log(`Total: ${elapsed}s`);

  const doc = data.document || {};
  const fields = [
    'complaints', 'anamnesis', 'outpatientExams', 'clinicalCourse',
    'allergyHistory', 'objectiveStatus', 'neurologicalStatus',
    'diagnosis', 'finalDiagnosis', 'conclusion',
    'doctorNotes', 'recommendations', 'riskAssessment', 'patient'
  ];
  for (const f of fields) {
    const v = doc[f];
    if (v === undefined || v === null) continue;
    const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
    console.log(`\n--- ${f} [${s.length} chars] ---`);
    console.log(s.length ? s : '(empty)');
  }
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
