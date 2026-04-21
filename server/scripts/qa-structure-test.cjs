#!/usr/bin/env node
/**
 * QA Structure-Only Test — feeds cached raw Whisper text into /api/structure
 * (bypasses transcription) so we can iterate on post-processing without re-running Whisper.
 * Usage: node scripts/qa-structure-test.cjs
 */
const fs = require('fs');
const path = require('path');

const SERVER = process.env.SERVER_URL || 'http://localhost:3001';
const AUTH_PASS = process.env.AUTH_PASS || 'meddok2026';
const CACHE = path.join(__dirname, '..', 'temp', 'qa_with_raw.txt');

function extractRawSections(text) {
  const out = [];
  const labelRe = /^\[(.+?)\] \(\d+ KB\)/mu;
  const parts = text.split(/^={40,}$/mu).map(p => p.trim()).filter(Boolean);
  for (let i = 0; i < parts.length - 1; i++) {
    const m = parts[i].match(labelRe);
    if (!m) continue;
    const label = m[1];
    const next = parts[i + 1];
    const rawMatch = next.match(/--- RAW WHISPER TEXT \[\d+ chars\] ---\s*\n([\s\S]*?)(?=\n---\s|\n={40,}|$)/u);
    if (!rawMatch) continue;
    out.push({ label, raw: rawMatch[1].trim() });
  }
  return out;
}

let authToken = null;

async function login() {
  const resp = await fetch(`${SERVER}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password: AUTH_PASS }),
  });
  const data = await resp.json();
  if (!data.token) throw new Error('Login failed: ' + JSON.stringify(data));
  authToken = data.token;
  console.log('Logged in successfully');
}

async function structure(label, rawText) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`[${label}] raw=${rawText.length} chars`);
  console.log('='.repeat(60));

  const start = Date.now();
  const resp = await fetch(`${SERVER}/api/structure`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${authToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: rawText }),
  });
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);

  if (!resp.ok) {
    const t = await resp.text();
    console.error(`ERROR ${resp.status}: ${t}`);
    return;
  }
  const data = await resp.json();
  console.log(`Total: ${elapsed}s`);

  const doc = data.document || {};
  const fields = [
    'complaints', 'anamnesis', 'outpatientExams', 'clinicalCourse',
    'allergyHistory', 'objectiveStatus', 'neurologicalStatus',
    'diagnosis', 'finalDiagnosis', 'conclusion',
    'doctorNotes', 'recommendations', 'riskAssessment'
  ];
  for (const f of fields) {
    const v = doc[f];
    if (v === undefined || v === null) continue;
    const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
    console.log(`\n--- ${f} [${s.length} chars] ---`);
    console.log(s.length ? s : '(empty)');
  }
}

async function main() {
  const text = fs.readFileSync(CACHE, 'utf-8');
  const sections = extractRawSections(text);
  if (!sections.length) throw new Error('No raw sections extracted');
  console.log(`Found ${sections.length} raw sections`);
  await login();
  for (const s of sections) {
    await structure(s.label, s.raw);
  }
  console.log(`\n${'='.repeat(80)}`);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
