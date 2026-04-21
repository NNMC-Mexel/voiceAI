#!/usr/bin/env node
/**
 * Прогоняет raw Whisper (из temp/_debug_*_raw.txt) через /api/structure,
 * печатает ПОЛНУЮ раскладку по полям документа + riskAssessment,
 * чтобы было видно: какое поле Claude заполнил, какое пусто, что галлюцинировал.
 *
 * Цель: диагностика путаницы полей (заключение в recommendations, шкала Морзе
 * как галлюцинация) на уровне выхода /api/structure ДО rendering.
 */
const fs = require('fs');
const path = require('path');

const SERVER = process.env.SERVER_URL || 'http://localhost:3001';
const AUTH_PASS = process.env.AUTH_PASS || 'meddok2026';
const RAW_FILE = process.argv[2] || path.resolve(__dirname, '..', 'temp', '_debug_1776677497623_raw.txt');

const FIELDS = [
  'complaints','anamnesis','clinicalCourse','allergyHistory','objectiveStatus',
  'neurologicalStatus','outpatientExams','diagnosis','finalDiagnosis','conclusion',
  'recommendations','doctorNotes',
];

async function main() {
  const raw = fs.readFileSync(RAW_FILE, 'utf-8').trim();
  console.log(`[probe] raw input: ${raw.length} chars from ${path.basename(RAW_FILE)}`);

  const loginResp = await fetch(`${SERVER}/api/auth/login`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password: AUTH_PASS }),
  });
  const { token } = await loginResp.json();

  const start = Date.now();
  const resp = await fetch(`${SERVER}/api/structure`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: raw }),
  });
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);

  if (!resp.ok) {
    console.error(`FAIL ${resp.status}: ${await resp.text()}`);
    process.exit(1);
  }
  const data = await resp.json();
  const doc = data.document || {};

  console.log(`[probe] elapsed: ${elapsed}s`);
  console.log(`\n── PATIENT ──`);
  console.log(JSON.stringify(doc.patient, null, 2));

  console.log(`\n── RISK ASSESSMENT (шкала Морзе, возможная галлюцинация) ──`);
  console.log(JSON.stringify(doc.riskAssessment, null, 2));

  console.log(`\n── TEXT FIELDS ──`);
  for (const f of FIELDS) {
    const v = doc[f];
    const len = typeof v === 'string' ? v.length : 0;
    const flag = len === 0 ? '∅   ' : '    ';
    console.log(`\n${flag}${f.padEnd(20)} (${len} chars)`);
    if (v) console.log(`    ${v.split('\n').join('\n    ')}`);
  }

  // Save for later inspection
  const outPath = RAW_FILE.replace('_raw.txt', '_structure_result.json');
  fs.writeFileSync(outPath, JSON.stringify(doc, null, 2), 'utf-8');
  console.log(`\n[probe] saved full document to ${path.basename(outPath)}`);
}

main().catch((e) => { console.error('Fatal:', e); process.exit(1); });
