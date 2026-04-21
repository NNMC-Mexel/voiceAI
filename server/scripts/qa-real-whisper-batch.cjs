#!/usr/bin/env node
/**
 * Прогоняет все файлы из temp/_real_whisper/*.txt через /api/structure.
 * Для каждого:
 *   - выводит длину raw + длину каждого поля
 *   - coverage %: доля предложений raw, находимых в финальном doc
 *   - список пропавших предложений (топ 5)
 * Результаты сохраняет в temp/_real_whisper/<slug>.result.json
 */
const fs = require('fs');
const path = require('path');

const SERVER = process.env.SERVER_URL || 'http://localhost:3001';
const AUTH_PASS = process.env.AUTH_PASS || 'meddok2026';
const DIR = path.resolve(__dirname, '..', 'temp', '_real_whisper');

const FIELDS = [
  'complaints', 'anamnesis', 'outpatientExams', 'clinicalCourse',
  'allergyHistory', 'objectiveStatus', 'neurologicalStatus',
  'diagnosis', 'finalDiagnosis', 'conclusion', 'doctorNotes', 'recommendations',
];

function normalize(s) {
  return String(s || '').toLowerCase().replace(/ё/g, 'е')
    .replace(/[^a-zа-я0-9\s]/gu, ' ').replace(/\s+/g, ' ').trim();
}

function coverageReport(raw, doc) {
  const docAll = FIELDS.map((f) => doc[f] || '').join(' ');
  const nDoc = normalize(docAll);
  const sents = raw.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter((s) => s.length >= 15);
  const missing = [];
  let found = 0;
  for (const s of sents) {
    const ns = normalize(s);
    if (ns.length < 15) continue;
    const fp1 = ns.substring(0, Math.min(25, ns.length));
    if (nDoc.includes(fp1)) { found++; continue; }
    const mid = Math.max(0, Math.floor(ns.length / 2) - 8);
    const fp2 = ns.substring(mid, mid + 15);
    if (fp2.length >= 12 && nDoc.includes(fp2)) { found++; continue; }
    missing.push(s);
  }
  return { total: sents.length, found, missing };
}

async function structureOne(label, raw, token) {
  const start = Date.now();
  const resp = await fetch(`${SERVER}/api/structure`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: raw }),
  });
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  if (!resp.ok) throw new Error(`${label}: ${resp.status} ${await resp.text()}`);
  const data = await resp.json();
  return { data, elapsed };
}

async function main() {
  const loginResp = await fetch(`${SERVER}/api/auth/login`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password: AUTH_PASS }),
  });
  const { token } = await loginResp.json();

  const files = fs.readdirSync(DIR).filter((f) => f.endsWith('.txt')).sort();
  console.log(`Running ${files.length} real Whisper test cases\n`);

  for (const f of files) {
    const raw = fs.readFileSync(path.join(DIR, f), 'utf-8').trim();
    const label = f.replace(/\.txt$/, '');

    console.log(`\n${'='.repeat(72)}\n[${label}]  raw=${raw.length} chars\n${'='.repeat(72)}`);
    let result;
    try {
      result = await structureOne(label, raw, token);
    } catch (e) {
      console.error(`  FAIL: ${e.message}`);
      continue;
    }
    const doc = result.data.document || {};
    if (typeof doc.patient === 'string') { try { doc.patient = JSON.parse(doc.patient); } catch {} }
    if (typeof doc.riskAssessment === 'string') { try { doc.riskAssessment = JSON.parse(doc.riskAssessment); } catch {} }

    console.log(`  time: ${result.elapsed}s`);
    console.log(`  fields filled:`);
    for (const fld of FIELDS) {
      const v = doc[fld];
      const len = typeof v === 'string' ? v.length : 0;
      const flag = len === 0 ? '  ∅' : '   ';
      console.log(`    ${flag} ${fld.padEnd(18)} ${len.toString().padStart(5)} chars`);
    }

    const cov = coverageReport(raw, doc);
    const pct = cov.total ? ((cov.found / cov.total) * 100).toFixed(0) : '0';
    console.log(`\n  COVERAGE: ${cov.found}/${cov.total} sentences (${pct}%)`);
    if (cov.missing.length > 0) {
      console.log(`  MISSING (top 6):`);
      for (const m of cov.missing.slice(0, 6)) {
        console.log(`    · ${m.substring(0, 120)}${m.length > 120 ? '…' : ''}`);
      }
    }

    // Сохраняем структурный результат для последующего ручного анализа
    const outJson = path.join(DIR, `${label}.result.json`);
    fs.writeFileSync(outJson, JSON.stringify({ raw, result: doc, coverage: cov, elapsed: result.elapsed }, null, 2), 'utf-8');
  }
}

main().catch((e) => { console.error('Fatal:', e); process.exit(1); });
