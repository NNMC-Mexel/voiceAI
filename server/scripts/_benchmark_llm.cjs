#!/usr/bin/env node
/* eslint-disable */
// Ad-hoc benchmark: сравнить LLM-модели (qwen3.5:9b vs deepseek-r1:14b и т.д.)
// на КЭШИРОВАННЫХ whisper-выходах (server/temp/_real_whisper/*_whisper.txt).
// Не коммитить — это временный замерочный скрипт, не часть прод-тестов.

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const SECRET = process.env.JWT_SECRET || 'dev-insecure-secret-change-in-production';

function b64url(buf) {
  return Buffer.from(buf).toString('base64').replace(/=+$/, '').replace(/\+/g, '-').replace(/\//g, '_');
}
function signJwt(payload, secret) {
  const header = { alg: 'HS256', typ: 'JWT' };
  const now = Math.floor(Date.now() / 1000);
  const body = { ...payload, iat: now, exp: now + 2 * 60 * 60 };
  const head = b64url(JSON.stringify(header));
  const pl   = b64url(JSON.stringify(body));
  const sig  = b64url(crypto.createHmac('sha256', secret).update(`${head}.${pl}`).digest());
  return `${head}.${pl}.${sig}`;
}
const jwt = { sign: (payload, secret) => signJwt(payload, secret) };
const API = process.env.API_URL || 'http://127.0.0.1:3001';
const WHISPER_DIR = path.join(__dirname, '..', 'temp', '_real_whisper');
const OUT_DIR = path.join(__dirname, '_bench_out');
const CASES = ['cardio', 'dyusenov', 'endo', 'eszhanov', 'kstalasova', 'labs_only'];

const TEXT_FIELDS = [
  'complaints', 'anamnesis', 'outpatientExams', 'clinicalCourse',
  'allergyHistory', 'objectiveStatus', 'neurologicalStatus',
  'diagnosis', 'finalDiagnosis', 'conclusion', 'doctorNotes', 'recommendations',
];

if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

const token = jwt.sign(
  { doctorId: 1, email: 'rustam35136@gmail.com', name: 'Уалиев Рустам', role: 'admin' },
  SECRET,
  { expiresIn: '2h' }
);

const tag = process.argv[2] || 'unknown';

(async () => {
  console.log(`Benchmark tag=${tag}, API=${API}, cases=${CASES.length}`);
  const results = [];
  for (const id of CASES) {
    const txtPath = path.join(WHISPER_DIR, `${id}_whisper.txt`);
    if (!fs.existsSync(txtPath)) {
      console.log(`SKIP ${id}: no file`);
      continue;
    }
    const text = fs.readFileSync(txtPath, 'utf8').trim();
    if (!text) {
      console.log(`SKIP ${id}: empty file`);
      continue;
    }
    const inputLen = text.length;
    process.stdout.write(`>> ${id}  in=${inputLen} ... `);
    const t0 = Date.now();
    let r;
    try {
      const res = await fetch(`${API}/api/structure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ text }),
      });
      const elapsed = Date.now() - t0;
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        console.log(`FAIL ${res.status}: ${(data && (data.error || data.message)) || res.statusText} (${elapsed}ms)`);
        r = { id, ok: false, elapsed, status: res.status, error: (data && (data.error || data.message)) || res.statusText };
      } else {
        const doc = (data && data.document) || data || {};
        const filled = TEXT_FIELDS.filter((f) => typeof doc[f] === 'string' && doc[f].trim().length > 5);
        const outChars = TEXT_FIELDS.reduce((s, f) => s + (typeof doc[f] === 'string' ? doc[f].length : 0), 0);
        console.log(`OK ${elapsed}ms  in=${inputLen} out=${outChars} fields=${filled.length}/${TEXT_FIELDS.length}`);
        r = { id, ok: true, elapsed, inputLen, outChars, filledFields: filled.length, filledList: filled, doc };
        fs.writeFileSync(path.join(OUT_DIR, `${tag}_${id}.json`), JSON.stringify(doc, null, 2));
      }
    } catch (e) {
      console.log(`ERR: ${e.message}`);
      r = { id, ok: false, error: e.message };
    }
    results.push(r);
  }

  const ok = results.filter((r) => r.ok);
  const avgMs = ok.length ? Math.round(ok.reduce((s, r) => s + r.elapsed, 0) / ok.length) : 0;
  const avgFilled = ok.length ? (ok.reduce((s, r) => s + r.filledFields, 0) / ok.length).toFixed(1) : '0';
  const totalChars = ok.reduce((s, r) => s + r.outChars, 0);
  console.log('---');
  console.log(`Summary [${tag}]: ${ok.length}/${results.length} ok, avg ${avgMs}ms, avg ${avgFilled}/${TEXT_FIELDS.length} fields, total out ${totalChars} chars`);

  fs.writeFileSync(
    path.join(OUT_DIR, `_summary_${tag}.json`),
    JSON.stringify({ tag, when: new Date().toISOString(), summary: { ok: ok.length, total: results.length, avgMs, avgFilled, totalChars }, results }, null, 2)
  );
})();
