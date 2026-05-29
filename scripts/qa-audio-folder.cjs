#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = process.env.QA_AUDIO_ROOT || 'C:\\Users\\Айдар\\OneDrive\\Рабочий стол\\ВЫПИСКИ';
const SERVER = (process.env.SERVER_URL || 'http://127.0.0.1:3001').replace(/\/+$/, '');
const OUT_DIR = process.env.QA_OUT_DIR || path.resolve('temp', 'qa-audio-folder');
const LIMIT = Number.parseInt(process.env.QA_LIMIT || '0', 10);
const SKIP = Number.parseInt(process.env.QA_SKIP || '0', 10);
const START_AFTER = process.env.QA_START_AFTER || '';
const ONLY = process.env.QA_ONLY || '';
const LIST_FILE = process.env.QA_LIST_FILE || '';
const RESUME = process.env.QA_RESUME === 'true';
const USE_JOBS = process.env.QA_USE_JOBS === 'true';
const JOB_POLL_MS = Number.parseInt(process.env.QA_JOB_POLL_MS || '2000', 10);
const JOB_TIMEOUT_MS = Number.parseInt(process.env.QA_JOB_TIMEOUT_MS || `${15 * 60 * 1000}`, 10);
const AUDIO_RE = /\.(wav|mp3|ogg|webm|m4a)$/i;
const FIELDS = [
  'complaints',
  'anamnesis',
  'outpatientExams',
  'clinicalCourse',
  'allergyHistory',
  'objectiveStatus',
  'neurologicalStatus',
  'diagnosis',
  'finalDiagnosis',
  'conclusion',
  'doctorNotes',
  'recommendations',
];

function b64url(input) {
  return Buffer.from(input).toString('base64url');
}

function signJwt(payload, secret) {
  const header = { alg: 'HS256', typ: 'JWT' };
  const body = {
    ...payload,
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + 24 * 60 * 60,
  };
  const unsigned = `${b64url(JSON.stringify(header))}.${b64url(JSON.stringify(body))}`;
  const sig = crypto.createHmac('sha256', secret).update(unsigned).digest('base64url');
  return `${unsigned}.${sig}`;
}

function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, acc);
    else if (entry.isFile() && AUDIO_RE.test(entry.name)) acc.push(full);
  }
  return acc;
}

function pairText(audioPath) {
  const parsed = path.parse(audioPath);
  const candidates = [
    path.join(parsed.dir, `${parsed.name.replace(/_audio$/i, '')}_text.txt`),
    path.join(parsed.dir, `${parsed.name}_text.txt`),
  ];
  return candidates.find((p) => fs.existsSync(p)) || '';
}

function rel(p) {
  return path.relative(ROOT, p);
}

function loadListFilter() {
  if (!LIST_FILE) return null;
  const resolved = path.resolve(LIST_FILE);
  const wanted = fs.readFileSync(resolved, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  return new Set(wanted.map((item) => item.toLowerCase()));
}

function slug(s) {
  return s
    .replace(/[\\/:*?"<>|]+/g, '_')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 180);
}

function normalize(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}\s]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenSet(s) {
  return new Set(
    normalize(s)
      .split(/\s+/)
      .filter((x) => x.length >= 4),
  );
}

function coverage(reference, actual) {
  const ref = tokenSet(reference);
  const got = tokenSet(actual);
  if (!ref.size) return null;
  let found = 0;
  for (const t of ref) if (got.has(t)) found++;
  return Math.round((found / ref.size) * 100);
}

function sentenceCoverage(reference, actual) {
  const doc = normalize(actual);
  const docTokens = tokenSet(actual);
  const sents = String(reference || '')
    .split(/(?<=[.!?])\s+|\n+/u)
    .map((s) => s.trim())
    .filter((s) => normalize(s).length >= 25);
  let found = 0;
  const missing = [];
  for (const s of sents) {
    const ns = normalize(s);
    const probes = [
      ns.slice(0, Math.min(35, ns.length)),
      ns.slice(Math.max(0, Math.floor(ns.length / 2) - 18), Math.floor(ns.length / 2) + 18),
      ns.slice(Math.max(0, ns.length - 35)),
    ].filter((x) => x.length >= 18);
    const sentTokens = [...tokenSet(s)];
    const overlap = sentTokens.filter((t) => docTokens.has(t)).length;
    const enoughTokenOverlap = sentTokens.length >= 3 && overlap / sentTokens.length >= 0.55;
    if (probes.some((p) => doc.includes(p)) || enoughTokenOverlap) found++;
    else missing.push(s);
  }
  return {
    total: sents.length,
    found,
    percent: sents.length ? Math.round((found / sents.length) * 100) : null,
    missing: missing.slice(0, 8),
  };
}

function flattenDocument(doc) {
  const patient = doc?.patient || {};
  const risk = doc?.riskAssessment || {};
  return [
    patient.fullName || '',
    patient.age || '',
    patient.gender || '',
    patient.birthDate || '',
    risk.fallInLast3Months || '',
    risk.dizzinessOrWeakness || '',
    risk.needsEscort || '',
    risk.painScore || '',
    ...FIELDS.map((f) => doc?.[f] || ''),
  ].join('\n');
}

function hasMarker(text, marker) {
  return marker.test(normalize(text));
}

function isProcedureLike(text) {
  const n = normalize(text);
  return /\b(?:манипуляция|оптическая когерентная томография|окт|лазеркапсулотом|скан|узи|уздг|консультация витреоретинального хирурга)\b/u.test(n);
}

function analyzeResult(reference, data) {
  const transcription = data?.transcription?.text || '';
  const sourceText = reference || transcription;
  const doc = data?.document || {};
  const structuredText = flattenDocument(doc);
  const fieldLengths = Object.fromEntries(FIELDS.map((f) => [f, String(doc[f] || '').length]));
  const filled = FIELDS.filter((f) => fieldLengths[f] > 0);
  const issues = [];
  const expects = {
    patient: hasMarker(sourceText, /\b(?:фио|пациент|пациентка|больной|больная)\b/u),
    diagnosis: hasMarker(sourceText, /\b(?:диагноз|диагнос|дз|основной диагноз|клинический диагноз)\b/u),
    recommendations: hasMarker(sourceText, /\b(?:рекомендац|рекомендовано|назначени|лечение|лечени|план лечения|контроль|повторный осмотр|повторный прием)\b/u),
    complaints: hasMarker(sourceText, /\b(?:жалоб|жалобы|жалоба|шағым)\b/u),
  };
  const procedureLike = isProcedureLike(sourceText);

  if (!transcription.trim()) issues.push('empty_transcription');
  if (filled.length === 0) issues.push('empty_structured_document');
  if (expects.patient && !doc?.patient?.fullName) issues.push('patient_name_empty');
  if (expects.diagnosis && !procedureLike && !fieldLengths.diagnosis && !fieldLengths.finalDiagnosis) issues.push('diagnosis_empty');
  if (expects.recommendations && fieldLengths.recommendations === 0) issues.push('recommendations_empty');
  if (expects.complaints && !procedureLike && fieldLengths.complaints === 0) issues.push('complaints_empty');

  const sttCoverage = coverage(reference, transcription);
  const llmCoverage = coverage(reference, structuredText);
  const sentenceCov = sentenceCoverage(reference, structuredText);
  if (sttCoverage !== null && sttCoverage < 70) issues.push(`low_stt_token_coverage:${sttCoverage}`);
  const sttGoodEnoughForLlmQa = sttCoverage === null || sttCoverage >= 70;
  if (sttGoodEnoughForLlmQa && llmCoverage !== null && llmCoverage < 60) issues.push(`low_llm_token_coverage:${llmCoverage}`);
  if (sttGoodEnoughForLlmQa && sentenceCov.percent !== null && sentenceCov.percent < 50) issues.push(`low_llm_sentence_coverage:${sentenceCov.percent}`);

  return {
    patient: doc.patient || {},
    fieldLengths,
    filled,
    warnings: data?.warnings || [],
    qualityWarnings: data?.qualityWarnings || [],
    sttCoverage,
    llmCoverage,
    sentenceCoverage: sentenceCov,
    expects,
    procedureLike,
    issues,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function requestJson(url, options) {
  const resp = await fetch(url, options);
  const bodyText = await resp.text();
  let data;
  try {
    data = JSON.parse(bodyText);
  } catch {
    data = { rawBody: bodyText };
  }
  return { resp, data };
}

async function processViaJob(form, token) {
  const { resp: startResp, data: started } = await requestJson(`${SERVER}/api/jobs/audio`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });

  if (!startResp.ok || !started.jobId) {
    return { status: startResp.status, ok: false, data: started };
  }

  const deadline = Date.now() + JOB_TIMEOUT_MS;
  let lastJob = started;
  while (Date.now() < deadline) {
    await sleep(JOB_POLL_MS);
    const { resp, data } = await requestJson(`${SERVER}/api/jobs/${encodeURIComponent(started.jobId)}`, {
      method: 'GET',
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!resp.ok) {
      return { status: resp.status, ok: false, data };
    }
    lastJob = data;
    if (data.status === 'done' && data.result) {
      return { status: 200, ok: true, data: data.result, job: data };
    }
    if (data.status === 'failed') {
      return {
        status: data.statusCode || 500,
        ok: false,
        data: {
          success: false,
          error: data.error || 'audio_job_failed',
          message: data.message,
          transcription: data.transcription,
        },
        job: data,
      };
    }
  }

  return {
    status: 408,
    ok: false,
    data: {
      success: false,
      error: 'audio_job_timeout',
      message: `Job did not finish in ${JOB_TIMEOUT_MS}ms`,
      job: lastJob,
    },
    job: lastJob,
  };
}

async function processViaSync(form, token) {
  const { resp, data } = await requestJson(`${SERVER}/api/process`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  return { status: resp.status, ok: resp.ok, data };
}

async function processOne(file, token, index, total) {
  const textPath = pairText(file);
  const reference = textPath ? fs.readFileSync(textPath, 'utf-8') : '';
  const buf = fs.readFileSync(file);
  const form = new FormData();
  form.set('file', new Blob([buf]), path.basename(file));

  const started = Date.now();
  const processed = USE_JOBS ? await processViaJob(form, token) : await processViaSync(form, token);
  const elapsedSec = Math.round((Date.now() - started) / 100) / 10;
  const data = processed.data;

  const base = {
    index,
    total,
    audioPath: file,
    relativePath: rel(file),
    textPath,
    textRelativePath: textPath ? rel(textPath) : '',
    status: processed.status,
    ok: processed.ok,
    mode: USE_JOBS ? 'job' : 'sync',
    jobId: processed.job?.id || '',
    jobStatus: processed.job?.status || '',
    elapsedSec,
    audioBytes: buf.length,
  };

  const result = processed.ok
    ? { ...base, analysis: analyzeResult(reference, data), response: data }
    : { ...base, analysis: { issues: [`http_${processed.status}`] }, response: data };

  const outName = `${String(index).padStart(4, '0')}_${slug(rel(file))}.json`;
  fs.writeFileSync(path.join(OUT_DIR, outName), JSON.stringify(result, null, 2), 'utf-8');
  return result;
}

function writeReports(results) {
  const jsonlPath = path.join(OUT_DIR, 'results.jsonl');
  fs.writeFileSync(jsonlPath, results.map((r) => JSON.stringify(r)).join('\n') + '\n', 'utf-8');

  const csvRows = [
    [
      'index',
      'ok',
      'status',
      'elapsedSec',
      'relativePath',
      'hasText',
      'sttCoverage',
      'llmCoverage',
      'llmSentenceCoverage',
      'patient',
      'filledFields',
      'warnings',
      'issues',
    ],
  ];
  for (const r of results) {
    const a = r.analysis || {};
    csvRows.push([
      r.index,
      r.ok,
      r.status,
      r.elapsedSec,
      r.relativePath,
      Boolean(r.textPath),
      a.sttCoverage ?? '',
      a.llmCoverage ?? '',
      a.sentenceCoverage?.percent ?? '',
      a.patient?.fullName || '',
      (a.filled || []).join('|'),
      (a.warnings || []).join('|'),
      (a.issues || []).join('|'),
    ]);
  }
  const csv = csvRows
    .map((row) => row.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','))
    .join('\n');
  fs.writeFileSync(path.join(OUT_DIR, 'summary.csv'), csv, 'utf-8');

  const issueCounts = new Map();
  for (const r of results) {
    for (const issue of r.analysis?.issues || []) {
      issueCounts.set(issue, (issueCounts.get(issue) || 0) + 1);
    }
  }
  const md = [
    '# QA audio folder report',
    '',
    `Root: ${ROOT}`,
    `Server: ${SERVER}`,
    `Processed: ${results.length}`,
    `Success: ${results.filter((r) => r.ok).length}`,
    `Failed: ${results.filter((r) => !r.ok).length}`,
    '',
    '## Issue counts',
    '',
    ...[...issueCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([issue, count]) => `- ${issue}: ${count}`),
    '',
    '## Worst cases',
    '',
    ...results
      .filter((r) => r.analysis?.issues?.length)
      .slice(0, 30)
      .map((r) => `- #${r.index} ${r.relativePath}: ${(r.analysis.issues || []).join(', ')}`),
    '',
  ].join('\n');
  fs.writeFileSync(path.join(OUT_DIR, 'report.md'), md, 'utf-8');
}

function loadExistingResults() {
  const jsonlPath = path.join(OUT_DIR, 'results.jsonl');
  if (!RESUME || !fs.existsSync(jsonlPath)) return [];
  return fs.readFileSync(jsonlPath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line.replace(/^\uFEFF/, '')));
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const token = process.env.AUTH_TOKEN || signJwt({
    doctorId: Number.parseInt(process.env.QA_DOCTOR_ID || '1', 10),
    email: process.env.QA_DOCTOR_EMAIL || 'rustam35136@gmail.com',
    name: process.env.QA_DOCTOR_NAME || 'QA',
    role: process.env.QA_DOCTOR_ROLE || 'admin',
  }, process.env.JWT_SECRET || 'dev-insecure-secret-change-in-production');

  let files = walk(ROOT).sort((a, b) => rel(a).localeCompare(rel(b), 'ru'));
  const listFilter = loadListFilter();
  if (listFilter) {
    files = files.filter((f) => {
      const absolute = path.resolve(f).toLowerCase();
      const relative = rel(f).toLowerCase();
      return listFilter.has(absolute) || listFilter.has(relative);
    });
  }
  if (ONLY) files = files.filter((f) => rel(f).toLowerCase().includes(ONLY.toLowerCase()));
  if (SKIP > 0) files = files.slice(SKIP);
  if (START_AFTER) {
    const pos = files.findIndex((f) => rel(f) === START_AFTER || f === START_AFTER);
    if (pos >= 0) files = files.slice(pos + 1);
  }
  if (LIMIT > 0) files = files.slice(0, LIMIT);

  console.log(`[qa] root=${ROOT}`);
  console.log(`[qa] server=${SERVER}`);
  console.log(`[qa] out=${OUT_DIR}`);
  console.log(`[qa] files=${files.length}`);
  console.log(`[qa] resume=${RESUME}`);

  const results = loadExistingResults();
  const done = new Set(results.map((r) => r.relativePath));
  if (results.length) {
    console.log(`[qa] loaded existing results=${results.length}`);
  }
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const label = rel(file);
    if (done.has(label)) {
      console.log(`\n[${i + 1}/${files.length}] SKIP existing ${label}`);
      continue;
    }
    console.log(`\n[${i + 1}/${files.length}] ${label}`);
    try {
      const result = await processOne(file, token, i + 1, files.length);
      results.push(result);
      const a = result.analysis || {};
      console.log(`  status=${result.status} elapsed=${result.elapsedSec}s stt=${a.sttCoverage ?? '-'} llm=${a.llmCoverage ?? '-'} issues=${(a.issues || []).join('|') || '-'}`);
    } catch (err) {
      const result = {
        index: i + 1,
        total: files.length,
        audioPath: file,
        relativePath: label,
        ok: false,
        status: 0,
        elapsedSec: 0,
        analysis: { issues: [`exception:${err.message}`] },
      };
      results.push(result);
      fs.writeFileSync(path.join(OUT_DIR, `${String(i + 1).padStart(4, '0')}_${slug(label)}.error.json`), JSON.stringify(result, null, 2), 'utf-8');
      console.error(`  ERROR ${err.stack || err.message}`);
    }
    writeReports(results);
  }

  console.log(`\n[qa] done. Report: ${path.join(OUT_DIR, 'report.md')}`);
  console.log(`[qa] summary: ${path.join(OUT_DIR, 'summary.csv')}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
