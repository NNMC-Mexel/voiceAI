#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const AUDIO_EXT_RE = /\.wav$/i;
const TEXT_FIELDS = [
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

const SECTION_TO_FIELD = [
  { field: 'complaints', re: /жалоб[ыа)]|шағым/i },
  { field: 'anamnesis', re: /анамнез\s+заболев|ауру\s+анамнез/i },
  { field: 'clinicalCourse', re: /перенесенн|перенесённ|анамнез\s+жизни|бастан/i },
  { field: 'allergyHistory', re: /аллерголог|аллергиялық/i },
  { field: 'objectiveStatus', re: /объектив|объективті/i },
  { field: 'neurologicalStatus', re: /невролог/i },
  { field: 'outpatientExams', re: /обследован|анализ|оак|оам|биохим|экг|эхо|узи|рентген|кт|мрт/i },
  { field: 'diagnosis', re: /предварительн[а-я\s]*диагноз|диагноз/i },
  { field: 'finalDiagnosis', re: /заключительн[а-я\s]*диагноз|қорытынды/i },
  { field: 'doctorNotes', re: /план\s+обследован|тактика/i },
  { field: 'recommendations', re: /рекомендац|ұсыным/i },
];

const COMMON = new Set([
  'пациент', 'пациентка', 'жалобы', 'анамнез', 'диагноз', 'объективный', 'статус',
  'рекомендации', 'лечение', 'данные', 'исследования', 'года', 'лет', 'при',
  'для', 'или', 'без', 'после', 'перед', 'есть', 'нет', 'день', 'дня', 'дней',
  'раз', 'раза', 'врач', 'осмотр', 'состояние',
]);

function parseArgs(argv) {
  const args = {
    root: 'C:\\Users\\AI\\Desktop\\prepared_blind_test',
    server: process.env.SERVER_URL || 'http://127.0.0.1:1337',
    out: '',
    concurrency: 1,
    limit: 0,
    offset: 0,
    email: process.env.QA_EMAIL || 'blind.qa@example.test',
    password: process.env.QA_PASSWORD || 'BlindQaPassword123!',
    retry: 1,
    infraRetries: Number(process.env.QA_INFRA_RETRIES || 720),
    infraDelayMs: Number(process.env.QA_INFRA_DELAY_MS || 5000),
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const read = () => argv[++i];
    if (a === '--root') args.root = read();
    else if (a.startsWith('--root=')) args.root = a.slice(7);
    else if (a === '--server') args.server = read();
    else if (a.startsWith('--server=')) args.server = a.slice(9);
    else if (a === '--out') args.out = read();
    else if (a.startsWith('--out=')) args.out = a.slice(6);
    else if (a === '--concurrency') args.concurrency = Number(read()) || 1;
    else if (a.startsWith('--concurrency=')) args.concurrency = Number(a.slice(14)) || 1;
    else if (a === '--limit') args.limit = Number(read()) || 0;
    else if (a.startsWith('--limit=')) args.limit = Number(a.slice(8)) || 0;
    else if (a === '--offset') args.offset = Number(read()) || 0;
    else if (a.startsWith('--offset=')) args.offset = Number(a.slice(9)) || 0;
    else if (a === '--email') args.email = read();
    else if (a === '--password') args.password = read();
    else if (a === '--retry') args.retry = Number(read()) || 0;
    else if (a === '--infra-retries') args.infraRetries = Number(read()) || 0;
    else if (a === '--infra-delay-ms') args.infraDelayMs = Number(read()) || 1000;
    else if (a === '--help' || a === '-h') {
      console.log('Usage: node scripts/qa-blind-batch.cjs --root <prepared_blind_test> [--out <dir>] [--concurrency 1] [--limit N]');
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${a}`);
    }
  }
  args.root = path.resolve(args.root);
  if (!args.out) {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    args.out = path.resolve('temp', 'blind-qa', stamp);
  } else {
    args.out = path.resolve(args.out);
  }
  args.concurrency = Math.max(1, Math.min(4, args.concurrency));
  args.infraRetries = Math.max(0, args.infraRetries);
  args.infraDelayMs = Math.max(1000, args.infraDelayMs);
  return args;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      row.push(cell);
      cell = '';
    } else if (ch === '\n') {
      row.push(cell.replace(/\r$/, ''));
      rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += ch;
    }
  }
  if (cell || row.length) {
    row.push(cell.replace(/\r$/, ''));
    rows.push(row);
  }
  const [header, ...data] = rows;
  return data
    .filter((r) => r.length && r.some(Boolean))
    .map((r) => Object.fromEntries(header.map((h, idx) => [h, r[idx] ?? ''])));
}

function readUtf8(file) {
  return fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, '');
}

function loadCases(root) {
  const manifest = path.join(root, '_manifest.csv');
  if (fs.existsSync(manifest)) {
    return parseCsv(readUtf8(manifest))
      .filter((r) => r.Audio && AUDIO_EXT_RE.test(r.Audio))
      .map((r, idx) => ({
        id: String(idx + 1).padStart(4, '0'),
        doctor: r.Doctor,
        patient: r.Patient,
        textPath: path.join(root, r.Text),
        sourceTextPath: r.SourceText ? path.join(root, r.SourceText) : '',
        audioPath: path.join(root, r.Audio),
        textChars: Number(r.TextChars) || 0,
        audioBytes: Number(r.AudioBytes) || 0,
      }));
  }

  const files = [];
  walk(root, files);
  return files
    .filter((f) => AUDIO_EXT_RE.test(f))
    .sort((a, b) => a.localeCompare(b, 'ru'))
    .map((audioPath, idx) => {
      const base = audioPath.replace(/_audio\.wav$/i, '');
      return {
        id: String(idx + 1).padStart(4, '0'),
        doctor: path.relative(root, audioPath).split(path.sep)[0] || '',
        patient: path.relative(root, audioPath).split(path.sep)[1] || '',
        textPath: `${base}_text.txt`,
        sourceTextPath: `${base}_source_text.txt`,
        audioPath,
        textChars: 0,
        audioBytes: fs.statSync(audioPath).size,
      };
    });
}

function walk(dir, out) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else if (entry.isFile()) out.push(full);
  }
}

function normalize(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}/.,%+-]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokens(text) {
  return normalize(text)
    .split(/\s+/u)
    .filter((t) => t.length >= 2);
}

function contentTokens(text) {
  return tokens(text).filter((t) => t.length >= 4 && !COMMON.has(t) && !/^\d+$/.test(t));
}

function multisetMetrics(expected, actual) {
  const exp = contentTokens(expected);
  const act = contentTokens(actual);
  const actCounts = new Map();
  for (const t of act) actCounts.set(t, (actCounts.get(t) || 0) + 1);
  let hit = 0;
  for (const t of exp) {
    const c = actCounts.get(t) || 0;
    if (c > 0) {
      hit += 1;
      actCounts.set(t, c - 1);
    }
  }
  const precision = act.length ? hit / act.length : 0;
  const recall = exp.length ? hit / exp.length : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return {
    expectedTokens: exp.length,
    actualTokens: act.length,
    hit,
    precision: round(precision * 100),
    recall: round(recall * 100),
    f1: round(f1 * 100),
  };
}

function levenshtein(a, b, maxLen = 24) {
  if (a === b) return 0;
  if (!a || !b) return Math.max(a.length, b.length);
  if (Math.max(a.length, b.length) > maxLen) return maxLen + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    const curr = [i];
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    prev = curr;
  }
  return prev[b.length];
}

function extractNumbers(text) {
  return (String(text || '').match(/\d+(?:[.,]\d+)?(?:\s*\/\s*\d+(?:[.,]\d+)?)?/g) || [])
    .map((n) => n.replace(/\s+/g, '').replace(',', '.'))
    .filter((n) => n.length > 0);
}

function numberRecall(expected, actual) {
  const exp = extractNumbers(expected);
  const actSet = new Set(extractNumbers(actual));
  const meaningful = exp.filter((n) => !/^(?:0|1|2|3|4|5|10)$/.test(n));
  const uniq = [...new Set(meaningful)];
  const hit = uniq.filter((n) => actSet.has(n)).length;
  return { total: uniq.length, hit, recall: uniq.length ? round((hit / uniq.length) * 100) : null };
}

function parseSections(text) {
  const lines = String(text || '').split(/\r?\n/u);
  const sections = {};
  let current = 'unmapped';
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    const heading = SECTION_TO_FIELD.find((s) => s.re.test(line.slice(0, 120)));
    if (heading) current = heading.field;
    sections[current] = sections[current] ? `${sections[current]}\n${line}` : line;
  }
  return sections;
}

function documentText(doc) {
  if (!doc || typeof doc !== 'object') return '';
  const patient = doc.patient && typeof doc.patient === 'object' ? doc.patient : {};
  return [
    patient.fullName, patient.age, patient.gender, patient.complaintDate,
    ...TEXT_FIELDS.map((field) => doc[field]),
  ].filter(Boolean).join('\n');
}

function sectionMetrics(expectedText, doc) {
  const sections = parseSections(expectedText);
  const out = {};
  for (const [field, sectionText] of Object.entries(sections)) {
    if (field === 'unmapped') continue;
    const expectedLen = normalize(sectionText).length;
    if (expectedLen < 20) continue;
    out[field] = {
      expectedChars: sectionText.length,
      actualChars: String(doc?.[field] || '').length,
      ...multisetMetrics(sectionText, doc?.[field] || ''),
    };
  }
  return out;
}

function classify(result) {
  if (result.error) return 'FAIL';
  const w = result.whisper || {};
  const llm = result.llm || {};
  if ((w.recall ?? 100) < 55 || (w.f1 ?? 100) < 55) return 'FAIL';
  if ((llm.documentRecall ?? 100) < 35) return 'FAIL';
  if ((w.recall ?? 100) < 72 || (llm.documentRecall ?? 100) < 50 || (result.apiWarnings || []).length) return 'WARN';
  return 'PASS';
}

function collectSubstitutionCandidates(expected, actual, bucket) {
  const exp = contentTokens(expected).filter((t) => /[а-яё]/i.test(t) && t.length >= 5);
  const act = contentTokens(actual).filter((t) => /[а-яё]/i.test(t) && t.length >= 5);
  const window = 8;
  const limit = Math.min(exp.length, 2500);
  for (let i = 0; i < limit; i += 1) {
    const correct = exp[i];
    if (act.includes(correct)) continue;
    const start = Math.max(0, i - window);
    const end = Math.min(act.length, i + window + 1);
    let best = null;
    for (let j = start; j < end; j += 1) {
      const wrong = act[j];
      if (wrong === correct || wrong.length < 5) continue;
      const d = levenshtein(wrong, correct);
      const maxAllowed = Math.max(2, Math.floor(Math.max(wrong.length, correct.length) * 0.34));
      if (d <= maxAllowed && (!best || d < best.d)) best = { wrong, correct, d };
    }
    if (best) {
      const key = `${best.wrong}\t${best.correct}`;
      const item = bucket.get(key) || { wrong: best.wrong, correct: best.correct, count: 0, cases: [] };
      item.count += 1;
      bucket.set(key, item);
    }
  }
}

function round(n) {
  return Math.round(n * 10) / 10;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { raw: text };
  }
  if (!response.ok) {
    const err = new Error(`HTTP ${response.status}: ${JSON.stringify(body).slice(0, 500)}`);
    err.status = response.status;
    err.body = body;
    throw err;
  }
  return body;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isInfrastructureError(error) {
  const status = Number(error?.status || 0);
  const message = String(error?.message || error || '').toLowerCase();
  return (
    status === 0 ||
    status === 429 ||
    status >= 500 ||
    message.includes('fetch failed') ||
    message.includes('econnrefused') ||
    message.includes('econnreset') ||
    message.includes('etimedout') ||
    message.includes('socket') ||
    message.includes('terminated') ||
    message.includes('network') ||
    message.includes('timeout')
  );
}

async function waitForHealth(args, label) {
  let last = '';
  for (let attempt = 1; attempt <= args.infraRetries; attempt += 1) {
    try {
      const health = await requestJson(`${args.server}/api/health`);
      const services = health.services || {};
      if (!services.whisper || services.whisper === 'ready') {
        return health;
      }
      last = `health=${JSON.stringify(services)}`;
    } catch (error) {
      last = error instanceof Error ? error.message : String(error);
    }
    if (attempt === 1 || attempt % 12 === 0) {
      console.warn(`[infra] ${label}: waiting for API health (${attempt}/${args.infraRetries}) ${last}`);
    }
    await sleep(args.infraDelayMs);
  }
  throw new Error(`Infrastructure unavailable after ${args.infraRetries} health checks: ${last}`);
}

async function withInfrastructureRetry(fn, args, label) {
  let last;
  for (let attempt = 0; attempt <= args.infraRetries; attempt += 1) {
    try {
      return await withRetry(fn, args.retry);
    } catch (error) {
      last = error;
      if (!isInfrastructureError(error) || attempt >= args.infraRetries) break;
      const msg = error instanceof Error ? error.message : String(error);
      console.warn(`[infra] ${label}: ${msg}; retrying same case after health check`);
      await waitForHealth(args, label);
      await sleep(args.infraDelayMs);
    }
  }
  throw last;
}

async function login(args) {
  const setup = await requestJson(`${args.server}/api/auth/setup-status`).catch(() => ({ setupRequired: false }));
  if (setup.setupRequired) {
    await requestJson(`${args.server}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Blind QA', email: args.email, password: args.password, specialty: 'QA' }),
    });
  }
  const data = await requestJson(`${args.server}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: args.email, password: args.password }),
  });
  if (!data.token) throw new Error('Login response has no token');
  return data.token;
}

async function processAudio(server, token, audioPath) {
  const audioBytes = await fs.promises.readFile(audioPath);
  const form = new FormData();
  form.append(
    'file',
    new Blob([audioBytes], { type: 'audio/wav' }),
    path.basename(audioPath),
  );
  const response = await fetch(`${server}/api/process`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  const text = await response.text();
  let parsed = null;
  try { parsed = text ? JSON.parse(text) : null; } catch {}
  if (!response.ok) {
    const err = new Error(`HTTP ${response.status}: ${text.slice(0, 500)}`);
    err.status = response.status;
    err.body = parsed || text;
    throw err;
  }
  if (!parsed) throw new Error(`Non-JSON response: ${text.slice(0, 500)}`);
  return parsed;
}

async function runCase(testCase, token, args, candidates) {
  const started = Date.now();
  const expectedText = fs.existsSync(testCase.textPath) ? readUtf8(testCase.textPath) : '';
  const sourceText = testCase.sourceTextPath && fs.existsSync(testCase.sourceTextPath)
    ? readUtf8(testCase.sourceTextPath)
    : '';
  const base = {
    id: testCase.id,
    doctor: testCase.doctor,
    patient: testCase.patient,
    audio: path.relative(args.root, testCase.audioPath),
    text: path.relative(args.root, testCase.textPath),
    audioBytes: testCase.audioBytes || fs.statSync(testCase.audioPath).size,
    expectedChars: expectedText.length,
    sourceChars: sourceText.length,
  };

  try {
    const api = await withInfrastructureRetry(
      () => processAudio(args.server, token, testCase.audioPath),
      args,
      `case ${testCase.id}`,
    );
    const transcript = api.transcription?.text || '';
    const doc = api.document || {};
    const docAll = documentText(doc);
    const sec = sectionMetrics(expectedText, doc);
    const lowSections = Object.entries(sec)
      .filter(([, m]) => m.expectedChars >= 80 && (m.recall ?? 0) < 35)
      .map(([field, m]) => `${field}:${m.recall}`);
    const result = {
      ...base,
      status: 'PASS',
      elapsedMs: Date.now() - started,
      processingTime: api.processingTime,
      apiWarnings: api.warnings || [],
      whisper: {
        chars: transcript.length,
        ...multisetMetrics(expectedText, transcript),
        numbers: numberRecall(expectedText, transcript),
        warnings: api.transcription?.warnings || [],
      },
      llm: {
        documentChars: docAll.length,
        documentRecall: multisetMetrics(expectedText, docAll).recall,
        numbers: numberRecall(expectedText, docAll),
        meaningfulFields: TEXT_FIELDS.filter((f) => String(doc[f] || '').trim().length >= 20),
        sectionMetrics: sec,
        lowSections,
      },
      sample: {
        expectedStart: normalize(expectedText).slice(0, 240),
        transcriptStart: normalize(transcript).slice(0, 240),
      },
    };
    result.status = classify(result);
    collectSubstitutionCandidates(expectedText, transcript, candidates);
    return result;
  } catch (error) {
    return {
      ...base,
      status: 'FAIL',
      elapsedMs: Date.now() - started,
      error: error instanceof Error ? error.message : String(error),
      errorBody: error && error.body ? error.body : undefined,
    };
  }
}

async function withRetry(fn, retries) {
  let last;
  for (let i = 0; i <= retries; i += 1) {
    try {
      return await fn();
    } catch (error) {
      last = error;
      if (i < retries) await new Promise((r) => setTimeout(r, 1500 * (i + 1)));
    }
  }
  throw last;
}

function loadDone(jsonlPath) {
  const done = new Set();
  if (!fs.existsSync(jsonlPath)) return done;
  for (const line of fs.readFileSync(jsonlPath, 'utf8').split(/\r?\n/u)) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      if (row.id) done.add(row.id);
    } catch {}
  }
  return done;
}

function aggregate(results) {
  const summary = {
    total: results.length,
    pass: 0,
    warn: 0,
    fail: 0,
    avgWhisperRecall: 0,
    avgWhisperF1: 0,
    avgLlmRecall: 0,
    avgElapsedSec: 0,
    warnings: {},
    worstWhisper: [],
    worstLlm: [],
    lowSections: {},
  };
  const ok = results.filter((r) => !r.error);
  for (const r of results) summary[String(r.status || 'fail').toLowerCase()] += 1;
  if (ok.length) {
    summary.avgWhisperRecall = round(ok.reduce((s, r) => s + (r.whisper?.recall || 0), 0) / ok.length);
    summary.avgWhisperF1 = round(ok.reduce((s, r) => s + (r.whisper?.f1 || 0), 0) / ok.length);
    summary.avgLlmRecall = round(ok.reduce((s, r) => s + (r.llm?.documentRecall || 0), 0) / ok.length);
    summary.avgElapsedSec = round(ok.reduce((s, r) => s + (r.elapsedMs || 0), 0) / ok.length / 1000);
  }
  for (const r of ok) {
    for (const w of r.apiWarnings || []) summary.warnings[w] = (summary.warnings[w] || 0) + 1;
    for (const item of r.llm?.lowSections || []) {
      const field = item.split(':')[0];
      summary.lowSections[field] = (summary.lowSections[field] || 0) + 1;
    }
  }
  summary.worstWhisper = [...ok]
    .sort((a, b) => (a.whisper?.recall || 0) - (b.whisper?.recall || 0))
    .slice(0, 25)
    .map((r) => ({ id: r.id, doctor: r.doctor, patient: r.patient, recall: r.whisper.recall, f1: r.whisper.f1, audio: r.audio }));
  summary.worstLlm = [...ok]
    .sort((a, b) => (a.llm?.documentRecall || 0) - (b.llm?.documentRecall || 0))
    .slice(0, 25)
    .map((r) => ({ id: r.id, doctor: r.doctor, patient: r.patient, recall: r.llm.documentRecall, warnings: r.apiWarnings, audio: r.audio }));
  return summary;
}

function writeReports(args, results, candidates) {
  const summary = aggregate(results);
  const candidateList = [...candidates.values()]
    .filter((c) => c.count >= 4 && c.wrong !== c.correct)
    .sort((a, b) => b.count - a.count)
    .slice(0, 200);
  fs.writeFileSync(path.join(args.out, 'summary.json'), JSON.stringify(summary, null, 2), 'utf8');
  fs.writeFileSync(path.join(args.out, 'dictionary-candidates.json'), JSON.stringify(candidateList, null, 2), 'utf8');

  const failures = results
    .filter((r) => r.status === 'FAIL' || r.status === 'WARN')
    .map((r) => [
      r.status, r.id, r.doctor, r.patient,
      r.whisper?.recall ?? '', r.whisper?.f1 ?? '', r.llm?.documentRecall ?? '',
      (r.apiWarnings || []).join('|'), r.error || '', r.audio,
    ]);
  const csv = [
    ['status', 'id', 'doctor', 'patient', 'whisper_recall', 'whisper_f1', 'llm_recall', 'warnings', 'error', 'audio'],
    ...failures,
  ].map((row) => row.map((v) => `"${String(v ?? '').replace(/"/g, '""')}"`).join(',')).join('\n');
  fs.writeFileSync(path.join(args.out, 'warn-fail.csv'), csv, 'utf8');
  return { summary, candidateList };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  fs.mkdirSync(args.out, { recursive: true });
  const jsonlPath = path.join(args.out, 'cases.jsonl');
  const progressPath = path.join(args.out, 'progress.json');

  const allCases = loadCases(args.root);
  const selected = allCases.slice(args.offset, args.limit ? args.offset + args.limit : undefined);
  const done = loadDone(jsonlPath);
  const pending = selected.filter((c) => !done.has(c.id));
  const previous = [];
  if (fs.existsSync(jsonlPath)) {
    for (const line of fs.readFileSync(jsonlPath, 'utf8').split(/\r?\n/u)) {
      if (!line.trim()) continue;
      try { previous.push(JSON.parse(line)); } catch {}
    }
  }

  console.log(`Blind QA: cases=${selected.length}, pending=${pending.length}, out=${args.out}`);
  const health = await requestJson(`${args.server}/api/health`);
  console.log(`Health: ${JSON.stringify(health.services || health)}`);
  const token = await login(args);
  const candidates = new Map();
  const results = [...previous];
  const stream = fs.createWriteStream(jsonlPath, { flags: 'a', encoding: 'utf8' });
  let index = 0;
  let completed = results.length;
  const startedAt = Date.now();

  async function worker(workerId) {
    while (index < pending.length) {
      const current = pending[index++];
      const n = completed + 1;
      const rel = path.relative(args.root, current.audioPath);
      console.log(`[${new Date().toISOString()}] worker=${workerId} start ${n}/${selected.length} id=${current.id} ${rel}`);
      const result = await runCase(current, token, args, candidates);
      completed += 1;
      results.push(result);
      stream.write(`${JSON.stringify(result)}\n`);
      const elapsedSec = (Date.now() - startedAt) / 1000;
      const rate = completed / Math.max(1, elapsedSec);
      const etaSec = (selected.length - completed) / Math.max(rate, 0.0001);
      const progress = {
        generatedAt: new Date().toISOString(),
        out: args.out,
        total: selected.length,
        completed,
        pending: selected.length - completed,
        pass: results.filter((r) => r.status === 'PASS').length,
        warn: results.filter((r) => r.status === 'WARN').length,
        fail: results.filter((r) => r.status === 'FAIL').length,
        last: {
          id: result.id,
          status: result.status,
          whisperRecall: result.whisper?.recall,
          llmRecall: result.llm?.documentRecall,
          elapsedSec: round((result.elapsedMs || 0) / 1000),
          error: result.error,
        },
        etaHours: round(etaSec / 3600),
      };
      fs.writeFileSync(progressPath, JSON.stringify(progress, null, 2), 'utf8');
      if (completed % 10 === 0 || result.status === 'FAIL') {
        writeReports(args, results, candidates);
      }
      console.log(`[${new Date().toISOString()}] done ${completed}/${selected.length} id=${result.id} status=${result.status} whisper=${result.whisper?.recall ?? 'err'} llm=${result.llm?.documentRecall ?? 'err'} eta=${progress.etaHours}h`);
    }
  }

  await Promise.all(Array.from({ length: args.concurrency }, (_, i) => worker(i + 1)));
  await new Promise((resolve) => stream.end(resolve));
  const final = writeReports(args, results, candidates);
  console.log(`Done: ${JSON.stringify(final.summary)}`);
  console.log(`Reports: ${args.out}`);
}

main().catch((error) => {
  console.error(`Fatal: ${error.stack || error.message || error}`);
  process.exit(1);
});
