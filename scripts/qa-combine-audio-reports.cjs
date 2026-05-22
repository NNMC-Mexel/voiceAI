#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const dirs = (process.env.QA_REPORT_DIRS || [
  'temp/qa-audio-folder',
  'temp/qa-audio-folder-136-235',
  'temp/qa-audio-folder-219-318',
  'temp/qa-audio-folder-319-418',
  'temp/qa-audio-folder-382-469',
  'temp/qa-audio-folder-443-469',
].join(';')).split(';').filter(Boolean);

const out = process.env.QA_COMBINED_OUT || 'temp/qa-audio-folder-combined';
const DEDUPE_BY_PATH = process.env.QA_DEDUPE_BY_PATH === 'true';
const REANALYZE = process.env.QA_REANALYZE === 'true';
fs.mkdirSync(out, { recursive: true });

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

function normalize(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^\p{L}\p{N}\s]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function coverage(reference, actual) {
  const refTokens = new Set(normalize(reference).split(/\s+/).filter((x) => x.length >= 4));
  if (refTokens.size === 0) return null;
  const actualTokens = new Set(normalize(actual).split(/\s+/).filter((x) => x.length >= 4));
  let found = 0;
  for (const t of refTokens) if (actualTokens.has(t)) found++;
  return Math.round((found / refTokens.size) * 100);
}

function sentenceCoverage(reference, actual) {
  const doc = normalize(actual);
  const docTokens = new Set(normalize(actual).split(/\s+/).filter((x) => x.length >= 4));
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
    const sentTokens = normalize(s).split(/\s+/).filter((x) => x.length >= 4);
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

let rows = [];
for (const d of dirs) {
  const p = path.join(d, 'results.jsonl');
  const lines = fs.readFileSync(p, 'utf8').trim().split(/\n/).filter(Boolean);
  for (const line of lines) {
    const r = JSON.parse(line);
    r.batch = d;
    rows.push(r);
  }
}

if (DEDUPE_BY_PATH) {
  const latestByPath = new Map();
  for (const r of rows) latestByPath.set(r.relativePath, r);
  rows = [...latestByPath.values()];
}
if (REANALYZE) {
  for (const r of rows) {
    const reference = r.textPath && fs.existsSync(r.textPath) ? fs.readFileSync(r.textPath, 'utf8') : '';
    if (r.ok) r.analysis = analyzeResult(reference, r.response);
  }
}
rows.forEach((r, i) => { r.globalIndex = i + 1; });

const issueCounts = {};
const warningCounts = {};
const statusCounts = {};
const categories = {
  low_stt: 0,
  low_llm_token: 0,
  low_llm_sentence: 0,
  patient_name_empty: 0,
  complaints_empty: 0,
  diagnosis_empty: 0,
  recommendations_empty: 0,
  http_or_fetch_failed: 0,
};

let ok = 0;
let totalElapsed = 0;
const stt = [];
const llm = [];
const sent = [];

for (const r of rows) {
  if (r.ok) ok++;
  statusCounts[r.status] = (statusCounts[r.status] || 0) + 1;
  totalElapsed += Number(r.elapsedSec) || 0;
  const a = r.analysis || {};
  if (a.sttCoverage != null) stt.push(a.sttCoverage);
  if (a.llmCoverage != null) llm.push(a.llmCoverage);
  if (a.sentenceCoverage?.percent != null) sent.push(a.sentenceCoverage.percent);

  for (const w of a.warnings || []) warningCounts[w] = (warningCounts[w] || 0) + 1;
  for (const w of a.qualityWarnings || []) {
    const key = w.code || String(w);
    warningCounts[key] = (warningCounts[key] || 0) + 1;
  }

  for (const issue of a.issues || []) {
    issueCounts[issue] = (issueCounts[issue] || 0) + 1;
    if (issue === 'patient_name_empty') categories.patient_name_empty++;
    if (issue === 'complaints_empty') categories.complaints_empty++;
    if (issue === 'diagnosis_empty') categories.diagnosis_empty++;
    if (issue === 'recommendations_empty') categories.recommendations_empty++;
    if (issue.startsWith('low_stt_token_coverage')) categories.low_stt++;
    if (issue.startsWith('low_llm_token_coverage')) categories.low_llm_token++;
    if (issue.startsWith('low_llm_sentence_coverage')) categories.low_llm_sentence++;
    if (issue.startsWith('http_') || issue.startsWith('exception:fetch')) categories.http_or_fetch_failed++;
  }
}

function avg(a) {
  return a.length ? Math.round(a.reduce((x, y) => x + y, 0) / a.length) : null;
}

function pct(arr, p) {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((s.length - 1) * p))];
}

function csvEscape(v) {
  return `"${String(v ?? '').replace(/"/g, '""')}"`;
}

const csv = [[
  'globalIndex',
  'batchIndex',
  'ok',
  'status',
  'elapsedSec',
  'relativePath',
  'sttCoverage',
  'llmCoverage',
  'sentenceCoverage',
  'patient',
  'warnings',
  'issues',
].map(csvEscape).join(',')];

for (const r of rows) {
  const a = r.analysis || {};
  csv.push([
    r.globalIndex,
    r.index,
    r.ok,
    r.status,
    r.elapsedSec,
    r.relativePath,
    a.sttCoverage ?? '',
    a.llmCoverage ?? '',
    a.sentenceCoverage?.percent ?? '',
    a.patient?.fullName || '',
    (a.warnings || []).join('|'),
    (a.issues || []).join('|'),
  ].map(csvEscape).join(','));
}

const failures = rows
  .filter((r) => !r.ok)
  .map((r) => ({
    globalIndex: r.globalIndex,
    status: r.status,
    relativePath: r.relativePath,
    issues: r.analysis?.issues || [],
    response: r.response,
  }));

const worst = rows
  .filter((r) => r.ok)
  .map((r) => ({
    globalIndex: r.globalIndex,
    relativePath: r.relativePath,
    stt: r.analysis?.sttCoverage,
    llm: r.analysis?.llmCoverage,
    sent: r.analysis?.sentenceCoverage?.percent,
    issues: r.analysis?.issues || [],
    warnings: r.analysis?.warnings || [],
  }))
  .sort((a, b) => (a.sent ?? 999) - (b.sent ?? 999) || (a.llm ?? 999) - (b.llm ?? 999))
  .slice(0, 60);

const summary = {
  total: rows.length,
  success: ok,
  failed: rows.length - ok,
  statusCounts,
  totalElapsedSec: Math.round(totalElapsed),
  avgElapsedSec: Math.round((totalElapsed / rows.length) * 10) / 10,
  avgSttCoverage: avg(stt),
  avgLlmCoverage: avg(llm),
  avgSentenceCoverage: avg(sent),
  p10Stt: pct(stt, 0.10),
  p10Llm: pct(llm, 0.10),
  p10Sentence: pct(sent, 0.10),
  categories,
};

function top(obj, n = 25) {
  return Object.entries(obj)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n);
}

const md = [
  '# Combined QA audio report',
  '',
  `Processed: ${rows.length}`,
  `Success: ${ok}`,
  `Failed: ${rows.length - ok}`,
  `Status counts: ${JSON.stringify(statusCounts)}`,
  `Total elapsed: ${Math.round(totalElapsed)} sec (${Math.round(totalElapsed / 60)} min)`,
  `Average elapsed: ${summary.avgElapsedSec} sec/file`,
  '',
  '## Coverage',
  '',
  `- Avg STT token coverage: ${summary.avgSttCoverage}%`,
  `- Avg LLM token coverage: ${summary.avgLlmCoverage}%`,
  `- Avg LLM sentence coverage: ${summary.avgSentenceCoverage}%`,
  `- P10 STT/LLM/sentence: ${summary.p10Stt}% / ${summary.p10Llm}% / ${summary.p10Sentence}%`,
  '',
  '## Category counts',
  '',
  ...Object.entries(categories).map(([k, v]) => `- ${k}: ${v}`),
  '',
  '## Top issues',
  '',
  ...top(issueCounts).map(([k, v]) => `- ${k}: ${v}`),
  '',
  '## Top warnings',
  '',
  ...top(warningCounts).map(([k, v]) => `- ${k}: ${v}`),
  '',
  '## Failures',
  '',
  ...failures.map((r) => `- #${r.globalIndex} status=${r.status} ${r.relativePath}: ${r.issues.join('|')}`),
  '',
].join('\n');

fs.writeFileSync(path.join(out, 'combined.csv'), `\uFEFF${csv.join('\n')}`, 'utf8');
fs.writeFileSync(path.join(out, 'combined.json'), JSON.stringify({
  summary,
  issueCounts,
  warningCounts,
  failures,
  worst,
}, null, 2), 'utf8');
fs.writeFileSync(path.join(out, 'combined-report.md'), md, 'utf8');

console.log(JSON.stringify(summary, null, 2));
