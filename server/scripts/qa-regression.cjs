#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const MANIFEST_PATH = path.join(__dirname, 'qa-corpus.json');
const SERVER = process.env.SERVER_URL || 'http://localhost:3001';
const AUTH_PASS = process.env.AUTH_PASS || 'meddok2026';

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

const NON_EXAM_FIELDS = TEXT_FIELDS.filter((field) => field !== 'outpatientExams');

const SECTION_HEADER_RE =
  /^(?:жалобы|анамнез(?:\s+(?:заболевани[а-яё\w]*|жизни))?|объективн[а-яё\w]*\s+статус|объективно|перенесенные\s+заболевани[а-яё\w]*|перенесённые\s+заболевани[а-яё\w]*|аллерголог[а-яё\w]*\s+анамнез|неврологическ[а-яё\w]*\s+статус|диагноз(?:\s+(?:предварительн[а-яё\w]*|заключительн[а-яё\w]*|основн[а-яё\w]*))?|план(?:\s+(?:обследовани[а-яё\w]*|лечени[а-яё\w]*))?|рекомендации|рекомендация|заключение|сопутствующ[а-яё\w]*\s+диагноз|амбулаторн[а-яё\w]*\s+терапи[а-яё\w]*|данные|статус)\.?$/iu;

const COMMON_TOKENS = new Set([
  'пациент',
  'пациентка',
  'жалобы',
  'анамнез',
  'диагноз',
  'данные',
  'год',
  'года',
  'лет',
  'при',
  'для',
  'или',
  'это',
  'что',
  'как',
  'без',
  'после',
  'перед',
  'есть',
  'нет',
  'под',
  'над',
  'the',
  'and',
]);

function parseArgs(argv) {
  const out = {
    mode: 'cached',
    cases: [],
    server: SERVER,
    jsonOnly: false,
    failOnWarn: false,
    writeDoc: true,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--mode') {
      out.mode = argv[++i] || out.mode;
    } else if (arg.startsWith('--mode=')) {
      out.mode = arg.slice('--mode='.length);
    } else if (arg === '--case') {
      out.cases.push(argv[++i]);
    } else if (arg.startsWith('--case=')) {
      out.cases.push(arg.slice('--case='.length));
    } else if (arg === '--server') {
      out.server = argv[++i] || out.server;
    } else if (arg.startsWith('--server=')) {
      out.server = arg.slice('--server='.length);
    } else if (arg === '--json-only') {
      out.jsonOnly = true;
    } else if (arg === '--fail-on-warn') {
      out.failOnWarn = true;
    } else if (arg === '--no-doc') {
      out.writeDoc = false;
    } else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!['cached', 'source', 'audio'].includes(out.mode)) {
    throw new Error(`Invalid --mode "${out.mode}". Expected cached, source, or audio.`);
  }

  return out;
}

function printHelp() {
  console.log(`Usage: node scripts/qa-regression.cjs [options]

Options:
  --mode cached       Use cached Whisper text when available, otherwise source text.
  --mode source       Use source transcript text only.
  --mode audio        Send audio files to /api/process. Slow.
  --case <id>         Run a single case. Can be repeated.
  --server <url>      API server URL. Default: ${SERVER}
  --json-only         Print only the report path and summary JSON.
  --fail-on-warn      Exit with code 1 when warnings exist.
  --no-doc            Do not store structured documents in the JSON report.

Environment:
  AUTH_PASS           Login password.
  SERVER_URL          API server URL.
  QA_AUDIO_DIR        External audio folder. Defaults to manifest audioRootDefault.`);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function expandPercentEnv(value) {
  return String(value || '').replace(/%([^%]+)%/g, (_, name) => process.env[name] || '');
}

function resolveRepoPath(relativePath) {
  return path.resolve(REPO_ROOT, relativePath);
}

function resolveAudioRoot(manifest) {
  const envName = manifest.audioRootEnv || 'QA_AUDIO_DIR';
  return process.env[envName] || expandPercentEnv(manifest.audioRootDefault || '');
}

function resolveAudioPath(manifest, testCase) {
  if (testCase.audioPath) {
    const candidate = expandPercentEnv(testCase.audioPath);
    return path.isAbsolute(candidate) ? candidate : resolveRepoPath(candidate);
  }
  if (testCase.audioFile) {
    const root = resolveAudioRoot(manifest);
    return root ? path.join(root, testCase.audioFile) : '';
  }
  return '';
}

function fileExists(filePath) {
  return Boolean(filePath) && fs.existsSync(filePath) && fs.statSync(filePath).isFile();
}

function pickInput(manifest, testCase, mode, selectedExplicitly) {
  if (mode !== 'audio' && testCase.enabledByDefault === false && !selectedExplicitly) {
    return { skip: true, reason: 'disabled_by_default' };
  }

  if (mode === 'audio') {
    const audioPath = resolveAudioPath(manifest, testCase);
    if (!fileExists(audioPath)) return { skip: true, reason: 'audio_missing', path: audioPath };
    return { kind: 'audio', path: audioPath };
  }

  if (mode === 'source') {
    const sourcePath = testCase.sourceTextPath ? resolveRepoPath(testCase.sourceTextPath) : '';
    if (!fileExists(sourcePath)) return { skip: true, reason: 'source_text_missing', path: sourcePath };
    return { kind: 'source-text', path: sourcePath, text: fs.readFileSync(sourcePath, 'utf8').trim() };
  }

  const rawPath = testCase.rawTextPath ? resolveRepoPath(testCase.rawTextPath) : '';
  if (fileExists(rawPath)) {
    return { kind: 'cached-whisper', path: rawPath, text: fs.readFileSync(rawPath, 'utf8').trim() };
  }

  const sourcePath = testCase.sourceTextPath ? resolveRepoPath(testCase.sourceTextPath) : '';
  if (fileExists(sourcePath)) {
    return { kind: 'source-text', path: sourcePath, text: fs.readFileSync(sourcePath, 'utf8').trim() };
  }

  return { skip: true, reason: 'no_cached_or_source_text', path: rawPath || sourcePath };
}

function mergeCase(defaults, testCase) {
  return {
    ...defaults,
    ...testCase,
    requiredMeaningfulFields:
      testCase.requiredMeaningfulFields || defaults.requiredMeaningfulFields || [],
  };
}

async function login(server) {
  let response;
  try {
    response = await fetch(`${server}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: AUTH_PASS }),
    });
  } catch (error) {
    throw new Error(`Cannot reach API server at ${server}. Start the backend first. ${error.message}`);
  }

  let data = {};
  try {
    data = await response.json();
  } catch {
    throw new Error(`Login returned non-JSON response with status ${response.status}`);
  }

  if (!response.ok || !data.token) {
    throw new Error(`Login failed with status ${response.status}: ${JSON.stringify(data)}`);
  }
  return data.token;
}

async function structureText(server, token, text) {
  const startedAt = Date.now();
  const response = await fetch(`${server}/api/structure`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });
  const elapsedMs = Date.now() - startedAt;
  const bodyText = await response.text();
  let body = {};
  try {
    body = bodyText ? JSON.parse(bodyText) : {};
  } catch {
    body = { raw: bodyText };
  }
  return { ok: response.ok, status: response.status, body, elapsedMs };
}

function buildMultipart(filePath, fieldName) {
  const basename = path.basename(filePath);
  const boundary = `----FormBoundary${Math.random().toString(36).slice(2)}`;
  const fileData = fs.readFileSync(filePath);
  const header = Buffer.from(
    `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="${fieldName}"; filename="${basename}"\r\n` +
      'Content-Type: application/octet-stream\r\n\r\n',
  );
  const footer = Buffer.from(`\r\n--${boundary}--\r\n`);
  return {
    body: Buffer.concat([header, fileData, footer]),
    contentType: `multipart/form-data; boundary=${boundary}`,
  };
}

async function processAudio(server, token, filePath) {
  const { body, contentType } = buildMultipart(filePath, 'file');
  const startedAt = Date.now();
  const response = await fetch(`${server}/api/process`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': contentType,
    },
    body,
  });
  const elapsedMs = Date.now() - startedAt;
  const bodyText = await response.text();
  let parsed = {};
  try {
    parsed = bodyText ? JSON.parse(bodyText) : {};
  } catch {
    parsed = { raw: bodyText };
  }
  return { ok: response.ok, status: response.status, body: parsed, elapsedMs };
}

function fieldToString(value) {
  if (value === undefined || value === null) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function getPathValue(doc, dottedPath) {
  return dottedPath.split('.').reduce((value, key) => {
    if (value && typeof value === 'object') return value[key];
    return undefined;
  }, doc);
}

function stripSectionHeaders(text) {
  return String(text || '')
    .split(/[.!?\n]+/u)
    .map((part) => part.trim())
    .filter((part) => part && !SECTION_HEADER_RE.test(part))
    .join('. ')
    .trim();
}

function isMeaningfulField(field, value) {
  const stripped = stripSectionHeaders(value);
  if (!stripped) return false;
  if (field === 'diagnosis' || field === 'finalDiagnosis' || field === 'conclusion') {
    return stripped.length >= 3 && /[а-яёa-z]/iu.test(stripped);
  }
  if (stripped.length < 10) return false;
  const words = stripped.match(/[а-яёa-z]{3,}/giu) || [];
  return words.length >= 2;
}

function normalizeText(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^a-zа-я0-9\s]/giu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(text) {
  return normalizeText(text)
    .split(/\s+/u)
    .filter((token) => token.length >= 4 && !COMMON_TOKENS.has(token));
}

function sentenceCoverage(inputText, doc) {
  const docText = TEXT_FIELDS.map((field) => fieldToString(doc[field])).join(' ');
  const normalizedDoc = normalizeText(docText);
  const sentences = String(inputText || '')
    .split(/(?<=[.!?])\s+|\n+/u)
    .map((sentence) => sentence.trim())
    .filter((sentence) => normalizeText(sentence).length >= 20);

  const missing = [];
  let found = 0;

  for (const sentence of sentences) {
    const normalized = normalizeText(sentence);
    if (!normalized) continue;

    const prefix = normalized.slice(0, Math.min(36, normalized.length));
    const middleStart = Math.max(0, Math.floor(normalized.length / 2) - 16);
    const middle = normalized.slice(middleStart, middleStart + 32);
    const sentenceTokens = tokenize(sentence);
    const tokenHits = sentenceTokens.filter((token) => normalizedDoc.includes(token)).length;
    const tokenHitRate = sentenceTokens.length ? tokenHits / sentenceTokens.length : 0;

    if (
      (prefix.length >= 18 && normalizedDoc.includes(prefix)) ||
      (middle.length >= 18 && normalizedDoc.includes(middle)) ||
      (sentenceTokens.length >= 4 && tokenHitRate >= 0.5)
    ) {
      found += 1;
    } else {
      missing.push(sentence);
    }
  }

  const pct = sentences.length ? Math.round((found / sentences.length) * 100) : 0;
  return { total: sentences.length, found, pct, missing: missing.slice(0, 8) };
}

function lexicalRecall(sourceText, actualText) {
  if (!sourceText || !actualText) return null;
  const sourceTokens = new Set(tokenize(sourceText));
  const actualTokens = new Set(tokenize(actualText));
  if (!sourceTokens.size || !actualTokens.size) return null;

  let found = 0;
  for (const token of sourceTokens) {
    if (actualTokens.has(token)) found += 1;
  }

  return {
    sourceTokens: sourceTokens.size,
    found,
    pct: Math.round((found / sourceTokens.size) * 100),
  };
}

function looksLikeMojibake(text) {
  const s = String(text || '');
  if (!s) return false;
  if (/[�ÐÑ]/u.test(s)) return true;
  const tokens = s.split(/\s+/u).filter(Boolean).length || 1;
  const suspiciousPairs = (s.match(/[РС][А-Яа-яЁё]/gu) || []).length;
  return suspiciousPairs > 40 && suspiciousPairs / tokens > 0.45;
}

function analyzeDocument(testCase, inputText, doc, sourceText) {
  const issues = [];
  const warnings = [];
  const fieldLengths = {};
  const meaningfulFields = [];
  const placeholderFields = [];
  const emptyFields = [];

  for (const field of TEXT_FIELDS) {
    const value = fieldToString(doc[field]);
    fieldLengths[field] = value.length;
    if (!value.trim()) {
      emptyFields.push(field);
    } else if (isMeaningfulField(field, value)) {
      meaningfulFields.push(field);
    } else {
      placeholderFields.push(field);
    }
  }

  const meaningfulNonExams = meaningfulFields.filter((field) => NON_EXAM_FIELDS.includes(field));
  const onlyLabs = meaningfulFields.length === 1 && meaningfulFields[0] === 'outpatientExams';

  if (looksLikeMojibake(inputText)) {
    issues.push('input_mojibake_suspected');
  }

  if (meaningfulFields.length === 0) {
    issues.push('document_appears_empty');
  }

  if (onlyLabs && testCase.profile !== 'labs_only') {
    warnings.push('document_labs_only_unexpected');
  }

  if (onlyLabs && testCase.profile === 'labs_only') {
    warnings.push('document_labs_only_expected');
  }

  if (testCase.profile === 'full_case' && meaningfulNonExams.length === 0) {
    issues.push('no_meaningful_non_exam_clinical_fields');
  }

  const requiredMissing = [];
  for (const field of testCase.requiredMeaningfulFields || []) {
    const value = fieldToString(getPathValue(doc, field));
    const topField = field.split('.')[0];
    if (!isMeaningfulField(topField, value)) requiredMissing.push(field);
  }
  if (requiredMissing.length) {
    issues.push(`required_fields_missing:${requiredMissing.join(',')}`);
  }

  const minMeaningfulFields = Number(testCase.minMeaningfulFields || 0);
  if (minMeaningfulFields && meaningfulFields.length < minMeaningfulFields) {
    issues.push(`too_few_meaningful_fields:${meaningfulFields.length}/${minMeaningfulFields}`);
  }

  if (placeholderFields.length) {
    warnings.push(`placeholder_or_too_short_fields:${placeholderFields.join(',')}`);
  }

  const coverage = sentenceCoverage(inputText, doc);
  const minCoverage = Number(testCase.minInputCoveragePct || 0);
  if (coverage.total > 0 && minCoverage && coverage.pct < minCoverage) {
    warnings.push(`low_input_sentence_coverage:${coverage.pct}/${minCoverage}`);
  }

  const recall = lexicalRecall(sourceText, inputText);
  const minRecall = Number(testCase.minSourceRecallPct || 0);
  if (recall && minRecall && recall.pct < minRecall) {
    warnings.push(`low_source_to_input_recall:${recall.pct}/${minRecall}`);
  }

  const patient = doc.patient && typeof doc.patient === 'object' ? doc.patient : {};
  const patientFilled = ['fullName', 'age', 'gender', 'complaintDate'].filter((field) =>
    fieldToString(patient[field]).trim(),
  );

  return {
    status: issues.length ? 'FAIL' : warnings.length ? 'WARN' : 'PASS',
    issues,
    warnings,
    metrics: {
      fieldLengths,
      meaningfulFields,
      emptyFields,
      placeholderFields,
      patientFilled,
      inputCoverage: coverage,
      sourceRecall: recall,
    },
  };
}

function loadSourceText(testCase) {
  if (!testCase.sourceTextPath) return '';
  const sourcePath = resolveRepoPath(testCase.sourceTextPath);
  return fileExists(sourcePath) ? fs.readFileSync(sourcePath, 'utf8').trim() : '';
}

async function runCase(manifest, testCase, input, token, args) {
  const sourceText = loadSourceText(testCase);
  const startedAt = Date.now();
  const response =
    input.kind === 'audio'
      ? await processAudio(args.server, token, input.path)
      : await structureText(args.server, token, input.text);

  const inputText =
    input.kind === 'audio'
      ? fieldToString(response.body.transcription && response.body.transcription.text)
      : input.text;

  const base = {
    id: testCase.id,
    title: testCase.title || testCase.id,
    profile: testCase.profile,
    inputKind: input.kind,
    inputPath: input.path ? path.relative(REPO_ROOT, input.path) : '',
    inputChars: inputText.length,
    httpStatus: response.status,
    elapsedMs: response.elapsedMs,
    totalMs: Date.now() - startedAt,
  };

  if (!response.ok) {
    return {
      ...base,
      status: 'FAIL',
      issues: [`api_error:${response.status}`],
      warnings: [],
      apiError: response.body,
    };
  }

  const doc = response.body.document || {};
  const analysis = analyzeDocument(testCase, inputText, doc, sourceText);
  return {
    ...base,
    ...analysis,
    apiWarnings: response.body.warnings || [],
    document: args.writeDoc ? doc : undefined,
  };
}

function summarize(results) {
  return results.reduce(
    (acc, item) => {
      const key = item.status && acc[item.status.toLowerCase()] !== undefined ? item.status.toLowerCase() : 'skipped';
      acc[key] += 1;
      return acc;
    },
    { pass: 0, warn: 0, fail: 0, skipped: 0 },
  );
}

function printResult(result) {
  if (result.status === 'SKIP') {
    console.log(`SKIP ${result.id} (${result.reason})`);
    return;
  }

  const coverage = result.metrics && result.metrics.inputCoverage;
  const recall = result.metrics && result.metrics.sourceRecall;
  const coverageText = coverage ? ` coverage=${coverage.pct}%` : '';
  const recallText = recall ? ` sourceRecall=${recall.pct}%` : '';
  const fields =
    result.metrics && result.metrics.meaningfulFields
      ? ` fields=${result.metrics.meaningfulFields.length}`
      : '';

  console.log(
    `${result.status} ${result.id} [${result.inputKind}] chars=${result.inputChars}${fields}${coverageText}${recallText} time=${(
      result.elapsedMs / 1000
    ).toFixed(1)}s`,
  );

  for (const issue of result.issues || []) {
    console.log(`  issue: ${issue}`);
  }
  for (const warning of result.warnings || []) {
    console.log(`  warn: ${warning}`);
  }
}

function writeReport(args, report) {
  const outDir = path.join(REPO_ROOT, 'server', 'temp', 'qa-regression');
  fs.mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const reportPath = path.join(outDir, `${stamp}.json`);
  const latestPath = path.join(outDir, 'latest-summary.json');
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');
  fs.writeFileSync(latestPath, JSON.stringify(report, null, 2), 'utf8');
  return { reportPath, latestPath };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const manifest = readJson(MANIFEST_PATH);
  const requestedCases = new Set(args.cases.filter(Boolean));
  const explicitCaseMode = requestedCases.size > 0;
  const configuredCases = manifest.cases.map((testCase) => mergeCase(manifest.defaults || {}, testCase));

  const casesToRun = configuredCases.filter((testCase) => !explicitCaseMode || requestedCases.has(testCase.id));
  const unknownCases = [...requestedCases].filter((id) => !configuredCases.some((testCase) => testCase.id === id));
  if (unknownCases.length) {
    throw new Error(`Unknown case id(s): ${unknownCases.join(', ')}`);
  }

  if (!args.jsonOnly) {
    console.log(`QA regression: mode=${args.mode}, server=${args.server}, cases=${casesToRun.length}`);
  }

  const token = await login(args.server);
  const results = [];

  for (const testCase of casesToRun) {
    const input = pickInput(manifest, testCase, args.mode, explicitCaseMode);
    if (input.skip) {
      const skipped = {
        id: testCase.id,
        title: testCase.title || testCase.id,
        status: 'SKIP',
        reason: input.reason,
        inputPath: input.path ? path.relative(REPO_ROOT, input.path) : '',
      };
      results.push(skipped);
      if (!args.jsonOnly) printResult(skipped);
      continue;
    }

    const result = await runCase(manifest, testCase, input, token, args);
    results.push(result);
    if (!args.jsonOnly) printResult(result);
  }

  const summary = summarize(results);
  const report = {
    generatedAt: new Date().toISOString(),
    mode: args.mode,
    server: args.server,
    summary,
    results,
  };
  const paths = writeReport(args, report);

  if (args.jsonOnly) {
    console.log(JSON.stringify({ summary, reportPath: paths.reportPath }, null, 2));
  } else {
    console.log(
      `Done: pass=${summary.pass}, warn=${summary.warn}, fail=${summary.fail}, skipped=${summary.skipped}`,
    );
    console.log(`Report: ${paths.reportPath}`);
  }

  if (summary.fail > 0 || (args.failOnWarn && summary.warn > 0)) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Fatal:', error.message || error);
  process.exit(1);
});
