#!/usr/bin/env node
/**
 * QA Batch Test — sends audio files to /api/process and prints structured results.
 * Usage: node scripts/qa-batch-test.cjs <file1> [file2] ...
 */
const fs = require('fs');
const path = require('path');

const SERVER = process.env.SERVER_URL || 'http://localhost:3001';
const AUTH_PASS = process.env.AUTH_PASS || 'meddok2026';

function buildMultipart(filePath, fieldName) {
  const basename = path.basename(filePath);
  const boundary = '----FormBoundary' + Math.random().toString(36).slice(2);
  const fileData = fs.readFileSync(filePath);

  const header = Buffer.from(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="${fieldName}"; filename="${basename}"\r\n` +
    `Content-Type: application/octet-stream\r\n\r\n`
  );
  const footer = Buffer.from(`\r\n--${boundary}--\r\n`);

  return {
    body: Buffer.concat([header, fileData, footer]),
    contentType: `multipart/form-data; boundary=${boundary}`,
  };
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

async function processFile(filePath) {
  const absPath = path.resolve(filePath);
  const basename = path.basename(absPath);
  const stat = fs.statSync(absPath);
  console.log(`\n${'='.repeat(60)}`);
  console.log(`[${basename}] (${Math.round(stat.size / 1024)} KB)`);
  console.log('='.repeat(60));

  const { body, contentType } = buildMultipart(absPath, 'file');

  const start = Date.now();
  const resp = await fetch(`${SERVER}/api/process`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${authToken}`,
      'Content-Type': contentType,
    },
    body: body,
  });

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);

  if (!resp.ok) {
    const text = await resp.text();
    console.error(`ERROR ${resp.status}: ${text}`);
    return;
  }

  const data = await resp.json();
  console.log(`Total: ${elapsed}s`);

  const raw = data.transcription?.text || '';
  console.log(`\n--- RAW WHISPER TEXT [${raw.length} chars] ---`);
  console.log(raw);

  const doc = data.document || {};
  const fields = [
    'complaints', 'anamnesis', 'outpatientExams', 'clinicalCourse',
    'allergyHistory', 'objectiveStatus', 'neurologicalStatus',
    'diagnosis', 'finalDiagnosis', 'conclusion',
    'doctorNotes', 'recommendations', 'riskAssessment'
  ];

  for (const field of fields) {
    const val = doc[field];
    if (val === undefined || val === null) continue;
    const str = typeof val === 'object' ? JSON.stringify(val) : String(val);
    console.log(`\n--- ${field} [${str.length} chars] ---`);
    if (str.length === 0) {
      console.log('(empty)');
    } else {
      console.log(str);
    }
  }
}

async function main() {
  const files = process.argv.slice(2);
  if (files.length === 0) {
    console.error('Usage: node scripts/qa-batch-test.cjs <file1> [file2] ...');
    process.exit(1);
  }

  const unique = [...new Set(files.map(f => path.resolve(f)))];
  console.log(`Testing ${unique.length} unique audio files`);

  await login();

  for (const f of unique) {
    await processFile(f);
  }

  console.log(`\n${'='.repeat(80)}`);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
