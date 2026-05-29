#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function loadEnvFile(file) {
  if (!fs.existsSync(file)) return;
  const content = fs.readFileSync(file, 'utf8');
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const match = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (!match || process.env[match[1]]) continue;
    let value = match[2].trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    process.env[match[1]] = value;
  }
}

loadEnvFile(path.resolve('server', '.env'));

const SERVER = (process.env.SERVER_URL || 'http://127.0.0.1:1337').replace(/\/+$/, '');
const OUT_DIR = process.env.QA_OUT_DIR || path.resolve('temp', 'qa-live-dictation');
const LIMIT = Number.parseInt(process.env.QA_LIMIT || '0', 10);
const TIMEOUT_MS = Number.parseInt(process.env.QA_TIMEOUT_MS || `${2 * 60 * 1000}`, 10);

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

const AUTH_TOKEN = process.env.AUTH_TOKEN || signJwt({
  doctorId: Number.parseInt(process.env.QA_DOCTOR_ID || '1', 10),
  email: process.env.QA_DOCTOR_EMAIL || 'qa@local.test',
  name: process.env.QA_DOCTOR_NAME || 'QA',
  role: process.env.QA_DOCTOR_ROLE || 'admin',
}, process.env.JWT_SECRET || 'dev-insecure-secret-change-in-production');

const CASES = [
  {
    id: 'live_with_patient_and_reco',
    text: [
      'Пациент Иванов Иван Иванович, 45 лет.',
      'Жалобы на головную боль и повышение артериального давления до 160 на 95.',
      'Анамнез заболевания: ухудшение в течение двух недель.',
      'Объективно: состояние удовлетворительное, пульс 78 в минуту.',
      'Диагноз: артериальная гипертензия второй степени.',
      'Рекомендовано: контроль артериального давления, бисопролол 5 мг утром, повторный прием кардиолога через месяц.',
    ].join(' '),
    expect: {
      patient: ['Иванов'],
      complaints: ['головн', '160/95'],
      diagnosis: ['гипертенз'],
      recommendations: ['бисопролол', 'кардиолог'],
      warningsAbsent: ['patient_identity_missing', 'unsupportedRecommendation', 'patientNameFromFilename'],
    },
  },
  {
    id: 'live_without_patient',
    text: [
      'Жалобы на боль в грудной клетке при физической нагрузке.',
      'Анамнез: симптомы появились около месяца назад.',
      'Диагноз: стабильная стенокардия напряжения.',
      'Рекомендовано: ЭКГ, ЭхоКГ, консультация кардиолога.',
    ].join(' '),
    expect: {
      complaints: ['грудн', 'нагруз'],
      diagnosis: ['стенокард'],
      recommendations: ['экг', 'кардиолог'],
      warningsPresent: ['patient_identity_missing'],
      warningsAbsent: ['patientNameFromFilename'],
    },
  },
  {
    id: 'live_no_recommendations_no_generic',
    text: [
      'Пациентка Петрова Мария Сергеевна, 62 года.',
      'Жалобы на слабость и головокружение.',
      'Анамнез заболевания: ухудшение самочувствия последние три дня.',
      'Объективно: кожные покровы обычной окраски, пульс 82.',
      'Диагноз: анемия неуточненная.',
    ].join(' '),
    expect: {
      patient: ['Петрова'],
      complaints: ['слабост', 'головокруж'],
      diagnosis: ['анем'],
      recommendationsAbsent: ['алкогол', 'диет', 'питани', 'физическ'],
      warningsAbsent: ['unsupportedRecommendation', 'patientNameFromFilename'],
    },
  },
  {
    id: 'live_labs_and_plan',
    text: [
      'Пациент Садыков Руслан Ерланович, 50 лет.',
      'Жалобы на жажду и сухость во рту.',
      'Амбулаторные обследования: глюкоза крови 8,9 миллимоль на литр, гликированный гемоглобин 7,2 процента.',
      'Диагноз: сахарный диабет второго типа.',
      'План обследования: контроль глюкозы, общий анализ мочи, консультация эндокринолога.',
    ].join(' '),
    expect: {
      patient: ['Садыков'],
      complaints: ['жажд', 'сухост'],
      outpatientExams: ['8,9', '7,2'],
      diagnosis: ['диабет'],
      doctorNotes: ['глюкоз', 'эндокринолог'],
      warningsAbsent: ['patient_identity_missing', 'patientNameFromFilename'],
    },
  },
  {
    id: 'live_current_therapy_vs_reco',
    text: [
      'Пациент Ахметов Нурлан Болатович, 58 лет.',
      'Жалобы на одышку при нагрузке.',
      'Амбулаторно принимает амлодипин 5 мг вечером постоянно.',
      'Диагноз: хроническая сердечная недостаточность.',
      'Рекомендовано: контроль ЭхоКГ и повторный прием через три месяца.',
    ].join(' '),
    expect: {
      patient: ['Ахметов'],
      conclusion: ['амлодипин'],
      diagnosis: ['сердечн'],
      recommendations: ['эхокг', 'повторн'],
      warningsAbsent: ['unsupportedRecommendation', 'patientNameFromFilename'],
    },
  },
  {
    id: 'live_ent_routing',
    text: [
      'Пациентка Алиева Дина Маратовна, 34 года.',
      'Жалобы на заложенность носа и боль в правом ухе.',
      'Лор статус: носовое дыхание затруднено, барабанная перепонка справа гиперемирована.',
      'Диагноз: острый средний отит справа.',
      'Рекомендовано: капли в ухо три раза в день, повторный осмотр отоларинголога через пять дней.',
    ].join(' '),
    expect: {
      patient: ['Алиева'],
      complaints: ['заложен', 'ух'],
      objectiveStatus: ['барабан', 'перепон'],
      diagnosis: ['отит'],
      recommendations: ['капли', 'отоларинголог'],
      warningsAbsent: ['patient_identity_missing', 'patientNameFromFilename'],
    },
  },
];

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function csvCell(value) {
  const s = Array.isArray(value) ? value.join('|') : String(value ?? '');
  return `"${s.replace(/"/g, '""')}"`;
}

function normalize(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/\s+/g, ' ')
    .trim();
}

function docText(document) {
  if (!document) return '';
  return [
    document.patient?.fullName,
    document.patient?.age,
    document.patient?.gender,
    document.complaints,
    document.anamnesis,
    document.outpatientExams,
    document.clinicalCourse,
    document.allergyHistory,
    document.objectiveStatus,
    document.neurologicalStatus,
    document.diagnosis,
    document.finalDiagnosis,
    document.conclusion,
    document.doctorNotes,
    document.recommendations,
    document.manualCheck,
  ].filter(Boolean).join('\n');
}

function fieldValue(document, field) {
  if (!document) return '';
  if (field === 'patient') return document.patient?.fullName || '';
  return document[field] || '';
}

function includesAll(value, terms = []) {
  const n = normalize(value);
  return terms.every((term) => n.includes(normalize(term)));
}

function missesAll(value, terms = []) {
  const n = normalize(value);
  return terms.every((term) => !n.includes(normalize(term)));
}

async function postStructure(text) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${SERVER}/api/structure`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        Authorization: `Bearer ${AUTH_TOKEN}`,
      },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    const raw = await res.text();
    let json;
    try {
      json = JSON.parse(raw);
    } catch {
      json = { raw };
    }
    return { status: res.status, ok: res.ok, json };
  } finally {
    clearTimeout(timer);
  }
}

function evaluate(testCase, response) {
  const document = response.json?.document;
  const warnings = response.json?.warnings || [];
  const failures = [];

  if (!response.ok || response.json?.success === false) {
    failures.push(`HTTP/status failed: ${response.status}`);
  }

  for (const [field, terms] of Object.entries(testCase.expect || {})) {
    if (field === 'warningsPresent' || field === 'warningsAbsent' || field === 'recommendationsAbsent') continue;
    if (!includesAll(fieldValue(document, field), terms)) {
      failures.push(`${field} missing terms: ${terms.join(', ')}`);
    }
  }

  if (testCase.expect?.recommendationsAbsent && !missesAll(document?.recommendations || '', testCase.expect.recommendationsAbsent)) {
    failures.push(`recommendations contains forbidden generic terms: ${testCase.expect.recommendationsAbsent.join(', ')}`);
  }

  for (const code of testCase.expect?.warningsPresent || []) {
    if (!warnings.includes(code)) failures.push(`warning not present: ${code}`);
  }
  for (const code of testCase.expect?.warningsAbsent || []) {
    if (warnings.includes(code)) failures.push(`warning unexpectedly present: ${code}`);
  }

  return {
    id: testCase.id,
    status: response.status,
    success: failures.length === 0,
    failures,
    warnings,
    qualityWarnings: response.json?.qualityWarnings || [],
    document,
    documentTextLength: docText(document).length,
  };
}

function writeReport(results) {
  const passed = results.filter((r) => r.success).length;
  const failed = results.length - passed;
  const warningCounts = new Map();
  for (const result of results) {
    for (const warning of result.warnings || []) {
      warningCounts.set(warning, (warningCounts.get(warning) || 0) + 1);
    }
  }

  const lines = [
    '# Live Dictation QA Report',
    '',
    `- Server: ${SERVER}`,
    `- Cases: ${results.length}`,
    `- Passed: ${passed}`,
    `- Failed: ${failed}`,
    '',
    '## Warning Counts',
    '',
    ...([...warningCounts.entries()].sort((a, b) => b[1] - a[1]).map(([code, count]) => `- ${code}: ${count}`)),
    '',
    '## Cases',
    '',
  ];

  for (const result of results) {
    lines.push(`### ${result.success ? 'PASS' : 'FAIL'} ${result.id}`);
    lines.push('');
    lines.push(`- Status: ${result.status}`);
    lines.push(`- Warnings: ${(result.warnings || []).join(', ') || '-'}`);
    lines.push(`- Document chars: ${result.documentTextLength}`);
    if (result.failures.length) {
      lines.push(`- Failures: ${result.failures.join('; ')}`);
    }
    lines.push('');
  }

  fs.writeFileSync(path.join(OUT_DIR, 'live-report.md'), `${lines.join('\n')}\n`, 'utf8');

  const csv = [
    ['id', 'success', 'status', 'warnings', 'failures', 'patient', 'complaints', 'diagnosis', 'recommendations'].map(csvCell).join(','),
    ...results.map((r) => [
      r.id,
      r.success,
      r.status,
      r.warnings,
      r.failures,
      r.document?.patient?.fullName || '',
      r.document?.complaints || '',
      r.document?.diagnosis || '',
      r.document?.recommendations || '',
    ].map(csvCell).join(',')),
  ].join('\n');
  fs.writeFileSync(path.join(OUT_DIR, 'live-results.csv'), `${csv}\n`, 'utf8');
  fs.writeFileSync(path.join(OUT_DIR, 'live-results.json'), JSON.stringify(results, null, 2), 'utf8');
}

async function main() {
  ensureDir(OUT_DIR);
  const cases = LIMIT > 0 ? CASES.slice(0, LIMIT) : CASES;
  const results = [];

  for (const testCase of cases) {
    process.stdout.write(`[live-qa] ${testCase.id} ... `);
    try {
      const response = await postStructure(testCase.text);
      const result = evaluate(testCase, response);
      results.push(result);
      process.stdout.write(`${result.success ? 'PASS' : 'FAIL'} (${response.status})\n`);
      if (!result.success) {
        for (const failure of result.failures) process.stdout.write(`  - ${failure}\n`);
      }
    } catch (error) {
      const result = {
        id: testCase.id,
        status: 0,
        success: false,
        failures: [error instanceof Error ? error.message : String(error)],
        warnings: [],
        qualityWarnings: [],
        document: null,
        documentTextLength: 0,
      };
      results.push(result);
      process.stdout.write('FAIL (exception)\n');
    }
  }

  writeReport(results);
  const failed = results.filter((r) => !r.success).length;
  console.log(`[live-qa] report: ${path.join(OUT_DIR, 'live-report.md')}`);
  console.log(`[live-qa] result: ${results.length - failed}/${results.length} passed`);
  process.exitCode = failed ? 1 : 0;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
