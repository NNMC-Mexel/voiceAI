import { readFileSync } from 'fs';
import assert from 'node:assert/strict';
import { findExamTemplate, parseExamValuesFromText, parseExamDate, formatExamLine } from './dist/data/examTemplates.js';

const BASE = 'http://localhost:3001';
const TOKEN = '02770628-d246-4de6-b9ce-e1cdf6a48c52';
const AUDIO_PATH = 'c:/Project/voicemed/sounds/WhatsApp Audio 2026-03-05 at 11.02.19.ogg';

// Mirror of the updated postProcessOutpatientExams from llm.ts
function postProcessOutpatientExams(text) {
  if (!text.trim()) return '';
  const lines = text
    .split(/\n/)
    .map(l => l.replace(/^\s*\d+[.)]\s*/, '').trim())
    .filter(l => l.length > 0)
    .filter(l => {
      const isRecommendationBullet =
        /^[-\u2013\u2022]\s+.{10,}/u.test(l) &&
        /провест|назначит|рекоменд|оценит|исключит|направит|контрол|наблюден|соблюда|избегат|проверит|получит|принима|лечени|терапи|обследовани/iu.test(l);
      return !isRecommendationBullet;
    });

  const processed = lines.map(line => {
    const template = findExamTemplate(line);
    if (!template || template.parameters.length === 0) return line;
    const date = parseExamDate(line) || undefined;
    const values = {};
    for (const param of template.parameters) {
      const esc = param.name.replace(/[.*+?^${}()|[\]\\]/g, String.raw`\$&`);
      const vp = new RegExp(String.raw`${esc}\s*[-\u2013\u2014:]\s*([\d.,]+)`, 'iu');
      const m = line.match(vp);
      if (m) values[param.name] = m[1].replace(/[,.]$/, '');
    }
    if (Object.keys(values).length === 0) Object.assign(values, parseExamValuesFromText(template.id, line));
    return formatExamLine(template, values, date);
  });
  return processed.map((l, i) => `${i + 1}. ${l}`).join('\n');
}

// ===== UNIT TESTS =====
console.log('\n======= UNIT TESTS: postProcessOutpatientExams =======\n');

// Test 1: recommendation bullets are removed
{
  const input = [
    'ОАК от 04.03.2026: Hb - 149, СОЭ - 15',
    '- Провести обследование на предмет гипотиреоза',
    '- Оценить функцию вестибулярного аппарата (вестибулярная диагностика, МРТ)',
    '- Назначить наблюдение у невролога',
    'ЭКГ от 05.03.2026: синусовый ритм, ЧСС 72',
  ].join('\n');
  const out = postProcessOutpatientExams(input);
  assert.ok(!out.includes('Провести'), 'recommendation "Провести" must be removed');
  assert.ok(!out.includes('Оценить функцию'), 'recommendation "Оценить" must be removed');
  assert.ok(!out.includes('Назначить'), 'recommendation "Назначить" must be removed');
  assert.ok(out.includes('Hb - 149'), 'OAK Hb value must be preserved');
  assert.ok(out.includes('ЭКГ'), 'ECG line must be preserved');
  console.log('✓ Test 1: recommendation bullets filtered from outpatientExams');
  console.log('  Result:', out.replace(/\n/g, ' | '));
}

// Test 2: лейкоформула params
{
  const input = 'ОАК от 04.03.2026: Hb - 149, Тр - 234, Л - 6.3, нейтрофилы - 60.3, эозинофилы - 2.9, базофилы - 0.8, моноциты - 5.7, лимфоциты - 30.3, СОЭ - 15';
  const out = postProcessOutpatientExams(input);
  assert.ok(out.includes('нейтрофилы - 60.3'), 'нейтрофилы must be in output');
  assert.ok(out.includes('лимфоциты - 30.3'), 'лимфоциты must be in output');
  assert.ok(out.includes('Hb - 149'), 'Hb must be in output');
  console.log('✓ Test 2: лейкоформула parsed correctly');
  console.log('  Result:', out.substring(0, 180) + '...');
}

// Test 3: new Б/х params
{
  const input = 'Б/х от 04.03.2026: глюкоза - 5.2, АЛТ - 25, ЛЖСС - 65.9, ОЖСС - 78.2, трансферрин - 3.5, TSat - 13.71, витамин В12 - 1056, витамин D - 42, HbA1c - 5.9, ГГТП - 32, фибриноген - 2.8';
  const out = postProcessOutpatientExams(input);
  assert.ok(out.includes('ЛЖСС'), 'ЛЖСС must be in output');
  assert.ok(out.includes('TSat'), 'TSat must be in output');
  assert.ok(out.includes('витамин В12'), 'витамин В12 must be in output');
  assert.ok(out.includes('ГГТП'), 'ГГТП must be in output');
  assert.ok(out.includes('фибриноген'), 'фибриноген must be in output');
  console.log('✓ Test 3: new Б/х params parsed correctly');
  console.log('  Result:', out.substring(0, 200) + '...');
}

// Test 4: partial data
{
  const input = [
    'ОАК от 12.03.2026: Hb - 132, Л - 7.1, СОЭ - 12',
    'Б/х от 12.03.2026: глюкоза - 6.1, витамин D - 42',
  ].join('\n');
  const out = postProcessOutpatientExams(input);
  assert.ok(out.includes('Hb - 132'), 'Hb in partial');
  assert.ok(out.includes('витамин D'), 'витамин D in partial');
  assert.ok(!out.includes('Эр -  '), 'Эр must NOT appear with empty value');
  console.log('✓ Test 4: partial data — only dictated params shown');
  console.log('  Result:', out.replace(/\n/g, ' | '));
}

// Test 5: extra exams with descriptions preserved
{
  const input = [
    'ОАК от 04.03.2026: Hb - 149',
    'ФГДС от 10.03.2026: эрозивный гастрит, H. pylori отрицательный',
    'ЭКГ от 04.03.2026: синусовый ритм, ЧСС 72 уд/мин, нормальная ЭОС',
  ].join('\n');
  const out = postProcessOutpatientExams(input);
  assert.ok(out.includes('ФГДС'), 'ФГДС must be preserved');
  assert.ok(out.includes('эрозивный гастрит'), 'ФГДС description must be preserved');
  assert.ok(out.includes('синусовый ритм'), 'ЭКГ description must be preserved');
  const lineCount = out.split('\n').length;
  assert.equal(lineCount, 3, `Expected 3 lines, got ${lineCount}`);
  console.log('✓ Test 5: extra exams with descriptions preserved and numbered correctly');
  console.log('  Result:', out.replace(/\n/g, ' | '));
}

// Test 6: mixed — real exams + recommendations + extra exams (realistic LLM output)
{
  const input = [
    'ОАК от 04.03.2026: Hb - 149, нейтрофилы - 60.3, лимфоциты - 30.3, СОЭ - 15',
    'Б/х от 04.03.2026: глюкоза - 5.2, витамин В12 - 1056, ЛЖСС - 65.9',
    '- Провести полисомнографию для оценки сна',
    '- Исключить патологии сердечно-сосудистой системы (ЭКГ, УЗИ сердца)',
    'ЭКГ от 05.03.2026: нарушений не выявлено',
    'ЭХОКГ от 05.03.2026: ФВ 62%, ДЗЛЖ 48 мм',
  ].join('\n');
  const out = postProcessOutpatientExams(input);
  assert.ok(!out.includes('Провести полисомнографию'), 'recommendation must be removed');
  assert.ok(!out.includes('Исключить патологии'), 'recommendation must be removed');
  assert.ok(out.includes('нейтрофилы'), 'нейтрофилы must remain');
  assert.ok(out.includes('витамин В12'), 'витамин В12 must remain');
  assert.ok(out.includes('ЭКГ'), 'ЭКГ must remain');
  assert.ok(out.includes('ЭХОКГ'), 'ЭХОКГ must remain');
  const lineCount = out.split('\n').length;
  assert.equal(lineCount, 4, `Expected 4 lines (OAK, Bx, ECG, ECHO), got ${lineCount}`);
  console.log('✓ Test 6: mixed input — recommendations removed, exams preserved');
  console.log('  Result:', out.replace(/\n/g, ' | '));
}

console.log('\n======= INTEGRATION TEST: Full audio pipeline =======\n');

let audioBuffer;
try {
  audioBuffer = readFileSync(AUDIO_PATH);
} catch {
  console.log('⚠ Audio file not found, skipping live API test');
  console.log('\n✓ ALL UNIT TESTS PASSED\n');
  process.exit(0);
}

const formData = new FormData();
formData.append('audio', new Blob([audioBuffer], { type: 'audio/ogg' }), 'test.ogg');
console.log(`Sending audio (${(audioBuffer.length / 1024).toFixed(1)} KB) to /api/process...`);
const t0 = Date.now();
const res = await fetch(`${BASE}/api/process`, { method: 'POST', body: formData, headers: { 'Authorization': 'Bearer ' + TOKEN } });
console.log(`HTTP ${res.status} in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);

if (!res.ok) {
  console.error('HTTP ERROR:', (await res.text()).substring(0, 500));
  process.exit(1);
}

const body = await res.json();
const doc = body.document || body;
console.log('Transcription preview:', (body.transcription?.text || '').substring(0, 120) + '...\n');
const fields = [
  'complaints', 'anamnesis', 'outpatientExams', 'clinicalCourse',
  'allergyHistory', 'objectiveStatus', 'diagnosis', 'recommendations',
  'diet', 'conclusion', 'doctorNotes',
];

let issues = 0;
for (const field of fields) {
  const val = (doc[field] || '').trim();
  if (!val) {
    console.log(`  ⚠ [${field}]: EMPTY`);
    continue;
  }
  const hasRecommInExams = field === 'outpatientExams' &&
    /провест|назначит|рекоменд|оценит|исключит/iu.test(val);
  if (hasRecommInExams) {
    console.log(`  ✗ [${field}]: STILL CONTAINS RECOMMENDATION TEXT`);
    console.log(`    "${val.substring(0, 150)}"`);
    issues++;
  } else {
    const preview = val.replace(/\n/g, ' ').substring(0, 90);
    console.log(`  ✓ [${field}]: "${preview}${val.length > 90 ? '...' : ''}"`);
  }
}

console.log('\nPatient:', JSON.stringify(doc.patient));
console.log('ProcessingTime:', body.processingTime, 'ms');

if (issues === 0) {
  console.log('\n✓ ALL TESTS PASSED\n');
} else {
  console.log(`\n✗ ${issues} ISSUE(S) FOUND\n`);
  process.exit(1);
}
