import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LLMService } from './services/llm.ts';
import type { LLMConfig, MedicalDocument } from './types.js';

const config: LLMConfig = {
  provider: 'llama',
  serverUrl: 'http://127.0.0.1:65535',
  model: 'test-model',
  maxTokens: 128,
  temperature: 0,
  parallelSlots: 1,
  requestTimeoutMs: 100,
  allowMockOnFailure: false,
};

function mkDoc(recommendations: string): MedicalDocument {
  return {
    patient: { fullName: '', age: '', gender: '', complaintDate: '' },
    riskAssessment: { fallInLast3Months: '', dizzinessOrWeakness: '', needsEscort: '', painScore: '' },
    complaints: '',
    anamnesis: '',
    outpatientExams: '',
    clinicalCourse: '',
    allergyHistory: '',
    objectiveStatus: '',
    neurologicalStatus: '',
    diagnosis: '',
    finalDiagnosis: '',
    conclusion: '',
    doctorNotes: '',
    recommendations,
  };
}

function cleanReco(doc: MedicalDocument): void {
  (new LLMService(config) as any).cleanRecommendations(doc);
}

function mkEmptyDoc(): MedicalDocument {
  return mkDoc('');
}

test('cleanRecommendations merges continuation tails («По 1 таблетке», «длительно»)', () => {
  const doc = mkDoc([
    '1. Бисопролол 2.5 мг',
    '2. По одной таблетке утром',
    '3. длительно',
    '4. Диета стол №10',
  ].join('\n'));
  cleanReco(doc);
  assert.equal(
    doc.recommendations,
    '1. Бисопролол 2.5 мг, по одной таблетке утром, длительно\n2. Диета стол №10'
  );
});

test('cleanRecommendations dedupes byte-identical items', () => {
  const doc = mkDoc([
    '1. Бисопролол 2.5 мг 1 раз в день',
    '2. Бисопролол 2.5 мг 1 раз в день',
    '3. Контроль АД',
  ].join('\n'));
  cleanReco(doc);
  assert.equal(doc.recommendations, '1. Бисопролол 2.5 мг 1 раз в день\n2. Контроль АД');
});

test('cleanRecommendations dedupes with case/punctuation differences', () => {
  const doc = mkDoc([
    '1. Консультация кардиолога.',
    '2. консультация кардиолога',
  ].join('\n'));
  cleanReco(doc);
  assert.equal(doc.recommendations, '1. Консультация кардиолога.');
});

test('cleanRecommendations does not merge independent capitalized drug items', () => {
  const doc = mkDoc([
    '1. Бисопролол 2.5 мг',
    '2. Ксарелто 20 мг',
    '3. Форсига 10 мг',
  ].join('\n'));
  cleanReco(doc);
  assert.equal(
    doc.recommendations,
    '1. Бисопролол 2.5 мг\n2. Ксарелто 20 мг\n3. Форсига 10 мг'
  );
});

test('cleanRecommendations does not merge directive items («Контроль АД»)', () => {
  const doc = mkDoc([
    '1. Бисопролол 2.5 мг',
    '2. Контроль АД 2 раза в день',
  ].join('\n'));
  cleanReco(doc);
  assert.equal(
    doc.recommendations,
    '1. Бисопролол 2.5 мг\n2. Контроль АД 2 раза в день'
  );
});

test('cleanRecommendations does not merge long-tail fragments (>100 chars)', () => {
  const long = 'по одной таблетке утром постоянно с постепенным увеличением до двух таблеток при сохранении целевого давления';
  const doc = mkDoc(['1. Бисопролол 2.5 мг', '2. ' + long].join('\n'));
  cleanReco(doc);
  const lines = doc.recommendations.split('\n');
  assert.equal(lines.length, 2, 'long tail must stay as a separate item');
});

// Регрессия: пункты с lowercase-началом без continuation-триггеров не должны
// схлопываться. Ранее startsLower-эвристика сшивала их все в один пункт.
test('cleanRecommendations keeps three lowercase-head independent items separate', () => {
  const doc = mkDoc([
    '1. бисопролол 2.5 мг 1 раз в день',
    '2. контроль АД',
    '3. консультация кардиолога',
  ].join('\n'));
  cleanReco(doc);
  const lines = doc.recommendations.split('\n');
  assert.equal(lines.length, 3, 'three independent items must not merge into one');
  assert.match(lines[0], /бисопролол/i);
  assert.match(lines[1], /контроль\s+АД/i);
  assert.match(lines[2], /консультация\s+кардиолога/i);
});

// Регрессия (Dyusenov): orphan continuation-фрагмент без drug-корня не должен
// приклеиваться к несвязанному пункту. Prev = «Памятка по питанию…» не drug,
// continuation «по одной таблетке…» дропается.
test('cleanRecommendations drops orphan continuation when prev has no drug context', () => {
  const doc = mkDoc([
    '1. Сбалансированное питание по правилу тарелки.',
    '2. Памятка по питанию выдана на руки.',
    '3. по одной таблетке одну раз в день',
    '4. по одной капсуле одну раз в день',
  ].join('\n'));
  cleanReco(doc);
  const lines = doc.recommendations.split('\n');
  assert.equal(lines.length, 2, 'orphan continuation fragments must be dropped');
  assert.match(lines[1], /^2\. Памятка по питанию выдана на руки\.?$/u);
});

test('routeMissingSentence routes ambulatory therapy sentence to conclusion', () => {
  const service = new LLMService(config) as any;
  const target = service.routeMissingSentence('Амбулаторная терапия: Амлодипин 5 мг, продолжает принимать ежедневно.');
  assert.equal(target, 'conclusion');
});

test('routeMissingSentence routes plan sentence to doctorNotes', () => {
  const service = new LLMService(config) as any;
  const target = service.routeMissingSentence('План обследования: контроль ОАМ через 7 дней.');
  assert.equal(target, 'doctorNotes');
});

test('recoverMissingContent skips cross-section echo sentence', () => {
  const service = new LLMService(config) as any;
  const doc = mkEmptyDoc();
  const raw = 'Амбулаторная терапия Амлодипин 5 мг продолжает принимать План обследования 1 контроль ОАМ через 7-10 дней.';
  service.recoverMissingContent(doc, raw, new Set<string>());

  assert.equal(doc.recommendations, '');
  assert.equal(doc.doctorNotes, '');
  assert.equal(doc.conclusion, '');
});

test('cleanRecommendations drops short heading-echo hybrid item', () => {
  const doc = mkDoc([
    '1. Рекомендации с ЛЧПЛАН лечения 1.',
    '2. Контроль АД утром и вечером.',
  ].join('\n'));
  cleanReco(doc);
  assert.equal(doc.recommendations, '1. Контроль АД утром и вечером.');
});

// Регрессия: висячий заголовок «Аллергологический анамнез.» в allergyHistory
// должен вырезаться (P5 иногда роутит его туда как отдельный sentence).
test('cleanAllergyHistory strips bare «Аллергологический анамнез.» trailing header', () => {
  const doc = mkEmptyDoc();
  doc.allergyHistory = 'Аллергическая реакция на амоксициллин в виде кожной сыпи. Пищевой аллергией не отмечает. Аллергологический анамнез.';
  (new LLMService(config) as any).cleanAllergyHistory(doc);
  assert.ok(!/аллерголог\S*\s+анамнез\.?\s*$/iu.test(doc.allergyHistory),
    'bare allergy header must be stripped');
  assert.match(doc.allergyHistory, /амоксициллин/i);
});

// Регрессия: бесконтентный двухсловный заголовок «Аллергологический анамнез.»
// из raw Whisper вообще не должен роутиться P5 (иначе cleanAllergyHistory ловит
// его на первом проходе, P5 возвращает обратно — и хвост остаётся в документе).
test('routeMissingSentence returns null for bare «Аллергологический анамнез.» header', () => {
  const service = new LLMService(config) as any;
  assert.equal(service.routeMissingSentence('Аллергологический анамнез.'), null);
  assert.equal(service.routeMissingSentence('Аллергологический анамнез'), null);
});

// Регрессия: LLM-ярлык «(ожирение N степени)» рядом с ИМТ в objectiveStatus
// снимается — степень ожирения принадлежит diagnosis.
test('stripObesityCommentaryFromObjectiveStatus removes obesity degree parenthetical', () => {
  const doc = mkEmptyDoc();
  doc.objectiveStatus = 'Рост 164 см, масса тела 82 кг, ИМТ 30,5 кг/м² (ожирение II степени). Кожные покровы чистые.';
  (new LLMService(config) as any).stripObesityCommentaryFromObjectiveStatus(doc);
  assert.ok(!/ожирение/iu.test(doc.objectiveStatus), 'obesity commentary must be gone');
  assert.match(doc.objectiveStatus, /ИМТ 30,5 кг\/м²/);
  assert.match(doc.objectiveStatus, /Кожные покровы/);
});

// Регрессия: в слепленном пункте с несколькими препаратами drug-signature
// собирается по каждому capitalized head, а не только по первому, чтобы
// P5-echo на второй/третий drug не проходил.
test('extractContinuedDrugsFromRecommendations adds signature for every drug head in composite item', () => {
  const doc = mkDoc([
    '1. Цефиксим 400 мг по 1 таблетке 1 раз в день после еды, курс 7 дней. Канефрон N по 2 таблетки 3 раза в день, 14 дней.',
  ].join('\n'));
  const hints = new Set<string>();
  (new LLMService(config) as any).extractContinuedDrugsFromRecommendations(doc, hints);
  const joined = Array.from(hints).join(' | ');
  assert.match(joined, /Цефиксим/, 'first drug head must be covered');
  assert.match(joined, /Канефрон/, 'second drug head must also be covered');
});
