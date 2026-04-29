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

function mkDoc(): MedicalDocument {
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
    recommendations: '',
  };
}

test('postProcessOutpatientExams preserves rich LLM lab lines instead of replacing them with empty templates', () => {
  const service = new LLMService(config) as any;
  const input = [
    '1. ОАК от 27.04.2026г.: Гемоглобин (HGB) 132 г/л, Эритроциты (RBC) 4,48 *10¹²/л, Лейкоциты (WBC) 9,8 *10⁹/л, Тромбоциты (PLT) 278 *10⁹/л, СОЭ (ESR) 24 мм/ч.',
    '2. Биохимический анализ крови от 27.04.2026г.: Креатинин (CREA) 92 мкмоль/л, Глюкоза (GLU) 6,2 ммоль/л, HbA1c 6,1%.',
  ].join('\n');

  const output = service.postProcessOutpatientExams(input);

  assert.match(output, /Гемоглобин \(HGB\) 132 г\/л/u);
  assert.match(output, /Креатинин \(CREA\) 92 мкмоль\/л/u);
  assert.match(output, /HbA1c 6,1%/u);
  assert.doesNotMatch(output, /Hb\s+-\s+г\/л/u);
});

test('rescueExamsFromRawText does not append truncated HbA1c when HbA1c already exists', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  doc.outpatientExams = '1. Биохимический анализ крови от 27.04.2026г.: Креатинин 92 мкмоль/л, HbA1c 6,1%.';
  const rawText = 'Гликированный гемоглобин (HbA1c) от 27.04.2026г. 6,1%. Общий анализ мочи от 27.04.2026г.';

  service.rescueExamsFromRawText(doc, rawText);

  assert.equal((doc.outpatientExams.match(/HbA1c/gu) || []).length, 1);
  assert.doesNotMatch(doc.outpatientExams, /Гликированный гемоглобин \(HbA1c\) от 27\./u);
});

test('cleanAllergyHistory keeps iodine contrast reaction sentence in allergyHistory', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  doc.allergyHistory = 'Аллергическая реакция на амоксициллин в виде кожной сыпи. Пищевую аллергию отрицает. Реакции на йод содержащие препараты ранее не отмечала.';

  service.cleanAllergyHistory(doc);

  assert.match(doc.allergyHistory, /йод содержащие препараты/u);
  assert.equal(doc.objectiveStatus, '');
});

test('parseDocumentWithRepair rejects empty LLM content instead of repairing it into garbage', async () => {
  const service = new LLMService(config) as any;
  await assert.rejects(
    () => service.parseDocumentWithRepair(''),
    /empty content/i
  );
});

test('parseDocumentWithRepair rejects boolean placeholder documents', async () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  for (const field of [
    'complaints',
    'anamnesis',
    'outpatientExams',
    'clinicalCourse',
    'allergyHistory',
    'objectiveStatus',
    'diagnosis',
    'finalDiagnosis',
    'conclusion',
    'doctorNotes',
    'recommendations',
  ] as const) {
    doc[field] = 'yes';
  }

  await assert.rejects(
    () => service.parseDocumentWithRepair(JSON.stringify(doc)),
    /boolean placeholders/i
  );
});

test('validateAndCleanDocument clears yes/no placeholders in text fields', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  doc.clinicalCourse = 'Yes';
  doc.allergyHistory = 'No';
  doc.conclusion = 'true';

  const cleaned = service.validateAndCleanDocument(doc);

  assert.equal(cleaned.clinicalCourse, '');
  assert.equal(cleaned.allergyHistory, '');
  assert.equal(cleaned.conclusion, '');
});

test('postProcessOutpatientExams preserves qualitative OAM values', () => {
  const service = new LLMService(config) as any;
  const input = '1. Общий анализ мочи от 27.04.2026г.: Относительная плотность 1,002, pH 6,0, белок следы, глюкоза отрицательна, кетоны отрицательно, нитриты положительно, лейкоциты 1822 в поле зрения, эритроциты 2,3 в поле зрения, бактерии +.';

  const output = service.postProcessOutpatientExams(input);

  assert.match(output, /нитриты положительно/u);
  assert.match(output, /бактерии \+/u);
  assert.match(output, /белок следы/u);
});

test('groupRecommendations keeps already numbered independent recommendations separate', () => {
  const service = new LLMService(config) as any;
  const input = [
    '1. Цефиксим 400 мг по одной таблетке 1 раз в день после еды курс 7 дней.',
    '2. Канефрон N по 2 таблетки 3 раза в день 14 дней.',
    '3. Увеличить питьевой режим до 1,8-2,0 литра в сутки.',
    '4. Амлодипин увеличить до 10 мг вечером.',
    '5. Ограничить соль до 5 граммов в сутки.',
    '6. Снижение массы тела на 5-7% за 6 месяцев.',
  ].join('\n');

  const output = service.groupRecommendations(input);
  const lines = output.split('\n');

  assert.equal(lines.length, 6);
  assert.match(lines[1], /^2\. Канефрон/u);
  assert.match(lines[4], /^5\. Ограничить/u);
});
