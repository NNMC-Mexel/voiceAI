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

function emptyDoc(): MedicalDocument {
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

test('raw rescue fills patient, complaints and diagnosis from compact new document text', () => {
  const service = new LLMService(config) as any;
  const doc = emptyDoc();
  const raw = 'Пациент Иванов Иван Иванович 45 лет. Жалобы на головную боль. Диагноз артериальная гипертензия.';

  service.rescueCoreClinicalFieldsFromRawText(doc, raw);
  service.rescueDiagnosisFromRawText(doc, raw);

  assert.equal(doc.patient.fullName, 'Иванов Иван Иванович');
  assert.match(doc.complaints, /головную боль/iu);
  assert.match(doc.diagnosis, /артериальная гипертензия/iu);
});

test('raw rescue reads recommendations block at end of text', () => {
  const service = new LLMService(config) as any;
  const doc = emptyDoc();
  const raw = 'Диагноз: ГЭРБ. Рекомендации: Нольпаза 20 мг 1 раз в день утром 4 недели. Контроль ФГДС в динамике.';

  service.rescueDiagnosisFromRawText(doc, raw);
  service.rescueRecommendationsFromRawText(doc, raw);

  assert.match(doc.diagnosis, /ГЭРБ/iu);
  assert.match(doc.recommendations, /Нольпаза 20 мг/iu);
  assert.match(doc.recommendations, /Контроль ФГДС/iu);
});

test('postprocess strips patient lead from complaints', () => {
  const service = new LLMService(config) as any;
  const doc = emptyDoc();
  doc.patient.fullName = 'Оли Фрустам';
  doc.complaints = 'Пациент Оли Фрустам болен, болит голова, отекают ноги, бронхиты.';

  const result = service.validateAndCleanDocument(doc);

  assert.equal(result.complaints, 'Болит голова, отекают ноги, бронхиты.');
});

test('postprocess expands short diet 1B recommendation', () => {
  const service = new LLMService(config) as any;
  const doc = emptyDoc();
  doc.recommendations = '1. Диета №1Б';

  const result = service.validateAndCleanDocument(doc);

  assert.match(result.recommendations, /^1\. Диета №1б/u);
  assert.match(result.recommendations, /при ГЭРБ/u);
});

test('empty structured result is rejected for meaningful source text', () => {
  const service = new LLMService(config) as any;
  const doc = emptyDoc();

  assert.throws(
    () => service.assertStructuredDocumentHasContent(
      doc,
      'Пациент Иванов Иван Иванович 45 лет. Жалобы на головную боль. Диагноз артериальная гипертензия.'
    ),
    /empty structured document/i
  );
});

test('semantic pass drops generic recommendations not supported by live dictation', () => {
  const service = new LLMService(config) as any;
  const doc = emptyDoc();
  doc.patient.fullName = 'Иванов Иван Иванович';
  doc.complaints = 'Головная боль';
  doc.diagnosis = 'Артериальная гипертензия';
  doc.recommendations = '1. Ограничить алкоголь и соблюдать диету\n2. Контроль веса';

  service.runSemanticRoutingPasses(
    doc,
    'Пациент Иванов Иван Иванович. Жалобы на головную боль. Диагноз артериальная гипертензия.'
  );

  assert.equal(doc.recommendations, '');
});

test('semantic pass moves recommendation tail out of diagnosis', () => {
  const service = new LLMService(config) as any;
  const doc = emptyDoc();
  doc.patient.fullName = 'Иванов Иван Иванович';
  doc.complaints = 'Головная боль';
  doc.diagnosis = 'Артериальная гипертензия. Рекомендовано: контроль АД, бисопролол 5 мг утром, повторный прием кардиолога.';

  service.runSemanticRoutingPasses(
    doc,
    'Пациент Иванов Иван Иванович. Жалобы на головную боль. Диагноз: артериальная гипертензия. Рекомендовано: контроль АД, бисопролол 5 мг утром, повторный прием кардиолога.'
  );

  assert.equal(doc.diagnosis, 'Артериальная гипертензия');
  assert.match(doc.recommendations, /бисопролол 5 мг/iu);
  assert.match(doc.recommendations, /кардиолога/iu);
});
