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
