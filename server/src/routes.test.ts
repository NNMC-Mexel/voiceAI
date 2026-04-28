import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  resolveUploadPath,
  toSafeUploadFilename,
  isValidMedicalDocument,
  assessDocumentUsefulness,
  isFieldMeaningful,
  collectDocumentQualityWarnings,
  collectDocumentQualityWarningDetails,
} from './routes.ts';
import type { MedicalDocument } from './types.js';

function mkDoc(overrides: Partial<MedicalDocument> = {}): MedicalDocument {
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
    ...overrides,
  };
}

test('toSafeUploadFilename removes unsafe chars', () => {
  assert.equal(toSafeUploadFilename('../../bad name?.webm'), 'bad_name_.webm');
});

test('resolveUploadPath blocks traversal outside upload root', () => {
  assert.throws(() => resolveUploadPath('./uploads', '../secret.txt'));
});

test('resolveUploadPath allows valid file path', () => {
  const resolved = resolveUploadPath('./uploads', '1700000_audio.webm');
  assert.match(resolved, /uploads/);
});

// ─── assessDocumentUsefulness — критерий «полезности» документа ──────────────

test('isFieldMeaningful rejects bare section headers', () => {
  assert.equal(isFieldMeaningful('Жалобы'), false);
  assert.equal(isFieldMeaningful('Жалобы.'), false);
  assert.equal(isFieldMeaningful('Диагноз'), false);
  assert.equal(isFieldMeaningful('Анамнез заболевания'), false);
  assert.equal(isFieldMeaningful('Аллергологический анамнез.'), false);
  assert.equal(isFieldMeaningful(''), false);
});

test('isFieldMeaningful accepts clinical content', () => {
  assert.equal(isFieldMeaningful('Жалобы. Головные боли с утра.'), true);
  assert.equal(isFieldMeaningful('Артериальная гипертензия II степени.'), true);
  assert.equal(isFieldMeaningful('Бисопролол 2,5 мг 1 раз в день.'), true);
});

test('isFieldMeaningful rejects bare diagnosis with too-short body', () => {
  // «Диагноз: АГ» — после strip остаётся ": АГ", меньше 10 символов и < 2 слов
  assert.equal(isFieldMeaningful('Диагноз: АГ'), false);
});

test('assessDocumentUsefulness flags placeholder-only document as empty', () => {
  // Точный кейс из QA: «Пациент Иванов. Жалобы. Диагноз.» → placeholder-документ
  const doc = mkDoc({
    patient: { fullName: 'Иванов', age: '', gender: '', complaintDate: '' },
    complaints: 'Жалобы',
    diagnosis: 'Диагноз',
  });
  const r = assessDocumentUsefulness(doc);
  assert.equal(r.status, 'empty');
  assert.equal(r.reason, 'document_appears_empty');
  assert.ok(r.placeholderFields.includes('complaints'));
  assert.ok(r.placeholderFields.includes('diagnosis'));
});

test('assessDocumentUsefulness accepts full clinical document', () => {
  const doc = mkDoc({
    patient: { fullName: 'Иванов Иван Петрович', age: '45 лет', gender: 'мужской', complaintDate: '2026-04-25' },
    complaints: 'Головные боли в течение 2 месяцев, повышение АД до 145/90 мм рт.ст.',
    anamnesis: 'Считает себя больным около 2 месяцев.',
    diagnosis: 'Артериальная гипертензия II степени, риск 3.',
    recommendations: '1. Бисопролол 2,5 мг 1 раз в день.\n2. Контроль АД.',
  });
  const r = assessDocumentUsefulness(doc);
  assert.equal(r.status, 'ok');
  assert.ok(r.meaningfulFields.includes('complaints'));
  assert.ok(r.meaningfulFields.includes('diagnosis'));
});

test('assessDocumentUsefulness flags labs-only document as labs_only', () => {
  // Аудио только с лабораторными данными: outpatientExams содержательный,
  // остальные клинические поля либо пустые, либо placeholder.
  const doc = mkDoc({
    complaints: 'Жалобы',
    diagnosis: 'Диагноз',
    outpatientExams:
      '1. ОАК от 22.04.2026: гемоглобин 128 г/л, эритроциты 4,32 ×10¹²/л, лейкоциты 10,8 ×10⁹/л. ' +
      '2. Биохимический анализ: креатинин 104,2 мкмоль/л, мочевина 8,6 ммоль/л, глюкоза 6,4 ммоль/л.',
  });
  const r = assessDocumentUsefulness(doc);
  assert.equal(r.status, 'labs_only');
  assert.ok(r.meaningfulFields.includes('outpatientExams'));
  assert.equal(r.meaningfulFields.filter((f) => f !== 'outpatientExams').length, 0);
});

test('assessDocumentUsefulness ignores patient.fullName alone', () => {
  // ФИО без клинического содержимого — не «спасает» документ
  const doc = mkDoc({
    patient: { fullName: 'Иванов Иван Петрович', age: '45 лет', gender: 'мужской', complaintDate: '2026-04-25' },
  });
  const r = assessDocumentUsefulness(doc);
  assert.equal(r.status, 'empty');
});

test('collectDocumentQualityWarnings reports missing BP value from raw text', () => {
  const doc = mkDoc({
    complaints: 'Повышение артериального давления до 165 мм рт.ст.',
    diagnosis: 'Артериальная гипертензия II степени.',
    recommendations: 'Контроль артериального давления утром и вечером.',
  });
  const usefulness = assessDocumentUsefulness(doc);
  const warnings = collectDocumentQualityWarnings('АД до 165/100 мм рт.ст.', doc, usefulness);
  assert.ok(warnings.includes('important_number_missing:bp_165/100'));
});

test('collectDocumentQualityWarnings reports suspicious unit garbage', () => {
  const doc = mkDoc({
    outpatientExams: 'Гемоглобин 128 СЛЧЭЛЬ, СОЭ 22 ММС ЛЩЕ.',
  });
  const usefulness = assessDocumentUsefulness(doc);
  const warnings = collectDocumentQualityWarnings('Гемоглобин 128 г/л.', doc, usefulness);
  assert.ok(warnings.includes('suspicious_unit_garbage_in_document'));
});

test('collectDocumentQualityWarningDetails reports advanced QA issues', () => {
  const doc = mkDoc({
    outpatientExams: '1. КТ миокарда в 58 лет, мать страдала сахарным диабетом 2 типа.',
    recommendations: '1. Алкоголь употребляет редко.\n2. Валсартан 160 мг и Амлодипин 5 мг принимать ежедневно.',
  });
  const usefulness = assessDocumentUsefulness(doc);
  const warnings = collectDocumentQualityWarningDetails(
    'Наследственность отягощена: отец перенес инфаркт миокарда в 58 лет. Рекомендации: Валсартан 160 мг принимать ежедневно.',
    doc,
    usefulness,
  );
  const codes = warnings.map((w) => w.code);
  assert.ok(codes.includes('suspiciousExamRescue'));
  assert.ok(codes.includes('sectionRoutingIssue'));
  assert.ok(codes.includes('drugListMayBeMerged'));
});

test('collectDocumentQualityWarningDetails reports possible lost lab value', () => {
  const doc = mkDoc({
    outpatientExams: '1. Биохимический анализ крови: креатинин 98 мкмоль/л.',
  });
  const usefulness = assessDocumentUsefulness(doc);
  const warnings = collectDocumentQualityWarningDetails(
    'Биохимический анализ крови: креатинин 98 микромоль на литр, глюкоза 8,9 миллимоль на литр.',
    doc,
    usefulness,
  );
  assert.ok(warnings.some((w) => w.code === 'possibleLostLabValue' && w.evidence === '8,9'));
});

test('isValidMedicalDocument validates minimal shape', () => {
  const valid = {
    patient: {
      fullName: '',
      age: '',
      gender: '',
      complaintDate: '',
    },
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
    diet: '',
  };

  assert.equal(isValidMedicalDocument(valid), true);
  assert.equal(isValidMedicalDocument({}), false);
});
