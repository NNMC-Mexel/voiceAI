import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  resolveUploadPath,
  toSafeUploadFilename,
  isValidMedicalDocument,
  documentFromExactSourceText,
  documentFromConsultationProtocolText,
  assessDocumentUsefulness,
  isFieldMeaningful,
  collectDocumentQualityWarnings,
  collectDocumentQualityWarningDetails,
  enrichDocumentFromSourceName,
  withTimeout,
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

test('withTimeout returns fallback when dependency health check hangs', async () => {
  const result = await withTimeout(new Promise<boolean>(() => undefined), 10, false);
  assert.equal(result, false);
});

test('quality warnings flag live dictation without patient identity', () => {
  const doc = mkDoc({
    complaints: 'Головная боль',
    diagnosis: 'Артериальная гипертензия',
    recommendations: '1. Контроль АД',
  });

  const warnings = collectDocumentQualityWarningDetails(
    'Жалобы на головную боль. Диагноз артериальная гипертензия. Рекомендован контроль АД.',
    doc,
    assessDocumentUsefulness(doc),
  );

  assert.ok(warnings.some((warning) => warning.code === 'patient_identity_missing'));
});

test('quality warnings flag generic recommendation without raw intent', () => {
  const doc = mkDoc({
    patient: { fullName: 'Иванов Иван Иванович', age: '45 лет', gender: 'мужской', complaintDate: '' },
    complaints: 'Головная боль',
    diagnosis: 'Артериальная гипертензия',
    recommendations: '1. Ограничить алкоголь и соблюдать диету',
  });

  const warnings = collectDocumentQualityWarningDetails(
    'Пациент Иванов Иван Иванович 45 лет. Жалобы на головную боль. Диагноз артериальная гипертензия.',
    doc,
    assessDocumentUsefulness(doc),
  );

  assert.ok(warnings.some((warning) => warning.code === 'unsupportedRecommendation'));
});

// ─── assessDocumentUsefulness — критерий «полезности» документа ──────────────

test('documentFromExactSourceText keeps patient and lab facts, drops OCR boilerplate', () => {
  const doc = documentFromExactSourceText(`
около 10 к Анамнез: заболевание проходил в 2020 году, с тех пор [неразборчиво]
Министерство здравоохранения
Форма №097/у
EFQM
Recognised for excellence
5 star

№ карты: 441480
ИНН: 000908551579
УАЛИЕВ РУСТАМ РАШАТОВИЧ (Муж)
Дата рождения: 08.09.2000
№ направления: 81007296
Заказ №26033103703633 от 31.03.2026
Кровь ЭДТА
Дата взятия: 31.03.2026 11:14
Выполнено: 31.03.2026 11:35
Показатель Результат Ед.изм. Реф.интервал
Общий анализ крови
Лейкоциты (WBC) 5.93 10E9/л (4.00 - 10.50)
Гемоглобин (HGB) 159.00 г/л (136.00 - 169.00)
СОЭ 4.00 мм/ч (0.00 - 15.00)
Интерпретацию полученных результатов проводит Врач
Адрес: г. Астана, пр. Абылай хан, 42
`);

  assert.equal(doc.patient.fullName, 'УАЛИЕВ РУСТАМ РАШАТОВИЧ');
  assert.equal(doc.patient.age, '25 лет');
  assert.equal(doc.patient.gender, 'мужской');
  assert.equal(doc.patient.complaintDate, '2026-03-31');
  assert.equal(doc.patient.birthDate, '2000-09-08');
  assert.match(doc.outpatientExams, /Лейкоциты \(WBC\) 5\.93 10E9\/л/u);
  assert.match(doc.outpatientExams, /Гемоглобин \(HGB\) 159\.00 г\/л/u);
  assert.doesNotMatch(doc.outpatientExams, /Анамнез|Министерство|EFQM|Адрес|неразборчиво/iu);
  assert.equal(doc.complaints, '');
  assert.equal(doc.anamnesis, '');
});

test('documentFromConsultationProtocolText preserves generated protocol sections', () => {
  const doc = documentFromConsultationProtocolText(`
КОНСУЛЬТАЦИЯ
Дата составления: 09.04.2026
ФИО пациента: -
Возраст: -
Пол: -
Дата обращения: 09.04.2026
Оценка риска (шкала Морзе)
Падал (3 мес.): нет
Головокружение: нет
Сопровождение: нет
Оценка боли: 0б
Амбулаторные обследования
1. ОАК от 11.03.2026: Hb - 139 г/л, Эр - 4,6 *10¹²/л, Тр - 299 *10⁹/л, Л - 4,5 *10⁹/л, СОЭ - 9 мм/ч.
2. Б/х анализ крови от 11.03.2026: креатинин - 75,8 мкмоль/л, глюкоза - 5,2 ммоль/л, АЛТ - 13,1 МЕ/л,
АСТ - 17,4 МЕ/л, общий билирубин - 10,1 мкмоль/л, прямой билирубин - 2,4 мкмоль/л, ТГ - 1,45
ммоль/л.
3. Ферритин от 05.10.2025г: 295 нг/мл
Подпись врача
Документ сформирован автоматически 09.04.2026
-- 1 of 1 --
`);

  assert.ok(doc);
  assert.equal(doc.patient.complaintDate, '2026-04-09');
  assert.equal(doc.riskAssessment.fallInLast3Months, 'нет');
  assert.equal(doc.riskAssessment.painScore, '0');
  assert.match(doc.outpatientExams, /АСТ - 17,4 МЕ\/л/u);
  assert.match(doc.outpatientExams, /Ферритин от 05\.10\.2025г: 295 нг\/мл/u);
  assert.equal(doc.recommendations, '');
  assert.doesNotMatch(doc.outpatientExams, /\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442\s+\u0441\u0444\u043e\u0440\u043c\u0438\u0440\u043e\u0432\u0430\u043d/u);
});

test('documentFromConsultationProtocolText does not let nested protocol text overwrite header fields', () => {
  const doc = documentFromConsultationProtocolText(`
КОНСУЛЬТАЦИЯ
Дата составления: 09.04.2026
Оценка риска (шкала Морзе)
Оценка боли: 0б
Рекомендации / План лечения
Оценка боли: 9б Амбулаторные обследования 1.
Дата обращения: 01.01.2099
`);

  assert.ok(doc);
  assert.equal(doc.patient.complaintDate, '2026-04-09');
  assert.equal(doc.riskAssessment.painScore, '0');
  assert.match(doc.recommendations, /Оценка боли: 9б/u);
  assert.match(doc.recommendations, /Дата обращения: 01\.01\.2099/u);
});

test('documentFromConsultationProtocolText preserves a single outpatient exams screenshot section', () => {
  const doc = documentFromConsultationProtocolText(`
Амбулаторные обследования

1. ОАК от 11.03.2026: Нь - 139 г/л, Эр - 4,6 *10^12/л, Тр - 299 *10^9/л, Л - 4,5 *10^9/л, СОЭ - 9 мм/ч.

2. Б/х анализ крови от 11.03.2026: креатинин - 75,8 мкмоль/л, глюкоза - 5,2 ммоль/л, АЛТ - 13,1 МЕ/л, АСТ - 17,4 МЕ/л, общий билирубин - 10,1 мкмоль/л, прямой билирубин - 2,4 мкмоль/л, ТГ - 1,45 ммоль/л.

3. Ферритин от 05.10.2025: 295 нг/мл

4. ОАМ от 11.03.2026: белок - 0 г/л, Л - 0 в п/з, Эр - 0 в п/з.

5. ЭхоКГ от 14.03.2023г: Признаки минимальной трикуспидальной регургитации, не связанной с легочной гипертензией, диастолическая функция в норме, FV 69.5%, NPV признаков застоя нет

6. УЗДГ БЦА от 13.02.2026г: КИМ не утолщен, атеросклеротического поражения артерии не выявлено, атеросклеротического поражения артерии нет

7. кт три Ферритин от 05.

8. ОАМ от 11: отн. плотность - , белок - г/л, Л - в п/з, Эр - в п/з.

9. кт 5 ЭХОК эхо кардиография от 14.

10. кт 5 Эхо кардиография от 14.

11. кт 6 УЗДГ БЦА – ультразвуковая допплерография брахиоцефальных артерий от 13.
`);

  assert.ok(doc);
  assert.match(doc.outpatientExams, /1\. ОАК от 11\.03\.2026: Hb - 139/u);
  assert.match(doc.outpatientExams, /2\. Б\/х анализ крови/u);
  assert.match(doc.outpatientExams, /5\. ЭхоКГ/u);
  assert.match(doc.outpatientExams, /11\. кт 6 УЗДГ БЦА/u);
  assert.equal((doc.outpatientExams.match(/^\d+\./gmu) || []).length, 11);
});

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

test('collectDocumentQualityWarnings does not flag real words that contain a garbage token', () => {
  const doc = mkDoc({
    outpatientExams: 'УЗДГ БЦА: КИМ не утолщен, атеросклеротического поражения артерии нет.',
  });
  const usefulness = assessDocumentUsefulness(doc);
  const warnings = collectDocumentQualityWarnings(doc.outpatientExams, doc, usefulness);
  assert.ok(!warnings.includes('suspicious_unit_garbage_in_document'));
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

test('collectDocumentQualityWarningDetails reports low document coverage', () => {
  const raw = [
    'Жалобы на боль в грудной клетке при нагрузке.',
    'Анамнез: в течение двух месяцев отмечает одышку, сердцебиение, повышение давления до 165/100.',
    'По ЭКГ синусовый ритм, по ЭхоКГ ФВ 48 процентов.',
    'Рекомендовано принимать бисопролол 5 мг и контроль кардиолога.',
  ].join(' ');
  const doc = mkDoc({
    complaints: 'Боль в грудной клетке при нагрузке.',
  });
  const usefulness = assessDocumentUsefulness(doc);
  const warnings = collectDocumentQualityWarningDetails(raw, doc, usefulness);
  assert.ok(warnings.some((w) => w.code === 'lowDocumentCoverage'));
});

test('collectDocumentQualityWarningDetails reports missing critical section by raw marker', () => {
  const doc = mkDoc({
    complaints: 'Боль в грудной клетке.',
    diagnosis: 'ИБС.',
  });
  const usefulness = assessDocumentUsefulness(doc);
  const warnings = collectDocumentQualityWarningDetails(
    'Жалобы на боль. Диагноз ИБС. Рекомендовано принимать препарат и контроль врача.',
    doc,
    usefulness,
  );
  assert.ok(warnings.some((w) => w.code === 'criticalFieldMissing' && w.field === 'recommendations'));
});

test('enrichDocumentFromSourceName fills weak patient name from audio filename', () => {
  const doc = mkDoc({
    patient: { fullName: 'Пациентка', age: '', gender: '', complaintDate: '' },
    complaints: 'Боль в ноге.',
  });
  const { document, warnings } = enrichDocumentFromSourceName(
    doc,
    'Иванова Мария Петровна, 12.07.1948г.р. УЗДГ вен нк__01_audio.wav',
  );
  assert.equal(document.patient.fullName, 'Иванова Мария Петровна');
  assert.equal(document.patient.birthDate, '1948-07-12');
  assert.ok(warnings.some((w) => w.code === 'patientNameFromFilename'));
});

test('collectDocumentQualityWarningDetails ignores date-like numbers in lab sources', () => {
  const doc = mkDoc({
    outpatientExams: 'HbA1c 7,2 %.',
  });
  const usefulness = assessDocumentUsefulness(doc);
  const warnings = collectDocumentQualityWarningDetails(
    'date 09.04.2026 HbA1c 7,2 %.',
    doc,
    usefulness,
  );
  assert.ok(!warnings.some((w) => w.code === 'possibleLostLabValue' && w.evidence === '09.04'));
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
