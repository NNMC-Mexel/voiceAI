import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LLMService } from './services/llm.ts';
import type { LLMConfig, MedicalDocument } from './types.js';

const config: LLMConfig = {
  provider: 'ollama',
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

test('validateAndCleanDocument returns life history tail from conclusion to clinicalCourse', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  doc.clinicalCourse = 'Амбулаторная терапия. АМЛОДИПЕН 5 мг по одной таблетке вечером продолжает принимать. ИНДАПАМИТ 1,5 мг по одной таблетки утром продолжает принимать нерегулярно. АТРВОСТАТИН 20 мг ранее назначался, принимает не регулярно. Туберкулез, вирусные гепатиты, ВИЧ-инфекцию и венерические заболевания отрицает. Операции - холецистектомия в 2018 году. Травм с госпитализацией не было. Наследственность отягощена.';

  const cleaned = service.validateAndCleanDocument(doc);

  assert.match(cleaned.conclusion, /АМЛОДИПЕН/u);
  assert.match(cleaned.conclusion, /ИНДАПАМИТ/u);
  assert.doesNotMatch(cleaned.conclusion, /Туберкулез/u);
  assert.match(cleaned.clinicalCourse, /Туберкулез/u);
  assert.match(cleaned.clinicalCourse, /Наследственность/u);
});

test('rescueExamsFromRawText preserves raw lab blocks when template parsing would drop values', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  const rawText = [
    'Амбулаторные обследования ОАК от 27.04.2026г. Гемоглобин 132 грамма на литр Эритроциты 4,48 *10¹²/л. Лейкоциты 9,8 *10⁹/л. Тромбоциты 278 *10⁹/л. СОЭ 24 мм/ч.',
    'Биохимический анализ крови от 27.04.2026г. Креатинин 92 микромольна литр. Мочевина 7,2 миллимольна литр. Мочевая кислота 386 микромольна литр. Глюкоза 6,2 ммоль/л. Общий холестерин 6,4 ммоль/л. ЛПНП 3,9 ммоль/л. ЛПВП 1,05 ммоль/л. Триглицерида 2,1 ммоль/л. АЛТ 36 единиц на литр АСТ 29 единиц на литр. С-реактивный белок 12,4 мг/л. Калий 4,2 ммоль/л. Натрий 139 ммоль/л. Гликированный гемоглобин от 27.04.2026г. 6,1%.',
    'Общий анализ мочи от 27.04.2026г. Относительная плотность 1,002, pH 6,0 белок следы, глюкоза отрицательна, кетоновые тела отрицательно, нитриты положительно, лейкоциты 18-22 в поле зрения, эритроциты 2-3 в поле зрения бактерии плюс.',
  ].join(' ');

  service.rescueExamsFromRawText(doc, rawText);

  assert.match(doc.outpatientExams, /СОЭ 24/u);
  assert.match(doc.outpatientExams, /С-реактивный белок 12,4/u);
  assert.match(doc.outpatientExams, /Калий 4,2/u);
  assert.match(doc.outpatientExams, /Натрий 139/u);
  assert.match(doc.outpatientExams, /Гликированный гемоглобин/u);
  assert.match(doc.outpatientExams, /нитриты положительно/u);
  assert.match(doc.outpatientExams, /бактерии плюс/u);
});

test('semantic passes remove heading echo and route peripheral edema to objective status', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  doc.diagnosis = 'Артериальная гипертензия II степени. Заключительный диагноз.';
  doc.objectiveStatus = 'Общее состояние относительно удовлетворительное.';
  const rawText = 'Объективный статус. Общее состояние относительно удовлетворительное. Периферические атаки, пастузность голеней к вечеру. Заключительный диагноз. Основной диагноз артериальная гипертензия.';

  service.runSemanticRoutingPasses(doc, rawText);

  assert.doesNotMatch(doc.diagnosis, /Заключительный диагноз/u);
  assert.match(doc.objectiveStatus, /Периферические отеки/u);
  assert.match(doc.objectiveStatus, /пастозность/u);
});

test('semantic passes normalize common Whisper artifacts in raw lab blocks', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  doc.allergyHistory = 'Аллергическая реакция на амоксицелин.';
  doc.conclusion = 'АМЛОДИПЕН 5 мг по одной таблетке вечером продолжает принимать. ИНДАПАМИТ 1,5 мг по одной таблетки утром продолжает принимать нерегулярно. АТРВОСТАТИН 20 мг ранее назначался, принимает не регулярно.';
  doc.outpatientExams = [
    '1. ОАК от 27.04.2026г. Гемоглобин 132 грамма на литр. Сою 24 ммч.',
    '2. Биохимический анализ крови от 27.04.2026г. Креатинин 92 микромольна литр. Глюкоза 6,2 ммол на литр. АЛ-36 единиц на литр с Т29 единиц на литр. С-реактивный белок 12,4 мг/л. Общий.',
    '3. Общий анализ мочи от 27.04.2026г. Относительная плотность 1,002, PH6-0 белок следы, лейкоциты 1822 в поле зрения, эритроциты 2,3 в поле зрения.',
    '4. ЭКО 28.04.2026г.-года, синусовый ритм.',
    '5. Эхакаге от 15.03.2026г. ФВ левого желудочка 61%.',
  ].join('\n');

  service.runSemanticRoutingPasses(doc, '');

  assert.match(doc.conclusion, /^1\. АМЛОДИПЕН/mu);
  assert.match(doc.conclusion, /^2\. ИНДАПАМИТ/mu);
  assert.match(doc.allergyHistory, /амоксициллин/u);
  assert.match(doc.outpatientExams, /СОЭ 24 мм\/ч/u);
  assert.match(doc.outpatientExams, /Креатинин 92 мкмоль\/л/u);
  assert.match(doc.outpatientExams, /Глюкоза 6,2 ммоль\/л/u);
  assert.match(doc.outpatientExams, /АЛТ 36 Ед\/л\. АСТ 29 Ед\/л/u);
  assert.match(doc.outpatientExams, /pH 6,0/u);
  assert.match(doc.outpatientExams, /лейкоциты 18-22 в поле зрения/u);
  assert.match(doc.outpatientExams, /эритроциты 2-3 в поле зрения/u);
  assert.match(doc.outpatientExams, /ЭКГ 28\.04\.2026/u);
  assert.match(doc.outpatientExams, /ЭхоКГ от 15\.03\.2026/u);
  assert.doesNotMatch(doc.outpatientExams, /Общий\.$/u);
});

test('rescueExamsFromRawText keeps instrumental exams with dotted dates', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  const rawText = [
    'ЭКО 28.04.2026г.-года, синусовый ритм, ЧСС – 82 уд/мин, электрическое ось сердца отклонена влево, признаки гипертрофии левого желудочка, острых ишемических изменений не зарегистрировано.',
    'Эхакаге от 15.03.2026г. ФВ левого желудочка 61%, концентрическая гипертрофия левого желудочка диастолическая дисфункции 1 типа.',
    'УЗИ почек и мочевого пузыря от 28.04.2026г.-года, почки обычных размеров, чашечно-лоханочная система не расширена, конкрементов не выявлено, остаточная моча 20 мл.',
    'Суточное монетарирование артериального давления от 20.04.2026г.-года, среднесуточное артериальное Недостаточное ночное снижение давления, эпизоды систолического давление до 176 мм рт.ст.',
    'Туберкулез отрицает.',
  ].join(' ');

  service.rescueExamsFromRawText(doc, rawText);
  service.runSemanticRoutingPasses(doc, '');

  assert.match(doc.outpatientExams, /ЭКГ 28\.04\.2026/u);
  assert.match(doc.outpatientExams, /ЭхоКГ от 15\.03\.2026/u);
  assert.match(doc.outpatientExams, /УЗИ почек и мочевого пузыря от 28\.04\.2026/u);
  assert.match(doc.outpatientExams, /Суточное мониторирование артериального давления от 20\.04\.2026/u);
  assert.doesNotMatch(doc.outpatientExams, /Туберкулез/u);
});

test('raw exam rescue does not create CT from infarct wording', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  const rawText = 'Анамнез жизни. Наследственность отягощена: отец перенес инфаркт миокарда в 58 лет, мать страдала сахарным диабетом 2 типа. Физическая активность низкая.';

  service.rescueExamsFromRawText(doc, rawText);

  assert.equal(doc.outpatientExams, '');
});

test('coverage recovery keeps alcohol history out of recommendations', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  doc.clinicalCourse = 'Туберкулез, вирусные гепатиты, ВИЧ-инфекцию отрицает. Алкоголь употребляет редко.';
  doc.recommendations = '1. Полный отказ от курения.';
  const rawText = [
    'Анамнез жизни. Туберкулез, вирусные гепатиты, ВИЧ-инфекцию отрицает.',
    'Алкоголь употребляют редко.',
    'Рекомендации и план лечения. Полный отказ от курения.',
  ].join(' ');

  service.runSemanticRoutingPasses(doc, rawText);

  assert.match(doc.clinicalCourse, /Алкоголь употребляет редко/u);
  assert.doesNotMatch(doc.recommendations, /Алкоголь употреб/u);
});

test('conclusion formatter splits glued current therapy medications', () => {
  const service = new LLMService(config) as any;
  const formatted = service.formatConclusionAsList(
    'Валсартан 160 мг по одной таблетке утром продолжает принимать, а Амлодипин 5 мг по 1 таблетке вечером продолжает принимать нерегулярно. Метформин 1000 мг по одной таблетке 2 раза в день принимает нерегулярно, а Торвастатин 20 мг принимает нерегулярно. Ацетилсалициловая кислота 75 мг ранее не принимал.'
  );

  assert.match(formatted, /^1\. Валсартан 160 мг/mu);
  assert.match(formatted, /^2\. Амлодипин 5 мг/mu);
  assert.match(formatted, /^3\. Метформин 1000 мг/mu);
  assert.match(formatted, /^4\. Торвастатин 20 мг/mu);
  assert.match(formatted, /^5\. Ацетилсалициловая кислота 75 мг/mu);
});

test('semantic passes normalize Demidov Whisper artifacts', () => {
  const service = new LLMService(config) as any;
  const doc = mkDoc();
  doc.complaints = 'Надавящие боли за грудиной. На дышку при подъеме на второй этаж.';
  doc.anamnesis = 'Появление давящих болей за грудины при ходьбе.';
  doc.conclusion = 'Торвастатин 20 мг принимает нерегулярно.';
  doc.recommendations = '1. Ацетилсолициловая кислота 75 мг ежедневно.\n2. Розовостатин 20 мг вечером.\n3. Эмпоглифлозин 10 мг утром.\n4. Решить вопрос с учетом положительного тридмил-теста.';
  doc.neurologicalStatus = 'Парестезий нет. Мышечная сила сохранена.';

  service.runSemanticRoutingPasses(doc, '');

  assert.match(doc.complaints, /На давящие боли/u);
  assert.match(doc.complaints, /На одышку/u);
  assert.match(doc.anamnesis, /за грудиной/u);
  assert.match(doc.conclusion, /Аторвастатин 20 мг/u);
  assert.match(doc.recommendations, /Ацетилсалициловая кислота/u);
  assert.match(doc.recommendations, /Розувастатин 20 мг/u);
  assert.match(doc.recommendations, /Эмпаглифлозин 10 мг/u);
  assert.match(doc.recommendations, /тредмил-теста/u);
  assert.match(doc.neurologicalStatus, /Парезов нет/u);
  assert.doesNotMatch(doc.neurologicalStatus, /Парестезий нет/u);
});
