import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DocEngine } from './radiology/doc-engine.js';
import { structureDictation } from './radiology/dictation.js';
import { RadiologyEngine } from './radiology/engine.js';
import { sectionize, sectionizeWithProvenance } from './radiology/sectionize.js';
import { verifyRadiologySafety } from './radiology/safety.js';
import { abdomenMikhailov } from './radiology/templates/abdomen-mikhailov.js';
import { ctAbdomen } from './radiology/templates/ct-abdomen.js';
import type { LLMCall } from './radiology/ollama.js';

const fakeLLM = (sections: Record<string, string>, unmatched = ''): LLMCall => async () => (
  JSON.stringify({ ...sections, unmatched })
);

const nullClassifier: LLMCall = async (_system, user) => {
  const atoms = (JSON.parse(user) as { atoms: { atomId: string }[] }).atoms;
  return JSON.stringify({
    assignments: atoms.map((atom) => ({ atomId: atom.atomId, sectionId: null })),
  });
};

test('секционизатор не зависит от порядка шаблона и агрегирует возврат к органу', () => {
  const transcript = [
    'Вводная фраза.',
    'Селезёнка размеры 12 на 6 на 5.',
    'Печень плотность 58.',
    'Почки без особенностей.',
    'Печень КВР 145.',
  ].join(' ');

  const result = sectionizeWithProvenance(abdomenMikhailov, transcript);
  assert.deepEqual(
    result.segments.map((segment) => segment.blockId),
    [null, 'spleen', 'liver', 'kidneys', 'liver'],
  );
  assert.equal(result.sections.liver.spans.length, 2);
  assert.match(result.sections.liver.text, /плотность 58/i);
  assert.match(result.sections.liver.text, /КВР 145/i);
  assert.equal(result.unmatchedText, 'Вводная фраза.');

  for (const span of [...result.sections.liver.spans, ...result.unmatched]) {
    assert.equal(transcript.slice(span.start, span.end), span.text);
  }

  // Старый компактный контракт остаётся доступен.
  assert.deepEqual(
    sectionize(abdomenMikhailov, transcript).map((segment) => segment.blockId),
    result.segments.map((segment) => segment.blockId),
  );
});

test('длинный якорь выигрывает у вложенного: «ворот почек» не создаёт ложный блок почек', () => {
  const result = sectionizeWithProvenance(abdomenMikhailov, 'Ворот почек лимфоузлы не увеличены.');
  assert.deepEqual(result.segments.map((segment) => segment.blockId), ['lymph_hilum']);
});

test('одиночное «в воротах» не является lymph-якорем и остаётся в текущем органе', () => {
  const transcript = 'Селезёнка без очагов, в воротах добавочная долька 9 мм. Почки норма.';
  const result = sectionizeWithProvenance(abdomenMikhailov, transcript);
  assert.match(result.sections.spleen.text, /в воротах добавочная долька 9 мм/iu);
  assert.equal(result.sections.lymph_hilum, undefined);

  const bare = sectionizeWithProvenance(abdomenMikhailov, 'В воротах добавочная долька 9 мм.');
  assert.equal(bare.assignments[0].sectionId, null);
  assert.ok(!bare.atoms[0].candidateSectionIds.includes('lymph_hilum'));
});

test('каждый TranscriptAtom имеет ровно одного владельца либо unmatched', () => {
  const result = sectionizeWithProvenance(
    abdomenMikhailov,
    'Селезёнка 12 мм. Печень 58 HU. Почки норма. Печень КВР 145 мм.',
  );
  assert.equal(result.assignments.length, result.atoms.length);
  assert.equal(new Set(result.assignments.map((item) => item.atomId)).size, result.atoms.length);
  for (const atom of result.atoms) {
    const owners = result.assignments.filter((item) => item.atomId === atom.id);
    assert.equal(owners.length, 1);
    assert.equal(result.transcript.slice(atom.start, atom.end), atom.text);
  }
});

test('правила КТ ОБП разделяют таз, кишечник и позвоночник без зависимости от порядка', () => {
  const transcript = [
    'В малом тазу небольшое количество свободной жидкости.',
    'Определяется утолщение стенок сигмовидной и нисходящей ободочной кишки до 5 мм.',
    'Дегенеративно дистрофические изменения поясничного отдела позвоночника.',
  ].join(' ');
  const result = sectionizeWithProvenance(abdomenMikhailov, transcript);

  assert.match(result.sections.pelvis.text, /свободной жидкости/iu);
  assert.doesNotMatch(result.sections.pelvis.text, /утолщение стенок/iu);
  assert.match(result.sections.bowel.text, /утолщение стенок.+ободочной кишки/iu);
  assert.match(result.sections.skeleton.text, /поясничного отдела позвоночника/iu);
});

test('описание компрессии чревного ствола sticky и продиктованное заключение извлекается точно', () => {
  const transcript = [
    'Печень норма.',
    'Поджелудочная норма.',
    'В артериальную фазу определяется деформация устья чредного ствола',
    'вследствие экстравазальной компрессии срединной дугообразной связкой.',
    'КТ признаков ишемических изменений печени селезёнки желудка не определяется.',
    'Заключение КТ признаки умеренной компрессии чревного ствола, стеноз 53 процента.',
  ].join(' ');
  const result = sectionizeWithProvenance(abdomenMikhailov, transcript);

  assert.equal(result.sections.pancreas.text, 'Поджелудочная норма.');
  assert.match(result.sections.celiac_trunk.text, /деформация.+ишемических изменений печени/iu);
  assert.doesNotMatch(result.sections.pancreas.text, /деформация|компресс/iu);
  assert.equal(
    result.dictatedConclusion?.text,
    'Заключение КТ признаки умеренной компрессии чревного ствола, стеноз 53 процента.',
  );
  assert.ok(result.sections.celiac_trunk.assignmentMethods.every((method) => method === 'rule'));
});

test('fill-in слоты связываются с числами при обоих порядках «параметр-число»', () => {
  for (const command of [
    'печень КВР 145 плотность 62',
    'печень 145 КВР 62 плотность',
    'печень 62 плотность 145 КВР',
    'печень 145 мм КВР 62 HU плотность',
  ]) {
    const engine = new DocEngine(abdomenMikhailov);
    assert.equal(engine.apply(command).ok, true, command);
    const liver = engine.build().blocks.find((block) => block.id === 'liver')!.text;
    assert.match(liver, /КВР 145 мм/, command);
    assert.match(liver, /\+62 HU/, command);
  }
});

test('schema-driven движок также принимает число перед параметром', () => {
  const engine = new RadiologyEngine(ctAbdomen);
  assert.equal(engine.apply('печень 56 плотность норма').ok, true);
  const liver = engine.build().sections.find((section) => section.id === 'liver')!.text;
  assert.match(liver, /56 HU/);
});

test('safety: безопасная перестановка фраз и синоним отрицания проходит', () => {
  const source = 'В правой почке конкремент 15 мм, метастазы не выявлены, исследование с контрастом.';
  const output = 'С контрастом, метастазы отсутствуют, конкремент 15 мм в правой почке.';
  const result = verifyRadiologySafety(source, output);
  assert.equal(result.ok, true);
  assert.equal(result.requiresReview, false);
});

test('safety hard-set ловит 15/50, мм/см, справа/слева, отрицание, контраст и новый факт', () => {
  const source = 'Справа образование 15 мм, метастазы не выявлены, исследование с контрастом.';
  const output = 'Слева образование 50 см, метастазы выявлены, без контраста, свободный газ.';
  const result = verifyRadiologySafety(source, output);
  const codes = new Set(result.issues.map((issue) => issue.code));

  assert.equal(result.ok, false);
  assert.equal(result.requiresReview, true);
  assert.ok(codes.has('number_added'));
  assert.ok(codes.has('number_lost'));
  assert.ok(codes.has('laterality_changed'));
  assert.ok(codes.has('negation_changed'));
  assert.ok(codes.has('contrast_changed'));
  assert.ok(codes.has('unsupported_critical_fact'));
  assert.ok(result.unsupportedCriticalFacts.some((entity) => entity.factId === 'free_gas'));
});

test('safety отдельно ловит замену единицы при том же числе', () => {
  const result = verifyRadiologySafety('Очаг 15 мм.', 'Очаг 15 см.');
  assert.equal(result.numbers.ok, true);
  assert.equal(result.numberUnits.ok, false);
  assert.ok(result.issues.some((issue) => issue.code === 'number_or_unit_changed'));
});

test('safety понимает общую единицу группы размеров', () => {
  const result = verifyRadiologySafety('Размеры 15 на 20 мм.', 'Размеры 15 мм × 20 мм.');
  assert.equal(result.ok, true);
});

test('safety допускает перестановку целых клинических фрагментов', () => {
  const source = 'Справа в почке конкремент 5 мм. Слева в почке образование 15 мм.';
  const output = 'Слева в почке образование 15 мм. Справа в почке конкремент 5 мм.';
  const result = verifyRadiologySafety(source, output);

  assert.equal(result.ok, true);
  assert.equal(result.requiresReview, false);
});

test('safety блокирует перестановку связей факт-сторона-размер при равных сущностях', () => {
  const source = 'Справа конкремент 5 мм / слева образование 15 мм.';
  const output = 'Справа образование 15 мм / слева конкремент 5 мм.';
  const result = verifyRadiologySafety(source, output);

  assert.equal(result.numberUnits.ok, true);
  assert.equal(result.lateralities.ok, true);
  assert.equal(result.criticalFacts.ok, true);
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.code === 'clinical_relation_changed'));
});

test('safety распознаёт document-style отрицания и их положительные пары', () => {
  for (const [source, output] of [
    ['Тромбоз не отмечается.', 'Тромбоз отмечается.'],
    ['Свободная жидкость не наблюдается.', 'Свободная жидкость наблюдается.'],
    ['Метастазы не подтверждаются.', 'Метастазы подтверждаются.'],
    ['Метастазы исключены.', 'Метастазы.'],
  ]) {
    const result = verifyRadiologySafety(source, output);
    assert.equal(result.ok, false, `${source} -> ${output}`);
    assert.ok(
      result.issues.some((issue) => (
        issue.code === 'negation_changed'
        || issue.code === 'critical_fact_lost'
        || issue.code === 'unsupported_critical_fact'
      )),
      `${source} -> ${output}`,
    );
  }
});

test('safety связывает несколько пар сторона-размер без известного fact-якоря', () => {
  const result = verifyRadiologySafety(
    'Справа очаг 5 мм и слева очаг 15 мм.',
    'Справа очаг 15 мм и слева очаг 5 мм.',
  );

  assert.equal(result.numberUnits.ok, true);
  assert.equal(result.lateralities.ok, true);
  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.code === 'clinical_relation_changed'));
});

test('safety не считает пунктуацию границей пар сторона-размер', () => {
  const result = verifyRadiologySafety(
    'Справа очаг 5 мм, слева очаг 15 мм.',
    'Справа очаг 5 мм и слева очаг 15 мм.',
  );

  assert.equal(result.ok, true);
  assert.equal(result.requiresReview, false);
});

test('safety не отрывает запятой размер и отрицание от критического факта', () => {
  const sizeSwap = verifyRadiologySafety(
    'Конкремент, размером 5 мм. Образование, размером 15 мм.',
    'Конкремент, размером 15 мм. Образование, размером 5 мм.',
  );
  assert.equal(sizeSwap.numberUnits.ok, true);
  assert.equal(sizeSwap.criticalFacts.ok, true);
  assert.equal(sizeSwap.ok, false);
  assert.ok(sizeSwap.issues.some((issue) => issue.code === 'clinical_relation_changed'));

  const negationSwap = verifyRadiologySafety(
    'Метастазы, не выявлены. Тромбоз, выявлен.',
    'Метастазы, выявлены. Тромбоз, не выявлен.',
  );
  assert.equal(negationSwap.negations.ok, true);
  assert.equal(negationSwap.ok, false);
  assert.ok(negationSwap.issues.some((issue) => (
    issue.code === 'critical_fact_lost' || issue.code === 'unsupported_critical_fact'
  )));
});

test('safety сохраняет organ heading scope через colon/newline и допускает reorder секций', () => {
  const source = 'Печень:\nМетастазы.\nПочки:\nТромбоз.';
  const unsafe = verifyRadiologySafety(
    source,
    'Печень:\nТромбоз.\nПочки:\nМетастазы.',
  );
  assert.equal(unsafe.ok, false);
  assert.ok(unsafe.issues.some((issue) => issue.code === 'clinical_relation_changed'));

  const safe = verifyRadiologySafety(
    source,
    'Почки:\nТромбоз.\nПечень:\nМетастазы.',
  );
  assert.equal(safe.ok, true);
  assert.equal(safe.requiresReview, false);
});

test('safety допускает reorder фактов внутри общего organ scope', () => {
  const result = verifyRadiologySafety(
    'В правой почке конкремент 5 мм и образование 15 мм.',
    'В правой почке образование 15 мм и конкремент 5 мм.',
  );

  assert.equal(result.ok, true);
  assert.equal(result.requiresReview, false);
});

test('safety связывает числа и полярность с любым локальным клиническим термином', () => {
  for (const [source, output] of [
    [
      'Киста 5 мм. Гемангиома 15 мм.',
      'Киста 15 мм. Гемангиома 5 мм.',
    ],
    [
      'Киста 5 мм и гемангиома 15 мм.',
      'Киста 15 мм и гемангиома 5 мм.',
    ],
    [
      'Киста не выявлена. Гемангиома выявлена.',
      'Киста выявлена. Гемангиома не выявлена.',
    ],
    [
      'В печени киста 5 мм. В печени гемангиома 15 мм.',
      'В печени киста 15 мм. В печени гемангиома 5 мм.',
    ],
    [
      'В правой почке киста 5 мм. В правой почке гемангиома 15 мм.',
      'В правой почке киста 15 мм. В правой почке гемангиома 5 мм.',
    ],
  ]) {
    const result = verifyRadiologySafety(source, output);
    assert.equal(result.ok, false, `${source} -> ${output}`);
    assert.ok(
      result.issues.some((issue) => issue.code === 'clinical_relation_changed'),
      `${source} -> ${output}`,
    );
  }
});

test('generic lexical safety допускает перестановку целых подтверждённых фрагментов', () => {
  const result = verifyRadiologySafety(
    'Киста 5 мм. Гемангиома 15 мм.',
    'Гемангиома 15 мм. Киста 5 мм.',
  );

  assert.equal(result.ok, true);
  assert.equal(result.requiresReview, false);
});

test('safety считает перфорацию критическим фактом и блокирует её добавление', () => {
  const result = verifyRadiologySafety(
    'Кишечник описан без дополнительных находок.',
    'Кишечник описан без дополнительных находок. Перфорация кишки.',
  );

  assert.equal(result.ok, false);
  assert.ok(result.issues.some((issue) => issue.code === 'unsupported_critical_fact'));
  assert.ok(result.unsupportedCriticalFacts.some((entity) => entity.factId === 'perforation'));
});

test('structureDictation привязывает evidence, сохраняет unmatched и блокирует неподтверждённый atom', async () => {
  const transcript = 'Вводная фраза. Селезёнка 12 на 6 на 5. Печень плотность 58. Селезёнка добавочная долька.';
  const result = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, nullClassifier);

  const liver = result.blocks.find((block) => block.id === 'liver')!;
  const spleen = result.blocks.find((block) => block.id === 'spleen')!;
  const kidney = result.blocks.find((block) => block.id === 'kidneys')!;

  assert.equal(liver.source, 'dictated');
  assert.equal(liver.provenanceStatus, 'linked');
  assert.match(liver.text, /плотность 58/i);
  assert.equal(spleen.evidence.length, 2);
  assert.equal(kidney.source, 'normal');
  assert.equal(kidney.normalReason, 'missing');
  assert.equal(result.unmatched, 'Вводная фраза.');
  assert.equal(result.unmatchedSpans.length, 1);
  assert.equal(result.numberCheck.ok, true);
  assert.equal(result.safety.ok, false);
  assert.ok(result.safety.issues.some((issue) => issue.code === 'missing_provenance'));
});

test('structureDictation отклоняет LLM-текст без source span вместо включения его в протокол', async () => {
  const result = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    'Неразборчивый фрагмент.',
    fakeLLM({ liver: 'В печени метастазы.' }),
  );
  const liver = result.blocks.find((block) => block.id === 'liver')!;
  assert.equal(liver.provenanceStatus, 'template');
  assert.equal(liver.origin, 'template_default');
  assert.equal(result.unmatched, 'Неразборчивый фрагмент.');
  assert.equal(result.structuringRun.llmValid, false);
  assert.equal(result.safety.ok, false);
  assert.ok(result.safety.issues.some((issue) => issue.code === 'missing_provenance'));
  assert.doesNotMatch(result.evidenceBackedText, /метастаз/iu);
});
