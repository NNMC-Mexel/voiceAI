// Тесты ядра «диктовка → структура»: сборка evidence-backed документа + сверка чисел.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { structureDictation } from './radiology/dictation.js';
import { verifyNumbers } from './radiology/number-check.js';
import type { LLMCall } from './radiology/ollama.js';
import { sectionizeWithProvenance } from './radiology/sectionize.js';
import type { DocTemplate } from './radiology/doc-model.js';

// Legacy/unsafe response is useful for proving that authored text is rejected.
function fakeLLM(sections: Record<string, string>, unmatched = ''): LLMCall {
  return async () => JSON.stringify({ ...sections, unmatched });
}

const classifyAll = (sectionId: string | null): LLMCall => async (_system, user) => {
  const atoms = (JSON.parse(user) as { atoms: { atomId: string }[] }).atoms;
  return JSON.stringify({
    assignments: atoms.map((atom) => ({ atomId: atom.atomId, sectionId })),
  });
};

const blk = (r: Awaited<ReturnType<typeof structureDictation>>, id: string) => r.blocks.find((b) => b.id === id)!;

test('продиктованные секции остаются дословными, дефолты отделены от evidence-backed текста', async () => {
  const transcript = 'печень плотность плюс 57 контуры неровные селезёнка 165 на 78 на 196 желчный норма';
  const r = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, fakeLLM({
    liver: 'печень плотность +57 контуры неровные',
    spleen: '165 на 78 на 196',
    gallbladder: 'норма',
  }));
  assert.equal(blk(r, 'liver').source, 'dictated');
  assert.match(blk(r, 'liver').text, /плюс 57 контуры неровные/);
  assert.equal(blk(r, 'spleen').source, 'dictated');
  // не упомянутый холедох — норма шаблона
  assert.equal(blk(r, 'choledoch').source, 'normal');
  assert.match(blk(r, 'choledoch').text, /до 5,5 мм/);
  assert.ok(r.templateDefaults.some((item) => item.sectionId === 'choledoch'));
  assert.doesNotMatch(r.evidenceBackedText, /5,5 мм/u);
  // Явное «норма» остаётся словами врача; разворачивать его в неподтверждённые
  // шаблонные факты structuring-слой больше не имеет права.
  assert.equal(blk(r, 'gallbladder').source, 'normal');
  assert.equal(blk(r, 'gallbladder').text, 'желчный норма');
  assert.equal(blk(r, 'gallbladder').origin, 'transcript');
  assert.ok(!r.templateDefaults.some((item) => item.sectionId === 'gallbladder'));
});

test('сверка чисел: всё совпало → ok', async () => {
  const transcript = 'печень плотность плюс пятьдесят семь селезёнка сто шестьдесят пять на семьдесят восемь на сто девяносто шесть';
  const r = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, fakeLLM({
    liver: 'плотность +57', spleen: '165 на 78 на 196',
  }));
  assert.equal(r.numberCheck.ok, true);
  assert.deepEqual(r.numberCheck.addedByModel, []);
  assert.deepEqual(r.numberCheck.lost, []);
});

test('LLM не может добавить число в детерминированно найденную секцию', async () => {
  const transcript = 'холедох расширен образований нет';
  const r = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, fakeLLM({
    choledoch: 'холедох расширен до 6 миллиметров', // 6 врач не называл
  }));
  assert.equal(r.numberCheck.ok, true);
  assert.doesNotMatch(blk(r, 'choledoch').text, /6/);
});

test('LLM не может потерять число в детерминированно найденной секции', async () => {
  const transcript = 'портальная вена расширена до девятнадцати миллиметров';
  const r = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, fakeLLM({
    vessels: 'портальная вена расширена', // 19 потеряно
  }));
  assert.equal(r.numberCheck.ok, true);
  assert.match(blk(r, 'vessels').text, /девятнадцати миллиметров/);
});

test('unmatched не выбрасывается и учитывается в сверке чисел', async () => {
  const transcript = 'что-то непонятное восемь миллиметров';
  const r = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, classifyAll(null));
  assert.match(r.unmatched, /непонятное/);
  assert.ok(r.numberCheck.matched.includes(8)); // 8 из unmatched засчитано, не потеряно
  assert.equal(r.safety.ok, false);
  assert.ok(r.safety.issues.some((issue) => issue.code === 'missing_provenance'));
});

test('generateConclusion по команде «формируй заключение»', async () => {
  const r = await structureDictation('CT_ABDOMEN_MIKHAILOV', 'печень норма формируй заключение', fakeLLM({ liver: 'норма' }));
  assert.equal(r.generateConclusion, true);
  assert.equal(blk(r, 'conclusion').text, '');
  assert.deepEqual(blk(r, 'conclusion').evidence, []);
});

test('автоматическое заключение не повышает отрицанную находку до патологии', async () => {
  const transcript = 'печень образование не выявлено не увеличена формируй заключение';
  const r = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    transcript,
    fakeLLM({ liver: 'печень образование не выявлено не увеличена' }),
  );
  assert.equal(blk(r, 'conclusion').text, '');
  assert.deepEqual(blk(r, 'conclusion').evidence, []);
});

test('патологическая находка никогда не получает дефолтное ложно-нормальное заключение', async () => {
  const transcript = 'печень метастатическое образование';
  const r = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    transcript,
    fakeLLM({ liver: transcript }),
  );
  const conclusion = blk(r, 'conclusion');
  assert.equal(conclusion.text, '');
  assert.doesNotMatch(r.fullText, /патологических изменений.+не выявлено/iu);
});

test('заключение по команде копирует только evidence-linked находки', async () => {
  const transcript = 'печень метастатическое образование формируй заключение';
  const r = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    transcript,
    fakeLLM({ liver: 'печень метастатическое образование' }),
  );
  const conclusion = blk(r, 'conclusion');
  assert.equal(conclusion.provenanceStatus, 'linked');
  assert.ok(conclusion.evidence.length > 0);
  assert.match(conclusion.text, /печень.+метастатическое образование/iu);
  assert.equal(conclusion.text, 'печень метастатическое образование');
  assert.doesNotMatch(conclusion.text, /патологических изменений.+не выявлено/iu);
});

test('LLM классифицирует только atomId и не может добавить клинические символы', async () => {
  const transcript = 'неразборчивый клинический фрагмент';
  const r = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    transcript,
    classifyAll('liver'),
  );
  const liver = blk(r, 'liver');
  assert.equal(liver.text, transcript);
  assert.equal(liver.assignmentMethod, 'llm');
  assert.equal(liver.evidence[0].text, transcript);
  assert.equal(r.structuringRun.llmValid, true);
  assert.equal(r.routing.unmatchedAtomIds.length, 0);
});

test('старый LLM-формат с медицинским текстом отклоняется целиком', async () => {
  const transcript = 'неразборчивый фрагмент';
  const r = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    transcript,
    fakeLLM({ liver: 'В печени метастазы 15 мм.' }),
  );
  assert.equal(r.structuringRun.llmValid, false);
  assert.ok(r.structuringRun.issues.some((issue) => issue.code === 'invalid_response_shape'));
  assert.equal(r.unmatched, transcript);
  assert.equal(blk(r, 'liver').origin, 'template_default');
  assert.doesNotMatch(r.evidenceBackedText, /метастаз|15/iu);
});

test('неизвестный или повторный atomId инвалидирует весь ответ LLM', async () => {
  const transcript = 'неразборчивый первый фрагмент';
  for (const llm of [
    async () => JSON.stringify({
      assignments: [{ atomId: 'unknown', sectionId: 'liver' }],
    }),
    async (_system: string, user: string) => {
      const atomId = (JSON.parse(user) as { atoms: { atomId: string }[] }).atoms[0].atomId;
      return JSON.stringify({
        assignments: [
          { atomId, sectionId: 'liver' },
          { atomId, sectionId: 'kidneys' },
        ],
      });
    },
  ] satisfies LLMCall[]) {
    const r = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, llm);
    assert.equal(r.structuringRun.llmValid, false);
    assert.equal(r.routing.unmatchedAtomIds.length, 1);
    assert.equal(r.unmatched, transcript);
  }
});

test('запрещённая candidate-секция оставляет «в воротах» unmatched', async () => {
  const transcript = 'в воротах добавочная долька 9 мм';
  const r = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    transcript,
    classifyAll('lymph_hilum'),
  );
  assert.equal(r.structuringRun.llmValid, false);
  assert.ok(r.structuringRun.issues.some((issue) => issue.code === 'forbidden_section'));
  assert.equal(r.unmatched, transcript);
  assert.equal(blk(r, 'lymph_hilum').origin, 'template_default');
});

test('одинаковый classifier-input кэшируется по SHA для одного LLMCall', async () => {
  let calls = 0;
  const llm: LLMCall = async (_system, user) => {
    calls++;
    const atoms = (JSON.parse(user) as { atoms: { atomId: string }[] }).atoms;
    return JSON.stringify({
      assignments: atoms.map((atom) => ({ atomId: atom.atomId, sectionId: null })),
    });
  };
  const transcript = 'служебная вводная без якоря';
  const first = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, llm);
  const second = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, llm);
  assert.equal(calls, 1);
  assert.equal(first.structuringRun.llmInputSha256, second.structuringRun.llmInputSha256);
  assert.equal(first.structuringRun.llmResponseSha256, second.structuringRun.llmResponseSha256);
});

test('allowLLM=false не вызывает модель и оставляет неизвестные atoms unmatched', async () => {
  let called = false;
  const llm: LLMCall = async () => {
    called = true;
    throw new Error('LLM не должен вызываться');
  };
  const r = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    'неразборчивый фрагмент',
    llm,
    { allowLLM: false },
  );
  assert.equal(called, false);
  assert.equal(r.structuringRun.llmAllowed, false);
  assert.equal(r.structuringRun.llmCalled, false);
  assert.equal(r.unmatched, 'неразборчивый фрагмент');
  assert.equal(r.safety.ok, false);
});

test('продиктованное заключение хранится точным source span без перефразирования', async () => {
  const transcript = 'печень норма заключение КТ признаки кисты печени 9 мм';
  const r = await structureDictation('CT_ABDOMEN_MIKHAILOV', transcript, classifyAll(null));
  const conclusion = blk(r, 'conclusion');
  assert.equal(conclusion.text, 'заключение КТ признаки кисты печени 9 мм');
  assert.equal(conclusion.origin, 'transcript');
  assert.equal(conclusion.evidence.length, 1);
  const evidence = conclusion.evidence[0];
  assert.equal(transcript.slice(evidence.start, evidence.end), conclusion.text);
});

test('canonical report keeps verbatim evidence and adds a filled template review draft', async () => {
  const transcript = 'печень КВР 150 плотность 60';
  const report = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    transcript,
    classifyAll(null),
  );

  const liver = blk(report, 'liver');
  assert.equal(liver.text, transcript);
  assert.equal(report.evidenceBackedText, `Печень: ${transcript}`);

  const draft = report.reviewDraft;
  assert.ok(draft);
  assert.equal(draft.templateId, 'CT_ABDOMEN_MIKHAILOV');
  assert.match(draft.fullText, /КВР 150 мм/iu);
  assert.match(draft.fullText, /\+60 HU/iu);
  assert.deepEqual(report.fieldAssignments, draft.fieldAssignments);

  const kvr = draft.fieldAssignments.find((assignment) => assignment.fieldId.endsWith('.kvr'));
  const density = draft.fieldAssignments.find((assignment) => assignment.fieldId.endsWith('.density'));
  assert.equal(kvr?.status, 'applied');
  assert.deepEqual(kvr?.values, [150]);
  assert.equal(density?.status, 'applied');
  assert.deepEqual(density?.values, [60]);
  for (const assignment of [kvr, density]) {
    assert.ok(assignment);
    assert.ok(assignment.evidence.length > 0);
    for (const evidence of assignment.evidence) {
      assert.equal(transcript.slice(evidence.start, evidence.end), evidence.text);
    }
  }
});

test('unique versioned field alias routes a value spoken before the organ without LLM', async () => {
  let called = false;
  const llm: LLMCall = async () => {
    called = true;
    throw new Error('field alias routing must not call the LLM');
  };
  const transcript = 'КВР 150 печень';
  const report = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    transcript,
    llm,
    { allowLLM: false },
  );

  assert.equal(called, false);
  assert.equal(report.structuringRun.llmAllowed, false);
  assert.equal(report.structuringRun.llmCalled, false);
  assert.equal(report.unmatched, '');
  assert.deepEqual(report.routing.unmatchedAtomIds, []);
  assert.equal(report.routing.atoms.length, 1);
  assert.deepEqual(report.routing.atoms[0].candidateSectionIds, ['liver']);
  assert.deepEqual(report.routing.assignments, [{
    atomId: report.routing.atoms[0].id,
    sectionId: 'liver',
    method: 'rule',
  }]);
  assert.match(
    report.routing.atoms[0].anchorRuleIds[0] ?? '',
    /^field-alias:ct-abdomen-field-routing-v1:liver\.kvr:квр$/u,
  );

  const kvr = report.fieldAssignments?.find((assignment) => assignment.fieldId === 'liver.kvr');
  assert.equal(kvr?.status, 'applied');
  assert.deepEqual(kvr?.values, [150]);
  assert.match(report.reviewDraft?.fullText ?? '', /КВР 150 мм/iu);
});

test('invalid LLM cannot override deterministic field-alias routing before the organ', async () => {
  let called = false;
  const invalidLLM: LLMCall = async () => {
    called = true;
    return JSON.stringify({
      assignments: [{ atomId: 'unknown', sectionId: 'spleen' }],
    });
  };
  const report = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    'ка вэ эр 150 печень',
    invalidLLM,
  );

  assert.equal(called, false);
  assert.equal(report.structuringRun.llmCalled, false);
  assert.equal(report.structuringRun.llmValid, true);
  assert.equal(report.unmatched, '');
  assert.deepEqual(report.routing.unmatchedAtomIds, []);
  assert.equal(report.routing.assignments[0]?.sectionId, 'liver');
  assert.equal(report.routing.assignments[0]?.method, 'rule');
  const kvr = report.fieldAssignments?.find((assignment) => assignment.fieldId === 'liver.kvr');
  assert.equal(kvr?.status, 'applied');
  assert.deepEqual(kvr?.values, [150]);
});

test('ambiguous field routing alias is rejected instead of using template order', () => {
  const template: DocTemplate = {
    id: 'AMBIGUOUS_FIELD_ALIAS_TEST',
    name: 'Ambiguous field alias test',
    modality: 'CT',
    title: 'Test',
    aliases: [],
    fieldRoutingVersion: 'ambiguous-routing-v1',
    blocks: [
      {
        id: 'first',
        label: 'First',
        anchors: ['первый'],
        nodes: [{
          kind: 'slot',
          slot: {
            name: 'value',
            fieldId: 'first.value',
            keywords: ['маркер'],
            routingAliases: ['маркер'],
            default: '___',
          },
        }],
      },
      {
        id: 'second',
        label: 'Second',
        anchors: ['второй'],
        nodes: [{
          kind: 'slot',
          slot: {
            name: 'value',
            fieldId: 'second.value',
            keywords: ['маркер'],
            routingAliases: ['маркер'],
            default: '___',
          },
        }],
      },
    ],
  };

  const routed = sectionizeWithProvenance(template, 'маркер 150');
  assert.equal(routed.atoms.length, 1);
  assert.deepEqual(routed.atoms[0].anchorRuleIds, []);
  assert.equal(routed.assignments[0]?.sectionId, null);
  assert.equal(routed.assignments[0]?.method, 'unmatched');
  assert.deepEqual(routed.atoms[0].candidateSectionIds, ['first', 'second']);
});

// ─── прямые тесты number-check ────────────────────────────────────────────────
test('verifyNumbers: слова и цифры сопоставляются', () => {
  const c = verifyNumbers('давление сто двадцать семь и плюс 57', 'значения 127 и 57');
  assert.equal(c.ok, true);
});

test('verifyNumbers: десятичные с запятой', () => {
  const c = verifyNumbers('холедох 5,5 мм', 'холедох до 5,5 мм');
  assert.equal(c.ok, true);
});
