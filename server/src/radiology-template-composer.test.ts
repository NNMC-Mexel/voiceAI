import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  composeTemplateReviewDraft,
  TEMPLATE_COMPOSER_VERSION,
  type TemplateSectionAtom,
} from './radiology/template-composer.js';
import { abdomenMikhailov } from './radiology/templates/abdomen-mikhailov.js';

function atom(
  transcript: string,
  atomId: string,
  sectionId: string,
  text: string,
  from = 0,
): TemplateSectionAtom {
  const start = transcript.indexOf(text, from);
  assert.notEqual(start, -1, `Missing atom text: ${text}`);
  return {
    atomId,
    sectionId,
    start,
    end: start + text.length,
    text,
  };
}

function compose(transcript: string, atoms: TemplateSectionAtom[]) {
  return composeTemplateReviewDraft(abdomenMikhailov, transcript, atoms);
}

test('composer fills liver.kvr with exact evidence and leaves acoustic text outside the draft', () => {
  const transcript = 'печень КВР 150';
  const result = compose(transcript, [atom(transcript, 'liver-1', 'liver', transcript)]);
  const assignment = result.fieldAssignments.find((item) => item.fieldId === 'liver.kvr');
  const liver = result.sections.find((section) => section.id === 'liver')!;
  const segment = result.segments.find((item) => item.fieldId === 'liver.kvr')!;

  assert.equal(result.version, TEMPLATE_COMPOSER_VERSION);
  assert.deepEqual(assignment?.values, [150]);
  assert.equal(assignment?.unit, 'mm');
  assert.equal(assignment?.unitSource, 'template_schema');
  assert.deepEqual(assignment?.evidence, [{
    atomId: 'liver-1',
    start: 11,
    end: 14,
    text: '150',
    source: 'transcript',
    normalized: {
      start: 11,
      end: 14,
      text: '150',
    },
    raw: null,
  }]);
  assert.match(liver.text, /КВР 150 мм/u);
  assert.equal(liver.mode, 'template_filled');
  assert.equal(segment.kind, 'transcript_value');
  assert.equal(segment.origin, 'transcript_slot');
  assert.equal(segment.confirmationRequired, false);
  assert.equal(result.residualAtomIds.length, 0);
  assert.ok(!result.issues.some((issue) => issue.fieldId === 'liver.kvr'));
});

test('composer binds KVR and density in either parameter/value order', () => {
  for (const transcript of [
    'печень КВР 145 плотность 62',
    'печень 145 КВР 62 плотность',
    'печень 62 плотность 145 КВР',
    'печень 145 мм КВР 62 HU плотность',
  ]) {
    const result = compose(transcript, [atom(transcript, 'liver', 'liver', transcript)]);
    const assignments = new Map(
      result.fieldAssignments.map((item) => [item.fieldId, item]),
    );
    assert.deepEqual(assignments.get('liver.kvr')?.values, [145], transcript);
    assert.deepEqual(assignments.get('liver.density')?.values, [62], transcript);
    assert.equal(result.residualAtomIds.length, 0, transcript);
    assert.match(
      result.sections.find((section) => section.id === 'liver')!.text,
      /КВР 145 мм[\s\S]*\+62 HU/u,
      transcript,
    );
  }
});

test('composer accepts only versioned spoken KVR aliases without fuzzy matching', () => {
  for (const alias of ['КВР', 'к в р', 'ка вэ эр']) {
    const transcript = `печень ${alias} 150`;
    const result = compose(transcript, [atom(transcript, alias, 'liver', transcript)]);
    const assignment = result.fieldAssignments.find(
      (item) => item.fieldId === 'liver.kvr',
    );

    assert.deepEqual(assignment?.values, [150], alias);
    assert.equal(result.residualAtomIds.length, 0, alias);
  }
});

test('returning to an earlier organ fills the same deterministic section state', () => {
  const transcript = 'печень КВР 145. почки норма. печень плотность 62';
  const firstLiver = 'печень КВР 145';
  const kidneys = 'почки норма';
  const secondLiver = 'печень плотность 62';
  const result = compose(transcript, [
    atom(transcript, 'liver-kvr', 'liver', firstLiver),
    atom(transcript, 'kidneys-normal', 'kidneys', kidneys),
    atom(transcript, 'liver-density', 'liver', secondLiver),
  ]);
  const liver = result.sections.find((section) => section.id === 'liver')!;

  assert.match(liver.text, /КВР 145 мм[\s\S]*\+62 HU/u);
  assert.equal(
    result.fieldAssignments.filter((item) => item.sectionId === 'liver').length,
    2,
  );
  assert.equal(result.residualAtomIds.length, 0);
});

test('dimensions retain exact source spans and produce a provenance-linked derived value', () => {
  const transcript = 'селезёнка 12 на 6 на 5';
  const result = compose(transcript, [atom(transcript, 'spleen', 'spleen', transcript)]);
  const dimensions = result.fieldAssignments.find(
    (item) => item.fieldId === 'spleen.dimensions',
  )!;
  const derived = result.segments.find((item) => item.fieldId === 'spleen.index')!;

  assert.deepEqual(dimensions.values, [120, 60, 50]);
  assert.deepEqual(dimensions.evidence.map((item) => item.text), ['12', '6', '5']);
  assert.equal(derived.kind, 'derived');
  assert.equal(derived.text, '360');
  assert.deepEqual(derived.evidence, dimensions.evidence);
  assert.match(
    result.sections.find((section) => section.id === 'spleen')!.text,
    /120,0х60,0х50,0 мм – СИ ≈ 360/u,
  );
});

test('word-number dimensions map evidence back to the immutable source words', () => {
  const transcript = 'селезёнка сто двадцать на шестьдесят на пятьдесят миллиметров';
  const result = compose(transcript, [atom(transcript, 'spleen', 'spleen', transcript)]);
  const assignment = result.fieldAssignments.find(
    (item) => item.fieldId === 'spleen.dimensions',
  )!;

  assert.deepEqual(assignment.values, [120, 60, 50]);
  assert.deepEqual(assignment.evidence.map((item) => item.text), [
    'сто двадцать',
    'шестьдесят',
    'пятьдесят',
    'миллиметров',
  ]);
  assert.equal(assignment.unitSource, 'transcript');
});

test('switch selection is target-constrained and suppresses the unsafe normal conclusion', () => {
  const transcript = 'почки конкремент 6';
  const result = compose(transcript, [atom(transcript, 'kidney', 'kidneys', transcript)]);
  const status = result.fieldAssignments.find(
    (item) => item.fieldId === 'kidneys.stone_status',
  );
  const conclusion = result.sections.find((section) => section.id === 'conclusion')!;

  assert.equal(status?.kind, 'switch');
  assert.equal(status?.optionId, 'stones');
  assert.match(
    result.sections.find((section) => section.id === 'kidneys')!.text,
    /определяется конкремент до 6 мм/u,
  );
  assert.equal(conclusion.text, '');
  assert.ok(result.issues.some((issue) => issue.code === 'default_conclusion_suppressed'));
});

test('explicit normal expands only the targeted section as a versioned macro', () => {
  const transcript = 'желчный пузырь норма';
  const result = compose(transcript, [atom(transcript, 'gb', 'gallbladder', transcript)]);
  const section = result.sections.find((item) => item.id === 'gallbladder')!;

  assert.equal(section.mode, 'explicit_normal');
  assert.match(section.text, /гомогенным содержимым/u);
  assert.equal(
    result.fieldAssignments.find((item) => item.kind === 'explicit_normal')?.fieldId,
    'gallbladder.normal',
  );
  assert.ok(
    section.segmentIds
      .map((id) => result.segments.find((segment) => segment.id === id)!)
      .every((segment) => segment.kind === 'template_choice'),
  );
});

test('cm values are deterministically converted to the mm field unit', () => {
  const transcript = 'печень КВР 15 см';
  const result = compose(transcript, [atom(transcript, 'liver', 'liver', transcript)]);
  const assignment = result.fieldAssignments.find((item) => item.fieldId === 'liver.kvr')!;

  assert.deepEqual(assignment.values, [150]);
  assert.equal(assignment.formattedText, '150');
  assert.equal(assignment.unit, 'mm');
  assert.equal(assignment.unitSource, 'transcript');
  assert.equal(assignment.conversionRuleId, 'cm-to-mm-v1');
  assert.match(result.sections.find((section) => section.id === 'liver')!.text, /КВР 150 мм/u);
});

test('signed abnormal density falls back verbatim while mixed dimensions preserve units', () => {
  const densityText = 'печень КВР 150 плотность минус 10 HU';
  const density = compose(densityText, [
    atom(densityText, 'signed-density', 'liver', densityText),
  ]);
  const densityAssignment = density.fieldAssignments.find(
    (item) => item.fieldId === 'liver.density',
  )!;
  assert.equal(densityAssignment.value, -10);
  assert.deepEqual(densityAssignment.values, [-10]);
  assert.equal(densityAssignment.status, 'incomplete');
  assert.equal(
    density.sections.find((section) => section.id === 'liver')!.text,
    densityText,
  );
  assert.ok(density.issues.some((issue) => (
    issue.code === 'value_outside_template_claim'
  )));
  assert.ok(!density.issues.some((issue) => issue.code === 'unused_number'));

  const dimensionsText = 'селезёнка 12 см на 6 мм на 5 мм';
  const dimensions = compose(dimensionsText, [
    atom(dimensionsText, 'mixed-units', 'spleen', dimensionsText),
  ]);
  const dimensionsAssignment = dimensions.fieldAssignments.find(
    (item) => item.fieldId === 'spleen.dimensions',
  )!;
  assert.deepEqual(dimensionsAssignment.value, [120, 6, 5]);
  assert.deepEqual(dimensionsAssignment.values, [120, 6, 5]);
  assert.equal(dimensionsAssignment.conversionRuleId, 'cm-to-mm-v1');

  const ambiguousUnitsText = 'селезёнка 12 см на 6 на 5';
  const ambiguousUnits = compose(ambiguousUnitsText, [
    atom(ambiguousUnitsText, 'ambiguous-units', 'spleen', ambiguousUnitsText),
  ]);
  assert.equal(ambiguousUnits.status, 'failed');
  assert.ok(ambiguousUnits.issues.some((issue) => (
    issue.code === 'ambiguous_dimension_unit'
  )));
});

test('wrong units, conflicts, ambiguous numbers and residual text fail closed', () => {
  const wrongUnit = 'печень плотность 15 см';
  const wrong = compose(wrongUnit, [atom(wrongUnit, 'wrong-unit', 'liver', wrongUnit)]);
  assert.ok(wrong.issues.some((issue) => issue.code === 'unit_mismatch'));
  assert.equal(wrong.sections.find((section) => section.id === 'liver')?.mode, 'verbatim_fallback');

  const conflictText = 'печень КВР 145. печень КВР 150';
  const secondStart = conflictText.lastIndexOf('печень');
  const conflict = compose(conflictText, [
    atom(conflictText, 'first', 'liver', 'печень КВР 145'),
    atom(conflictText, 'second', 'liver', 'печень КВР 150', secondStart),
  ]);
  assert.ok(conflict.issues.some((issue) => issue.code === 'field_conflict'));
  assert.equal(conflict.sections.find((section) => section.id === 'liver')?.mode, 'verbatim_fallback');

  const ambiguousText = 'печень КВР пятьдесят пятьдесят три';
  const ambiguous = compose(ambiguousText, [
    atom(ambiguousText, 'ambiguous', 'liver', ambiguousText),
  ]);
  assert.ok(ambiguous.issues.some((issue) => issue.code === 'ambiguous_number_sequence'));

  const residualText = 'печень КВР 150 контуры неровные';
  const residual = compose(residualText, [
    atom(residualText, 'residual', 'liver', residualText),
  ]);
  assert.ok(residual.issues.some((issue) => issue.code === 'residual_clinical_text'));
  assert.deepEqual(residual.residualAtomIds, ['residual']);
  const residualSection = residual.sections.find((section) => section.id === 'liver')!;
  assert.equal(residualSection.mode, 'template_filled');
  assert.match(residualSection.text, /КВР 150 мм/u);
  assert.match(residualSection.text, /контуры неровные$/u);
  assert.ok(residual.segments.some((segment) => (
    segment.sectionId === 'liver'
    && segment.origin === 'transcript_append'
    && segment.text === 'контуры неровные'
  )));

  const partialText = 'селезёнка 12 на 6';
  const partial = compose(partialText, [
    atom(partialText, 'partial-dimensions', 'spleen', partialText),
  ]);
  assert.equal(partial.status, 'failed');
  assert.equal(
    partial.fieldAssignments.find(
      (assignment) => assignment.fieldId === 'spleen.dimensions',
    )?.status,
    'incomplete',
  );
  assert.ok(partial.issues.some((issue) => (
    issue.code === 'partial_dimension' && issue.severity === 'critical'
  )));
  assert.equal(
    partial.sections.find((section) => section.id === 'spleen')?.mode,
    'verbatim_fallback',
  );

  const extraDimensionText = 'селезёнка 12 на 6 на 5 7';
  const extraDimension = compose(extraDimensionText, [
    atom(extraDimensionText, 'extra-dimension', 'spleen', extraDimensionText),
  ]);
  assert.equal(extraDimension.status, 'failed');
  assert.equal(
    extraDimension.fieldAssignments.find(
      (assignment) => assignment.fieldId === 'spleen.dimensions',
    )?.status,
    'ambiguous',
  );
  assert.ok(extraDimension.issues.some((issue) => (
    issue.code === 'unused_number' && issue.severity === 'critical'
  )));
  assert.equal(
    extraDimension.sections.find((section) => section.id === 'spleen')?.mode,
    'verbatim_fallback',
  );

  const extraScalarText = 'печень КВР 150 160 киста';
  const extraScalar = compose(extraScalarText, [
    atom(extraScalarText, 'extra-scalar', 'liver', extraScalarText),
  ]);
  assert.equal(extraScalar.status, 'failed');
  assert.equal(
    extraScalar.fieldAssignments.find(
      (assignment) => assignment.fieldId === 'liver.kvr',
    )?.status,
    'ambiguous',
  );
  assert.ok(extraScalar.issues.some((issue) => (
    issue.code === 'ambiguous_field_binding' && issue.severity === 'critical'
  )));

  const ambiguousBindingText = 'печень плотность КВР 60 150';
  const ambiguousBinding = compose(ambiguousBindingText, [
    atom(ambiguousBindingText, 'ambiguous-binding', 'liver', ambiguousBindingText),
  ]);
  assert.equal(ambiguousBinding.status, 'failed');
  assert.ok(ambiguousBinding.issues.some((issue) => (
    issue.code === 'ambiguous_field_binding'
  )));
  assert.ok(ambiguousBinding.fieldAssignments.some((assignment) => (
    assignment.status === 'ambiguous'
  )));
});

test('required placeholders remain incomplete even in unmentioned sections', () => {
  const transcript = 'желчный пузырь норма';
  const result = compose(transcript, [atom(transcript, 'gb', 'gallbladder', transcript)]);

  assert.equal(result.status, 'partial');
  assert.ok(result.issues.some((issue) => (
    issue.code === 'unresolved_template_placeholder'
    && issue.severity === 'warning'
    && issue.fieldId === 'liver.kvr'
  )));
  assert.ok(result.issues.some((issue) => (
    issue.code === 'unresolved_template_placeholder'
    && issue.fieldId === 'spleen.dimensions'
  )));
  for (const fieldId of ['liver.kvr', 'spleen.dimensions']) {
    assert.equal(
      result.fieldAssignments.find((assignment) => assignment.fieldId === fieldId)?.status,
      'incomplete',
      fieldId,
    );
  }

  const addressedText = 'печень плотность 60';
  const addressed = compose(addressedText, [
    atom(addressedText, 'liver', 'liver', addressedText),
  ]);
  assert.equal(addressed.status, 'failed');
  assert.ok(addressed.issues.some((issue) => (
    issue.code === 'required_placeholder' && issue.fieldId === 'liver.kvr'
  )));

  const activeSwitchText = 'почки конкремент';
  const activeSwitch = compose(activeSwitchText, [
    atom(activeSwitchText, 'kidneys', 'kidneys', activeSwitchText),
  ]);
  assert.equal(activeSwitch.status, 'failed');
  assert.ok(activeSwitch.issues.some((issue) => (
    issue.code === 'required_placeholder' && issue.fieldId === 'kidneys.stone_size'
  )));

  for (const segment of result.segments) {
    assert.equal(
      result.fullText.slice(segment.start, segment.end),
      segment.text,
      segment.id,
    );
  }
  assert.match(result.sha256, /^[a-f0-9]{64}$/u);
});

test('out-of-schema pathology is appended exactly and suppresses normal conclusion', () => {
  const transcript = 'печень КВР 150, в VII сегменте киста: 10×12 мм.';
  const result = compose(transcript, [
    atom(transcript, 'liver-pathology', 'liver', transcript),
  ]);
  const liver = result.sections.find((section) => section.id === 'liver')!;
  const conclusion = result.sections.find((section) => section.id === 'conclusion')!;
  const appendText = result.segments
    .filter((segment) => segment.origin === 'transcript_append')
    .map((segment) => segment.text)
    .join(' ');

  assert.match(liver.text, /КВР 150 мм/u);
  assert.equal(appendText, 'в VII сегменте киста: 10×12 мм.');
  assert.equal(conclusion.text, '');
  assert.deepEqual(result.residualAtomIds, ['liver-pathology']);
  assert.equal(result.status, 'partial');
});

test('normal switch phrases cannot be promoted to positive stones', () => {
  for (const [sectionId, transcript, fieldId] of [
    ['kidneys', 'почки конкрементов не выявлено', 'kidneys.stone_status'],
    ['gallbladder', 'желчный пузырь конкрементов не выявлено', 'gallbladder.content'],
  ] as const) {
    const result = compose(transcript, [
      atom(transcript, `${sectionId}-normal`, sectionId, transcript),
    ]);
    const assignment = result.fieldAssignments.find((item) => item.fieldId === fieldId);
    assert.equal(assignment?.optionId, 'norm', transcript);
    assert.ok(!result.issues.some((issue) => issue.code === 'switch_conflict'), transcript);
    assert.equal(result.residualAtomIds.length, 0, transcript);
  }

  const contradictory = 'почки конкремент не определяется 6 мм';
  const blocked = compose(contradictory, [
    atom(contradictory, 'negated-stone', 'kidneys', contradictory),
  ]);
  assert.equal(blocked.status, 'failed');
  assert.ok(blocked.issues.some((issue) => issue.code === 'ambiguous_switch_negation'));
});

test('punctuated and leading units bind deterministically, orphan units fail closed', () => {
  for (const transcript of [
    'печень КВР 15, см',
    'печень КВР 15-см',
    'печень КВР [см] 15',
    'печень КВР см 15',
  ]) {
    const result = compose(transcript, [atom(transcript, transcript, 'liver', transcript)]);
    const assignment = result.fieldAssignments.find((item) => item.fieldId === 'liver.kvr');
    assert.equal(assignment?.value, 150, transcript);
    assert.equal(assignment?.unitSource, 'transcript', transcript);
  }

  const dimensionsText = 'селезёнка 12 на 6 на 5, см';
  const dimensions = compose(dimensionsText, [
    atom(dimensionsText, 'trailing-unit', 'spleen', dimensionsText),
  ]);
  assert.deepEqual(
    dimensions.fieldAssignments.find(
      (item) => item.fieldId === 'spleen.dimensions',
    )?.value,
    [120, 60, 50],
  );

  const orphanText = 'печень КВР 150 мм HU';
  const orphan = compose(orphanText, [
    atom(orphanText, 'orphan-unit', 'liver', orphanText),
  ]);
  assert.equal(orphan.status, 'failed');
  assert.ok(orphan.issues.some((issue) => issue.code === 'orphan_unit'));
});

test('domain constraints and non-finite values never render unsafe template claims', () => {
  for (const [transcript, expectedCode] of [
    ['печень КВР 200 плотность 60', 'value_outside_template_claim'],
    ['печень КВР 150 плотность 30', 'value_outside_template_claim'],
    ['почки конкремент 0', 'value_out_of_domain'],
    ['селезёнка 999 на 999 на 999 мм', 'value_outside_template_claim'],
    ['холедох 100 мм', 'value_outside_template_claim'],
    ['аорта 999 мм', 'value_out_of_domain'],
    [`печень КВР ${'9'.repeat(309)}`, 'non_finite_number'],
  ]) {
    const sectionId = transcript.startsWith('почки')
      ? 'kidneys'
      : transcript.startsWith('селезёнка')
        ? 'spleen'
        : transcript.startsWith('холедох')
          ? 'choledoch'
          : transcript.startsWith('аорта')
            ? 'vessels'
            : 'liver';
    const result = compose(transcript, [
      atom(transcript, expectedCode, sectionId, transcript),
    ]);
    assert.equal(result.status, 'failed', transcript);
    assert.equal(
      result.sections.find((section) => section.id === sectionId)?.mode,
      'verbatim_fallback',
      transcript,
    );
    assert.ok(result.issues.some((issue) => issue.code === expectedCode), transcript);
    assert.ok(!result.fullText.includes('Infinity'), transcript);
    assert.ok(!JSON.stringify(result).includes('Infinity'), transcript);
  }
});

test('plus-minus and dash confusables are never interpreted as signed measurements', () => {
  for (const sign of [
    'плюс минус',
    'плюс-минус',
    '+/-',
    '±',
    '\u2010',
    '\u2011',
    '\u207B',
  ]) {
    const transcript = `печень КВР 150 плотность ${sign} 10 HU`;
    const result = compose(transcript, [
      atom(transcript, sign, 'liver', transcript),
    ]);
    assert.equal(result.status, 'failed', sign);
    assert.ok(result.issues.some((issue) => issue.code === 'ambiguous_numeric_sign'), sign);
  }
});
