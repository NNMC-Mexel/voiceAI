import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  denormalize,
  denormalizeDetailed,
  GIGAAM_DENORMALIZER_VERSION,
} from './services/gigaam-denormalize.js';
import { verifyNumbers } from './radiology/number-check.js';
import {
  normalizeNumberWordsDetailed,
  convertNumberWords,
} from './radiology/numbers.js';
import { verifyRawToNormalizedSafety } from './radiology/safety.js';

test('strict cardinal grammar keeps a valid tens+unit number', () => {
  const result = denormalizeDetailed('Плотность пятьдесят три единицы Хаунсфилда.');

  assert.equal(result.version, GIGAAM_DENORMALIZER_VERSION);
  assert.equal(result.text, 'Плотность 53 единицы Хаунсфилда.');
  assert.deepEqual(result.issues, []);
  assert.deepEqual(
    result.transformations
      .filter((item) => item.type === 'cardinal')
      .map((item) => item.values),
    [[53]],
  );
});

test('repeated tens are never summed into a fabricated value', () => {
  const result = denormalizeDetailed(
    'Плотность пятьдесят пятьдесят три единицы Хаунсфилда.',
  );

  assert.equal(result.text, 'Плотность 50 53 единицы Хаунсфилда.');
  assert.doesNotMatch(result.text, /\b103\b/);
  assert.equal(result.issues.length, 1);
  assert.equal(result.issues[0].code, 'ambiguous_number_sequence');
  assert.deepEqual(result.issues[0].values, [50, 53]);
  assert.equal(result.issues[0].sourceText, 'пятьдесят пятьдесят три');
});

test('dimensions remain separate and explicit connectors break number runs', () => {
  const result = denormalizeDetailed(
    'Размеры десять на двадцать девять на сто пятьдесят один миллиметр.',
  );

  assert.equal(result.text, 'Размеры 10 на 29 на 151 миллиметр.');
  assert.deepEqual(result.issues, []);
  assert.equal(verifyNumbers(
    'Размеры десять на двадцать девять на сто пятьдесят один миллиметр.',
    result.text,
  ).ok, true);
});

test('only an explicit range connector separates adjacent values without ambiguity', () => {
  const explicit = normalizeNumberWordsDetailed(
    'от пятидесяти до пятидесяти трех миллиметров',
  );
  assert.equal(explicit.text, 'от 50 до 53 миллиметров');
  assert.deepEqual(explicit.issues, []);

  const implicit = normalizeNumberWordsDetailed(
    'пятьдесят три пятьдесят пять миллиметров',
  );
  assert.equal(implicit.text, '53 55 миллиметров');
  assert.equal(implicit.issues[0]?.code, 'ambiguous_number_sequence');
  assert.deepEqual(implicit.issues[0]?.values, [53, 55]);
});

test('legacy denormalize export stays string-compatible', () => {
  assert.equal(denormalize('сто тридцать девять грамм на литр'), '139 г/л');
  assert.equal(
    denormalize('одиннадцатого марта две тысячи двадцать шестого года'),
    '11.03.2026г.',
  );
  assert.equal(denormalize('пять целых семь десятых процента'), '5,7%');
  assert.equal(denormalize('один раз'), 'один раз');
  assert.equal(convertNumberWords('сто пятьдесят шесть'), '156');
});

test('raw-to-normalized safety passes an unambiguous value', () => {
  const normalized = denormalizeDetailed('Плотность пятьдесят три HU.');
  const safety = verifyRawToNormalizedSafety(
    'Плотность пятьдесят три HU.',
    normalized,
  );

  assert.equal(safety.status, 'passed');
  assert.equal(safety.ok, true);
  assert.equal(safety.numbers.ok, true);
  assert.deepEqual(safety.issues, []);
});

test('preserved repeated tens require review, even though both values survived', () => {
  const raw = 'Плотность пятьдесят пятьдесят три HU.';
  const normalized = denormalizeDetailed(raw);
  const safety = verifyRawToNormalizedSafety(raw, normalized);

  assert.equal(normalized.text, 'Плотность 50 53 HU.');
  assert.equal(safety.numbers.ok, true);
  assert.equal(safety.status, 'incomplete');
  assert.equal(safety.requiresReview, true);
  assert.ok(safety.issues.some((issue) => issue.code === 'ambiguous_number_sequence'));
});

test('independent safety catches the former destructive 103 conversion', () => {
  const raw = 'Плотность пятьдесят пятьдесят три HU.';
  const safety = verifyRawToNormalizedSafety(raw, 'Плотность 103 HU.');

  assert.equal(safety.status, 'failed');
  assert.equal(safety.ok, false);
  assert.deepEqual(safety.numbers.addedByModel, [103]);
  assert.deepEqual(safety.numbers.lost, [50, 53]);
  assert.ok(safety.issues.some((issue) => issue.code === 'ambiguous_number_sequence'));
  assert.ok(safety.issues.some((issue) => issue.code === 'normalization_number_added'));
  assert.ok(safety.issues.some((issue) => issue.code === 'normalization_number_lost'));
});

test('raw-to-normalized safety accepts explicit dimensions and ranges', () => {
  for (const raw of [
    'Размеры десять на двадцать девять на сто пятьдесят один миллиметр.',
    'Плотность от пятидесяти до пятидесяти трех HU.',
  ]) {
    const normalized = denormalizeDetailed(raw);
    const safety = verifyRawToNormalizedSafety(raw, normalized);
    assert.equal(safety.status, 'passed', raw);
  }
});
