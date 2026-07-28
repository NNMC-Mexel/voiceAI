import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { structureDictation, type DictationReport } from './radiology/dictation.js';
import type { LLMCall } from './radiology/ollama.js';
import {
  denormalizeDetailed,
  type GigaAMNormalizationResult,
} from './services/gigaam-denormalize.js';

function readRaw(caseId: '01' | '02' | '03'): string {
  return readFileSync(fileURLToPath(new URL(
    `./fixtures/mikhalych-case-${caseId}.raw.txt`,
    import.meta.url,
  )), 'utf8').trim();
}

async function runCase(caseId: '01' | '02' | '03'): Promise<{
  normalization: GigaAMNormalizationResult;
  report: DictationReport;
  llmCalls: number;
}> {
  let llmCalls = 0;
  const forbiddenLLM: LLMCall = async () => {
    llmCalls += 1;
    throw new Error('LLM must not be called by the deterministic hard-set run');
  };
  const normalization = denormalizeDetailed(readRaw(caseId));
  const report = await structureDictation(
    'CT_ABDOMEN_MIKHAILOV',
    normalization.text,
    forbiddenLLM,
    { allowLLM: false },
  );
  return { normalization, report, llmCalls };
}

function exactNumberCount(text: string, value: number): number {
  return [...text.matchAll(new RegExp(`(^|\\D)${value}(?=$|\\D)`, 'gu'))].length;
}

function bodyEvidence(report: DictationReport): string {
  return report.blocks
    .filter((block) => block.id !== 'conclusion' && block.origin === 'transcript')
    .map((block) => block.text)
    .join(' ');
}

function assertClosedExtractiveContract(
  normalized: string,
  report: DictationReport,
  llmCalls: number,
): void {
  assert.equal(llmCalls, 0);
  assert.equal(report.structuringRun.llmAllowed, false);
  assert.equal(report.structuringRun.llmCalled, false);

  assert.equal(report.routing.assignments.length, report.routing.atoms.length);
  assert.equal(
    new Set(report.routing.assignments.map((assignment) => assignment.atomId)).size,
    report.routing.atoms.length,
  );
  for (const atom of report.routing.atoms) {
    assert.equal(normalized.slice(atom.start, atom.end), atom.text);
    const owners = report.routing.assignments.filter(
      (assignment) => assignment.atomId === atom.id,
    );
    assert.equal(owners.length, 1);
    if (owners[0].sectionId === null) {
      assert.equal(owners[0].method, 'unmatched');
    }
  }

  for (const block of report.blocks) {
    if (block.origin === 'template_default') continue;
    assert.equal(
      block.text,
      block.evidence.map((span) => normalized.slice(span.start, span.end)).join(' '),
      `${block.id} must contain only exact source spans`,
    );
  }
}

test('Михалыч case 1: 9 мм остаётся единственным spleen-span, router не множит overlap-числа', async () => {
  const { normalization, report, llmCalls } = await runCase('01');
  assertClosedExtractiveContract(normalization.text, report, llmCalls);

  assert.doesNotMatch(normalization.text, /(^|\D)103(?=$|\D)/u);
  const spleen = report.blocks.find((block) => block.id === 'spleen')!;
  const lymph = report.blocks.find((block) => block.id === 'lymph_hilum')!;
  assert.match(spleen.text, /9 миллиметр/iu);
  assert.equal(exactNumberCount(bodyEvidence(report), 9), 1);
  assert.doesNotMatch(lymph.text, /9 миллиметр/iu);

  // The saved pre-v2 raw fixture already contains the overlap-produced second
  // 13. The deterministic router must preserve, but never amplify, raw evidence.
  assert.equal(
    exactNumberCount(bodyEvidence(report), 13),
    exactNumberCount(normalization.text, 13),
  );
});

test('Михалыч case 2: ambiguous 50 53 is blocked and bowel/pelvis stay separate', async () => {
  const { normalization, report, llmCalls } = await runCase('02');
  assertClosedExtractiveContract(normalization.text, report, llmCalls);

  assert.match(normalization.text, /10 на 29 на 151 миллиметр/iu);
  assert.match(normalization.text, /плотность 50 53 единиц/iu);
  assert.doesNotMatch(normalization.text, /(^|\D)103(?=$|\D)/u);
  assert.deepEqual(
    normalization.issues.map((issue) => issue.code),
    ['ambiguous_number_sequence'],
  );

  const bowel = report.blocks.find((block) => block.id === 'bowel')!;
  const pelvis = report.blocks.find((block) => block.id === 'pelvis')!;
  assert.match(bowel.text, /утолщение стенок.+кишки.+5 миллиметров/iu);
  assert.doesNotMatch(bowel.text, /свободной жидкости/iu);
  assert.match(pelvis.text, /в малом тазу.+свободной жидкости/iu);
  assert.doesNotMatch(pelvis.text, /утолщение стенок/iu);
});

test('Михалыч case 3: vascular span belongs only to celiac_trunk and dictated conclusion is exact', async () => {
  const { normalization, report, llmCalls } = await runCase('03');
  assertClosedExtractiveContract(normalization.text, report, llmCalls);

  assert.doesNotMatch(normalization.text, /(^|\D)103(?=$|\D)/u);
  const body = report.blocks.filter(
    (block) => block.id !== 'conclusion' && block.origin === 'transcript',
  );
  assert.deepEqual(
    body
      .filter((block) => /экстравазальн|чре[дв]ного ствол/iu.test(block.text))
      .map((block) => block.id),
    ['celiac_trunk'],
  );
  assert.doesNotMatch(
    report.blocks.find((block) => block.id === 'pancreas')!.text,
    /экстравазальн|чре[дв]ного ствол/iu,
  );
  assert.doesNotMatch(
    report.blocks.find((block) => block.id === 'choledoch')!.text,
    /экстравазальн|чре[дв]ного ствол/iu,
  );

  const conclusion = report.blocks.find((block) => block.id === 'conclusion')!;
  assert.equal(conclusion.origin, 'transcript');
  assert.equal(conclusion.evidence.length, 1);
  assert.equal(
    conclusion.text,
    normalization.text.slice(
      conclusion.evidence[0].start,
      conclusion.evidence[0].end,
    ),
  );
  assert.match(conclusion.text, /^заключение заключение кт признаки/iu);
});
