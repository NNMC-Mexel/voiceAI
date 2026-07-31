import { createHash } from 'crypto';
import type {
  BlockNode,
  DocBlock,
  DocNode,
  DocTemplate,
  SlotDef,
  SlotValues,
} from './doc-model.js';
import {
  assignNumbersToKeywordGroupsDetailed,
  extractNumbers,
  hasPhrase,
  normalizeNumberWordsDetailed,
} from './numbers.js';

export const TEMPLATE_COMPOSER_VERSION = 'radiology-template-composer-v1';

export interface TemplateSectionAtom {
  atomId: string;
  sectionId: string;
  start: number;
  end: number;
  text: string;
}

export interface TemplateEvidenceSpan {
  atomId: string;
  start: number;
  end: number;
  text: string;
  source: 'transcript';
  normalized: {
    start: number;
    end: number;
    text: string;
  };
  raw: {
    start: number;
    end: number;
    text: string;
  } | null;
}

export interface TemplateNormalizationAlignmentSpan {
  sourceStart: number;
  sourceEnd: number;
  normalizedStart: number;
  normalizedEnd: number;
  kind?: string;
}

export interface TemplateComposerSourceContext {
  rawTranscript?: string;
  alignment?: TemplateNormalizationAlignmentSpan[];
}

export interface TemplateFieldAssignment {
  id: string;
  fieldId: string;
  sectionId: string;
  kind: 'slot' | 'switch' | 'explicit_normal';
  status: 'applied' | 'ambiguous' | 'conflict' | 'invalid_unit' | 'incomplete';
  value: unknown;
  canonicalUnit: 'mm' | 'cm' | 'HU' | 'percent' | null;
  unitSource: 'transcript' | 'template_schema' | null;
  ruleId: string;
  /** Compatibility detail for numeric consumers; `value` is canonical. */
  values?: number[];
  formattedText?: string;
  unit?: string;
  conversionRuleId?: 'cm-to-mm-v1' | 'mm-to-cm-v1';
  optionId?: string;
  evidence: TemplateEvidenceSpan[];
}

export interface TemplateCompositionIssue {
  code: string;
  severity: 'critical' | 'warning';
  message: string;
  sectionId?: string;
  fieldId?: string;
  atomId?: string;
  evidence?: TemplateEvidenceSpan[];
}

export interface TemplateDraftSegment {
  id: string;
  sectionId: string;
  fieldId?: string;
  kind:
    | 'template_literal'
    | 'template_default'
    | 'transcript_value'
    | 'template_choice'
    | 'derived'
    | 'verbatim';
  origin:
    | 'template_literal'
    | 'template_default_value'
    | 'transcript_slot'
    | 'transcript_switch'
    | 'transcript_append'
    | 'derived_from_transcript'
    | 'dictated_conclusion';
  text: string;
  start: number;
  end: number;
  evidence: TemplateEvidenceSpan[];
  confirmationRequired: boolean;
  defaultKind?: 'placeholder' | 'clinical_default';
  unit?: string;
}

export interface TemplateReviewDraftSection {
  id: string;
  label: string;
  text: string;
  mode:
    | 'template_default'
    | 'template_filled'
    | 'explicit_normal'
    | 'verbatim_fallback'
    | 'conclusion';
  segmentIds: string[];
  start: number;
  end: number;
  issues: TemplateCompositionIssue[];
}

export interface TemplateReviewDraft {
  version: string;
  composerVersion: string;
  templateId: string;
  templateSha256: string;
  title: string;
  fullText: string;
  sha256: string;
  status: 'complete' | 'partial' | 'failed';
  sections: TemplateReviewDraftSection[];
  segments: TemplateDraftSegment[];
  fieldAssignments: TemplateFieldAssignment[];
  residualAtomIds: string[];
  issues: TemplateCompositionIssue[];
}

interface PendingSegment extends Omit<TemplateDraftSegment, 'start' | 'end'> {}

interface StoredSlot {
  slot: SlotDef;
  values: number[];
  evidence: TemplateEvidenceSpan[];
  unitSource?: 'spoken' | 'template_schema';
}

interface StoredSwitch {
  optionId: string;
  evidence: TemplateEvidenceSpan[];
}

interface ComposerState {
  slots: Map<string, StoredSlot>;
  switches: Map<string, StoredSwitch>;
  explicitNormal: Map<string, TemplateEvidenceSpan[]>;
  assignments: TemplateFieldAssignment[];
  issues: TemplateCompositionIssue[];
  residualAtomIds: Set<string>;
  residualEvidenceBySection: Map<string, TemplateEvidenceSpan[]>;
  residualEvidenceKeys: Set<string>;
  fallbackSections: Set<string>;
  atomIssues: Map<string, TemplateCompositionIssue[]>;
  sourceContext: TemplateComposerSourceContext;
}

interface NormalizedMapRange {
  normalizedStart: number;
  normalizedEnd: number;
  sourceStart: number;
  sourceEnd: number;
}

interface NormalizedAtom {
  text: string;
  ranges: NormalizedMapRange[];
  issues: ReturnType<typeof normalizeNumberWordsDetailed>['issues'];
}

function sha256(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function canonicalValue(value: unknown): unknown {
  if (typeof value === 'function') {
    return { $function: Function.prototype.toString.call(value) };
  }
  if (value === undefined) return undefined;
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, canonicalValue(entry)]),
    );
  }
  return value;
}

export function templateSha256(template: DocTemplate): string {
  return sha256(JSON.stringify(canonicalValue(template)));
}

function fieldId(block: DocBlock, slot: SlotDef): string {
  return slot.fieldId ?? `${block.id}.${slot.name}`;
}

function switchFieldId(block: DocBlock, name: string, configured?: string): string {
  return configured ?? `${block.id}.${name}`;
}

function formatSlot(values: number[], slot: SlotDef): string {
  const parts = values.map((value) => {
    const text = slot.decimals === undefined ? String(value) : value.toFixed(slot.decimals);
    const normalized = text.replace('.', ',');
    return slot.signMode === 'always' && value >= 0 ? `+${normalized}` : normalized;
  });
  const arity = slot.arity ?? 1;
  while (parts.length < arity) parts.push('__');
  return parts.join(slot.join ?? ' ');
}

function collectSlots(nodes: (BlockNode | DocNode)[]): SlotDef[] {
  const result: SlotDef[] = [];
  for (const node of nodes) {
    if (node.kind === 'slot') result.push(node.slot);
    if (node.kind === 'switch') {
      for (const option of node.sw.options) result.push(...collectSlots(option.nodes));
    }
  }
  return result;
}

function normalizeWithMap(input: string): NormalizedAtom {
  const normalized = normalizeNumberWordsDetailed(input);
  const ranges: NormalizedMapRange[] = [];
  let sourceCursor = 0;
  let normalizedCursor = 0;
  for (const transformation of normalized.transformations) {
    const unchangedLength = transformation.start - sourceCursor;
    if (unchangedLength > 0) {
      ranges.push({
        normalizedStart: normalizedCursor,
        normalizedEnd: normalizedCursor + unchangedLength,
        sourceStart: sourceCursor,
        sourceEnd: transformation.start,
      });
      normalizedCursor += unchangedLength;
    }
    ranges.push({
      normalizedStart: normalizedCursor,
      normalizedEnd: normalizedCursor + transformation.normalizedText.length,
      sourceStart: transformation.start,
      sourceEnd: transformation.end,
    });
    normalizedCursor += transformation.normalizedText.length;
    sourceCursor = transformation.end;
  }
  if (sourceCursor < input.length) {
    ranges.push({
      normalizedStart: normalizedCursor,
      normalizedEnd: normalized.text.length,
      sourceStart: sourceCursor,
      sourceEnd: input.length,
    });
  }
  return {
    text: normalized.text
      .toLowerCase()
      .replace(/ё/g, 'е')
      .replace(/[\u2212\uFE63\uFF0D]/gu, '-'),
    ranges,
    issues: normalized.issues,
  };
}

function phraseMatchRanges(
  text: string,
  phrase: string,
): Array<{ start: number; end: number }> {
  const words = phrase
    .toLowerCase()
    .replace(/ё/g, 'е')
    .split(/\s+/u)
    .filter(Boolean)
    .map((word) => word.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&'))
    .map((word) => (word.length >= 4 ? `${word}[a-zа-я]*` : word));
  if (words.length === 0) return [];
  const expression = new RegExp(
    `(^|[^a-zа-я])(${words.join('\\s+')})(?=$|[^a-zа-я])`,
    'giu',
  );
  return [...text.matchAll(expression)].map((match) => {
    const start = (match.index ?? 0) + (match[1]?.length ?? 0);
    return {
      start,
      end: start + (match[2]?.length ?? 0),
    };
  });
}

function sourceRange(
  normalized: NormalizedAtom,
  start: number,
  end: number,
): { start: number; end: number } {
  const hits = normalized.ranges.filter((range) => (
    range.normalizedStart < end && range.normalizedEnd > start
  ));
  if (hits.length === 0) return { start, end };
  const first = hits[0];
  const last = hits[hits.length - 1];
  const mappedStart = first.normalizedEnd - first.normalizedStart
    === first.sourceEnd - first.sourceStart
    ? first.sourceStart + Math.max(0, start - first.normalizedStart)
    : first.sourceStart;
  const mappedEnd = last.normalizedEnd - last.normalizedStart
    === last.sourceEnd - last.sourceStart
    ? last.sourceStart + Math.min(last.sourceEnd - last.sourceStart, end - last.normalizedStart)
    : last.sourceEnd;
  return { start: mappedStart, end: mappedEnd };
}

function atomEvidence(
  transcript: string,
  atom: TemplateSectionAtom,
  localStart = 0,
  localEnd = atom.text.length,
  sourceContext: TemplateComposerSourceContext = {},
): TemplateEvidenceSpan {
  const start = atom.start + localStart;
  const end = atom.start + localEnd;
  const normalizedText = transcript.slice(start, end);
  const alignmentHits = (sourceContext.alignment ?? []).filter((span) => (
    span.normalizedStart < end && span.normalizedEnd > start
  ));
  const firstAlignment = alignmentHits[0];
  const lastAlignment = alignmentHits[alignmentHits.length - 1];
  const mapAlignmentBoundary = (
    span: TemplateNormalizationAlignmentSpan,
    normalizedOffset: number,
    boundary: 'start' | 'end',
  ): number => {
    const sourceLength = span.sourceEnd - span.sourceStart;
    const normalizedLength = span.normalizedEnd - span.normalizedStart;
    if (sourceLength === normalizedLength) {
      return span.sourceStart + Math.max(
        0,
        Math.min(sourceLength, normalizedOffset - span.normalizedStart),
      );
    }
    return boundary === 'start' ? span.sourceStart : span.sourceEnd;
  };
  const raw = sourceContext.rawTranscript !== undefined
    ? firstAlignment && lastAlignment
      ? {
          start: mapAlignmentBoundary(firstAlignment, start, 'start'),
          end: mapAlignmentBoundary(lastAlignment, end, 'end'),
          text: sourceContext.rawTranscript.slice(
            mapAlignmentBoundary(firstAlignment, start, 'start'),
            mapAlignmentBoundary(lastAlignment, end, 'end'),
          ),
        }
      : sourceContext.rawTranscript === transcript
        ? {
            start,
            end,
            text: sourceContext.rawTranscript.slice(start, end),
          }
        : null
    : null;
  return {
    atomId: atom.atomId,
    start,
    end,
    text: normalizedText,
    source: 'transcript',
    normalized: {
      start,
      end,
      text: normalizedText,
    },
    raw,
  };
}

function pushIssue(
  state: ComposerState,
  issue: TemplateCompositionIssue,
): void {
  state.issues.push(issue);
  if (issue.atomId) {
    const current = state.atomIssues.get(issue.atomId) ?? [];
    current.push(issue);
    state.atomIssues.set(issue.atomId, current);
  }
}

function addResidualEvidence(
  state: ComposerState,
  sectionId: string,
  atomId: string,
  evidence: TemplateEvidenceSpan[],
): void {
  state.residualAtomIds.add(atomId);
  const current = state.residualEvidenceBySection.get(sectionId) ?? [];
  for (const span of evidence) {
    const key = `${sectionId}:${span.atomId}:${span.start}:${span.end}`;
    if (state.residualEvidenceKeys.has(key)) continue;
    state.residualEvidenceKeys.add(key);
    current.push(span);
  }
  current.sort((left, right) => left.start - right.start || left.end - right.end);
  state.residualEvidenceBySection.set(sectionId, current);
}

function hasDistinctClinicalWords(value: string): boolean {
  const normalized = normalizeNumberWordsDetailed(value).text
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[+\-−]?\d+(?:[.,]\d+)?/gu, ' ')
    .replace(
      /(^|[^a-zа-я])(?:мм|см|hu|ху|миллиметр[а-я]*|сантиметр[а-я]*|хаунсфилд[а-я]*|процент[а-я]*)(?=$|[^a-zа-я])/giu,
      ' ',
    )
    .replace(
      /(^|[^a-zа-я])(?:на|до|и|плюс|минус|значение|значения|размер|размеры|диаметр|составляет|равен|равна|еще|также)(?=$|[^a-zа-я])/giu,
      ' ',
    );
  return /[a-zа-я]{3,}/u.test(normalized);
}

function unusedNumbersHaveDistinctClinicalContext(
  normalized: NormalizedAtom,
  block: DocBlock,
  slots: SlotDef[],
  usedNumberIndexes: ReadonlySet<number>,
  numberTokens: ReturnType<typeof extractNumbers>,
  residual: TemplateEvidenceSpan[],
): boolean {
  if (!hasDistinctClinicalWords(residual.map((span) => span.text).join(' '))) {
    return false;
  }
  const firstUnused = numberTokens.findIndex((_, index) => !usedNumberIndexes.has(index));
  if (firstUnused < 0) return false;
  const firstUnusedStart = numberTokens[firstUnused].start;
  const previousUsedEnd = numberTokens.reduce((latest, token, index) => (
    usedNumberIndexes.has(index) && token.end <= firstUnusedStart
      ? Math.max(latest, token.end)
      : latest
  ), 0);
  const prefix = normalized.text.slice(previousUsedEnd, firstUnusedStart);
  const masked = prefix.split('');
  const schemaPhrases = [
    ...block.anchors,
    ...slots.flatMap((slot) => [
      ...slot.keywords,
      ...(slot.routingAliases ?? []),
    ]),
    ...block.nodes.flatMap((node) => (
      node.kind === 'switch'
        ? node.sw.options.flatMap((option) => [
            ...option.triggers,
            ...(option.excludes ?? []),
          ])
        : []
    )),
  ];
  for (const phrase of schemaPhrases) {
    for (const range of phraseMatchRanges(prefix, phrase)) {
      masked.fill(' ', range.start, range.end);
    }
  }
  return hasDistinctClinicalWords(masked.join(''));
}

function defaultKind(slot: SlotDef): 'placeholder' | 'clinical_default' {
  if (slot.defaultKind) return slot.defaultKind;
  return /_{2,}/u.test(slot.default) ? 'placeholder' : 'clinical_default';
}

/**
 * Deterministically composes a physician-review draft from already-routed
 * transcript atoms. It never routes an atom and never invokes an LLM.
 */
export function composeTemplateReviewDraft(
  template: DocTemplate,
  transcript: string,
  atoms: TemplateSectionAtom[],
  sourceContext: TemplateComposerSourceContext = {},
): TemplateReviewDraft {
  const state: ComposerState = {
    slots: new Map(),
    switches: new Map(),
    explicitNormal: new Map(),
    assignments: [],
    issues: [],
    residualAtomIds: new Set(),
    residualEvidenceBySection: new Map(),
    residualEvidenceKeys: new Set(),
    fallbackSections: new Set(),
    atomIssues: new Map(),
    sourceContext,
  };
  const blockById = new Map(template.blocks.map((block) => [block.id, block]));
  const orderedAtoms = [...atoms].sort((left, right) => (
    left.start - right.start || left.end - right.end || left.atomId.localeCompare(right.atomId)
  ));

  for (const atom of orderedAtoms) {
    const block = blockById.get(atom.sectionId);
    const exact = transcript.slice(atom.start, atom.end);
    if (!block || exact !== atom.text || atom.start < 0 || atom.end < atom.start) {
      pushIssue(state, {
        code: block ? 'evidence_span_mismatch' : 'unknown_section',
        severity: 'critical',
        message: block
          ? `Atom ${atom.atomId} does not match its immutable transcript span.`
          : `Atom ${atom.atomId} targets unknown section ${atom.sectionId}.`,
        sectionId: atom.sectionId,
        atomId: atom.atomId,
      });
      state.residualAtomIds.add(atom.atomId);
      state.fallbackSections.add(atom.sectionId);
      continue;
    }
    if (block.id === template.conclusionBlockId) continue;
    parseAtom(template, transcript, block, atom, state);
  }

  return renderDraft(template, transcript, orderedAtoms, state);
}

function parseAtom(
  _template: DocTemplate,
  transcript: string,
  block: DocBlock,
  atom: TemplateSectionAtom,
  state: ComposerState,
): void {
  const normalized = normalizeWithMap(atom.text);
  const ambiguousSign = atom.text.match(
    /(?:плюс\s*[-/]?\s*минус|\+\s*\/\s*-|[±]|[\u00AD\u2010-\u2015\u2043\u207B\u208B](?=\s*\d))/iu,
  );
  if (ambiguousSign) {
    const localStart = ambiguousSign.index ?? 0;
    const evidence = atomEvidence(
      transcript,
      atom,
      localStart,
      localStart + ambiguousSign[0].length,
      state.sourceContext,
    );
    pushIssue(state, {
      code: 'ambiguous_numeric_sign',
      severity: 'critical',
      message: `Atom ${atom.atomId} contains an ambiguous numeric sign.`,
      sectionId: block.id,
      atomId: atom.atomId,
      evidence: [evidence],
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }
  if (normalized.issues.length > 0) {
    for (const issue of normalized.issues) {
      const evidence = atomEvidence(
        transcript,
        atom,
        issue.start,
        issue.end,
        state.sourceContext,
      );
      pushIssue(state, {
        code: issue.code,
        severity: 'critical',
        message: issue.message,
        sectionId: block.id,
        atomId: atom.atomId,
        evidence: [evidence],
      });
    }
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }

  const fullEvidence = atomEvidence(
    transcript,
    atom,
    0,
    atom.text.length,
    state.sourceContext,
  );
  const explicitNormal = /(^|[^а-я])норм(?:а|е)(?=$|[^а-я])/u.test(normalized.text)
    || /без\s+особенност/u.test(normalized.text);
  if (explicitNormal) {
    const field = `${block.id}.normal`;
    state.explicitNormal.set(block.id, [fullEvidence]);
    state.assignments.push({
      id: `${atom.atomId}:${field}`,
      fieldId: field,
      sectionId: block.id,
      kind: 'explicit_normal',
      status: 'applied',
      value: true,
      canonicalUnit: null,
      unitSource: null,
      ruleId: 'explicit-normal-v1',
      formattedText: atom.text,
      evidence: [fullEvidence],
    });
  }

  const selectedSwitches = new Map<string, {
    fieldId: string;
    optionId: string;
    evidence: TemplateEvidenceSpan;
  }>();
  for (const node of block.nodes) {
    if (node.kind !== 'switch') continue;
    const matching = node.sw.options.flatMap((option) => {
      const trigger = option.triggers
        .map((phrase) => phraseMatchRanges(normalized.text, phrase)[0])
        .find(Boolean);
      const excluded = (option.excludes ?? []).some((phrase) => (
        hasPhrase(normalized.text, phrase)
      ));
      return trigger && !excluded ? [{ option, trigger }] : [];
    });
    if (matching.length > 1) {
      const switchId = switchFieldId(block, node.sw.name, node.sw.fieldId);
      pushIssue(state, {
        code: 'switch_conflict',
        severity: 'critical',
        message: `Atom ${atom.atomId} selects multiple options for ${switchId}.`,
        sectionId: block.id,
        fieldId: switchId,
        atomId: atom.atomId,
        evidence: [fullEvidence],
      });
      state.residualAtomIds.add(atom.atomId);
      state.fallbackSections.add(block.id);
      continue;
    }
    if (matching.length === 1) {
      const switchId = switchFieldId(block, node.sw.name, node.sw.fieldId);
      const selectedOption = matching[0].option;
      const triggerLocal = sourceRange(
        normalized,
        matching[0].trigger.start,
        matching[0].trigger.end,
      );
      const triggerEvidence = atomEvidence(
        transcript,
        atom,
        triggerLocal.start,
        triggerLocal.end,
        state.sourceContext,
      );
      const negatedPositiveChoice = (
        selectedOption.id !== node.sw.default
        && /(^|[^a-zа-я])(?:не|нет|без|отсутств[а-я]*)(?=$|[^a-zа-я])/iu
          .test(normalized.text)
      );
      if (negatedPositiveChoice) {
        pushIssue(state, {
          code: 'ambiguous_switch_negation',
          severity: 'critical',
          message: `Atom ${atom.atomId} contains a negation that conflicts with positive switch ${switchId}.`,
          sectionId: block.id,
          fieldId: switchId,
          atomId: atom.atomId,
          evidence: [fullEvidence],
        });
        state.residualAtomIds.add(atom.atomId);
        state.fallbackSections.add(block.id);
        continue;
      }
      selectedSwitches.set(node.sw.name, {
        fieldId: switchId,
        optionId: selectedOption.id,
        evidence: triggerEvidence,
      });
      const existing = state.switches.get(switchId);
      const status = existing && existing.optionId !== selectedOption.id ? 'conflict' : 'applied';
      state.assignments.push({
        id: `${atom.atomId}:${switchId}`,
        fieldId: switchId,
        sectionId: block.id,
        kind: 'switch',
        status,
        value: selectedOption.id,
        canonicalUnit: null,
        unitSource: null,
        ruleId: 'switch-trigger-v1',
        optionId: selectedOption.id,
        evidence: [triggerEvidence],
      });
      if (status === 'conflict') {
        pushIssue(state, {
          code: 'field_conflict',
          severity: 'critical',
          message: `Conflicting values were dictated for ${switchId}.`,
          sectionId: block.id,
          fieldId: switchId,
          atomId: atom.atomId,
          evidence: [...existing!.evidence, triggerEvidence],
        });
        state.residualAtomIds.add(atom.atomId);
        state.fallbackSections.add(block.id);
      } else {
        state.switches.set(switchId, {
          optionId: selectedOption.id,
          evidence: [triggerEvidence],
        });
      }
    }
  }

  const eligibleSlots: SlotDef[] = [];
  for (const node of block.nodes) {
    if (node.kind === 'slot') eligibleSlots.push(node.slot);
    if (node.kind === 'switch') {
      const selected = selectedSwitches.get(node.sw.name);
      if (!selected) continue;
      const option = node.sw.options.find((candidate) => candidate.id === selected.optionId);
      if (option) eligibleSlots.push(...collectSlots(option.nodes));
    }
  }
  const numberTokens = extractNumbers(normalized.text);
  const orphanUnit = findOrphanSpokenUnit(normalized.text, numberTokens);
  if (orphanUnit) {
    const local = sourceRange(normalized, orphanUnit.start, orphanUnit.end);
    const evidence = atomEvidence(
      transcript,
      atom,
      local.start,
      local.end,
      state.sourceContext,
    );
    pushIssue(state, {
      code: 'orphan_unit',
      severity: 'critical',
      message: `Atom ${atom.atomId} contains a unit that is not bound to a numeric value.`,
      sectionId: block.id,
      atomId: atom.atomId,
      evidence: [evidence],
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }
  const used = new Set<number>();
  const scalarSlots = eligibleSlots.filter((slot) => (slot.arity ?? 1) <= 1);
  const scalarBinding = assignNumbersToKeywordGroupsDetailed(
    numberTokens,
    scalarSlots.map((slot) => slot.keywords),
    used,
  );
  if (scalarBinding.ambiguous) {
    for (const slot of scalarSlots) {
      state.assignments.push({
        id: `${atom.atomId}:${fieldId(block, slot)}:ambiguous`,
        fieldId: fieldId(block, slot),
        sectionId: block.id,
        kind: 'slot',
        status: 'ambiguous',
        value: null,
        canonicalUnit: slot.unit ?? null,
        unitSource: null,
        ruleId: 'slot-bipartite-tie-v1',
        evidence: [fullEvidence],
      });
    }
    pushIssue(state, {
      code: 'ambiguous_field_binding',
      severity: 'critical',
      message: `Atom ${atom.atomId} has multiple equally valid field/value assignments.`,
      sectionId: block.id,
      atomId: atom.atomId,
      evidence: [fullEvidence],
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }
  const scalarAssignments = scalarBinding.assignment;
  for (let slotIndex = 0; slotIndex < scalarSlots.length; slotIndex++) {
    const numberIndex = scalarAssignments[slotIndex];
    if (numberIndex === undefined) continue;
    used.add(numberIndex);
    storeSlot(
      transcript,
      block,
      atom,
      normalized,
      scalarSlots[slotIndex],
      [numberTokens[numberIndex]],
      state,
    );
  }
  for (const slot of eligibleSlots.filter((candidate) => (candidate.arity ?? 1) > 1)) {
    const free = numberTokens
      .map((token, index) => ({ token, index }))
      .filter(({ index }) => !used.has(index));
    const addressed = block.anchors.some((anchor) => hasPhrase(normalized.text, anchor))
      || slot.keywords.some((keyword) => hasPhrase(normalized.text, keyword));
    if (!addressed || free.length === 0) continue;
    const take = free.slice(0, slot.arity ?? 1);
    for (const item of take) used.add(item.index);
    storeSlot(
      transcript,
      block,
      atom,
      normalized,
      slot,
      take.map((item) => item.token),
      state,
    );
    if (take.length < (slot.arity ?? 1)) {
      const id = fieldId(block, slot);
      const assignment = [...state.assignments].reverse().find((item) => (
        item.fieldId === id && item.id.startsWith(`${atom.atomId}:`)
      ));
      if (assignment) assignment.status = 'incomplete';
      state.slots.delete(id);
      pushIssue(state, {
        code: 'partial_dimension',
        severity: 'critical',
        message: `Field ${fieldId(block, slot)} has ${take.length} of ${slot.arity} dimensions.`,
        sectionId: block.id,
        fieldId: fieldId(block, slot),
        atomId: atom.atomId,
        evidence: [fullEvidence],
      });
      state.residualAtomIds.add(atom.atomId);
      state.fallbackSections.add(block.id);
    }
  }

  const unused = numberTokens
    .map((token, index) => ({ token, index }))
    .filter(({ index }) => !used.has(index));

  const parsedSomething = explicitNormal
    || selectedSwitches.size > 0
    || used.size > 0;
  const residual = residualEvidence(
    transcript,
    atom,
    block,
    normalized,
    eligibleSlots,
    used,
    numberTokens,
    state.sourceContext,
  );
  if (unused.length > 0) {
    const evidence = unused.map(({ token }) => {
      const local = sourceRange(normalized, token.start, token.end);
      return atomEvidence(
        transcript,
        atom,
        local.start,
        local.end,
        state.sourceContext,
      );
    });
    const hasDistinctClinicalText = unusedNumbersHaveDistinctClinicalContext(
      normalized,
      block,
      eligibleSlots,
      used,
      numberTokens,
      residual,
    );
    pushIssue(state, {
      code: 'unused_number',
      severity: hasDistinctClinicalText ? 'warning' : 'critical',
      message: hasDistinctClinicalText
        ? `Atom ${atom.atomId} contains a number in residual clinical text and requires verbatim review.`
        : `Atom ${atom.atomId} contains an unbound numeric value with no distinct clinical context.`,
      sectionId: block.id,
      atomId: atom.atomId,
      evidence,
    });
    if (!hasDistinctClinicalText) {
      for (const assignment of state.assignments) {
        if (
          assignment.id.startsWith(`${atom.atomId}:`)
          && assignment.kind === 'slot'
          && assignment.status === 'applied'
        ) {
          assignment.status = 'ambiguous';
          state.slots.delete(assignment.fieldId);
        }
      }
      state.residualAtomIds.add(atom.atomId);
      state.fallbackSections.add(block.id);
    }
  }
  if (residual.length > 0) {
    pushIssue(state, {
      code: 'residual_clinical_text',
      severity: 'warning',
      message: `Atom ${atom.atomId} contains clinical text outside the deterministic template grammar.`,
      sectionId: block.id,
      atomId: atom.atomId,
      evidence: residual,
    });
    addResidualEvidence(state, block.id, atom.atomId, residual);
  } else if (!parsedSomething) {
    pushIssue(state, {
      code: 'empty_template_command',
      severity: 'warning',
      message: `Atom ${atom.atomId} names a section but contains no deterministic field value.`,
      sectionId: block.id,
      atomId: atom.atomId,
      evidence: [fullEvidence],
    });
    addResidualEvidence(state, block.id, atom.atomId, [fullEvidence]);
  }
}

function storeSlot(
  transcript: string,
  block: DocBlock,
  atom: TemplateSectionAtom,
  normalized: NormalizedAtom,
  slot: SlotDef,
  tokens: ReturnType<typeof extractNumbers>,
  state: ComposerState,
): void {
  const id = fieldId(block, slot);
  const evidence = tokens.map((token) => {
    const local = sourceRange(normalized, token.start, token.end);
    return atomEvidence(
      transcript,
      atom,
      local.start,
      local.end,
      state.sourceContext,
    );
  });
  if (tokens.some((token) => !Number.isFinite(token.value))) {
    state.assignments.push({
      id: `${atom.atomId}:${id}`,
      fieldId: id,
      sectionId: block.id,
      kind: 'slot',
      status: 'incomplete',
      value: null,
      canonicalUnit: slot.unit ?? null,
      unitSource: null,
      ruleId: slot.validation?.ruleId ?? 'finite-number-v1',
      evidence,
    });
    pushIssue(state, {
      code: 'non_finite_number',
      severity: 'critical',
      message: `Field ${id} contains a number outside the finite numeric domain.`,
      sectionId: block.id,
      fieldId: id,
      atomId: atom.atomId,
      evidence,
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }
  const directUnits = tokens.map((token) => detectSpokenUnit(normalized.text, token));
  const unitEvidenceKeys = new Set<string>();
  tokens
    .map((token) => adjacentSpokenUnit(normalized.text, token))
    .filter((mention): mention is SpokenUnitMention => Boolean(mention))
    .forEach((mention) => {
      const key = `${mention.start}:${mention.end}`;
      if (unitEvidenceKeys.has(key)) return;
      unitEvidenceKeys.add(key);
      const local = sourceRange(normalized, mention.start, mention.end);
      evidence.push(atomEvidence(
        transcript,
        atom,
        local.start,
        local.end,
        state.sourceContext,
      ));
    });
  evidence.sort((left, right) => left.start - right.start || left.end - right.end);
  const explicitUnits = directUnits
    .map((unit, index) => ({ unit, index }))
    .filter((item): item is { unit: NonNullable<typeof item.unit>; index: number } => (
      item.unit !== undefined
    ));
  const effectiveUnits = directUnits.map((unit) => unit ?? slot.implicitUnit);
  if (
    tokens.length > 1
    && explicitUnits.length === 1
    && explicitUnits[0].index === tokens.length - 1
  ) {
    effectiveUnits.fill(explicitUnits[0].unit);
  }
  const ambiguousDimensionUnit = (
    tokens.length > 1
    && explicitUnits.length > 0
    && explicitUnits.length < tokens.length
    && !(explicitUnits.length === 1 && explicitUnits[0].index === tokens.length - 1)
  );
  if (ambiguousDimensionUnit) {
    state.assignments.push({
      id: `${atom.atomId}:${id}`,
      fieldId: id,
      sectionId: block.id,
      kind: 'slot',
      status: 'ambiguous',
      value: tokens.map((token) => token.value),
      canonicalUnit: slot.unit ?? null,
      unitSource: 'transcript',
      ruleId: 'dimension-unit-coverage-v1',
      values: tokens.map((token) => token.value),
      evidence,
    });
    pushIssue(state, {
      code: 'ambiguous_dimension_unit',
      severity: 'critical',
      message: `Field ${id} mixes explicit and implicit units without a trailing common unit.`,
      sectionId: block.id,
      fieldId: id,
      atomId: atom.atomId,
      evidence,
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }
  const incompatibleUnitIndex = slot.unit
    ? effectiveUnits.findIndex((unit) => (
        unit !== undefined
        && unit !== slot.unit
        && !(
          (slot.unit === 'mm' && unit === 'cm')
          || (slot.unit === 'cm' && unit === 'mm')
        )
      ))
    : -1;
  const conversionIndexes = new Set(
    effectiveUnits
      .map((unit, index) => ({ unit, index }))
      .filter((item) => slot.unit === 'mm' && item.unit === 'cm')
      .map((item) => item.index),
  );
  const reverseConversionIndexes = new Set(
    effectiveUnits
      .map((unit, index) => ({ unit, index }))
      .filter((item) => slot.unit === 'cm' && item.unit === 'mm')
      .map((item) => item.index),
  );
  const conversionRuleId = conversionIndexes.size > 0
    ? 'cm-to-mm-v1' as const
    : reverseConversionIndexes.size > 0
      ? 'mm-to-cm-v1' as const
      : undefined;
  if (slot.unit && incompatibleUnitIndex >= 0) {
    const spokenUnit = effectiveUnits[incompatibleUnitIndex]!;
    state.assignments.push({
      id: `${atom.atomId}:${id}`,
      fieldId: id,
      sectionId: block.id,
      kind: 'slot',
      status: 'invalid_unit',
      value: tokens.length === 1
        ? tokens[0].value
        : tokens.map((token) => token.value),
      canonicalUnit: slot.unit ?? null,
      unitSource: 'transcript',
      ruleId: 'slot-unit-validation-v1',
      values: tokens.map((token) => token.value),
      unit: spokenUnit,
      evidence,
    });
    pushIssue(state, {
      code: 'unit_mismatch',
      severity: 'critical',
      message: `Field ${id} expects ${slot.unit}, but ${spokenUnit} was dictated.`,
      sectionId: block.id,
      fieldId: id,
      atomId: atom.atomId,
      evidence,
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }
  if (
    slot.unit
    && effectiveUnits.some((unit) => unit === undefined)
    && slot.allowImplicitUnit !== true
  ) {
    state.assignments.push({
      id: `${atom.atomId}:${id}`,
      fieldId: id,
      sectionId: block.id,
      kind: 'slot',
      status: 'invalid_unit',
      value: tokens.length === 1
        ? tokens[0].value
        : tokens.map((token) => token.value),
      canonicalUnit: slot.unit ?? null,
      unitSource: null,
      ruleId: 'slot-unit-validation-v1',
      values: tokens.map((token) => token.value),
      evidence,
    });
    pushIssue(state, {
      code: 'unit_required',
      severity: 'critical',
      message: `Field ${id} requires an explicit ${slot.unit} unit.`,
      sectionId: block.id,
      fieldId: id,
      atomId: atom.atomId,
      evidence,
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }

  const values = tokens.map((token, index) => (
    conversionIndexes.has(index)
      ? token.value * 10
      : reverseConversionIndexes.has(index)
        ? token.value / 10
        : token.value
  ));
  const validation = slot.validation;
  const defaultPhysicalMinExclusive = (
    slot.unit === 'mm' || slot.unit === 'cm'
  )
    ? 0
    : undefined;
  const defaultPhysicalMaxInclusive = slot.unit === 'percent'
    ? 100
    : slot.unit === 'mm' || slot.unit === 'cm'
      ? 2_000
      : undefined;
  const outsidePhysicalDomain = values.some((value) => (
    !Number.isFinite(value)
    || (
      (validation?.minExclusive ?? defaultPhysicalMinExclusive) !== undefined
      && value <= (validation?.minExclusive ?? defaultPhysicalMinExclusive)!
    )
    || (
      validation?.minInclusive !== undefined
      && value < validation.minInclusive
    )
    || (
      (validation?.maxInclusive ?? defaultPhysicalMaxInclusive) !== undefined
      && value > (validation?.maxInclusive ?? defaultPhysicalMaxInclusive)!
    )
  ));
  const outsideTemplateClaim = values.some((value) => (
    (
      validation?.templateClaimMinInclusive !== undefined
      && value < validation.templateClaimMinInclusive
    )
    || (
      validation?.templateClaimMaxInclusive !== undefined
      && value > validation.templateClaimMaxInclusive
    )
  )) || (
    validation?.aggregate?.operation === 'product'
    && validation.aggregate.maxInclusive !== undefined
    && (
      values.reduce((product, value) => product * value, 1)
      / (validation.aggregate.divisor ?? 1)
    ) > validation.aggregate.maxInclusive
  );
  if (outsidePhysicalDomain || outsideTemplateClaim) {
    const issueCode = outsidePhysicalDomain
      ? 'value_out_of_domain'
      : 'value_outside_template_claim';
    state.assignments.push({
      id: `${atom.atomId}:${id}`,
      fieldId: id,
      sectionId: block.id,
      kind: 'slot',
      status: 'incomplete',
      value: values.length === 1 ? values[0] : values,
      canonicalUnit: slot.unit ?? null,
      unitSource: explicitUnits.length > 0
        ? 'transcript'
        : slot.unit
          ? 'template_schema'
          : null,
      ruleId: validation?.ruleId ?? 'finite-number-v1',
      values,
      evidence,
    });
    pushIssue(state, {
      code: issueCode,
      severity: 'critical',
      message: outsidePhysicalDomain
        ? `Field ${id} violates the versioned physical value constraints.`
        : `Field ${id} contradicts the clinical claim encoded by this template section.`,
      sectionId: block.id,
      fieldId: id,
      atomId: atom.atomId,
      evidence,
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }
  const existing = state.slots.get(id);
  const conflict = existing && (
    existing.values.length !== values.length
    || existing.values.some((value, index) => value !== values[index])
  );
  const publicUnitSource = slot.unit
    ? explicitUnits.length > 0
      ? 'transcript' as const
      : 'template_schema' as const
    : null;
  const parserRuleId = [
    (slot.arity ?? 1) > 1 ? 'slot-dimensions-v1' : 'slot-keyword-number-v1',
    conversionRuleId,
  ].filter(Boolean).join('+');
  state.assignments.push({
    id: `${atom.atomId}:${id}`,
    fieldId: id,
    sectionId: block.id,
    kind: 'slot',
    status: conflict ? 'conflict' : 'applied',
    value: values.length === 1 ? values[0] : values,
    canonicalUnit: slot.unit ?? null,
    unitSource: publicUnitSource,
    ruleId: parserRuleId,
    values,
    formattedText: formatSlot(values, slot),
    ...(slot.unit ? { unit: slot.unit } : {}),
    ...(conversionRuleId ? { conversionRuleId } : {}),
    evidence,
  });
  if (conflict) {
    pushIssue(state, {
      code: 'field_conflict',
      severity: 'critical',
      message: `Conflicting values were dictated for ${id}.`,
      sectionId: block.id,
      fieldId: id,
      atomId: atom.atomId,
      evidence: [...existing!.evidence, ...evidence],
    });
    state.residualAtomIds.add(atom.atomId);
    state.fallbackSections.add(block.id);
    return;
  }
  state.slots.set(id, {
    slot,
    values,
    evidence: existing ? [...existing.evidence, ...evidence] : evidence,
    ...(slot.unit
      ? {
          unitSource: explicitUnits.length > 0
            ? 'spoken' as const
            : 'template_schema' as const,
        }
      : {}),
  });
}

function detectSpokenUnit(
  text: string,
  token: ReturnType<typeof extractNumbers>[number],
): 'mm' | 'cm' | 'HU' | 'percent' | undefined {
  return adjacentSpokenUnit(text, token)?.unit;
}

interface SpokenUnitMention {
  start: number;
  end: number;
  unit: 'mm' | 'cm' | 'HU' | 'percent';
}

function spokenUnitMentions(text: string): SpokenUnitMention[] {
  const pattern = /(^|[^a-zа-я])(?:мм|см|hu|ху|миллиметр[а-я]*|сантиметр[а-я]*|хаунсфилд[а-я]*|процент[а-я]*|%)(?=$|[^a-zа-я])/giu;
  const result: SpokenUnitMention[] = [];
  for (const match of text.matchAll(pattern)) {
    const prefixLength = match[1]?.length ?? 0;
    const start = (match.index ?? 0) + prefixLength;
    const spoken = text.slice(start, (match.index ?? 0) + match[0].length)
      .toLowerCase();
    const unit: SpokenUnitMention['unit'] = (
      spoken === 'мм' || spoken.startsWith('миллиметр')
    )
      ? 'mm'
      : spoken === 'см' || spoken.startsWith('сантиметр')
        ? 'cm'
        : spoken === '%' || spoken.startsWith('процент')
          ? 'percent'
          : 'HU';
    result.push({
      start,
      end: (match.index ?? 0) + match[0].length,
      unit,
    });
  }
  return result;
}

function adjacentSpokenUnit(
  text: string,
  token: ReturnType<typeof extractNumbers>[number],
): SpokenUnitMention | undefined {
  const onlyGap = (value: string): boolean => /^[\s,;:()[\]{}-]*$/u.test(value);
  return spokenUnitMentions(text)
    .filter((mention) => (
      (
        token.end <= mention.start
        && mention.start - token.end <= 28
        && onlyGap(text.slice(token.end, mention.start))
      )
      || (
        mention.end <= token.start
        && token.start - mention.end <= 28
        && onlyGap(text.slice(mention.end, token.start))
      )
    ))
    .sort((left, right) => {
      const leftDistance = Math.min(
        Math.abs(left.start - token.end),
        Math.abs(token.start - left.end),
      );
      const rightDistance = Math.min(
        Math.abs(right.start - token.end),
        Math.abs(token.start - right.end),
      );
      return leftDistance - rightDistance || left.start - right.start;
    })[0];
}

function findOrphanSpokenUnit(
  text: string,
  tokens: ReturnType<typeof extractNumbers>,
): SpokenUnitMention | null {
  for (const mention of spokenUnitMentions(text)) {
    const adjacent = tokens.some((token) => {
      const bound = adjacentSpokenUnit(text, token);
      return bound?.start === mention.start && bound.end === mention.end;
    });
    if (!adjacent) return mention;
  }
  return null;
}

function residualEvidence(
  transcript: string,
  atom: TemplateSectionAtom,
  block: DocBlock,
  normalized: NormalizedAtom,
  slots: SlotDef[],
  usedNumberIndexes: ReadonlySet<number>,
  numberTokens: ReturnType<typeof extractNumbers>,
  sourceContext: TemplateComposerSourceContext,
): TemplateEvidenceSpan[] {
  const covered = Array.from({ length: normalized.text.length }, () => false);
  const cover = (start: number, end: number): void => {
    for (let index = Math.max(0, start); index < Math.min(covered.length, end); index++) {
      covered[index] = true;
    }
  };
  const phrases = [
    ...block.anchors,
    ...slots.flatMap((slot) => slot.keywords),
    ...block.nodes.flatMap((node) => (
      node.kind === 'switch'
        ? node.sw.options.flatMap((option) => [
            ...option.triggers,
            ...(option.excludes ?? []),
          ])
        : []
    )),
    'без особенностей',
    'в остальном норма',
    'норма',
  ].filter(Boolean).sort((left, right) => right.length - left.length);
  for (const phrase of phrases) {
    const words = phrase
      .toLowerCase()
      .replace(/ё/g, 'е')
      .split(/\s+/u)
      .map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .map((word) => (word.length >= 4 ? `${word}[a-zа-я]*` : word));
    const expression = new RegExp(
      `(^|[^a-zа-я])(${words.join('\\s+')})(?=$|[^a-zа-я])`,
      'giu',
    );
    for (const match of normalized.text.matchAll(expression)) {
      const start = (match.index ?? 0) + (match[1]?.length ?? 0);
      cover(start, start + (match[2]?.length ?? 0));
    }
  }
  numberTokens.forEach((token, index) => {
    if (!usedNumberIndexes.has(index)) return;
    cover(token.start, token.end);
    for (const unit of spokenUnitMentions(normalized.text)) {
      const between = unit.start >= token.end
        ? normalized.text.slice(token.end, unit.start)
        : unit.end <= token.start
          ? normalized.text.slice(unit.end, token.start)
          : '';
      if (
        between.length <= 28
        && /^[\s,;:()[\]{}-]*$/u.test(between)
        && (unit.start >= token.end || unit.end <= token.start)
      ) {
        cover(unit.start, unit.end);
      }
    }
  });
  const generic = /(^|[^a-zа-я])(?:на|до|плюс|минус|и|значение|значения|размер|размеры|диаметр|составляет|равен|равна)(?=$|[^a-zа-я])/giu;
  for (const match of normalized.text.matchAll(generic)) {
    cover(
      (match.index ?? 0) + (match[1]?.length ?? 0),
      (match.index ?? 0) + match[0].length,
    );
  }
  for (let index = 0; index < normalized.text.length; index++) {
    if (!/[\p{L}\p{N}]/u.test(normalized.text[index] ?? '')) covered[index] = true;
  }

  const residualRanges: Array<{ start: number; end: number }> = [];
  let cursor = 0;
  while (cursor < covered.length) {
    while (cursor < covered.length && covered[cursor]) cursor++;
    if (cursor >= covered.length) break;
    const start = cursor;
    while (cursor < covered.length && !covered[cursor]) cursor++;
    residualRanges.push({ start, end: cursor });
  }
  const connectorGap = /^[\s\p{P}\p{S}]*(?:(?:на|до|плюс|минус|и|значение|значения|размер|размеры|диаметр|составляет|равен|равна)[\s\p{P}\p{S}]*)*$/iu;
  const mergedRanges: Array<{ start: number; end: number }> = [];
  for (const range of residualRanges) {
    const previous = mergedRanges[mergedRanges.length - 1];
    if (
      previous
      && connectorGap.test(normalized.text.slice(previous.end, range.start))
    ) {
      previous.end = range.end;
    } else {
      mergedRanges.push({ ...range });
    }
  }

  const result: TemplateEvidenceSpan[] = [];
  for (const range of mergedRanges) {
    let end = range.end;
    while (
      end < normalized.text.length
      && /[\p{P}\p{S}]/u.test(normalized.text[end] ?? '')
    ) {
      end++;
    }
    const local = sourceRange(normalized, range.start, end);
    result.push(atomEvidence(
      transcript,
      atom,
      local.start,
      local.end,
      sourceContext,
    ));
  }
  return result;
}

function renderDraft(
  template: DocTemplate,
  transcript: string,
  atoms: TemplateSectionAtom[],
  state: ComposerState,
): TemplateReviewDraft {
  const atomsBySection = new Map<string, TemplateSectionAtom[]>();
  for (const atom of atoms) {
    const list = atomsBySection.get(atom.sectionId) ?? [];
    list.push(atom);
    atomsBySection.set(atom.sectionId, list);
  }
  const legacySlots: SlotValues = {};
  for (const stored of state.slots.values()) legacySlots[stored.slot.name] = stored.values;
  const nonDefaultSwitchSelected = template.blocks.some((block) => (
    block.nodes.some((node) => {
      if (node.kind !== 'switch') return false;
      const selected = state.switches.get(
        switchFieldId(block, node.sw.name, node.sw.fieldId),
      );
      return Boolean(selected && selected.optionId !== node.sw.default);
    })
  ));
  const suppressDefaultConclusion = state.fallbackSections.size > 0
    || state.residualAtomIds.size > 0
    || nonDefaultSwitchSelected;

  const sections: TemplateReviewDraftSection[] = [];
  const segments: TemplateDraftSegment[] = [];
  let fullText = template.title;
  for (const block of template.blocks) {
    const sectionAtoms = (atomsBySection.get(block.id) ?? [])
      .sort((left, right) => left.start - right.start || left.end - right.end);
    let pending: PendingSegment[];
    let mode: TemplateReviewDraftSection['mode'];
    if (block.id === template.conclusionBlockId) {
      const unresolvedRequiredField = state.issues.some((issue) => (
        issue.code === 'required_placeholder'
        || issue.code === 'unresolved_template_placeholder'
      ));
      if (sectionAtoms.length > 0) {
        pending = verbatimSegments(
          block,
          transcript,
          sectionAtoms,
          'dictated_conclusion',
          state.sourceContext,
        );
      } else if (suppressDefaultConclusion || unresolvedRequiredField) {
        pending = [];
        if (!state.issues.some((issue) => issue.code === 'default_conclusion_suppressed')) {
          pushIssue(state, {
            code: 'default_conclusion_suppressed',
            severity: 'warning',
            message: 'The normal template conclusion was suppressed because the draft contains a finding or unresolved content.',
            sectionId: block.id,
          });
        }
      } else {
        pending = renderNodes(
          block,
          block.nodes,
          'root',
          state,
          legacySlots,
          true,
        );
      }
      mode = 'conclusion';
    } else if (state.fallbackSections.has(block.id)) {
      pending = verbatimSegments(
        block,
        transcript,
        sectionAtoms,
        'transcript_append',
        state.sourceContext,
      );
      mode = 'verbatim_fallback';
    } else {
      const explicitEvidence = state.explicitNormal.get(block.id);
      pending = renderNodes(
        block,
        block.nodes,
        'root',
        state,
        legacySlots,
        sectionAtoms.length === 0,
        explicitEvidence,
      );
      const residual = state.residualEvidenceBySection.get(block.id) ?? [];
      if (residual.length > 0) {
        if (pending.some((segment) => segment.text.trim().length > 0)) {
          pending.push({
            id: `${block.id}:residual-separator`,
            sectionId: block.id,
            kind: 'template_literal',
            origin: 'template_literal',
            text: ' ',
            evidence: [],
            confirmationRequired: false,
          });
        }
        residual.forEach((evidence, index) => {
          if (index > 0) {
            pending.push({
              id: `${block.id}:residual-separator:${index}`,
              sectionId: block.id,
              kind: 'template_literal',
              origin: 'template_literal',
              text: ' ',
              evidence: [],
              confirmationRequired: false,
            });
          }
          pending.push({
            id: `${block.id}:residual:${evidence.atomId}:${evidence.start}:${evidence.end}`,
            sectionId: block.id,
            kind: 'verbatim',
            origin: 'transcript_append',
            text: evidence.text,
            evidence: [evidence],
            confirmationRequired: false,
          });
        });
      }
      mode = sectionAtoms.length === 0
        ? 'template_default'
        : explicitEvidence && state.assignments.every((assignment) => (
            assignment.sectionId !== block.id || assignment.kind === 'explicit_normal'
          ))
          ? 'explicit_normal'
          : 'template_filled';
    }

    const body = pending.map((segment) => segment.text).join('').trim();
    const sectionIssues = state.issues.filter((issue) => issue.sectionId === block.id);
    const prefix = `\n${block.label}: `;
    fullText += prefix;
    const bodyStart = fullText.length;
    const segmentIds: string[] = [];
    let bodyCursor = bodyStart;
    for (const segment of pending) {
      // Trimming the complete body only affects leading/trailing whitespace.
      let text = segment.text;
      if (bodyCursor === bodyStart) text = text.trimStart();
      const isLast = segment === pending[pending.length - 1];
      if (isLast) text = text.trimEnd();
      if (!text) continue;
      const rendered: TemplateDraftSegment = {
        ...segment,
        text,
        start: bodyCursor,
        end: bodyCursor + text.length,
      };
      segments.push(rendered);
      segmentIds.push(rendered.id);
      fullText += text;
      bodyCursor += text.length;
    }
    sections.push({
      id: block.id,
      label: block.label,
      text: body,
      mode,
      segmentIds,
      start: bodyStart,
      end: bodyCursor,
      issues: sectionIssues,
    });
  }

  const status: TemplateReviewDraft['status'] = state.issues.some(
    (issue) => issue.severity === 'critical',
  )
    ? 'failed'
    : state.issues.length > 0 || state.residualAtomIds.size > 0
      ? 'partial'
      : 'complete';
  return {
    version: TEMPLATE_COMPOSER_VERSION,
    composerVersion: TEMPLATE_COMPOSER_VERSION,
    templateId: template.id,
    templateSha256: templateSha256(template),
    title: template.title,
    fullText,
    sha256: sha256(fullText),
    status,
    sections,
    segments,
    fieldAssignments: state.assignments,
    residualAtomIds: [...state.residualAtomIds],
    issues: state.issues,
  };
}

function verbatimSegments(
  block: DocBlock,
  transcript: string,
  atoms: TemplateSectionAtom[],
  origin: 'transcript_append' | 'dictated_conclusion',
  sourceContext: TemplateComposerSourceContext,
): PendingSegment[] {
  const result: PendingSegment[] = [];
  atoms.forEach((atom, index) => {
    if (index > 0) {
      result.push({
        id: `${block.id}:verbatim-separator:${index}`,
        sectionId: block.id,
        kind: 'template_literal',
        origin: 'template_literal',
        text: ' ',
        evidence: [],
        confirmationRequired: false,
      });
    }
    const evidence = atomEvidence(
      transcript,
      atom,
      0,
      atom.text.length,
      sourceContext,
    );
    result.push({
      id: `${block.id}:verbatim:${atom.atomId}`,
      sectionId: block.id,
      kind: 'verbatim',
      origin,
      text: evidence.text,
      evidence: [evidence],
      confirmationRequired: false,
    });
  });
  return result;
}

function renderNodes(
  block: DocBlock,
  nodes: (BlockNode | DocNode)[],
  path: string,
  state: ComposerState,
  legacySlots: SlotValues,
  missingSection: boolean,
  explicitNormalEvidence?: TemplateEvidenceSpan[],
  choice?: { fieldId: string; evidence: TemplateEvidenceSpan[] },
): PendingSegment[] {
  const result: PendingSegment[] = [];
  nodes.forEach((node, index) => {
    const nodePath = `${path}.${index}`;
    if (node.kind === 'text') {
      const evidence = explicitNormalEvidence ?? choice?.evidence ?? [];
      const kind: TemplateDraftSegment['kind'] = missingSection
        ? 'template_default'
        : explicitNormalEvidence || choice
          ? 'template_choice'
          : 'template_literal';
      result.push({
        id: `${block.id}:node:${nodePath}`,
        sectionId: block.id,
        ...(choice ? { fieldId: choice.fieldId } : {}),
        kind,
        origin: missingSection
          ? 'template_default_value'
          : explicitNormalEvidence || choice
            ? 'transcript_switch'
            : 'template_literal',
        text: node.text,
        evidence,
        confirmationRequired: explicitNormalEvidence ? false : true,
        ...(missingSection ? { defaultKind: 'clinical_default' as const } : {}),
      });
      return;
    }
    if (node.kind === 'slot') {
      const id = fieldId(block, node.slot);
      const stored = state.slots.get(id);
      if (stored && !missingSection) {
        result.push({
          id: `${block.id}:field:${id}`,
          sectionId: block.id,
          fieldId: id,
          kind: 'transcript_value',
          origin: 'transcript_slot',
          text: formatSlot(stored.values, node.slot),
          evidence: stored.evidence,
          confirmationRequired: false,
          ...(node.slot.unit ? { unit: node.slot.unit } : {}),
        });
      } else {
        if (
          node.slot.requiredForApproval === true
          && !state.issues.some((issue) => (
            (
              issue.code === 'required_placeholder'
              || issue.code === 'unresolved_template_placeholder'
            )
            && issue.fieldId === id
          ))
        ) {
          pushIssue(state, {
            code: missingSection
              ? 'unresolved_template_placeholder'
              : 'required_placeholder',
            severity: missingSection ? 'warning' : 'critical',
            message: missingSection
              ? `Template field ${id} remains an unconfirmed placeholder in an unmentioned section.`
              : `Required template field ${id} has not been provided for the addressed section.`,
            sectionId: block.id,
            fieldId: id,
          });
        }
        if (
          node.slot.requiredForApproval === true
          && !state.assignments.some((assignment) => assignment.fieldId === id)
        ) {
          state.assignments.push({
            id: `template:${id}:incomplete`,
            fieldId: id,
            sectionId: block.id,
            kind: 'slot',
            status: 'incomplete',
            value: null,
            canonicalUnit: node.slot.unit ?? null,
            unitSource: null,
            ruleId: 'required-template-field-v1',
            evidence: [],
          });
        }
        result.push({
          id: `${block.id}:field:${id}:default`,
          sectionId: block.id,
          fieldId: id,
          kind: 'template_default',
          origin: 'template_default_value',
          text: node.slot.default,
          evidence: [],
          confirmationRequired: true,
          defaultKind: defaultKind(node.slot),
          ...(node.slot.unit ? { unit: node.slot.unit } : {}),
        });
      }
      return;
    }
    if (node.kind === 'derived') {
      const id = node.fieldId ?? `${block.id}.${node.name}`;
      const dependencies = (node.dependsOn ?? [])
        .map((dependency) => state.slots.get(dependency))
        .filter((stored): stored is StoredSlot => Boolean(stored));
      const computedValue = node.compute(legacySlots);
      const numericComputedValue = Number(computedValue.replace(',', '.'));
      const value = (
        node.outputDivisor !== undefined
        && Number.isFinite(numericComputedValue)
        && node.outputDivisor > 0
      )
        ? String(Math.round(numericComputedValue / node.outputDivisor))
        : computedValue;
      const derived = !missingSection
        && dependencies.length === (node.dependsOn?.length ?? 0)
        && dependencies.length > 0
        && !/_/u.test(value);
      result.push({
        id: `${block.id}:field:${id}:${derived ? 'derived' : 'default'}`,
        sectionId: block.id,
        fieldId: id,
        kind: derived ? 'derived' : 'template_default',
        origin: derived
          ? 'derived_from_transcript'
          : 'template_default_value',
        text: value,
        evidence: derived
          ? dependencies.flatMap((stored) => stored.evidence)
          : [],
        confirmationRequired: !derived,
        ...(!derived ? { defaultKind: 'placeholder' as const } : {}),
        ...(node.unit ? { unit: node.unit } : {}),
      });
      return;
    }

    const switchId = switchFieldId(block, node.sw.name, node.sw.fieldId);
    const selected = state.switches.get(switchId);
    const optionId = selected?.optionId ?? node.sw.default;
    const option = node.sw.options.find((candidate) => candidate.id === optionId)
      ?? node.sw.options.find((candidate) => candidate.id === node.sw.default);
    if (!option) return;
    result.push(...renderNodes(
      block,
      option.nodes,
      `${nodePath}.switch.${option.id}`,
      state,
      legacySlots,
      missingSection,
      explicitNormalEvidence,
      selected
        ? { fieldId: switchId, evidence: selected.evidence }
        : choice,
    ));
  });
  return result;
}
