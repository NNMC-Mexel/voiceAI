import type { BlockNode, DocTemplate, SlotDef } from './doc-model.js';

export type AssignmentMethod = 'anchor' | 'rule' | 'llm' | 'unmatched';

export interface TranscriptAtom {
  id: string;
  start: number;
  end: number;
  text: string;
  candidateSectionIds: string[];
  anchorRuleIds: string[];
}

export interface SpanAssignment {
  atomId: string;
  sectionId: string | null;
  method: AssignmentMethod;
}

export interface Segment {
  blockId: string | null;
  text: string;
}

/** Точный фрагмент исходного транскрипта. Индексы соответствуют JS slice(start, end). */
export interface EvidenceSpan {
  start: number;
  end: number;
  text: string;
  source: 'transcript';
}

export interface AnchorSpan extends EvidenceSpan {
  blockId: string;
  anchor: string;
  ruleId: string;
  method: 'anchor' | 'rule';
}

export interface ProvenancedSegment extends Segment {
  atomId: string;
  span: EvidenceSpan;
  assignmentMethod: AssignmentMethod;
  anchor?: AnchorSpan;
}

export interface SectionizedBlock {
  blockId: string;
  text: string;
  spans: EvidenceSpan[];
  anchors: AnchorSpan[];
  atomIds: string[];
  assignmentMethods: AssignmentMethod[];
}

export interface SectionizedTranscript {
  transcript: string;
  atoms: TranscriptAtom[];
  assignments: SpanAssignment[];
  unmatchedAtomIds: string[];
  segments: ProvenancedSegment[];
  sections: Record<string, SectionizedBlock>;
  unmatched: EvidenceSpan[];
  unmatchedText: string;
  dictatedConclusion?: EvidenceSpan;
  generateConclusion: boolean;
}

interface Hit {
  pos: number;
  end: number;
  blockId: string;
  label: string;
  ruleId: string;
  method: 'anchor' | 'rule';
  len: number;
  blockOrder: number;
  sticky: boolean;
}

interface Range {
  start: number;
  end: number;
}

const norm = (value: string): string => value.toLowerCase().replace(/ё/g, 'е');
const esc = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function phrasePattern(value: string): string {
  const tokens = norm(value).trim().split(/\s+/u).filter(Boolean);
  return tokens.map((token) => (
    token.length <= 2 ? esc(token) : `${esc(token)}[a-zа-я]*`
  )).join('\\s+');
}

function trimSpan(transcript: string, start: number, end: number): EvidenceSpan | undefined {
  while (start < end && /\s/u.test(transcript[start])) start++;
  while (end > start && /\s/u.test(transcript[end - 1])) end--;
  if (start >= end) return undefined;
  return {
    start,
    end,
    text: transcript.slice(start, end),
    source: 'transcript',
  };
}

function findRanges(transcript: string, expression: RegExp): Range[] {
  const ranges: Range[] = [];
  const normalized = norm(transcript);
  expression.lastIndex = 0;
  for (let match = expression.exec(normalized); match; match = expression.exec(normalized)) {
    const leading = match[1]?.length ?? 0;
    ranges.push({
      start: match.index + leading,
      end: match.index + match[0].length,
    });
  }
  return ranges;
}

const CONCLUSION_CONTROL_RE = /(^|[^а-яё])(?:с?формиру(?:й|ю)|собери|сделай)\s+заключени[ея](?=$|[^а-яё])[.!]?/giu;
const CONCLUSION_WORD_RE = /(^|[^а-яё])заключени[ея](?=$|[^а-яё])/giu;

function conclusionParts(transcript: string): {
  bodyEnd: number;
  controlRanges: Range[];
  dictatedConclusion?: EvidenceSpan;
  generateConclusion: boolean;
} {
  const controlRanges = findRanges(transcript, CONCLUSION_CONTROL_RE);
  const conclusionWords = findRanges(transcript, CONCLUSION_WORD_RE);
  const explicit = conclusionWords.find((word) => (
    !controlRanges.some((control) => word.start >= control.start && word.end <= control.end)
  ));
  const dictatedConclusion = explicit
    ? trimSpan(transcript, explicit.start, transcript.length)
    : undefined;
  return {
    bodyEnd: explicit?.start ?? transcript.length,
    controlRanges: controlRanges.filter((range) => range.start < (explicit?.start ?? transcript.length)),
    dictatedConclusion,
    generateConclusion: controlRanges.length > 0,
  };
}

function hitForPhrase(
  transcript: string,
  phrase: string,
  blockId: string,
  blockOrder: number,
  method: 'anchor' | 'rule',
  ruleId: string,
  sticky: boolean,
): Hit[] {
  const text = norm(transcript);
  const body = phrasePattern(phrase);
  if (!body) return [];
  const expression = new RegExp(`(^|[^a-zа-я])(${body})(?=$|[^a-zа-я])`, 'gu');
  const hits: Hit[] = [];
  for (let match = expression.exec(text); match; match = expression.exec(text)) {
    const pos = match.index + match[1].length;
    hits.push({
      pos,
      end: pos + match[2].length,
      blockId,
      label: phrase,
      ruleId,
      method,
      len: match[2].length,
      blockOrder,
      sticky,
    });
  }
  return hits;
}

function exactPhrasePattern(value: string): string {
  return norm(value)
    .trim()
    .split(/\s+/u)
    .filter(Boolean)
    .map(esc)
    .join('\\s+');
}

function hitForExactFieldAlias(
  transcript: string,
  alias: string,
  blockId: string,
  blockOrder: number,
  fieldId: string,
  routingVersion: string,
): Hit[] {
  const text = norm(transcript);
  const body = exactPhrasePattern(alias);
  if (!body) return [];
  const expression = new RegExp(`(^|[^a-zа-я])(${body})(?=$|[^a-zа-я])`, 'gu');
  const hits: Hit[] = [];
  for (let match = expression.exec(text); match; match = expression.exec(text)) {
    const pos = match.index + match[1].length;
    hits.push({
      pos,
      end: pos + match[2].length,
      blockId,
      label: alias,
      ruleId: `field-alias:${routingVersion}:${fieldId}:${norm(alias)}`,
      method: 'rule',
      len: match[2].length,
      blockOrder,
      sticky: false,
    });
  }
  return hits;
}

interface FieldRoutingAlias {
  alias: string;
  blockId: string;
  blockOrder: number;
  fieldId: string;
}

function slotsFromNodes(nodes: BlockNode[]): SlotDef[] {
  const slots: SlotDef[] = [];
  for (const node of nodes) {
    if (node.kind === 'slot') {
      slots.push(node.slot);
      continue;
    }
    if (node.kind === 'switch') {
      for (const option of node.sw.options) {
        slots.push(...slotsFromNodes(option.nodes));
      }
    }
  }
  return slots;
}

/**
 * Builds the routing table from the versioned template schema. An alias is
 * usable only when it names one stable field in one section; duplicates are
 * rejected rather than resolved by template order.
 */
function uniqueFieldRoutingAliases(tpl: DocTemplate): FieldRoutingAlias[] {
  if (!tpl.fieldRoutingVersion) return [];
  const byAlias = new Map<string, FieldRoutingAlias[]>();
  for (let blockOrder = 0; blockOrder < tpl.blocks.length; blockOrder++) {
    const block = tpl.blocks[blockOrder];
    if (block.id === tpl.conclusionBlockId) continue;
    for (const slot of slotsFromNodes(block.nodes)) {
      if (!slot.fieldId) continue;
      for (const configuredAlias of slot.routingAliases ?? []) {
        const alias = norm(configuredAlias).trim().replace(/\s+/gu, ' ');
        if (!alias) continue;
        const records = byAlias.get(alias) ?? [];
        records.push({
          alias,
          blockId: block.id,
          blockOrder,
          fieldId: slot.fieldId,
        });
        byAlias.set(alias, records);
      }
    }
  }
  return [...byAlias.values()].flatMap((records) => {
    const identities = new Set(records.map((record) => (
      `${record.blockId}\u0000${record.fieldId}`
    )));
    return identities.size === 1 ? [records[0]] : [];
  });
}

function detectHits(tpl: DocTemplate, transcript: string, bodyEnd: number): Hit[] {
  const body = transcript.slice(0, bodyEnd);
  const hits: Hit[] = [];
  for (const fieldAlias of uniqueFieldRoutingAliases(tpl)) {
    hits.push(...hitForExactFieldAlias(
      body,
      fieldAlias.alias,
      fieldAlias.blockId,
      fieldAlias.blockOrder,
      fieldAlias.fieldId,
      tpl.fieldRoutingVersion!,
    ));
  }
  for (let blockOrder = 0; blockOrder < tpl.blocks.length; blockOrder++) {
    const block = tpl.blocks[blockOrder];
    if (block.id === tpl.conclusionBlockId) continue;
    for (const anchor of block.anchors) {
      hits.push(...hitForPhrase(
        body,
        anchor,
        block.id,
        blockOrder,
        'anchor',
        `anchor:${block.id}:${norm(anchor)}`,
        false,
      ));
    }
    for (const rule of block.routingRules ?? []) {
      for (const phrase of rule.phrases) {
        hits.push(...hitForPhrase(
          body,
          phrase,
          block.id,
          blockOrder,
          'rule',
          `rule:${block.id}:${rule.id}`,
          rule.sticky === true,
        ));
      }
    }
  }

  hits.sort((left, right) => (
    left.pos - right.pos
    || Number(right.sticky) - Number(left.sticky)
    || right.end - left.end
    || Number(right.method === 'rule') - Number(left.method === 'rule')
    || right.len - left.len
    || left.blockOrder - right.blockOrder
  ));

  const nonOverlapping: Hit[] = [];
  for (const hit of hits) {
    const previous = nonOverlapping[nonOverlapping.length - 1];
    if (previous && hit.pos < previous.end) continue;
    nonOverlapping.push(hit);
  }

  // A sticky rule denotes one continuous clinical finding (for example the
  // celiac-trunk compression description). Nested organ names are objects of
  // that finding, not new section headings.
  const sticky = nonOverlapping.find((hit) => hit.sticky);
  if (!sticky) return nonOverlapping;
  return nonOverlapping.filter((hit) => (
    hit.pos < sticky.pos || hit.blockId === sticky.blockId
  ));
}

function subtractRanges(
  transcript: string,
  start: number,
  end: number,
  excluded: Range[],
): EvidenceSpan[] {
  const spans: EvidenceSpan[] = [];
  let cursor = start;
  for (const range of excluded) {
    if (range.end <= cursor || range.start >= end) continue;
    const before = trimSpan(transcript, cursor, Math.min(range.start, end));
    if (before) spans.push(before);
    cursor = Math.max(cursor, range.end);
    if (cursor >= end) break;
  }
  const tail = trimSpan(transcript, cursor, end);
  if (tail) spans.push(tail);
  return spans;
}

function containsRenalHilumPhrase(value: string): boolean {
  const text = norm(value);
  return /ворот[а-я]*\s+почек/u.test(text)
    || /почечн[а-я]*\s+ворот[а-я]*/u.test(text)
    || /лимф(?:атическ[а-я]*\s+узл[а-я]*|оузл[а-я]*)/u.test(text);
}

function unresolvedCandidates(tpl: DocTemplate, text: string): string[] {
  const ids = tpl.blocks
    .filter((block) => block.id !== tpl.conclusionBlockId)
    .map((block) => block.id);
  if (
    ids.includes('lymph_hilum')
    && /(^|[^а-я])ворот[а-я]*(?=$|[^а-я])/u.test(norm(text))
    && !containsRenalHilumPhrase(text)
  ) {
    return ids.filter((id) => id !== 'lymph_hilum');
  }
  return ids;
}

function collapseBounds(hits: Hit[]): Hit[] {
  const bounds: Hit[] = [];
  for (const hit of hits) {
    const last = bounds[bounds.length - 1];
    if (last && last.pos === hit.pos) continue;
    if (last && last.blockId === hit.blockId) continue;
    bounds.push(hit);
  }
  return bounds;
}

interface AtomSeed {
  span: EvidenceSpan;
  hit?: Hit;
}

function atomSeeds(
  transcript: string,
  bodyEnd: number,
  bounds: Hit[],
  excluded: Range[],
): AtomSeed[] {
  const seeds: AtomSeed[] = [];
  if (bounds.length === 0) {
    return subtractRanges(transcript, 0, bodyEnd, excluded).map((span) => ({ span }));
  }
  for (const span of subtractRanges(transcript, 0, bounds[0].pos, excluded)) {
    seeds.push({ span });
  }
  for (let index = 0; index < bounds.length; index++) {
    const from = bounds[index].pos;
    const to = index + 1 < bounds.length ? bounds[index + 1].pos : bodyEnd;
    for (const span of subtractRanges(transcript, from, to, excluded)) {
      seeds.push({ span, hit: bounds[index] });
    }
  }
  return seeds.sort((left, right) => left.span.start - right.span.start);
}

function evidenceForAtom(atom: TranscriptAtom): EvidenceSpan {
  return {
    start: atom.start,
    end: atom.end,
    text: atom.text,
    source: 'transcript',
  };
}

export function applySpanAssignments(
  base: Pick<
    SectionizedTranscript,
    'transcript' | 'atoms' | 'dictatedConclusion' | 'generateConclusion'
  >,
  requestedAssignments: SpanAssignment[],
): SectionizedTranscript {
  const atomById = new Map(base.atoms.map((atom) => [atom.id, atom]));
  const assignmentLists = new Map<string, SpanAssignment[]>();
  for (const assignment of requestedAssignments) {
    if (!atomById.has(assignment.atomId)) continue;
    const list = assignmentLists.get(assignment.atomId) ?? [];
    list.push(assignment);
    assignmentLists.set(assignment.atomId, list);
  }

  const assignments: SpanAssignment[] = base.atoms.map((atom) => {
    const list = assignmentLists.get(atom.id) ?? [];
    if (list.length !== 1) {
      return { atomId: atom.id, sectionId: null, method: 'unmatched' };
    }
    const requested = list[0];
    if (
      requested.sectionId === null
      || !atom.candidateSectionIds.includes(requested.sectionId)
      || requested.method === 'unmatched'
    ) {
      return { atomId: atom.id, sectionId: null, method: 'unmatched' };
    }
    return requested;
  });

  const sections: Record<string, SectionizedBlock> = {};
  const unmatched: EvidenceSpan[] = [];
  const segments: ProvenancedSegment[] = [];
  for (let index = 0; index < base.atoms.length; index++) {
    const atom = base.atoms[index];
    const assignment = assignments[index];
    const span = evidenceForAtom(atom);
    const ruleId = atom.anchorRuleIds[0];
    const deterministicMethod: 'anchor' | 'rule' | undefined =
      assignment.method === 'anchor' || assignment.method === 'rule'
        ? assignment.method
        : undefined;
    const anchor: AnchorSpan | undefined = deterministicMethod && assignment.sectionId && ruleId
      ? {
        ...span,
        end: Math.min(
          atom.end,
          atom.start + Math.max(1, atom.text.split(/\s+/u)[0]?.length ?? 1),
        ),
        text: base.transcript.slice(
          atom.start,
          Math.min(
            atom.end,
            atom.start + Math.max(1, atom.text.split(/\s+/u)[0]?.length ?? 1),
          ),
        ),
        blockId: assignment.sectionId,
        anchor: ruleId,
        ruleId,
        method: deterministicMethod,
      }
      : undefined;
    segments.push({
      atomId: atom.id,
      blockId: assignment.sectionId,
      text: atom.text,
      span,
      assignmentMethod: assignment.method,
      anchor,
    });
    if (assignment.sectionId === null) {
      unmatched.push(span);
      continue;
    }
    const section = sections[assignment.sectionId] ??= {
      blockId: assignment.sectionId,
      text: '',
      spans: [],
      anchors: [],
      atomIds: [],
      assignmentMethods: [],
    };
    section.spans.push(span);
    section.atomIds.push(atom.id);
    section.assignmentMethods.push(assignment.method);
    if (anchor) section.anchors.push(anchor);
    section.text = section.spans.map((item) => item.text).join(' ');
  }

  const unmatchedAtomIds = assignments
    .filter((assignment) => assignment.sectionId === null)
    .map((assignment) => assignment.atomId);
  return {
    transcript: base.transcript,
    atoms: base.atoms,
    assignments,
    unmatchedAtomIds,
    segments,
    sections,
    unmatched,
    unmatchedText: unmatched.map((span) => span.text).join(' '),
    dictatedConclusion: base.dictatedConclusion,
    generateConclusion: base.generateConclusion,
  };
}

/**
 * Deterministically splits a normalized transcript into immutable atoms. Every
 * atom gets exactly one owner or the explicit unmatched state.
 */
export function sectionizeWithProvenance(
  tpl: DocTemplate,
  transcript: string,
): SectionizedTranscript {
  const conclusion = conclusionParts(transcript);
  const bounds = collapseBounds(detectHits(tpl, transcript, conclusion.bodyEnd));
  const seeds = atomSeeds(
    transcript,
    conclusion.bodyEnd,
    bounds,
    conclusion.controlRanges,
  );
  const atoms: TranscriptAtom[] = seeds.map((seed, index) => {
    const deterministicSection = seed.hit?.blockId;
    return {
      id: `a${String(index + 1).padStart(4, '0')}`,
      start: seed.span.start,
      end: seed.span.end,
      text: seed.span.text,
      candidateSectionIds: deterministicSection
        ? [deterministicSection]
        : unresolvedCandidates(tpl, seed.span.text),
      anchorRuleIds: seed.hit ? [seed.hit.ruleId] : [],
    };
  });
  const deterministicAssignments: SpanAssignment[] = atoms.map((atom, index) => {
    const hit = seeds[index].hit;
    return {
      atomId: atom.id,
      sectionId: hit?.blockId ?? null,
      method: hit?.method ?? 'unmatched',
    };
  });
  return applySpanAssignments({
    transcript,
    atoms,
    dictatedConclusion: conclusion.dictatedConclusion,
    generateConclusion: conclusion.generateConclusion,
  }, deterministicAssignments);
}

/** Старый компактный контракт сохранён для существующих потребителей. */
export function sectionize(tpl: DocTemplate, transcript: string): Segment[] {
  return sectionizeWithProvenance(tpl, transcript).segments.map(({ blockId, text }) => ({
    blockId,
    text,
  }));
}
