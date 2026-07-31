import {
  ARRANGER_PROMPT_VERSION,
  arrangeDictation,
  type ArrangeIssue,
  type ArrangeResult,
} from './arranger.js';
import { createDocEngine, getDocTemplate } from './doc-registry.js';
import { verifyNumbers, type NumberCheck } from './number-check.js';
import type { LLMCall } from './ollama.js';
import {
  applySpanAssignments,
  sectionizeWithProvenance,
  type AssignmentMethod,
  type EvidenceSpan,
  type SpanAssignment,
  type TranscriptAtom,
} from './sectionize.js';
import { verifyRadiologySafety, type RadiologySafetyReport } from './safety.js';
import {
  composeTemplateReviewDraft,
  type TemplateFieldAssignment,
  type TemplateNormalizationAlignmentSpan,
  type TemplateReviewDraft,
  type TemplateSectionAtom,
} from './template-composer.js';

export interface DictationBlock {
  id: string;
  label: string;
  text: string;
  source: 'dictated' | 'normal' | 'conclusion';
  evidence: EvidenceSpan[];
  provenanceStatus: 'linked' | 'template' | 'unverified';
  normalReason?: 'missing' | 'explicit';
  origin?: 'transcript' | 'template_default' | 'generated_extract';
  assignmentMethod?: AssignmentMethod | 'mixed';
}

export interface TemplateDefault {
  sectionId: string;
  label: string;
  text: string;
}

export interface DictationProvenance {
  sections: Record<string, EvidenceSpan[]>;
  unmatched: EvidenceSpan[];
}

export interface DictationRouting {
  atoms: TranscriptAtom[];
  assignments: SpanAssignment[];
  unmatchedAtomIds: string[];
  dictatedConclusion?: EvidenceSpan;
}

export interface StructuringRun {
  routerVersion: 'radiology-span-router-v2';
  promptVersion: string;
  llmAllowed: boolean;
  llmCalled: boolean;
  llmValid: boolean;
  llmInputSha256: string | null;
  llmResponseSha256: string | null;
  issues: ArrangeIssue[];
}

export interface DictationReport {
  title: string;
  blocks: DictationBlock[];
  /** Legacy display field. Missing-section defaults are intentionally omitted. */
  fullText: string;
  /** Only speech-backed section text (plus deterministic labels), without template defaults. */
  evidenceBackedText: string;
  templateDefaults: TemplateDefault[];
  routing: DictationRouting;
  structuringRun: StructuringRun;
  unmatched: string;
  unmatchedSpans: EvidenceSpan[];
  generateConclusion: boolean;
  numberCheck: NumberCheck;
  safety: RadiologySafetyReport;
  provenance: DictationProvenance;
  /**
   * Optional additive v2 fields. Existing persisted v2 artifacts remain
   * readable without them; newly composed reports expose the immutable,
   * physician-reviewable template draft without changing evidenceBackedText.
   */
  fieldAssignments?: TemplateFieldAssignment[];
  reviewDraft?: TemplateReviewDraft;
}

export interface StructureDictationOptions {
  /**
   * false is used when an earlier integrity gate is incomplete/ambiguous.
   * Unresolved atoms remain unmatched; the LLM is not called.
   */
  allowLLM?: boolean;
  rawTranscript?: string;
  normalizationAlignment?: TemplateNormalizationAlignmentSpan[];
}

const explicitNormal = (value: string): boolean => {
  const text = value.trim().toLowerCase().replace(/ё/g, 'е');
  return /(^|[^а-я])норм(?:а|е)(?=$|[^а-я])/u.test(text)
    || /без\s+особенност/u.test(text);
};

function assignmentMethod(methods: AssignmentMethod[]): AssignmentMethod | 'mixed' | undefined {
  const unique = [...new Set(methods)];
  if (unique.length === 0) return undefined;
  return unique.length === 1 ? unique[0] : 'mixed';
}

const PATHOLOGY_MARKERS = [
  /метаст/iu,
  /кист/iu,
  /босняк/iu,
  /липомат/iu,
  /стеатоз/iu,
  /цирроз/iu,
  /гепатомегал|спленомегал|лимфаденопат/iu,
  /стеноз|компресс|деформац/iu,
  /расширен|увеличен|утолщен|утолщение/iu,
  /неровн|бугрист/iu,
  /дегенератив|дистрофич/iu,
  /шунт|коллатерал/iu,
  /кальцин|атеросклер|тромб|аневризм/iu,
  /конкремент|камн|перфорац/iu,
  /асцит|инфильтрат|воспален|от[её]к/iu,
  /ишеми|дилатац|гипертенз/iu,
  /свободн\w*\s+(?:жидкост|газ)/iu,
  /образован|очаг/iu,
] as const;

const NEGATED_PATHOLOGY = [
  /(?:не\s+(?:выявлен|определя|обнаруж|визуализ)|отсутств)\w*[\s\S]{0,40}(?:образован|очаг|метаст|жидкост|газ|стеноз|компресс|ишеми|атеросклер|коллатерал|тромб|аневризм)/iu,
  /(?:образован|очаг|метаст|жидкост|газ|стеноз|компресс|ишеми|атеросклер|коллатерал|тромб|аневризм)[\s\S]{0,40}(?:не\s+(?:выявлен|определя|обнаруж|визуализ)|отсутств)/iu,
  /не\s+(?:увеличен|расширен|утолщен|деформирован)/iu,
  /без\s*(?:очаг|патолог|особенност)/iu,
  /(?:жидкост|газ)\w*\s+нет/iu,
] as const;

/**
 * Conservative deterministic filter for the optional automatic conclusion.
 * False negatives are preferable here: a doctor can dictate a conclusion,
 * while a normal/negated atom must never be promoted as a pathological claim.
 */
function hasConfirmedPathology(text: string): boolean {
  for (const marker of PATHOLOGY_MARKERS) {
    const flags = marker.flags.includes('g') ? marker.flags : `${marker.flags}g`;
    const matches = text.matchAll(new RegExp(marker.source, flags));
    for (const match of matches) {
      const start = match.index ?? 0;
      const window = text.slice(
        Math.max(0, start - 48),
        Math.min(text.length, start + match[0].length + 48),
      );
      if (!NEGATED_PATHOLOGY.some((pattern) => pattern.test(window))) return true;
    }
  }
  return false;
}

function conclusionFromVerifiedBlocks(blocks: DictationBlock[]): {
  text: string;
  evidence: EvidenceSpan[];
} {
  const verified = blocks.filter((block) => (
    block.id !== 'conclusion'
    && block.origin === 'transcript'
    && block.provenanceStatus === 'linked'
    && block.text.trim().length > 0
    && block.source !== 'normal'
    && hasConfirmedPathology(block.text)
  ));
  return {
    // Extractive-only: every clinical character comes from an evidence span.
    text: verified.map((block) => block.text).join(' '),
    evidence: verified.flatMap((block) => block.evidence),
  };
}

function bodyText(rendered: string): string {
  return rendered.replace(/^[^:]+:\s*/u, '');
}

function addMissingProvenanceIssue(
  safety: RadiologySafetyReport,
  atom: TranscriptAtom,
): void {
  safety.issues.push({
    code: 'missing_provenance',
    severity: 'critical',
    message: `Фрагмент ${atom.id} не назначен ни одной секции: «${atom.text}».`,
  });
  safety.ok = false;
  safety.requiresReview = true;
}

export async function structureDictation(
  templateId: string,
  transcript: string,
  llm: LLMCall,
  options: StructureDictationOptions = {},
): Promise<DictationReport> {
  const tpl = getDocTemplate(templateId);
  if (!tpl) throw new Error(`Неизвестный шаблон: ${templateId}`);

  const deterministic = sectionizeWithProvenance(tpl, transcript);
  const deterministicByAtom = new Map(
    deterministic.assignments.map((assignment) => [assignment.atomId, assignment]),
  );
  const unresolved = deterministic.atoms.filter((atom) => (
    deterministicByAtom.get(atom.id)?.sectionId === null
  ));
  const llmAllowed = options.allowLLM !== false;
  const arranged: ArrangeResult = llmAllowed
    ? await arrangeDictation(tpl, unresolved, llm)
    : {
      assignments: unresolved.map((atom) => ({
        atomId: atom.id,
        sectionId: null,
        method: 'unmatched',
      })),
      unmatchedAtomIds: unresolved.map((atom) => atom.id),
      called: false,
      valid: true,
      promptVersion: ARRANGER_PROMPT_VERSION,
      inputSha256: null,
      responseSha256: null,
      issues: [],
    };
  const arrangedByAtom = new Map(
    arranged.assignments.map((assignment) => [assignment.atomId, assignment]),
  );
  const combinedAssignments = deterministic.atoms.map((atom) => {
    const fixed = deterministicByAtom.get(atom.id);
    if (fixed?.sectionId !== null) return fixed!;
    return arrangedByAtom.get(atom.id) ?? {
      atomId: atom.id,
      sectionId: null,
      method: 'unmatched' as const,
    };
  });
  const routed = applySpanAssignments(deterministic, combinedAssignments);
  const sectionByAtom = new Map(
    routed.assignments.map((assignment) => [assignment.atomId, assignment.sectionId]),
  );
  const compositionAtoms: TemplateSectionAtom[] = routed.atoms.flatMap((atom) => {
    const sectionId = sectionByAtom.get(atom.id);
    if (!sectionId) return [];
    return [{
      atomId: atom.id,
      sectionId,
      start: atom.start,
      end: atom.end,
      text: atom.text,
    }];
  });
  if (routed.dictatedConclusion && tpl.conclusionBlockId) {
    compositionAtoms.push({
      atomId: 'dictated-conclusion',
      sectionId: tpl.conclusionBlockId,
      start: routed.dictatedConclusion.start,
      end: routed.dictatedConclusion.end,
      text: routed.dictatedConclusion.text,
    });
  }
  const reviewDraft = templateId === 'CT_ABDOMEN_MIKHAILOV'
    ? composeTemplateReviewDraft(tpl, transcript, compositionAtoms, {
        ...(options.rawTranscript !== undefined
          ? { rawTranscript: options.rawTranscript }
          : {}),
        ...(options.normalizationAlignment
          ? { alignment: options.normalizationAlignment }
          : {}),
      })
    : undefined;

  const renderedDefaults = createDocEngine(templateId).build().blocks;
  const defaultById = new Map(renderedDefaults.map((block) => [
    block.id,
    bodyText(block.text),
  ]));
  const blockById = new Map(tpl.blocks.map((block) => [block.id, block]));
  const blocks: DictationBlock[] = [];
  const templateDefaults: TemplateDefault[] = [];
  const provenanceSections: Record<string, EvidenceSpan[]> = {};
  const safetyParts: string[] = [];

  for (const rendered of renderedDefaults) {
    const definition = blockById.get(rendered.id);
    const label = definition?.label ?? rendered.label;
    if (rendered.id === tpl.conclusionBlockId) continue;
    const routedSection = routed.sections[rendered.id];
    if (routedSection) {
      const said = routedSection.spans.map((span) => span.text).join(' ');
      const normal = explicitNormal(said);
      const block: DictationBlock = {
        id: rendered.id,
        label,
        text: said,
        source: normal ? 'normal' : 'dictated',
        evidence: routedSection.spans,
        provenanceStatus: 'linked',
        normalReason: normal ? 'explicit' : undefined,
        origin: 'transcript',
        assignmentMethod: assignmentMethod(routedSection.assignmentMethods),
      };
      blocks.push(block);
      provenanceSections[rendered.id] = routedSection.spans;
      safetyParts.push(said);
      continue;
    }

    const defaultText = defaultById.get(rendered.id) ?? '';
    blocks.push({
      id: rendered.id,
      label,
      text: defaultText,
      source: 'normal',
      evidence: [],
      provenanceStatus: 'template',
      normalReason: 'missing',
      origin: 'template_default',
    });
    templateDefaults.push({
      sectionId: rendered.id,
      label,
      text: defaultText,
    });
    provenanceSections[rendered.id] = [];
  }

  const conclusionDefinition = tpl.conclusionBlockId
    ? blockById.get(tpl.conclusionBlockId)
    : undefined;
  if (conclusionDefinition) {
    let conclusion: DictationBlock;
    if (routed.dictatedConclusion) {
      conclusion = {
        id: conclusionDefinition.id,
        label: conclusionDefinition.label,
        text: routed.dictatedConclusion.text,
        source: 'conclusion',
        evidence: [routed.dictatedConclusion],
        provenanceStatus: 'linked',
        origin: 'transcript',
        assignmentMethod: 'anchor',
      };
      safetyParts.push(routed.dictatedConclusion.text);
    } else if (routed.generateConclusion) {
      const generated = conclusionFromVerifiedBlocks(blocks);
      conclusion = {
        id: conclusionDefinition.id,
        label: conclusionDefinition.label,
        text: generated.text,
        source: 'conclusion',
        evidence: generated.evidence,
        provenanceStatus: generated.evidence.length ? 'linked' : 'unverified',
        origin: 'generated_extract',
      };
    } else {
      conclusion = {
        id: conclusionDefinition.id,
        label: conclusionDefinition.label,
        text: '',
        source: 'conclusion',
        evidence: [],
        provenanceStatus: 'unverified',
        origin: 'generated_extract',
      };
    }
    blocks.push(conclusion);
    provenanceSections[conclusion.id] = conclusion.evidence;
  }

  if (routed.unmatchedText) safetyParts.push(routed.unmatchedText);
  const safetyComparableText = safetyParts.join(' ');
  const numberCheck = verifyNumbers(transcript, safetyComparableText);
  const safety = verifyRadiologySafety(transcript, safetyComparableText);
  for (const atomId of routed.unmatchedAtomIds) {
    const atom = routed.atoms.find((candidate) => candidate.id === atomId);
    if (atom) addMissingProvenanceIssue(safety, atom);
  }

  const evidenceBlocks = blocks.filter((block) => (
    block.origin !== 'template_default' && block.text.trim().length > 0
  ));
  const evidenceBackedText = evidenceBlocks
    .map((block) => `${block.label}: ${block.text}`)
    .join('\n');
  const fullText = evidenceBackedText
    ? `${tpl.title}\n${evidenceBackedText}`
    : tpl.title;

  return {
    title: tpl.title,
    blocks,
    fullText,
    evidenceBackedText,
    templateDefaults,
    routing: {
      atoms: routed.atoms,
      assignments: routed.assignments,
      unmatchedAtomIds: routed.unmatchedAtomIds,
      dictatedConclusion: routed.dictatedConclusion,
    },
    structuringRun: {
      routerVersion: 'radiology-span-router-v2',
      promptVersion: arranged.promptVersion,
      llmAllowed,
      llmCalled: arranged.called,
      llmValid: arranged.valid,
      llmInputSha256: arranged.inputSha256,
      llmResponseSha256: arranged.responseSha256,
      issues: arranged.issues,
    },
    unmatched: routed.unmatchedText,
    unmatchedSpans: routed.unmatched,
    generateConclusion: routed.generateConclusion,
    numberCheck,
    safety,
    provenance: {
      sections: provenanceSections,
      unmatched: routed.unmatched,
    },
    ...(reviewDraft
      ? {
          fieldAssignments: reviewDraft.fieldAssignments,
          reviewDraft,
        }
      : {}),
  };
}
