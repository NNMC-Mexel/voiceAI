import { createHash } from 'node:crypto';
import type { DocTemplate } from './doc-model.js';
import type { LLMCall } from './ollama.js';
import type { SpanAssignment, TranscriptAtom } from './sectionize.js';

export const ARRANGER_PROMPT_VERSION = 'radiology-span-router-v2';

export interface ArrangeIssue {
  code:
    | 'invalid_json'
    | 'invalid_response_shape'
    | 'unknown_atom'
    | 'duplicate_atom'
    | 'missing_atom'
    | 'forbidden_section';
  message: string;
  atomId?: string;
}

export interface ArrangeResult {
  assignments: SpanAssignment[];
  unmatchedAtomIds: string[];
  called: boolean;
  valid: boolean;
  promptVersion: string;
  inputSha256: string | null;
  responseSha256: string | null;
  issues: ArrangeIssue[];
}

interface AssignmentResponse {
  assignments: Array<{
    atomId: string;
    sectionId: string | null;
  }>;
}

interface PreparedPrompt {
  system: string;
  user: string;
  atomIds: string[];
}

const callCache = new WeakMap<LLMCall, Map<string, Promise<string>>>();

function sha256(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function clinicalBlocks(tpl: DocTemplate): Set<string> {
  return new Set(
    tpl.blocks
      .filter((block) => block.id !== tpl.conclusionBlockId)
      .map((block) => block.id),
  );
}

/**
 * The model is a constrained classifier. It may choose only from the supplied
 * IDs; no clinical text is accepted back from it.
 */
export function buildArrangerPrompt(
  tpl: DocTemplate,
  atoms: TranscriptAtom[],
): PreparedPrompt {
  const sectionLabels = new Map(
    tpl.blocks
      .filter((block) => block.id !== tpl.conclusionBlockId)
      .map((block) => [block.id, block.label]),
  );
  const atomIds = atoms.map((atom) => atom.id);
  const input = atoms.map((atom) => ({
    atomId: atom.id,
    text: atom.text,
    candidates: atom.candidateSectionIds.map((id) => ({
      sectionId: id,
      label: sectionLabels.get(id) ?? id,
    })),
  }));
  const system = `Ты классифицируешь уже выделенные фрагменты диктовки рентгенолога.
Верни только JSON строго такого вида:
{"assignments":[{"atomId":"a0001","sectionId":"liver"}]}

Правила:
- Для каждого входного atomId верни ровно одну запись.
- atomId копируй без изменений.
- sectionId выбирай только из candidates этого atom.
- Если уверенно выбрать секцию нельзя, верни sectionId:null.
- Не возвращай текст диктовки, исправления, комментарии или дополнительные поля.
- Не меняй медицинские термины, числа, единицы, отрицания и латеральность.
- Порядок секций шаблона не задаёт порядок диктовки.

Версия контракта: ${ARRANGER_PROMPT_VERSION}.`;
  const user = JSON.stringify({ atoms: input });
  return { system, user, atomIds };
}

function extractJson(raw: string): unknown {
  const end = raw.lastIndexOf('}');
  if (end < 0) throw new Error('LLM не вернул JSON');
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;
  for (let i = end; i >= 0; i--) {
    const ch = raw[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === '\\') {
      escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === '}') depth++;
    if (ch === '{') {
      depth--;
      if (depth === 0) {
        start = i;
        break;
      }
    }
  }
  if (start < 0) throw new Error('LLM не вернул JSON');
  return JSON.parse(raw.slice(start, end + 1)) as unknown;
}

function isAssignmentResponse(value: unknown): value is AssignmentResponse {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const obj = value as Record<string, unknown>;
  if (Object.keys(obj).length !== 1 || !Array.isArray(obj.assignments)) return false;
  return obj.assignments.every((item) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return false;
    const assignment = item as Record<string, unknown>;
    const keys = Object.keys(assignment).sort();
    return keys.length === 2
      && keys[0] === 'atomId'
      && keys[1] === 'sectionId'
      && typeof assignment.atomId === 'string'
      && (typeof assignment.sectionId === 'string' || assignment.sectionId === null);
  });
}

async function cachedCall(llm: LLMCall, key: string, system: string, user: string): Promise<string> {
  let cache = callCache.get(llm);
  if (!cache) {
    cache = new Map<string, Promise<string>>();
    callCache.set(llm, cache);
  }
  let result = cache.get(key);
  if (!result) {
    result = llm(system, user);
    cache.set(key, result);
    void result.catch(() => cache?.delete(key));
  }
  return result;
}

function failedResult(
  atoms: TranscriptAtom[],
  called: boolean,
  inputSha256: string | null,
  responseSha256: string | null,
  issues: ArrangeIssue[],
): ArrangeResult {
  return {
    assignments: atoms.map((atom) => ({
      atomId: atom.id,
      sectionId: null,
      method: 'unmatched',
    })),
    unmatchedAtomIds: atoms.map((atom) => atom.id),
    called,
    valid: false,
    promptVersion: ARRANGER_PROMPT_VERSION,
    inputSha256,
    responseSha256,
    issues,
  };
}

export async function arrangeDictation(
  tpl: DocTemplate,
  atoms: TranscriptAtom[],
  llm: LLMCall,
): Promise<ArrangeResult> {
  if (atoms.length === 0) {
    return {
      assignments: [],
      unmatchedAtomIds: [],
      called: false,
      valid: true,
      promptVersion: ARRANGER_PROMPT_VERSION,
      inputSha256: null,
      responseSha256: null,
      issues: [],
    };
  }

  const allowedTemplateSections = clinicalBlocks(tpl);
  const { system, user } = buildArrangerPrompt(tpl, atoms);
  const inputHash = sha256(`${system}\n${user}`);
  let raw: string;
  try {
    raw = await cachedCall(llm, inputHash, system, user);
  } catch (error) {
    return failedResult(atoms, true, inputHash, null, [{
      code: 'invalid_json',
      message: `LLM-классификатор недоступен: ${error instanceof Error ? error.message : String(error)}`,
    }]);
  }

  const responseHash = sha256(raw);
  let parsed: unknown;
  try {
    parsed = extractJson(raw);
  } catch (error) {
    return failedResult(atoms, true, inputHash, responseHash, [{
      code: 'invalid_json',
      message: error instanceof Error ? error.message : 'LLM не вернул JSON',
    }]);
  }
  if (!isAssignmentResponse(parsed)) {
    return failedResult(atoms, true, inputHash, responseHash, [{
      code: 'invalid_response_shape',
      message: 'LLM вернул данные вне закрытого контракта assignments.',
    }]);
  }

  const atomById = new Map(atoms.map((atom) => [atom.id, atom]));
  const seen = new Set<string>();
  const issues: ArrangeIssue[] = [];
  for (const assignment of parsed.assignments) {
    const atom = atomById.get(assignment.atomId);
    if (!atom) {
      issues.push({
        code: 'unknown_atom',
        atomId: assignment.atomId,
        message: `LLM вернул неизвестный atomId: ${assignment.atomId}.`,
      });
      continue;
    }
    if (seen.has(assignment.atomId)) {
      issues.push({
        code: 'duplicate_atom',
        atomId: assignment.atomId,
        message: `LLM повторно назначил atomId: ${assignment.atomId}.`,
      });
      continue;
    }
    seen.add(assignment.atomId);
    if (
      assignment.sectionId !== null
      && (
        !allowedTemplateSections.has(assignment.sectionId)
        || !atom.candidateSectionIds.includes(assignment.sectionId)
      )
    ) {
      issues.push({
        code: 'forbidden_section',
        atomId: assignment.atomId,
        message: `Секция ${assignment.sectionId} не разрешена для ${assignment.atomId}.`,
      });
    }
  }
  for (const atom of atoms) {
    if (!seen.has(atom.id)) {
      issues.push({
        code: 'missing_atom',
        atomId: atom.id,
        message: `LLM не вернул назначение для ${atom.id}.`,
      });
    }
  }

  // Fail closed for the complete classifier response. Mixing validated and
  // invalid records would make retries depend on response order.
  if (issues.length > 0) {
    return failedResult(atoms, true, inputHash, responseHash, issues);
  }

  const assignments: SpanAssignment[] = parsed.assignments.map((assignment) => ({
    atomId: assignment.atomId,
    sectionId: assignment.sectionId,
    method: assignment.sectionId === null ? 'unmatched' : 'llm',
  }));
  return {
    assignments,
    unmatchedAtomIds: assignments
      .filter((assignment) => assignment.sectionId === null)
      .map((assignment) => assignment.atomId),
    called: true,
    valid: true,
    promptVersion: ARRANGER_PROMPT_VERSION,
    inputSha256: inputHash,
    responseSha256: responseHash,
    issues: [],
  };
}
