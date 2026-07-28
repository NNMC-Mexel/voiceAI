// Детерминированные safety-проверки лучевой диктовки.
// Сравнивается только текст, действительно продиктованный врачом, с производным
// текстом structuring-слоя. Шаблонные нормы сюда передавать нельзя: они не
// являются утверждениями из аудио и по определению не имеют speech-provenance.

import {
  findAmbiguousNumberSequences,
  verifyNumbers,
  type AmbiguousNumberSequence,
  type NumberCheck,
} from './number-check.js';
import { convertNumberWords } from './numbers.js';

export type SafetyEntityType =
  | 'number_unit'
  | 'negation'
  | 'laterality'
  | 'contrast'
  | 'critical_fact';

export interface SafetyEntity {
  type: SafetyEntityType;
  normalized: string;
  text: string;
  start: number;
  end: number;
  value?: number;
  unit?: 'mm' | 'cm' | 'hu' | null;
  polarity?: 'positive' | 'negative';
  factId?: string;
}

export interface MatchedSafetyEntity {
  source: SafetyEntity;
  output: SafetyEntity;
}

export interface SafetyEntityCheck {
  matched: MatchedSafetyEntity[];
  addedByOutput: SafetyEntity[];
  lostFromOutput: SafetyEntity[];
  ok: boolean;
}

export type SafetyIssueCode =
  | 'number_added'
  | 'number_lost'
  | 'normalization_number_added'
  | 'normalization_number_lost'
  | 'ambiguous_number_sequence'
  | 'normalization_issue'
  | 'number_or_unit_changed'
  | 'negation_changed'
  | 'laterality_changed'
  | 'contrast_changed'
  | 'clinical_relation_changed'
  | 'unsupported_critical_fact'
  | 'critical_fact_lost'
  | 'missing_provenance'
  | 'longform_degraded'
  | 'overlap_seam_conflict';

export interface SafetyIssue {
  code: SafetyIssueCode;
  severity: 'critical' | 'warning';
  message: string;
  source?: SafetyEntity;
  output?: SafetyEntity;
}

export interface RadiologySafetyReport {
  ok: boolean;
  requiresReview: boolean;
  numbers: NumberCheck;
  numberUnits: SafetyEntityCheck;
  negations: SafetyEntityCheck;
  lateralities: SafetyEntityCheck;
  contrast: SafetyEntityCheck;
  criticalFacts: SafetyEntityCheck;
  unsupportedCriticalFacts: SafetyEntity[];
  issues: SafetyIssue[];
}

export interface NormalizationSafetyEvidence {
  text: string;
  issues?: ReadonlyArray<{
    code: string;
    severity: 'critical' | 'warning';
    message?: string;
    start?: number;
    end?: number;
    sourceText?: string;
    normalizedText?: string;
  }>;
}

export interface NormalizationSafetyStageResult {
  stage: 'raw_to_normalized';
  status: 'passed' | 'failed' | 'incomplete';
  ok: boolean;
  requiresReview: boolean;
  numbers: NumberCheck;
  ambiguities: AmbiguousNumberSequence[];
  issues: SafetyIssue[];
}

interface NumericMention {
  value: number;
  start: number;
  end: number;
  text: string;
}

const LETTER = 'a-zа-яё';

function numericWordValue(word: string): number | undefined {
  const converted = convertNumberWords(word.toLowerCase().replace(/ё/g, 'е'));
  if (!/^\d+$/.test(converted)) return undefined;
  return Number(converted);
}

function extractNumericMentions(text: string): NumericMention[] {
  const result: NumericMention[] = [];

  for (const match of text.matchAll(/[+-]?\d+(?:[.,]\d+)?/g)) {
    const raw = match[0];
    const start = match.index;
    result.push({
      value: Number(raw.replace(',', '.')),
      start,
      end: start + raw.length,
      text: raw,
    });
  }

  const words = [...text.matchAll(/[а-яё]+/gi)];
  for (let i = 0; i < words.length;) {
    if (numericWordValue(words[i][0]) === undefined) {
      i++;
      continue;
    }
    const start = words[i].index;
    let end = start + words[i][0].length;
    let j = i + 1;
    while (j < words.length) {
      const gap = text.slice(end, words[j].index);
      if (!/^[\s-]+$/.test(gap) || numericWordValue(words[j][0]) === undefined) break;
      end = words[j].index + words[j][0].length;
      j++;
    }
    const raw = text.slice(start, end);
    const converted = convertNumberWords(raw);
    if (/^\d+$/.test(converted)) {
      result.push({ value: Number(converted), start, end, text: raw });
    }
    i = j;
  }

  return result.sort((a, b) => a.start - b.start);
}

const UNIT_TOKEN = `(?:мм|миллиметр[${LETTER}]*|см|сантиметр[${LETTER}]*|hu|ху|единиц[${LETTER}]*\\s+хаунсфилд[${LETTER}]*)`;
const UNIT_AFTER_RE = new RegExp(`^\\s*(${UNIT_TOKEN})(?=$|[^${LETTER}])`, 'i');
const UNIT_BEFORE_RE = new RegExp(`(${UNIT_TOKEN})(?=$|[^${LETTER}])\\s*$`, 'i');

function canonicalUnit(raw: string | undefined): 'mm' | 'cm' | 'hu' | null {
  if (!raw) return null;
  const unit = raw.toLowerCase();
  if (unit === 'мм' || unit.startsWith('миллиметр')) return 'mm';
  if (unit === 'см' || unit.startsWith('сантиметр')) return 'cm';
  return 'hu';
}

function unitAt(text: string, mention: NumericMention): 'mm' | 'cm' | 'hu' | null {
  const after = text.slice(mention.end, mention.end + 40).match(UNIT_AFTER_RE)?.[1];
  if (after) return canonicalUnit(after);
  const before = text.slice(Math.max(0, mention.start - 40), mention.start).match(UNIT_BEFORE_RE)?.[1];
  return canonicalUnit(before);
}

export function extractNumberUnitEntities(text: string): SafetyEntity[] {
  const mentions = extractNumericMentions(text);
  const units = mentions.map((mention) => unitAt(text, mention));
  const dimensionConnector = /^\s*(?:на|[xх×*])\s*$/i;

  // В «15 на 20 мм» единица относится ко всей группе размеров, а не только
  // к последнему числу. Распространяем её только через явный размерный связник.
  for (let i = units.length - 2; i >= 0; i--) {
    if (!units[i] && units[i + 1] && dimensionConnector.test(text.slice(mentions[i].end, mentions[i + 1].start))) {
      units[i] = units[i + 1];
    }
  }
  for (let i = 1; i < units.length; i++) {
    if (!units[i] && units[i - 1] && dimensionConnector.test(text.slice(mentions[i - 1].end, mentions[i].start))) {
      units[i] = units[i - 1];
    }
  }

  return mentions.map((mention, index) => {
    const unit = units[index];
    return {
      type: 'number_unit',
      normalized: `${mention.value}|${unit ?? ''}`,
      text: mention.text,
      start: mention.start,
      end: mention.end,
      value: mention.value,
      unit,
    };
  });
}

function entitiesFromRegex(
  text: string,
  type: SafetyEntityType,
  normalized: string,
  regex: RegExp,
  occupied: Array<[number, number]> = [],
): SafetyEntity[] {
  const result: SafetyEntity[] = [];
  for (const match of text.matchAll(regex)) {
    const start = match.index;
    const end = start + match[0].length;
    if (occupied.some(([from, to]) => start < to && end > from)) continue;
    result.push({ type, normalized, text: match[0], start, end });
  }
  return result;
}

const NEGATABLE_FINDING_STEMS = [
  'выяв',
  'определ',
  'обнаруж',
  'визуализ',
  'расшир',
  'увелич',
  'смещ',
  'накоп',
  'отмеч',
  'наблюд',
  'прослеж',
  'регистр',
  'подтвержд',
  'получ',
];
const NEGATABLE_FINDING_VERB = NEGATABLE_FINDING_STEMS
  .map((stem) => `${stem}[${LETTER}]*`)
  .join('|');

const POSITIVE_FINDING_STEMS = [
  'выяв',
  'определ',
  'обнаруж',
  'визуализ',
  'отмеч',
  'наблюд',
  'прослеж',
  'регистр',
  'подтвержд',
];
const POSITIVE_FINDING_VERB = POSITIVE_FINDING_STEMS
  .map((stem) => `${stem}[${LETTER}]*`)
  .join('|');

const NEGATION_RE = new RegExp(
  `(?:^|[^${LETTER}])(?:нет|отсутств[${LETTER}]*|без\\s+признак[${LETTER}]*|не\\s+(?:${NEGATABLE_FINDING_VERB})|(?<!не\\s)исключ[${LETTER}]*)`,
  'gi',
);
const POSITIVE_RE = new RegExp(
  `(?:^|[^${LETTER}])(?:есть|имеет[${LETTER}]*|присутств[${LETTER}]*|${POSITIVE_FINDING_VERB})`,
  'gi',
);

export function extractNegationEntities(text: string): SafetyEntity[] {
  const negative = entitiesFromRegex(text, 'negation', 'negative', NEGATION_RE)
    .map((entity) => ({ ...entity, polarity: 'negative' as const }));
  const occupied = negative.map((entity) => [entity.start, entity.end] as [number, number]);
  const positive = entitiesFromRegex(text, 'negation', 'positive', POSITIVE_RE, occupied)
    .map((entity) => ({ ...entity, polarity: 'positive' as const }));
  return [...negative, ...positive].sort((a, b) => a.start - b.start);
}

const RIGHT_RE = new RegExp(
  `(?:^|[^${LETTER}])(?:справа|правосторонн[${LETTER}]*|прав(?:ый|ая|ое|ые|ого|ой|ому|ом|ую|ым|ыми|ых))(?=$|[^${LETTER}])`,
  'gi',
);
const LEFT_RE = new RegExp(
  `(?:^|[^${LETTER}])(?:слева|левосторонн[${LETTER}]*|лев(?:ый|ая|ое|ые|ого|ой|ому|ом|ую|ым|ыми|ых))(?=$|[^${LETTER}])`,
  'gi',
);

export function extractLateralityEntities(text: string): SafetyEntity[] {
  return [
    ...entitiesFromRegex(text, 'laterality', 'right', RIGHT_RE),
    ...entitiesFromRegex(text, 'laterality', 'left', LEFT_RE),
  ].sort((a, b) => a.start - b.start);
}

const NO_CONTRAST_RE = new RegExp(
  `(?:^|[^${LETTER}])без\\s+(?:внутривенн[${LETTER}]*\\s+)?контраст[${LETTER}]*`,
  'gi',
);
const WITH_CONTRAST_RE = new RegExp(
  `(?:^|[^${LETTER}])(?:(?:с|после)\\s+(?:внутривенн[${LETTER}]*\\s+)?контраст[${LETTER}]*|контрастн[${LETTER}]*\\s+усилен[${LETTER}]*|контрастирован[${LETTER}]*)`,
  'gi',
);

export function extractContrastEntities(text: string): SafetyEntity[] {
  const negative = entitiesFromRegex(text, 'contrast', 'without', NO_CONTRAST_RE);
  const occupied = negative.map((entity) => [entity.start, entity.end] as [number, number]);
  const positive = entitiesFromRegex(text, 'contrast', 'with', WITH_CONTRAST_RE, occupied);
  return [...negative, ...positive].sort((a, b) => a.start - b.start);
}

interface CriticalFactPattern {
  id: string;
  pattern: RegExp;
}

const CRITICAL_FACTS: CriticalFactPattern[] = [
  { id: 'free_gas', pattern: /пневмоперитонеум|свободн[а-яё]*\s+газ[а-яё]*/gi },
  { id: 'free_fluid', pattern: /асцит[а-яё]*|свободн[а-яё]*\s+жидкост[а-яё]*/gi },
  { id: 'perforation', pattern: /перфораци[а-яё]*/gi },
  { id: 'metastasis', pattern: /метастаз[а-яё]*/gi },
  { id: 'thrombosis', pattern: /тромб(?:оз)?[а-яё]*/gi },
  { id: 'embolism', pattern: /эмбол[а-яё]*/gi },
  { id: 'hemorrhage', pattern: /кровоизлиян[а-яё]*/gi },
  { id: 'infarction', pattern: /инфаркт[а-яё]*/gi },
  { id: 'aneurysm', pattern: /аневризм[а-яё]*/gi },
  { id: 'obstruction', pattern: /обструкц[а-яё]*|непроходимост[а-яё]*/gi },
  { id: 'hydronephrosis', pattern: /гидронефроз[а-яё]*/gi },
  { id: 'abscess', pattern: /абсцесс[а-яё]*/gi },
  { id: 'mass', pattern: /опухол[а-яё]*|образовани[а-яё]*/gi },
  { id: 'stone', pattern: /конкремент[а-яё]*|кам(?:ень|ня|ни|ней|нями)/gi },
];

function clauseStart(text: string, at: number): number {
  const separator = Math.max(
    text.lastIndexOf('.', at - 1),
    text.lastIndexOf(';', at - 1),
    text.lastIndexOf('\n', at - 1),
  );
  return separator + 1;
}

function clauseEnd(text: string, at: number): number {
  const candidates = [text.indexOf('.', at), text.indexOf(';', at), text.indexOf('\n', at)]
    .filter((index) => index >= 0);
  return candidates.length ? Math.min(...candidates) : text.length;
}

function factPolarity(text: string, start: number, end: number): 'positive' | 'negative' {
  const before = text.slice(Math.max(clauseStart(text, start), start - 80), start).toLowerCase();
  const after = text.slice(end, Math.min(clauseEnd(text, end), end + 80)).toLowerCase();
  const punctuation = '[\\s,:()\\-–—]*';
  const reportQualifier = [
    `(?:по\\s+данн[${LETTER}]*(?:\\s+[${LETTER}]+){0,2})`,
    `(?:согласно\\s+(?:данн|результат)[${LETTER}]*(?:\\s+[${LETTER}]+){0,2})`,
    `(?:на\\s+момент(?:\\s+[${LETTER}]+){0,2})`,
    `(?:достоверн[${LETTER}]*|убедительн[${LETTER}]*|отч[её]тлив[${LETTER}]*)`,
  ].join('|');
  const negativeBefore = new RegExp(
    [
      `(?:без|нет)\\s+(?:[${LETTER}]+\\s+){0,3}$`,
      `(?:не\\s+(?:${NEGATABLE_FINDING_VERB})|(?<!не\\s)исключ[${LETTER}]*)\\s+(?:[${LETTER}]+\\s+){0,2}$`,
    ].join('|'),
    'i',
  );
  const negativeAfter = new RegExp(
    `^${punctuation}(?:(?:${reportQualifier})${punctuation}){0,2}(?:не\\s+(?:${NEGATABLE_FINDING_VERB})|нет|отсутств[${LETTER}]*|(?<!не\\s)исключ[${LETTER}]*)`,
    'i',
  );
  return negativeBefore.test(before) || negativeAfter.test(after) ? 'negative' : 'positive';
}

export function extractCriticalFactEntities(text: string): SafetyEntity[] {
  const result: SafetyEntity[] = [];
  for (const fact of CRITICAL_FACTS) {
    for (const match of text.matchAll(fact.pattern)) {
      const start = match.index;
      const end = start + match[0].length;
      const polarity = factPolarity(text, start, end);
      result.push({
        type: 'critical_fact',
        normalized: `${fact.id}:${polarity}`,
        text: match[0],
        start,
        end,
        polarity,
        factId: fact.id,
      });
    }
  }
  return result.sort((a, b) => a.start - b.start || a.end - b.end);
}

interface RelationMention {
  normalized: string;
  start: number;
  end: number;
}

interface RelationSignature {
  key: string;
  description: string;
}

interface RelationSignatureCheck {
  addedByOutput: RelationSignature[];
  lostFromOutput: RelationSignature[];
  ok: boolean;
}

interface OrganPattern {
  id: string;
  pattern: RegExp;
}

// Это не секционизатор и не попытка понять весь медицинский текст. Список
// задаёт устойчивые якоря для safety-связей: орган ↔ сторона/факт/размер.
// Паттерны намеренно морфологически широкие, но лексически специфичные.
const ORGAN_PATTERNS: OrganPattern[] = [
  { id: 'gallbladder', pattern: /желчн[а-яё]*\s+пузыр[а-яё]*/gi },
  { id: 'urinary_bladder', pattern: /мочев[а-яё]*\s+пузыр[а-яё]*/gi },
  { id: 'pancreas', pattern: /поджелудочн[а-яё]*(?:\s+желез[а-яё]*)?/gi },
  { id: 'bile_ducts', pattern: /желчн[а-яё]*\s+проток[а-яё]*|холедох[а-яё]*/gi },
  { id: 'adrenal', pattern: /надпочечник[а-яё]*/gi },
  { id: 'kidney', pattern: /поч(?:к[а-яё]*|ечн[а-яё]*)/gi },
  { id: 'liver', pattern: /печ(?:ень|ен[а-яё]*)/gi },
  { id: 'spleen', pattern: /селез[её]н[а-яё]*/gi },
  { id: 'bowel', pattern: /киш(?:к[а-яё]*|ечник[а-яё]*)/gi },
  { id: 'stomach', pattern: /желуд(?:ок|к[а-яё]*)/gi },
  { id: 'ureter', pattern: /мочеточник[а-яё]*/gi },
  { id: 'lung', pattern: /л[её]г(?:к[а-яё]*|очн[а-яё]*)/gi },
  { id: 'pleura', pattern: /плевр[а-яё]*/gi },
  { id: 'brain', pattern: /головн[а-яё]*\s+мозг[а-яё]*|мозг[а-яё]*/gi },
  { id: 'sinus', pattern: /пазух[а-яё]*/gi },
  { id: 'lymph_node', pattern: /лимф(?:оуз(?:ел|л[а-яё]*)|атическ[а-яё]*\s+узл[а-яё]*)/gi },
  { id: 'aorta', pattern: /аорт[а-яё]*/gi },
  { id: 'vessel', pattern: /сосуд[а-яё]*/gi },
  { id: 'bone', pattern: /кост(?:ь|и|ей|н[а-яё]*)|скелет[а-яё]*/gi },
  { id: 'uterus', pattern: /мат(?:к[а-яё]*|очн[а-яё]*)/gi },
  { id: 'ovary', pattern: /яичник[а-яё]*/gi },
  { id: 'prostate', pattern: /предстательн[а-яё]*\s+желез[а-яё]*|простата/gi },
];

function extractOrganMentions(text: string): RelationMention[] {
  const candidates: RelationMention[] = [];
  for (const organ of ORGAN_PATTERNS) {
    for (const match of text.matchAll(organ.pattern)) {
      candidates.push({
        normalized: organ.id,
        start: match.index,
        end: match.index + match[0].length,
      });
    }
  }

  // Длинный специфичный якорь выигрывает у вложенного (например, желчный
  // пузырь не должен одновременно стать неопределённым «пузырём»).
  return candidates
    .sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start))
    .filter((candidate, index, all) => !all.some(
      (other, otherIndex) => otherIndex < index
        && other.start <= candidate.start
        && other.end >= candidate.end,
    ));
}

function splitRelationClauses(text: string): string[] {
  const rawClauses: string[] = [];
  let start = 0;

  const append = (end: number): void => {
    const clause = text.slice(start, end).trim();
    if (clause) rawClauses.push(clause);
  };

  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    const decimalSeparator = (char === ',' || char === '.')
      && /\d/.test(text[index - 1] ?? '')
      && /\d/.test(text[index + 1] ?? '');
    if (decimalSeparator) continue;

    // A colon-terminated organ heading owns the text on the following line:
    //   Печень:
    //   Метастазы не выявлены.
    // Treating that newline as a hard boundary would detach the finding from
    // its organ and let findings be moved between formatted report sections.
    if (char === '\n') {
      let previous = index - 1;
      while (previous >= start && /\s/.test(text[previous])) previous--;
      if (text[previous] === ':') continue;
    }

    // A comma often separates a finding from its size or predicate
    // ("конкремент, размером 5 мм", "метастазы, не выявлены"). It is not a
    // reliable clinical-scope boundary. Positional association below keeps
    // independent side/number pairs distinct without severing these phrases.
    if (char === '.' || char === ';' || char === '\n' || char === '/' || char === '|') {
      append(index);
      start = index + 1;
    }
  }
  append(text.length);

  // Also support a plain standalone organ heading without a trailing colon.
  // This is deliberately conservative: only a clause made entirely of one
  // known organ anchor is carried into the immediately following clause.
  const clauses: string[] = [];
  for (let index = 0; index < rawClauses.length; index++) {
    const clause = rawClauses[index];
    const organs = extractOrganMentions(clause);
    const onlyOrgan = organs.length === 1
      && `${clause.slice(0, organs[0].start)}${clause.slice(organs[0].end)}`
        .replace(/[\s:()[\]{}\-–—]+/gu, '') === '';
    if (onlyOrgan && index + 1 < rawClauses.length) {
      clauses.push(`${clause}: ${rawClauses[index + 1]}`);
      index++;
    } else {
      clauses.push(clause);
    }
  }
  return clauses;
}

function mentionDistance(anchor: RelationMention, mention: RelationMention): number {
  if (mention.end <= anchor.start) return anchor.start - mention.end;
  if (mention.start >= anchor.end) return mention.start - anchor.end;
  return 0;
}

function positionalBoundary(
  text: string,
  left: RelationMention,
  right: RelationMention,
): number {
  const between = text.slice(left.end, right.start);
  const candidates: number[] = [];

  for (let index = 0; index < between.length; index++) {
    if (between[index] !== ',') continue;
    const absolute = left.end + index;
    const decimalComma = /\d/.test(text[absolute - 1] ?? '')
      && /\d/.test(text[absolute + 1] ?? '');
    if (!decimalComma) candidates.push(absolute + 0.5);
  }
  for (const match of between.matchAll(/(?:^|[\s,])(?:и|либо|также|а\s+также)(?=$|[\s,])/giu)) {
    candidates.push(left.end + match.index + (match[0].length / 2));
  }

  // The last separator is normally the one introducing the next positional
  // group ("справа очаг 5 мм, дополнительно ... и слева ...").
  return candidates.length
    ? Math.max(...candidates)
    : (left.end + right.start) / 2;
}

function positionalGroupIndex(
  text: string,
  anchors: RelationMention[],
  mention: RelationMention,
): number {
  const center = (mention.start + mention.end) / 2;
  for (let index = 0; index < anchors.length - 1; index++) {
    if (center < positionalBoundary(text, anchors[index], anchors[index + 1])) return index;
  }
  return anchors.length - 1;
}

function relationKey(parts: {
  anchors?: string[];
  facts: string[];
  organs: string[];
  lateralities: string[];
  negations: string[];
  numbers: string[];
  contrast: string[];
}): string {
  const sorted = (values: string[]): string => [...values].sort().join(',');
  return [
    `anchor=${sorted(parts.anchors ?? [])}`,
    `fact=${sorted(parts.facts)}`,
    `organ=${sorted(parts.organs)}`,
    `side=${sorted(parts.lateralities)}`,
    `neg=${sorted(parts.negations)}`,
    `num=${sorted(parts.numbers)}`,
    `contrast=${sorted(parts.contrast)}`,
  ].join('|');
}

interface GenericRelationRange {
  start: number;
  end: number;
}

interface GenericRelationGroup extends GenericRelationRange {
  rangeIndex: number;
  anchors: string[];
  negations: string[];
  numbers: string[];
}

const GENERIC_RELATION_STOP_WORDS = new Set([
  'а',
  'без',
  'в',
  'во',
  'для',
  'до',
  'есть',
  'и',
  'из',
  'к',
  'ко',
  'либо',
  'на',
  'не',
  'нет',
  'ни',
  'но',
  'от',
  'по',
  'при',
  'с',
  'со',
  'также',
  'у',
  'это',
  'эта',
  'этот',
  'эти',
  'данным',
  'данные',
  'исследование',
  'исследования',
  'момент',
  'результат',
  'результатам',
  'кт',
  'мрт',
]);

function genericRelationRanges(text: string): GenericRelationRange[] {
  const separators: Array<{ start: number; end: number }> = [];

  for (let index = 0; index < text.length; index++) {
    if (text[index] !== ',') continue;
    const decimalComma = /\d/.test(text[index - 1] ?? '')
      && /\d/.test(text[index + 1] ?? '');
    if (!decimalComma) separators.push({ start: index, end: index + 1 });
  }

  const conjunction = new RegExp(
    `(?:^|[^${LETTER}])(а\\s+также|и|либо|также)(?=$|[^${LETTER}])`,
    'giu',
  );
  for (const match of text.matchAll(conjunction)) {
    const token = match[1];
    const offset = match[0].lastIndexOf(token);
    separators.push({
      start: match.index + offset,
      end: match.index + offset + token.length,
    });
  }

  separators.sort((a, b) => a.start - b.start || a.end - b.end);
  const ranges: GenericRelationRange[] = [];
  let cursor = 0;
  for (const separator of separators) {
    if (separator.start < cursor) {
      cursor = Math.max(cursor, separator.end);
      continue;
    }
    if (text.slice(cursor, separator.start).trim()) {
      ranges.push({ start: cursor, end: separator.start });
    }
    cursor = separator.end;
  }
  if (text.slice(cursor).trim()) ranges.push({ start: cursor, end: text.length });
  return ranges.length ? ranges : [{ start: 0, end: text.length }];
}

function isGenericRelationContentToken(raw: string): boolean {
  const token = raw.toLowerCase().replace(/ё/gu, 'е');
  if (token.length < 2 || GENERIC_RELATION_STOP_WORDS.has(token)) return false;
  if (numericWordValue(token) !== undefined) return false;
  if (/^(?:мм|см|hu|ху)$/iu.test(token)) return false;
  if (/^(?:миллиметр|сантиметр|единиц|хаунсфилд|размер|диаметр)/iu.test(token)) return false;
  if (/^(?:отсутств|присутств|исключ|име)/iu.test(token)) return false;
  if (
    NEGATABLE_FINDING_STEMS.some((stem) => token.startsWith(stem))
    || POSITIVE_FINDING_STEMS.some((stem) => token.startsWith(stem))
  ) {
    return false;
  }
  return true;
}

function genericRelationAnchors(text: string): string[] {
  const tokenRegex = new RegExp(`[${LETTER}]+`, 'giu');
  const tokens = [...text.matchAll(tokenRegex)]
    .map((match) => match[0].toLowerCase().replace(/ё/gu, 'е'))
    .filter(isGenericRelationContentToken);
  return [...new Set(tokens)].sort();
}

function maskRelationMentions(text: string, mentions: RelationMention[]): string {
  const chars = [...text];
  for (const mention of mentions) {
    for (let index = mention.start; index < mention.end; index++) chars[index] = ' ';
  }
  return chars.join('');
}

function genericRelationSignatures(
  clause: string,
  numbers: SafetyEntity[],
  negations: SafetyEntity[],
): RelationSignature[] | null {
  if (!numbers.length && !negations.length) return null;

  const ranges = genericRelationRanges(clause);
  const groups: GenericRelationGroup[] = ranges
    .map((range, rangeIndex) => ({
      ...range,
      rangeIndex,
      anchors: genericRelationAnchors(clause.slice(range.start, range.end)),
      negations: [] as string[],
      numbers: [] as string[],
    }))
    .filter((group) => group.anchors.length > 0);
  if (!groups.length) return null;

  const groupFor = (mention: RelationMention): GenericRelationGroup => {
    const center = (mention.start + mention.end) / 2;
    const containingRangeIndex = ranges.findIndex(
      (range) => center >= range.start && center <= range.end,
    );
    const containing = groups.find((group) => group.rangeIndex === containingRangeIndex);
    if (containing) return containing;

    // Attribute-only continuations normally describe the preceding content
    // span: "киста, размером 5 мм", "киста, по данным КТ, не выявлена".
    const preceding = [...groups]
      .reverse()
      .find((group) => group.rangeIndex < containingRangeIndex);
    if (preceding) return preceding;
    const following = groups.find((group) => group.rangeIndex > containingRangeIndex);
    return following ?? groups[0];
  };

  for (const number of numbers) groupFor(number).numbers.push(number.normalized);
  for (const negation of negations) groupFor(negation).negations.push(negation.normalized);

  return groups
    .filter((group) => group.numbers.length > 0 || group.negations.length > 0)
    .map((group) => {
      const key = relationKey({
        anchors: group.anchors,
        facts: [],
        organs: [],
        lateralities: [],
        negations: group.negations,
        numbers: group.numbers,
        contrast: [],
      });
      return { key, description: key };
    });
}

function relationSignatures(text: string): RelationSignature[] {
  const result: RelationSignature[] = [];

  for (const clause of splitRelationClauses(text)) {
    const facts = extractCriticalFactEntities(clause).map((entity) => ({
      normalized: entity.normalized,
      start: entity.start,
      end: entity.end,
    }));
    const organs = extractOrganMentions(clause);
    const lateralities = extractLateralityEntities(clause);
    const negations = extractNegationEntities(clause);
    const numbers = extractNumberUnitEntities(clause);
    const contrast = extractContrastEntities(clause);
    const attributes: Array<RelationMention & {
      kind: 'organs' | 'lateralities' | 'negations' | 'numbers' | 'contrast';
    }> = [
      ...organs.map((mention) => ({ ...mention, kind: 'organs' as const })),
      ...lateralities.map((entity) => ({
        normalized: entity.normalized,
        start: entity.start,
        end: entity.end,
        kind: 'lateralities' as const,
      })),
      ...negations.map((entity) => ({
        normalized: entity.normalized,
        start: entity.start,
        end: entity.end,
        kind: 'negations' as const,
      })),
      ...numbers.map((entity) => ({
        normalized: entity.normalized,
        start: entity.start,
        end: entity.end,
        kind: 'numbers' as const,
      })),
      ...contrast.map((entity) => ({
        normalized: entity.normalized,
        start: entity.start,
        end: entity.end,
        kind: 'contrast' as const,
      })),
    ];
    const genericLexicalRelations = facts.length
      ? null
      : genericRelationSignatures(
        maskRelationMentions(clause, [...organs, ...lateralities, ...contrast]),
        numbers,
        negations,
      );

    // Факт — самый сильный якорь. Каждый атрибут относится только к ближайшему
    // факту данного фрагмента, поэтому глобальное совпадение «справа/слева» и
    // «5/15 мм» уже не маскирует перестановку их связей.
    if (facts.length) {
      const grouped = facts.map((fact) => ({
        anchor: fact,
        facts: [fact.normalized],
        organs: [] as string[],
        lateralities: [] as string[],
        negations: [] as string[],
        numbers: [] as string[],
        contrast: [] as string[],
      }));

      const firstFact = facts[0];
      const lastFact = facts[facts.length - 1];
      const coordinatedFacts = facts.length > 1 && facts.slice(1).every((fact, index) => {
        const between = clause.slice(facts[index].end, fact.start);
        return /(?:^|[\s,])(?:и|либо|также|а\s+также)(?=$|[\s,])/iu.test(between);
      });
      const hasSharedScopePosition = (mention: RelationMention): boolean => (
        mention.end <= firstFact.start || mention.start >= lastFact.end
      );
      const sharedAttributes = new Set<RelationMention>();

      // A single leading/trailing locative scopes a coordinated list:
      // "в правой почке конкремент ... и образование ...". Replicate that
      // scope into each fact signature, so a safe reordering of the list does
      // not assign the organ to whichever fact happens to be closest.
      if (coordinatedFacts && organs.length === 1 && hasSharedScopePosition(organs[0])) {
        for (const group of grouped) group.organs.push(organs[0].normalized);
        const attribute = attributes.find(
          (item) => item.kind === 'organs'
            && item.start === organs[0].start
            && item.end === organs[0].end,
        );
        if (attribute) sharedAttributes.add(attribute);
      }
      if (
        coordinatedFacts
        && lateralities.length === 1
        && hasSharedScopePosition(lateralities[0])
      ) {
        for (const group of grouped) group.lateralities.push(lateralities[0].normalized);
        const attribute = attributes.find(
          (item) => item.kind === 'lateralities'
            && item.start === lateralities[0].start
            && item.end === lateralities[0].end,
        );
        if (attribute) sharedAttributes.add(attribute);
      }
      // Contrast is study-level rather than finding-level when it appears
      // once in a clause with several findings.
      if (facts.length > 1 && contrast.length === 1) {
        for (const group of grouped) group.contrast.push(contrast[0].normalized);
        const attribute = attributes.find(
          (item) => item.kind === 'contrast'
            && item.start === contrast[0].start
            && item.end === contrast[0].end,
        );
        if (attribute) sharedAttributes.add(attribute);
      }

      for (const attribute of attributes) {
        if (sharedAttributes.has(attribute)) continue;
        const nearest = grouped.reduce((best, group) => {
          const distance = mentionDistance(group.anchor, attribute);
          return distance < best.distance ? { group, distance } : best;
        }, { group: grouped[0], distance: Number.POSITIVE_INFINITY }).group;
        nearest[attribute.kind].push(attribute.normalized);
      }
      for (const group of grouped) {
        const key = relationKey(group);
        result.push({ key, description: key });
      }
      continue;
    }

    // Если критического факта нет, орган становится якорем для размеров,
    // латеральности, отрицаний и контраста.
    if (organs.length) {
      // One shared organ can contain several independently sided findings
      // ("в почках справа очаг 5 мм и слева очаг 15 мм"). Side mentions are
      // the only safe positional anchors in that shape.
      if (organs.length === 1 && lateralities.length > 1) {
        const grouped = lateralities.map((laterality) => ({
          anchor: laterality,
          facts: [] as string[],
          organs: [organs[0].normalized],
          lateralities: [laterality.normalized],
          negations: [] as string[],
          numbers: [] as string[],
          contrast: [] as string[],
        }));
        const remaining = attributes.filter(
          (item) => item.kind !== 'organs' && item.kind !== 'lateralities',
        );
        for (const attribute of remaining) {
          const group = grouped[positionalGroupIndex(clause, lateralities, attribute)];
          group[attribute.kind].push(attribute.normalized);
        }
        for (const group of grouped) {
          const key = relationKey(group);
          result.push({ key, description: key });
        }
        if (genericLexicalRelations?.length) result.push(...genericLexicalRelations);
        continue;
      }

      const grouped = organs.map((organ) => ({
        anchor: organ,
        facts: [] as string[],
        organs: [organ.normalized],
        lateralities: [] as string[],
        negations: [] as string[],
        numbers: [] as string[],
        contrast: [] as string[],
      }));
      for (const attribute of attributes.filter((item) => item.kind !== 'organs')) {
        const nearest = grouped.reduce((best, group) => {
          const distance = mentionDistance(group.anchor, attribute);
          return distance < best.distance ? { group, distance } : best;
        }, { group: grouped[0], distance: Number.POSITIVE_INFINITY }).group;
        nearest[attribute.kind].push(attribute.normalized);
      }
      for (const group of grouped) {
        const key = relationKey(group);
        result.push({ key, description: key });
      }
      if (genericLexicalRelations?.length) result.push(...genericLexicalRelations);
      continue;
    }

    // Even without a known critical-fact or organ anchor, explicit side
    // mentions establish independent positional groups. This catches a value
    // swap in "справа очаг 5 мм и слева очаг 15 мм" without having to classify
    // every possible radiology finding as a critical fact.
    if (lateralities.length) {
      const grouped = lateralities.map((laterality) => ({
        anchor: laterality,
        facts: [] as string[],
        organs: [] as string[],
        lateralities: [laterality.normalized],
        negations: [] as string[],
        numbers: [] as string[],
        contrast: [] as string[],
      }));
      for (const attribute of attributes.filter((item) => item.kind !== 'lateralities')) {
        const group = grouped[positionalGroupIndex(clause, lateralities, attribute)];
        group[attribute.kind].push(attribute.normalized);
      }
      for (const group of grouped) {
        const key = relationKey(group);
        result.push({ key, description: key });
      }
      if (genericLexicalRelations?.length) result.push(...genericLexicalRelations);
      continue;
    }

    const generic = genericLexicalRelations;
    if (generic?.length) {
      result.push(...generic);
      if (contrast.length) {
        const key = relationKey({
          facts: [],
          organs: [],
          lateralities: [],
          negations: [],
          numbers: [],
          contrast: contrast.map((entity) => entity.normalized),
        });
        result.push({ key, description: key });
      }
      continue;
    }

    // Глобальные признаки исследования (например, «с контрастом») и
    // фрагменты без органного якоря всё равно сравниваются как одна сигнатура.
    if (attributes.length) {
      const key = relationKey({
        facts: [],
        organs: [],
        lateralities: lateralities.map((entity) => entity.normalized),
        negations: negations.map((entity) => entity.normalized),
        numbers: numbers.map((entity) => entity.normalized),
        contrast: contrast.map((entity) => entity.normalized),
      });
      result.push({ key, description: key });
    }
  }

  return result;
}

function compareRelationSignatures(sourceText: string, outputText: string): RelationSignatureCheck {
  const source = relationSignatures(sourceText);
  const output = relationSignatures(outputText);
  const used = new Set<number>();
  const addedByOutput: RelationSignature[] = [];

  for (const candidate of output) {
    const index = source.findIndex(
      (signature, sourceIndex) => !used.has(sourceIndex) && signature.key === candidate.key,
    );
    if (index < 0) addedByOutput.push(candidate);
    else used.add(index);
  }

  const lostFromOutput = source.filter((_signature, index) => !used.has(index));
  return {
    addedByOutput,
    lostFromOutput,
    ok: addedByOutput.length === 0 && lostFromOutput.length === 0,
  };
}

function compareEntities(source: SafetyEntity[], output: SafetyEntity[]): SafetyEntityCheck {
  const used = new Set<number>();
  const matched: MatchedSafetyEntity[] = [];
  const addedByOutput: SafetyEntity[] = [];

  for (const candidate of output) {
    const index = source.findIndex(
      (entity, sourceIndex) => !used.has(sourceIndex) && entity.normalized === candidate.normalized,
    );
    if (index < 0) addedByOutput.push(candidate);
    else {
      used.add(index);
      matched.push({ source: source[index], output: candidate });
    }
  }

  const lostFromOutput = source.filter((_entity, index) => !used.has(index));
  return {
    matched,
    addedByOutput,
    lostFromOutput,
    ok: addedByOutput.length === 0 && lostFromOutput.length === 0,
  };
}

function changedPair(check: SafetyEntityCheck): { source?: SafetyEntity; output?: SafetyEntity } {
  return { source: check.lostFromOutput[0], output: check.addedByOutput[0] };
}

/**
 * Проверяет, что structuring-слой не изменил критические сущности относительно
 * дословной расшифровки. outputText должен содержать только продиктованные
 * секции и unmatched, без норм шаблона.
 */
export function verifyRadiologySafety(transcript: string, outputText: string): RadiologySafetyReport {
  const numbers = verifyNumbers(transcript, outputText);
  const numberUnits = compareEntities(
    extractNumberUnitEntities(transcript),
    extractNumberUnitEntities(outputText),
  );
  const negations = compareEntities(extractNegationEntities(transcript), extractNegationEntities(outputText));
  const lateralities = compareEntities(
    extractLateralityEntities(transcript),
    extractLateralityEntities(outputText),
  );
  const contrast = compareEntities(extractContrastEntities(transcript), extractContrastEntities(outputText));
  const criticalFacts = compareEntities(
    extractCriticalFactEntities(transcript),
    extractCriticalFactEntities(outputText),
  );
  const relations = compareRelationSignatures(transcript, outputText);

  const issues: SafetyIssue[] = [];
  for (const value of numbers.addedByModel) {
    issues.push({
      code: 'number_added',
      severity: 'critical',
      message: `В структурированный текст добавлено неподтверждённое число ${value}.`,
    });
  }
  for (const value of numbers.lost) {
    issues.push({
      code: 'number_lost',
      severity: 'critical',
      message: `Из структурированного текста потеряно продиктованное число ${value}.`,
    });
  }
  if (numbers.ok && !numberUnits.ok) {
    issues.push({
      code: 'number_or_unit_changed',
      severity: 'critical',
      message: 'Изменена или потеряна единица измерения при сохранённом числовом значении.',
      ...changedPair(numberUnits),
    });
  }
  if (!negations.ok) {
    issues.push({
      code: 'negation_changed',
      severity: 'critical',
      message: 'Изменена полярность утверждения (наличие/отрицание).',
      ...changedPair(negations),
    });
  }
  if (!lateralities.ok) {
    issues.push({
      code: 'laterality_changed',
      severity: 'critical',
      message: 'Изменена или потеряна латеральность (справа/слева).',
      ...changedPair(lateralities),
    });
  }
  if (!contrast.ok) {
    issues.push({
      code: 'contrast_changed',
      severity: 'critical',
      message: 'Изменён признак выполнения исследования с/без контраста.',
      ...changedPair(contrast),
    });
  }
  // Не дублируем более конкретные ошибки сущностей. Эта проверка нужна именно
  // для случая, когда все значения сохранены, но их клинические связи изменены.
  if (numbers.ok
    && numberUnits.ok
    && negations.ok
    && lateralities.ok
    && contrast.ok
    && criticalFacts.ok
    && !relations.ok) {
    issues.push({
      code: 'clinical_relation_changed',
      severity: 'critical',
      message: [
        'Изменена связь между органом, находкой, латеральностью, отрицанием, числом/единицей или контрастом.',
        relations.lostFromOutput[0]
          ? `Не сохранена связь: ${relations.lostFromOutput[0].description}.`
          : '',
        relations.addedByOutput[0]
          ? `Добавлена неподтверждённая связь: ${relations.addedByOutput[0].description}.`
          : '',
      ].filter(Boolean).join(' '),
    });
  }
  for (const fact of criticalFacts.addedByOutput) {
    issues.push({
      code: 'unsupported_critical_fact',
      severity: 'critical',
      message: `Добавлен критический факт без подтверждения в транскрипте: ${fact.text}.`,
      output: fact,
    });
  }
  for (const fact of criticalFacts.lostFromOutput) {
    issues.push({
      code: 'critical_fact_lost',
      severity: 'critical',
      message: `Потерян критический факт из транскрипта: ${fact.text}.`,
      source: fact,
    });
  }

  return {
    ok: issues.length === 0,
    requiresReview: issues.some((issue) => issue.severity === 'critical'),
    numbers,
    numberUnits,
    negations,
    lateralities,
    contrast,
    criticalFacts,
    unsupportedCriticalFacts: criticalFacts.addedByOutput,
    issues,
  };
}

/**
 * Первый safety-gate конвейера. Он запускается до секционизатора и LLM.
 *
 * Проверка повторов разрядов намеренно реализована независимо от
 * denormalizer parser: даже если преобразователь снова начнёт складывать
 * «пятьдесят пятьдесят три» в 103, gate увидит исходную неоднозначность.
 *
 * `failed` означает разрушительное изменение чисел. `incomplete` означает,
 * что значения сохранены, но исходная речь неоднозначна и требует врача.
 */
export function verifyRawToNormalizedSafety(
  rawText: string,
  normalized: string | NormalizationSafetyEvidence,
): NormalizationSafetyStageResult {
  const evidence: NormalizationSafetyEvidence = typeof normalized === 'string'
    ? { text: normalized }
    : normalized;
  const numbers = verifyNumbers(rawText, evidence.text);
  const ambiguities = findAmbiguousNumberSequences(rawText);
  const issues: SafetyIssue[] = [];

  for (const value of numbers.addedByModel) {
    issues.push({
      code: 'normalization_number_added',
      severity: 'critical',
      message: `Нормализация добавила или изменила числовое значение: ${value}.`,
    });
  }
  for (const value of numbers.lost) {
    issues.push({
      code: 'normalization_number_lost',
      severity: 'critical',
      message: `Нормализация потеряла числовое значение из raw ASR: ${value}.`,
    });
  }

  const ambiguityKeys = new Set<string>();
  for (const ambiguity of ambiguities) {
    const key = `${ambiguity.start}:${ambiguity.end}:${ambiguity.text}`;
    ambiguityKeys.add(key);
    issues.push({
      code: 'ambiguous_number_sequence',
      severity: 'critical',
      message: [
        `Raw ASR содержит неоднозначную последовательность числительных «${ambiguity.text}».`,
        'Она не может автоматически считаться суммой или диапазоном.',
      ].join(' '),
    });
  }

  let hasOtherCriticalNormalizationIssue = false;
  for (const issue of evidence.issues ?? []) {
    if (issue.code === 'ambiguous_number_sequence') {
      const sourceText = issue.sourceText ?? (
        issue.start !== undefined && issue.end !== undefined
          ? rawText.slice(issue.start, issue.end)
          : ''
      );
      const key = `${issue.start ?? -1}:${issue.end ?? -1}:${sourceText}`;
      if (!ambiguityKeys.has(key)) {
        issues.push({
          code: 'ambiguous_number_sequence',
          severity: 'critical',
          message: issue.message
            ?? `Нормализатор обнаружил неоднозначную последовательность «${sourceText}».`,
        });
      }
      continue;
    }
    if (issue.severity === 'critical') hasOtherCriticalNormalizationIssue = true;
    issues.push({
      code: 'normalization_issue',
      severity: issue.severity,
      message: issue.message ?? `Нормализатор сообщил проблему ${issue.code}.`,
    });
  }

  const status: NormalizationSafetyStageResult['status'] = (
    !numbers.ok || hasOtherCriticalNormalizationIssue
  )
    ? 'failed'
    : issues.some((issue) => issue.code === 'ambiguous_number_sequence')
      ? 'incomplete'
      : 'passed';

  return {
    stage: 'raw_to_normalized',
    status,
    ok: status === 'passed',
    requiresReview: status !== 'passed',
    numbers,
    ambiguities,
    issues,
  };
}
