// Числовой токенизатор голосовых команд рентгенолога.
// Детерминистично, без LLM. Две задачи:
//   1. converttNumberWords — «восемьдесят на десять» → «80 на 10»
//      (страховка на случай, если ASR/денормализация оставили слова-числа).
//   2. extractNumbers — вытащить все числа с привязкой к предшествующему
//      ключевому слову («стенка 5» → number 5, precededBy 'стенка').
//
// Команды из ТЗ используют цифры («плотность 56», «размеры 14 7 5»),
// поэтому основной путь — цифры; слова-числа поддержаны для устойчивости.

// Именительный + косвенные падежи (врач диктует «до девятнадцати», «до тринадцати»).
const UNITS: Record<string, number> = {
  ноль: 0, один: 1, одна: 1, одно: 1, одного: 1, одной: 1,
  два: 2, две: 2, двух: 2, три: 3, трёх: 3, трех: 3, четыре: 4, четырёх: 4, четырех: 4,
  пять: 5, пяти: 5, шесть: 6, шести: 6, семь: 7, семи: 7, восемь: 8, восьми: 8, девять: 9, девяти: 9,
};
const TEENS: Record<string, number> = {
  десять: 10, десяти: 10, одиннадцать: 11, одиннадцати: 11, двенадцать: 12, двенадцати: 12,
  тринадцать: 13, тринадцати: 13, четырнадцать: 14, четырнадцати: 14, пятнадцать: 15, пятнадцати: 15,
  шестнадцать: 16, шестнадцати: 16, семнадцать: 17, семнадцати: 17, восемнадцать: 18, восемнадцати: 18,
  девятнадцать: 19, девятнадцати: 19,
};
const TENS: Record<string, number> = {
  двадцать: 20, двадцати: 20, тридцать: 30, тридцати: 30, сорок: 40, сорока: 40,
  пятьдесят: 50, пятидесяти: 50, шестьдесят: 60, шестидесяти: 60, семьдесят: 70, семидесяти: 70,
  восемьдесят: 80, восьмидесяти: 80, девяносто: 90, девяноста: 90,
};
const HUNDREDS: Record<string, number> = {
  сто: 100, ста: 100, двести: 200, двухсот: 200, триста: 300, трёхсот: 300, трехсот: 300,
  четыреста: 400, четырёхсот: 400, четырехсот: 400, пятьсот: 500, пятисот: 500,
  шестьсот: 600, шестисот: 600, семьсот: 700, семисот: 700, восемьсот: 800, восьмисот: 800,
  девятьсот: 900, девятисот: 900,
};
const THOUSANDS = new Set([
  'тысяча', 'тысячи', 'тысяч',
]);

// Слова, которые НЕ являются числами, но разделяют/связывают их — пропускаем.
const NUMBER_WORD = new Set([
  ...Object.keys(UNITS), ...Object.keys(TEENS), ...Object.keys(TENS), ...Object.keys(HUNDREDS),
  ...THOUSANDS,
]);

type NumberWordKind = 'unit' | 'teen' | 'tens' | 'hundreds' | 'thousand';

interface NumberWordLexeme {
  raw: string;
  normalized: string;
  start: number;
  end: number;
  kind: NumberWordKind;
  value: number;
}

export interface NumberWordTransformation {
  type: 'cardinal';
  start: number;
  end: number;
  sourceText: string;
  normalizedText: string;
  values: number[];
}

export interface NumberWordNormalizationIssue {
  code: 'ambiguous_number_sequence';
  severity: 'critical';
  start: number;
  end: number;
  sourceText: string;
  normalizedText: string;
  values: number[];
  message: string;
}

export interface NumberWordNormalizationResult {
  text: string;
  transformations: NumberWordTransformation[];
  issues: NumberWordNormalizationIssue[];
}

export interface NormalizeNumberWordsOptions {
  /**
   * GigaAM denormalization historically leaves a single «один/одна» intact:
   * outside an explicit numeric context it can be a pronoun. Command parsing,
   * on the other hand, keeps the old behaviour and converts it.
   */
  preserveStandaloneOne?: boolean;
}

function classifyNumberWord(raw: string): Pick<NumberWordLexeme, 'kind' | 'value'> | null {
  const word = raw.toLowerCase().replace(/ё/g, 'е');
  if (word in HUNDREDS) return { kind: 'hundreds', value: HUNDREDS[word] };
  if (word in TENS) return { kind: 'tens', value: TENS[word] };
  if (word in TEENS) return { kind: 'teen', value: TEENS[word] };
  if (word in UNITS) return { kind: 'unit', value: UNITS[word] };
  if (THOUSANDS.has(word)) return { kind: 'thousand', value: 1000 };
  return null;
}

interface ParsedNumber {
  value: number;
  end: number;
}

/**
 * Разбирает ровно одну допустимую русскую числовую группу до 999.
 * Порядок разрядов строгий: сотни → десятки → единицы. Повтор разряда
 * не складывается, а завершает текущее число.
 */
function parseSubThousand(tokens: NumberWordLexeme[], start: number): ParsedNumber | null {
  let index = start;
  let value = 0;
  let consumed = false;

  if (tokens[index]?.kind === 'hundreds') {
    value += tokens[index].value;
    index++;
    consumed = true;
  }

  if (tokens[index]?.kind === 'teen') {
    value += tokens[index].value;
    index++;
    return { value, end: index };
  }

  if (tokens[index]?.kind === 'tens') {
    value += tokens[index].value;
    index++;
    consumed = true;
    if (tokens[index]?.kind === 'unit' && tokens[index].value > 0) {
      value += tokens[index].value;
      index++;
    }
    return { value, end: index };
  }

  if (tokens[index]?.kind === 'unit') {
    // «сто ноль» и «двести ноль» — не одно число: ноль не дополняет сотни.
    if (!consumed || tokens[index].value > 0) {
      value += tokens[index].value;
      index++;
      consumed = true;
    }
  }

  return consumed ? { value, end: index } : null;
}

function parseOneNumber(tokens: NumberWordLexeme[], start: number): ParsedNumber {
  if (tokens[start].kind === 'thousand') {
    const remainder = parseSubThousand(tokens, start + 1);
    if (remainder && tokens[remainder.end]?.kind !== 'thousand') {
      return { value: 1000 + remainder.value, end: remainder.end };
    }
    return { value: 1000, end: start + 1 };
  }

  const prefix = parseSubThousand(tokens, start);
  // Все токены заранее классифицированы, поэтому этот случай недостижим.
  if (!prefix) return { value: tokens[start].value, end: start + 1 };

  if (tokens[prefix.end]?.kind !== 'thousand') return prefix;

  const thousandValue = prefix.value * 1000;
  const afterThousand = prefix.end + 1;
  const remainder = parseSubThousand(tokens, afterThousand);
  // Не поглощаем «две» из «две тысячи две тысячи»: это две неоднозначно
  // стоящие рядом группы, а не 2002 тысячи.
  if (remainder && tokens[remainder.end]?.kind !== 'thousand') {
    return { value: thousandValue + remainder.value, end: remainder.end };
  }
  return { value: thousandValue, end: afterThousand };
}

function splitStrictNumberRun(tokens: NumberWordLexeme[]): ParsedNumber[] {
  const result: ParsedNumber[] = [];
  let index = 0;
  while (index < tokens.length) {
    const parsed = parseOneNumber(tokens, index);
    result.push(parsed);
    index = Math.max(parsed.end, index + 1);
  }
  return result;
}

function numberWordRuns(input: string): NumberWordLexeme[][] {
  const words: NumberWordLexeme[] = [];
  for (const match of input.matchAll(/\p{L}+/gu)) {
    const classified = classifyNumberWord(match[0]);
    if (!classified) continue;
    words.push({
      raw: match[0],
      normalized: match[0].toLowerCase().replace(/ё/g, 'е'),
      start: match.index,
      end: match.index + match[0].length,
      ...classified,
    });
  }

  const runs: NumberWordLexeme[][] = [];
  for (const word of words) {
    const current = runs[runs.length - 1];
    if (
      current
      && /^[\s-]+$/u.test(input.slice(current[current.length - 1].end, word.start))
    ) {
      current.push(word);
    } else {
      runs.push([word]);
    }
  }
  return runs;
}

/**
 * Заменяет последовательности числительных на цифры.
 * «восемьдесят на десять» → «80 на 10», «сто пятьдесят шесть» → «156».
 *
 * Важно: недопустимые последовательности не складываются. Например,
 * «пятьдесят пятьдесят три» превращается в «50 53» и получает critical issue,
 * а не в клинически опасное «103».
 */
export function normalizeNumberWordsDetailed(
  input: string,
  options: NormalizeNumberWordsOptions = {},
): NumberWordNormalizationResult {
  if (!input) return { text: input, transformations: [], issues: [] };

  const transformations: NumberWordTransformation[] = [];
  const issues: NumberWordNormalizationIssue[] = [];
  let cursor = 0;
  let text = '';

  for (const run of numberWordRuns(input)) {
    const start = run[0].start;
    const end = run[run.length - 1].end;
    const sourceText = input.slice(start, end);
    const parsed = splitStrictNumberRun(run);
    const values = parsed.map((item) => item.value);
    const preserveStandaloneOne = options.preserveStandaloneOne === true
      && run.length === 1
      && values.length === 1
      && values[0] === 1;
    const normalizedText = preserveStandaloneOne ? sourceText : values.join(' ');

    text += input.slice(cursor, start);
    text += normalizedText;
    cursor = end;

    if (normalizedText !== sourceText) {
      transformations.push({
        type: 'cardinal',
        start,
        end,
        sourceText,
        normalizedText,
        values,
      });
    }
    if (parsed.length > 1) {
      issues.push({
        code: 'ambiguous_number_sequence',
        severity: 'critical',
        start,
        end,
        sourceText,
        normalizedText,
        values,
        message: [
          `Последовательность «${sourceText}» не образует одно число по строгой грамматике.`,
          `Она сохранена как отдельные значения: ${values.join(', ')}.`,
        ].join(' '),
      });
    }
  }
  text += input.slice(cursor);
  return { text, transformations, issues };
}

export function convertNumberWords(input: string): string {
  return normalizeNumberWordsDetailed(input).text;
}

export interface NumberToken {
  value: number;       // числовое значение (десятичные через точку внутри)
  raw: string;         // как в тексте
  start: number;
  end: number;
  precededBy: string;  // ближайшее слово-тег перед числом (lowercase, без пунктуации)
  followedBy: string;  // ближайшее слово-тег после числа; нужно для «145 КВР»
  precedingWords: string[]; // ближайшие слова слева, от ближнего к дальнему
  followingWords: string[]; // ближайшие слова справа, от ближнего к дальнему
}

const WORD_RE = /[a-zа-яё]+/i;

/**
 * Извлекает все числа команды с привязкой к предшествующему слову-тегу.
 * «холедох 11 мм камень 6 мм» → [{value:11,precededBy:'холедох'},{value:6,precededBy:'камень'}]
 * Игнорирует слова-числа (они уже переведены convertNumberWords).
 */
export function extractNumbers(text: string): NumberToken[] {
  const result: NumberToken[] = [];
  const numRe = /(?:(?<![a-zа-яё])(?:минус|плюс)\s+|[+\-\u2212]\s*)?\d+(?:[.,]\d+)?/giu;
  let m: RegExpExecArray | null;
  while ((m = numRe.exec(text)) !== null) {
    const raw = m[0];
    const normalizedRaw = raw.toLowerCase().replace(/\s+/gu, ' ').trim();
    const sign = /^(?:минус(?=$|[^a-zа-яё])|[-\u2212])/u.test(normalizedRaw) ? -1 : 1;
    const numeric = normalizedRaw.match(/\d+(?:[.,]\d+)?/u)?.[0] ?? '';
    const value = sign * parseFloat(numeric.replace(',', '.'));
    // ищем слово-тег: последнее «словесное» слово слева, пропуская другие числа/предлоги
    const before = text.slice(0, m.index);
    const words = before.match(/[a-zа-яё]+/gi) || [];
    let precededBy = '';
    for (let i = words.length - 1; i >= 0; i--) {
      const w = words[i].toLowerCase();
      if (NUMBER_WORD.has(w)) continue; // «от», предлоги оставляем как тег? нет — только числа-слова пропускаем
      precededBy = w;
      break;
    }
    const after = text.slice(m.index + raw.length);
    const followedBy = (after.match(/[a-zа-яё]+/i)?.[0] ?? '').toLowerCase();
    const precedingWords = words
      .map((word) => word.toLowerCase().replace(/ё/g, 'е'))
      .filter((word) => !NUMBER_WORD.has(word))
      .slice(-4)
      .reverse();
    const followingWords = (after.match(/[a-zа-яё]+/gi) ?? [])
      .map((word) => word.toLowerCase().replace(/ё/g, 'е'))
      .filter((word) => !NUMBER_WORD.has(word))
      .slice(0, 4);
    result.push({
      value,
      raw,
      start: m.index,
      end: m.index + raw.length,
      precededBy: precededBy.replace(/ё/g, 'е'),
      followedBy: followedBy.replace(/ё/g, 'е'),
      precedingWords,
      followingWords,
    });
  }
  return result;
}

function wordMatchesKeyword(word: string, keyword: string): boolean {
  const normalized = keyword.toLowerCase().replace(/ё/g, 'е');
  return word === normalized || word.startsWith(normalized);
}

function keywordPhraseDistance(
  words: string[],
  keyword: string,
  reverse: boolean,
): number | undefined {
  const parts = keyword
    .toLowerCase()
    .replace(/ё/g, 'е')
    .trim()
    .split(/\s+/u)
    .filter(Boolean);
  if (parts.length === 0) return undefined;
  const expected = reverse ? [...parts].reverse() : parts;
  for (let offset = 0; offset <= words.length - expected.length; offset++) {
    if (
      expected.every((part, index) => (
        wordMatchesKeyword(words[offset + index], part)
      ))
    ) {
      return offset;
    }
  }
  return undefined;
}

/**
 * Возвращает цену связи числа с параметром. Меньше — ближе и надёжнее.
 * Слово перед числом получает небольшой приоритет, но учитываются обе стороны,
 * включая единицу между значением и названием параметра.
 */
export function numberKeywordDistance(token: NumberToken, keywords: string[]): number | undefined {
  let best: number | undefined;
  for (const keyword of keywords) {
    const before = keywordPhraseDistance(token.precedingWords, keyword, true);
    if (before !== undefined) {
      const score = before * 2;
      if (best === undefined || score < best) best = score;
    }
    const after = keywordPhraseDistance(token.followingWords, keyword, false);
    if (after !== undefined) {
      const score = after * 2 + 1;
      if (best === undefined || score < best) best = score;
    }
  }
  return best;
}

/**
 * Глобально связывает группы ключевых слов с числами один-к-одному.
 * В отличие от последовательного greedy это корректно разбирает и
 * «КВР 145 плотность 62», и «145 КВР 62 плотность».
 */
export interface NumberKeywordAssignmentResult {
  assignment: Array<number | undefined>;
  ambiguous: boolean;
}

export function assignNumbersToKeywordGroupsDetailed(
  tokens: NumberToken[],
  keywordGroups: string[][],
  unavailable: ReadonlySet<number> = new Set<number>(),
): NumberKeywordAssignmentResult {
  interface Candidate {
    assignment: Array<number | undefined>;
    assigned: number;
    cost: number;
  }
  let best: Candidate = {
    assignment: Array<number | undefined>(keywordGroups.length).fill(undefined),
    assigned: -1,
    cost: Number.POSITIVE_INFINITY,
  };
  let ambiguous = false;
  const current = Array<number | undefined>(keywordGroups.length).fill(undefined);
  const used = new Set<number>(unavailable);

  const visit = (groupIndex: number, assigned: number, cost: number): void => {
    if (groupIndex === keywordGroups.length) {
      if (assigned > best.assigned || (assigned === best.assigned && cost < best.cost)) {
        best = { assignment: [...current], assigned, cost };
        ambiguous = false;
      } else if (
        assigned === best.assigned
        && cost === best.cost
        && current.some((value, index) => value !== best.assignment[index])
      ) {
        ambiguous = true;
      }
      return;
    }

    const options = tokens
      .map((token, tokenIndex) => ({
        tokenIndex,
        distance: numberKeywordDistance(token, keywordGroups[groupIndex]),
      }))
      .filter((option): option is { tokenIndex: number; distance: number } => (
        option.distance !== undefined && !used.has(option.tokenIndex)
      ))
      .sort((a, b) => a.distance - b.distance || a.tokenIndex - b.tokenIndex);

    for (const option of options) {
      current[groupIndex] = option.tokenIndex;
      used.add(option.tokenIndex);
      visit(groupIndex + 1, assigned + 1, cost + option.distance);
      used.delete(option.tokenIndex);
      current[groupIndex] = undefined;
    }
    visit(groupIndex + 1, assigned, cost);
  };

  visit(0, 0, 0);
  return {
    assignment: best.assignment,
    ambiguous,
  };
}

export function assignNumbersToKeywordGroups(
  tokens: NumberToken[],
  keywordGroups: string[][],
  unavailable: ReadonlySet<number> = new Set<number>(),
): Array<number | undefined> {
  return assignNumbersToKeywordGroupsDetailed(
    tokens,
    keywordGroups,
    unavailable,
  ).assignment;
}

/**
 * Есть ли фраза в тексте команды. Фраза нормализуется (lowercase, ё→е).
 * Для одиночных слов длиной ≥4 — префиксный матч по началу слова, чтобы
 * «метастаз» ловил «метастазы», «надпочечник» → «надпочечнике» (русские словоформы).
 * Короткие слова (<4, напр. «газ») — только как отдельное слово, без ложных «газета».
 */
export function hasPhrase(normalizedText: string, phrase: string): boolean {
  const p = phrase.toLowerCase().replace(/ё/g, 'е').trim();
  if (!p) return false;
  if (WORD_RE.test(p) && !/\s/.test(p)) {
    const body = p.length >= 4
      ? `(^|[^a-zа-я])${escapeRe(p)}`                    // префикс: словоформы
      : `(^|[^a-zа-я])${escapeRe(p)}([^a-zа-я]|$)`;      // точное слово
    return new RegExp(body, 'i').test(normalizedText);
  }
  return normalizedText.includes(p);
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Нормализация команды: lowercase, ё→е, цифры из слов, схлопнуть пробелы. */
export function normalizeCommand(input: string): string {
  return convertNumberWords(input)
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[«»"']/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
