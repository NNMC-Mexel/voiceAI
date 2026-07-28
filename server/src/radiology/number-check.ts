// Детерминированная сверка чисел: гарантия документа, что LLM не выдумал и не потерял ни одной цифры.
// Сравниваем мультимножество чисел в РАСШИФРОВКЕ и в тексте, который врач реально надиктовал
// (только продиктованные секции + unmatched — НЕ нормы-дефолты шаблона, там свои числа).

import { convertNumberWords } from './numbers.js';

export interface NumberCheck {
  matched: number[];       // числа, совпавшие в речи и в выводе
  addedByModel: number[];  // есть в выводе, НЕТ в речи → возможная выдумка/досочинение (красный флаг)
  lost: number[];          // есть в речи, НЕТ в выводе → потеряно (флаг)
  ok: boolean;             // true, если ничего не добавлено и не потеряно
}

export function numbersOf(text: string): number[] {
  const norm = convertNumberWords(text);
  const out: number[] = [];
  for (const m of norm.matchAll(/\d+(?:[.,]\d+)?/g)) out.push(parseFloat(m[0].replace(',', '.')));
  return out;
}

export interface AmbiguousNumberSequence {
  code: 'ambiguous_number_sequence';
  start: number;
  end: number;
  text: string;
}

type IndependentNumberRank = 'unit' | 'teen' | 'tens' | 'hundreds' | 'thousand';

// Этот словарь намеренно локален и не использует parser нормализатора:
// raw→normalized gate должен обнаружить повтор разряда даже при регрессии
// основного преобразователя.
const INDEPENDENT_NUMBER_RANKS: Readonly<Record<string, IndependentNumberRank>> = {
  ноль: 'unit',
  один: 'unit', одна: 'unit', одно: 'unit', одну: 'unit', одного: 'unit', одной: 'unit',
  два: 'unit', две: 'unit', двух: 'unit',
  три: 'unit', трех: 'unit',
  четыре: 'unit', четырех: 'unit',
  пять: 'unit', пяти: 'unit',
  шесть: 'unit', шести: 'unit',
  семь: 'unit', семи: 'unit',
  восемь: 'unit', восьми: 'unit',
  девять: 'unit', девяти: 'unit',
  десять: 'teen', десяти: 'teen',
  одиннадцать: 'teen', одиннадцати: 'teen',
  двенадцать: 'teen', двенадцати: 'teen',
  тринадцать: 'teen', тринадцати: 'teen',
  четырнадцать: 'teen', четырнадцати: 'teen',
  пятнадцать: 'teen', пятнадцати: 'teen',
  шестнадцать: 'teen', шестнадцати: 'teen',
  семнадцать: 'teen', семнадцати: 'teen',
  восемнадцать: 'teen', восемнадцати: 'teen',
  девятнадцать: 'teen', девятнадцати: 'teen',
  двадцать: 'tens', двадцати: 'tens',
  тридцать: 'tens', тридцати: 'tens',
  сорок: 'tens', сорока: 'tens',
  пятьдесят: 'tens', пятидесяти: 'tens',
  шестьдесят: 'tens', шестидесяти: 'tens',
  семьдесят: 'tens', семидесяти: 'tens',
  восемьдесят: 'tens', восьмидесяти: 'tens',
  девяносто: 'tens', девяноста: 'tens',
  сто: 'hundreds', ста: 'hundreds',
  двести: 'hundreds', двухсот: 'hundreds',
  триста: 'hundreds', трехсот: 'hundreds',
  четыреста: 'hundreds', четырехсот: 'hundreds',
  пятьсот: 'hundreds', пятисот: 'hundreds',
  шестьсот: 'hundreds', шестисот: 'hundreds',
  семьсот: 'hundreds', семисот: 'hundreds',
  восемьсот: 'hundreds', восьмисот: 'hundreds',
  девятьсот: 'hundreds', девятисот: 'hundreds',
  тысяча: 'thousand', тысячи: 'thousand', тысяч: 'thousand',
};

interface RankedWord {
  rank: IndependentNumberRank;
  normalized: string;
  start: number;
  end: number;
}

function isInvalidRankTransition(
  previous: RankedWord,
  current: RankedWord,
  thousandSeen: boolean,
): boolean {
  if (current.rank === 'thousand') {
    return thousandSeen || previous.rank === 'thousand';
  }
  if (previous.rank === 'thousand') return false;
  if (previous.rank === 'hundreds') {
    return current.rank === 'hundreds'
      || (current.rank === 'unit' && current.normalized === 'ноль');
  }
  if (previous.rank === 'tens') {
    return current.rank !== 'unit' || current.normalized === 'ноль';
  }
  // Teen и единица завершают число. Следующее числительное без явного
  // связника («на», «до», «тире») уже является соседним значением.
  return previous.rank === 'teen' || previous.rank === 'unit';
}

/**
 * Независимая от преобразователя проверка порядка разрядов.
 * Связники и пунктуация разрывают run, поэтому «50 на 53» и «от 50 до 53»
 * не считаются неоднозначными, а «пятьдесят пятьдесят три» считается.
 */
export function findAmbiguousNumberSequences(text: string): AmbiguousNumberSequence[] {
  const recognized: RankedWord[] = [];
  for (const match of text.matchAll(/\p{L}+/gu)) {
    const normalized = match[0].toLowerCase().replace(/ё/g, 'е');
    const rank = INDEPENDENT_NUMBER_RANKS[normalized];
    if (!rank) continue;
    recognized.push({
      rank,
      normalized,
      start: match.index,
      end: match.index + match[0].length,
    });
  }

  const runs: RankedWord[][] = [];
  for (const word of recognized) {
    const run = runs[runs.length - 1];
    if (run && /^[\s-]+$/u.test(text.slice(run[run.length - 1].end, word.start))) {
      run.push(word);
    } else {
      runs.push([word]);
    }
  }

  const issues: AmbiguousNumberSequence[] = [];
  for (const run of runs) {
    if (run.length < 2) continue;
    let thousandSeen = false;
    let invalid = false;
    for (let index = 1; index < run.length; index++) {
      if (run[index - 1].rank === 'thousand') thousandSeen = true;
      if (isInvalidRankTransition(run[index - 1], run[index], thousandSeen)) {
        invalid = true;
        break;
      }
      if (run[index].rank === 'thousand') thousandSeen = true;
    }
    if (!invalid) continue;
    const start = run[0].start;
    const end = run[run.length - 1].end;
    issues.push({
      code: 'ambiguous_number_sequence',
      start,
      end,
      text: text.slice(start, end),
    });
  }
  return issues;
}

// Разница мультимножеств a\b (по количеству вхождений).
function multisetDiff(a: number[], b: number[]): number[] {
  const counts = new Map<number, number>();
  for (const n of b) counts.set(n, (counts.get(n) ?? 0) + 1);
  const diff: number[] = [];
  for (const n of a) {
    const c = counts.get(n) ?? 0;
    if (c > 0) counts.set(n, c - 1);
    else diff.push(n);
  }
  return diff;
}

export function verifyNumbers(transcript: string, dictatedText: string): NumberCheck {
  const said = numbersOf(transcript);
  const doc = numbersOf(dictatedText);
  const addedByModel = multisetDiff(doc, said); // в doc, но не в said
  const lost = multisetDiff(said, doc);          // в said, но не в doc
  const matched = multisetDiff(doc, addedByModel); // всё из doc минус добавленное = совпавшее
  return { matched, addedByModel, lost, ok: addedByModel.length === 0 && lost.length === 0 };
}
