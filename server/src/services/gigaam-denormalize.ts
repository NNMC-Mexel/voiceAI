// Денормализатор: цифры/единицы/даты прописью → цифровые сокращения.
// Вызывается ПОСЛЕ applyMedicalDictionary и ДО передачи в LLM.
// GigaAM выдаёт «сто тридцать девять грамм на литр» → нужно «139 г/л».
//
// Все регексы используют lookahead/lookbehind вместо \b — стандартный JS \b
// в Unicode-режиме НЕ работает на кириллице (т.к. \w = [a-zA-Z0-9_] по умолчанию).

import {
  normalizeNumberWordsDetailed,
  type NumberWordNormalizationIssue,
} from '../radiology/numbers.js';

// Класс «буква или цифра» в любом письме (Cyrillic + Latin + digit)
const WC = '[\\p{L}\\p{N}_]';
// «Не словесный символ» = граница слова
const WB_BEFORE = `(?<!${WC})`;
const WB_AFTER  = `(?!${WC})`;

const CARDINAL: Record<string, number> = {
  ноль: 0,
  один: 1, одна: 1, одно: 1, одну: 1,
  два: 2, две: 2,
  три: 3, четыре: 4, пять: 5, шесть: 6, семь: 7, восемь: 8, девять: 9,
  десять: 10,
  одиннадцать: 11, двенадцать: 12, тринадцать: 13, четырнадцать: 14,
  пятнадцать: 15, шестнадцать: 16, семнадцать: 17, восемнадцать: 18, девятнадцать: 19,
  двадцать: 20, тридцать: 30, сорок: 40, пятьдесят: 50, шестьдесят: 60,
  семьдесят: 70, восемьдесят: 80, девяносто: 90,
  сто: 100, двести: 200, триста: 300, четыреста: 400, пятьсот: 500,
  шестьсот: 600, семьсот: 700, восемьсот: 800, девятьсот: 900,
};

const ORDINAL_DAY: Record<string, number> = {
  первого: 1, второго: 2, третьего: 3,
  четвертого: 4, четвёртого: 4,
  пятого: 5, шестого: 6, седьмого: 7,
  восьмого: 8, девятого: 9, десятого: 10,
  одиннадцатого: 11, двенадцатого: 12, тринадцатого: 13,
  четырнадцатого: 14, пятнадцатого: 15, шестнадцатого: 16,
  семнадцатого: 17, восемнадцатого: 18, девятнадцатого: 19,
  двадцатого: 20, тридцатого: 30,
};

const MONTHS: Record<string, number> = {
  января: 1, февраля: 2, марта: 3, апреля: 4, мая: 5, июня: 6,
  июля: 7, августа: 8, сентября: 9, октября: 10, ноября: 11, декабря: 12,
};

const THOUSAND = new Set(['тысяча', 'тысячи', 'тысяч']);

function parseRusNumber(text: string): number | null {
  const ordinalCardinal: Record<number, string> = {
    1: 'один',
    2: 'два',
    3: 'три',
    4: 'четыре',
    5: 'пять',
    6: 'шесть',
    7: 'семь',
    8: 'восемь',
    9: 'девять',
    10: 'десять',
    11: 'одиннадцать',
    12: 'двенадцать',
    13: 'тринадцать',
    14: 'четырнадцать',
    15: 'пятнадцать',
    16: 'шестнадцать',
    17: 'семнадцать',
    18: 'восемнадцать',
    19: 'девятнадцать',
    20: 'двадцать',
    30: 'тридцать',
  };
  const cardinalized = text
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => {
      const ordinal = ORDINAL_DAY[token];
      return ordinal === undefined ? token : ordinalCardinal[ordinal];
    })
    .join(' ');
  if (!cardinalized) return null;
  const parsed = normalizeNumberWordsDetailed(cardinalized);
  if (parsed.issues.length || !/^\d+$/.test(parsed.text)) return null;
  return Number(parsed.text);
}

function pad2(n: number): string { return String(n).padStart(2, '0'); }

const sortedKeys = (obj: Record<string, unknown>) => Object.keys(obj).sort((a, b) => b.length - a.length);

const CARDINAL_RE = sortedKeys(CARDINAL).join('|');
const ORDINAL_DAY_RE = sortedKeys(ORDINAL_DAY).join('|');
const MONTH_RE = Object.keys(MONTHS).join('|');
const THOUSAND_RE = 'тысяч[аи]?';
const NUMBER_TOKEN = `(?:${CARDINAL_RE}|${THOUSAND_RE})`;
const NUMBER_RUN = `${NUMBER_TOKEN}(?:\\s+${NUMBER_TOKEN})*`;
// Для года: cardinals + thousand + завершающий ordinal («…двадцать шестого»)
const YEAR_TOKEN = `(?:${CARDINAL_RE}|${THOUSAND_RE}|${ORDINAL_DAY_RE})`;
const YEAR_RUN = `${YEAR_TOKEN}(?:\\s+${YEAR_TOKEN})*`;

// ────── 1. Даты ─────────────────────────────────────────────────────────────
function denormalizeDates(s: string): string {
  // С годом: «одиннадцатого марта две тысячи двадцать шестого года»
  const withYear = new RegExp(
    `${WB_BEFORE}(${ORDINAL_DAY_RE})\\s+(${MONTH_RE})\\s+(${YEAR_RUN})(?:\\s+(?:год[а]?|г\\.?))${WB_AFTER}`,
    'giu',
  );
  let out = s.replace(withYear, (_m, day, month, yearExpr) => {
    const dayN = ORDINAL_DAY[String(day).toLowerCase()];
    const monthN = MONTHS[String(month).toLowerCase()];
    const yearN = parseRusNumber(String(yearExpr));
    if (!dayN || !monthN || !yearN) return _m;
    if (yearN < 1900 || yearN > 2100) return _m;
    return `${pad2(dayN)}.${pad2(monthN)}.${yearN}г.`;
  });

  // Без года: «одиннадцатого марта» → 11.03
  const noYear = new RegExp(`${WB_BEFORE}(${ORDINAL_DAY_RE})\\s+(${MONTH_RE})${WB_AFTER}`, 'giu');
  out = out.replace(noYear, (_m, day, month) => {
    const dayN = ORDINAL_DAY[String(day).toLowerCase()];
    const monthN = MONTHS[String(month).toLowerCase()];
    if (!dayN || !monthN) return _m;
    return `${pad2(dayN)}.${pad2(monthN)}`;
  });

  return out;
}

// ────── 2. Десятичные дроби ─────────────────────────────────────────────────
function denormalizeDecimals(s: string): string {
  // Поддерживаем все роды:
  //   целых / целая / целое  ←  intPart
  //   десятых / десятая / десятые / сотых / сотая  ←  scale + ending
  const re = new RegExp(
    `${WB_BEFORE}(${NUMBER_RUN})\\s+цел(?:ых|ая|ое|ых|ые)\\s+(${NUMBER_RUN})\\s+(десят|сот)(?:ых|ая|ые|ой|ую)${WB_AFTER}`,
    'giu',
  );
  return s.replace(re, (_m, intPart, fracPart, scale) => {
    const intN = parseRusNumber(String(intPart));
    const fracN = parseRusNumber(String(fracPart));
    if (intN === null || fracN === null) return _m;
    if (scale === 'десят') return `${intN},${fracN}`;
    if (scale === 'сот') return `${intN},${fracN < 10 ? '0' + fracN : fracN}`;
    return _m;
  });
}

// ────── 3. Кардинальные числа ───────────────────────────────────────────────
function denormalizeCardinals(s: string): string {
  return normalizeNumberWordsDetailed(s, { preserveStandaloneOne: true }).text;
}

// ────── 4. Единицы измерения ────────────────────────────────────────────────
// Используем строки с явными границами вместо литералов с \b.
function buildUnitRegex(body: string): RegExp {
  return new RegExp(`${WB_BEFORE}${body}${WB_AFTER}`, 'giu');
}

const UNIT_REPLACEMENTS: Array<[RegExp, string]> = [
  // Концентрации (per литр)
  [buildUnitRegex(`грамм(?:а|ов|)\\s+на\\s+литр`), 'г/л'],
  [buildUnitRegex(`миллиграмм(?:а|ов|)\\s+на\\s+литр`), 'мг/л'],
  [buildUnitRegex(`миллимол[ьея][ийя]?\\s+на\\s+литр`), 'ммоль/л'],
  [buildUnitRegex(`микромол[ьея][ийя]?\\s+на\\s+литр`), 'мкмоль/л'],
  [buildUnitRegex(`наномол[ьея][ийя]?\\s+на\\s+литр`), 'нмоль/л'],
  [buildUnitRegex(`пикомол[ьея][ийя]?\\s+на\\s+литр`), 'пмоль/л'],

  // Per миллилитр
  [buildUnitRegex(`нанограмм(?:а|ов|)\\s+на\\s+миллилитр`), 'нг/мл'],
  [buildUnitRegex(`пикограмм(?:а|ов|)\\s+на\\s+миллилитр`), 'пг/мл'],
  [buildUnitRegex(`миллиграмм(?:а|ов|)\\s+на\\s+миллилитр`), 'мг/мл'],
  [buildUnitRegex(`микрограмм(?:а|ов|)\\s+на\\s+миллилитр`), 'мкг/мл'],
  [buildUnitRegex(`международных\\s+единиц\\s+на\\s+миллилитр`), 'МЕ/мл'],

  // Единицы / per литр
  [buildUnitRegex(`единиц(?:а|ы|)\\s+на\\s+литр`), 'Ед/л'],
  [buildUnitRegex(`международных\\s+единиц\\s+на\\s+литр`), 'МЕ/л'],

  // Физиология
  [buildUnitRegex(`миллиметр(?:ов|а|)\\s+ртутного\\s+столба`), 'мм рт.ст.'],
  [buildUnitRegex(`ударов\\s+в\\s+минуту`), 'уд/мин'],
  [buildUnitRegex(`миллиметр(?:ов|а|)\\s+в\\s+час`), 'мм/ч'],
  [buildUnitRegex(`раз(?:а|у|)\\s+в\\s+минуту`), '/мин'],

  // Одиночные единицы (после концентраций!)
  // Процент жадно съедает предшествующий пробел, чтобы получить «5,7%» а не «5,7 %».
  [new RegExp(`\\s*${WB_BEFORE}процент(?:а|ов|)${WB_AFTER}`, 'giu'), '%'],
  [buildUnitRegex(`килограмм(?:а|ов|)`), 'кг'],
  [buildUnitRegex(`миллиграмм(?:а|ов|)`), 'мг'],
  [buildUnitRegex(`микрограмм(?:а|ов|)`), 'мкг'],
  [buildUnitRegex(`нанограмм(?:а|ов|)`), 'нг'],
  [buildUnitRegex(`миллилитр(?:ов|а|)`), 'мл'],
  [buildUnitRegex(`литр(?:ов|а|)`), 'л'],
  [buildUnitRegex(`километр(?:ов|а|)`), 'км'],
];

function denormalizeUnitsOnly(s: string): string {
  let out = s;
  for (const [re, repl] of UNIT_REPLACEMENTS) out = out.replace(re, repl);
  return out;
}

// ────── 5. Нормализация уже-цифровых форматов ───────────────────────────────
// Многие документы пишут даты криво: «27.04.26г.», «11.23г», «05.05.26г».
// Приводим к канону «27.04.2026», «27.04.2026», «05.05.2026».
// Recall на датах был 67% — большая часть потерь именно из-за нестандартного «г.».
function normalizeNumericFormats(s: string): string {
  let out = s;

  // Гибридные годы: GigaAM digitizes частично, оставляет ordinal как слово.
  // «20 шестого года» (т.е. «две тысячи двадцать шестого года» → 2026 года)
  out = out.replace(
    /(?<![\p{L}\p{N}_])20\s+(первого|второго|третьего|четвертого|четвёртого|пятого|шестого|седьмого|восьмого|девятого)\s+(?=год)/giu,
    (_m, ord) => {
      const ords: Record<string, number> = { первого:1, второго:2, третьего:3, четвертого:4, четвёртого:4, пятого:5, шестого:6, седьмого:7, восьмого:8, девятого:9 };
      const val = 2020 + (ords[ord.toLowerCase()] || 0);
      return `${val} `;
    },
  );

  // «2020 шестого года» (уже 4-значный «двадцать = 2020», добавляем ordinal)
  // Срабатывает только если последняя цифра 0 (т.е. «X0 шестого года»).
  out = out.replace(
    /(?<![\p{L}\p{N}_])(\d{3}0)\s+(первого|второго|третьего|четвертого|четвёртого|пятого|шестого|седьмого|восьмого|девятого)\s+(?=год)/giu,
    (_m, year, ord) => {
      const ords: Record<string, number> = { первого:1, второго:2, третьего:3, четвертого:4, четвёртого:4, пятого:5, шестого:6, седьмого:7, восьмого:8, девятого:9 };
      const val = parseInt(year, 10) + (ords[ord.toLowerCase()] || 0);
      if (val < 1900 || val > 2100) return _m;
      return `${val} `;
    },
  );

  // Гибридная дата: «ДД.ММ <year-expr> года?» → «ДД.ММ.<year>г.»
  // Покрывает «от 15.05 20 шестого года» → «от 15.05.2026г.»
  out = out.replace(
    /(?<![\p{L}\p{N}_])(\d{1,2})\.(\d{1,2})\s+(\d{4})\s*г(?:\.|ода?)?/giu,
    (_m, dd, mm, yyyy) => {
      const d = parseInt(dd, 10), m = parseInt(mm, 10), y = parseInt(yyyy, 10);
      if (d < 1 || d > 31 || m < 1 || m > 12 || y < 1900 || y > 2100) return _m;
      return `${String(d).padStart(2,'0')}.${String(m).padStart(2,'0')}.${y}г.`;
    },
  );

  // Дата ДД.ММ.ГГ или ДД.ММ.ГГГГ, опционально с «г.»/«г» в конце
  out = out.replace(
    new RegExp(`${WB_BEFORE}(\\d{1,2})\\.(\\d{1,2})\\.(\\d{2,4})\\s*г\\.?${WB_AFTER}`, 'gu'),
    (_m, dd, mm, yy) => {
      const day = String(parseInt(dd, 10)).padStart(2, '0');
      const month = String(parseInt(mm, 10)).padStart(2, '0');
      let year = yy.length === 2 ? (parseInt(yy, 10) >= 50 ? '19' + yy : '20' + yy) : yy;
      // Валидация диапазонов
      if (parseInt(day) > 31 || parseInt(month) > 12) return _m;
      return `${day}.${month}.${year}г.`;
    },
  );
  // То же без «г.» в конце — приводим 2-значный год к 4-значному
  out = out.replace(
    new RegExp(`${WB_BEFORE}(\\d{1,2})\\.(\\d{1,2})\\.(\\d{2})${WB_AFTER}`, 'gu'),
    (_m, dd, mm, yy) => {
      if (parseInt(dd) > 31 || parseInt(mm) > 12) return _m;
      const year = parseInt(yy, 10) >= 50 ? '19' + yy : '20' + yy;
      return `${dd.padStart(2, '0')}.${mm.padStart(2, '0')}.${year}`;
    },
  );

  // ИМТ: «27 кг/м2», «28,4 кг/м2», «30кг/м2» → «27 кг/м²»
  out = out.replace(/(\d+(?:[,.]\d+)?)\s*кг\s*\/\s*м\s*[2²]/giu, '$1 кг/м²');

  // Степени для лабов: «4,5 на 10 9 степени», «эритроциты 4,5 ×10 12 степени»
  // — все варианты с опциональным предлогом «на» и опциональным умножением.
  out = out.replace(/(\d+(?:[,.]\d+)?)\s*(?:на\s+|[*x×]\s*)?10\s*9\s*степени/giu, '$1 ×10⁹');
  out = out.replace(/(\d+(?:[,.]\d+)?)\s*(?:на\s+|[*x×]\s*)?10\s*12\s*степени/giu, '$1 ×10¹²');
  // После «×10⁹»/«×10¹²» часто идёт «на литр» / «на л» → «/л»
  out = out.replace(/(×10[⁹¹²]+)\s*(?:на\s+(?:литр|л))/giu, '$1/л');

  // Артериальное давление — две формы:
  //   «160/90 мм рт.ст.» (письменная)
  //   «160 на 90 мм рт.ст.» (разговорная, частая в диктовке)
  out = out.replace(/(\d{2,3})\s*\/\s*(\d{2,3})\s*мм\s*рт\.?\s*ст\.?/giu, '$1/$2 мм рт.ст.');
  out = out.replace(/(\d{2,3})\s+на\s+(\d{2,3})\s+мм\s*рт\.?\s*ст\.?/giu, '$1/$2 мм рт.ст.');

  return out;
}

export const GIGAAM_DENORMALIZER_VERSION = 'gigaam-denormalizer-strict-v2.1';

export type NormalizationTransformationType =
  | 'date'
  | 'decimal'
  | 'cardinal'
  | 'unit'
  | 'numeric_format'
  | 'whitespace';

export interface NormalizationTransformation {
  type: NormalizationTransformationType;
  /**
   * Смещения относятся к тексту на входе указанного этапа. Для cardinal
   * это точный span; для остальных этапов — минимальный изменившийся span.
   */
  start: number;
  end: number;
  sourceText: string;
  normalizedText: string;
  values?: number[];
}

export interface NormalizationIssue {
  code: 'ambiguous_number_sequence';
  severity: 'critical';
  stage: 'cardinal';
  start: number;
  end: number;
  sourceText: string;
  normalizedText: string;
  values: number[];
  message: string;
}

export interface GigaAMNormalizationResult {
  text: string;
  version: typeof GIGAAM_DENORMALIZER_VERSION;
  transformations: NormalizationTransformation[];
  issues: NormalizationIssue[];
}

function stageTransformation(
  before: string,
  after: string,
  type: Exclude<NormalizationTransformationType, 'cardinal'>,
): NormalizationTransformation | null {
  if (before === after) return null;
  let prefix = 0;
  const maxPrefix = Math.min(before.length, after.length);
  while (prefix < maxPrefix && before[prefix] === after[prefix]) prefix++;

  let suffix = 0;
  const maxSuffix = Math.min(before.length - prefix, after.length - prefix);
  while (
    suffix < maxSuffix
    && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  ) {
    suffix++;
  }

  return {
    type,
    start: prefix,
    end: before.length - suffix,
    sourceText: before.slice(prefix, before.length - suffix),
    normalizedText: after.slice(prefix, after.length - suffix),
  };
}

function mapCardinalIssue(issue: NumberWordNormalizationIssue): NormalizationIssue {
  return {
    ...issue,
    stage: 'cardinal',
  };
}

/**
 * Структурированная нормализация для clinical-integrity контура.
 * Неоднозначная последовательность никогда не складывается: она остаётся
 * несколькими значениями и обязательно попадает в issues.
 */
export function denormalizeDetailed(text: string): GigaAMNormalizationResult {
  if (!text) {
    return {
      text,
      version: GIGAAM_DENORMALIZER_VERSION,
      transformations: [],
      issues: [],
    };
  }

  const transformations: NormalizationTransformation[] = [];
  let current = text;

  const applyStage = (
    type: Exclude<NormalizationTransformationType, 'cardinal'>,
    transform: (value: string) => string,
  ): void => {
    const next = transform(current);
    const ledgerItem = stageTransformation(current, next, type);
    if (ledgerItem) transformations.push(ledgerItem);
    current = next;
  };

  applyStage('date', denormalizeDates);
  applyStage('decimal', denormalizeDecimals);

  const cardinals = normalizeNumberWordsDetailed(current, { preserveStandaloneOne: true });
  transformations.push(...cardinals.transformations);
  current = cardinals.text;

  applyStage('unit', denormalizeUnitsOnly);
  applyStage('numeric_format', normalizeNumericFormats);
  // Raw ASR remains byte-for-byte immutable, while the normalized transcript
  // has canonical outer boundaries. Keep both removals in the ledger instead
  // of silently trimming the evidence source.
  applyStage('whitespace', (value) => value.trimStart());
  applyStage('whitespace', (value) => value.trimEnd());

  return {
    text: current,
    version: GIGAAM_DENORMALIZER_VERSION,
    transformations,
    issues: cardinals.issues.map(mapCardinalIssue),
  };
}

// ────── Главная совместимая точка входа ─────────────────────────────────────
export function denormalize(text: string): string {
  return denormalizeDetailed(text).text;
}

export const _internals = {
  parseRusNumber,
  denormalizeDates,
  denormalizeDecimals,
  denormalizeCardinals,
  denormalizeUnitsOnly,
  normalizeNumericFormats,
};
