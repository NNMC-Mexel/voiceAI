// Числовой токенизатор голосовых команд рентгенолога.
// Детерминистично, без LLM. Две задачи:
//   1. converttNumberWords — «восемьдесят на десять» → «80 на 10»
//      (страховка на случай, если ASR/денормализация оставили слова-числа).
//   2. extractNumbers — вытащить все числа с привязкой к предшествующему
//      ключевому слову («стенка 5» → number 5, precededBy 'стенка').
//
// Команды из ТЗ используют цифры («плотность 56», «размеры 14 7 5»),
// поэтому основной путь — цифры; слова-числа поддержаны для устойчивости.

const UNITS: Record<string, number> = {
  ноль: 0, один: 1, одна: 1, одно: 1, два: 2, две: 2, три: 3, четыре: 4,
  пять: 5, шесть: 6, семь: 7, восемь: 8, девять: 9,
};
const TEENS: Record<string, number> = {
  десять: 10, одиннадцать: 11, двенадцать: 12, тринадцать: 13, четырнадцать: 14,
  пятнадцать: 15, шестнадцать: 16, семнадцать: 17, восемнадцать: 18, девятнадцать: 19,
};
const TENS: Record<string, number> = {
  двадцать: 20, тридцать: 30, сорок: 40, пятьдесят: 50, шестьдесят: 60,
  семьдесят: 70, восемьдесят: 80, девяносто: 90,
};
const HUNDREDS: Record<string, number> = {
  сто: 100, двести: 200, триста: 300, четыреста: 400, пятьсот: 500,
  шестьсот: 600, семьсот: 700, восемьсот: 800, девятьсот: 900,
};

// Слова, которые НЕ являются числами, но разделяют/связывают их — пропускаем.
const NUMBER_WORD = new Set([
  ...Object.keys(UNITS), ...Object.keys(TEENS), ...Object.keys(TENS), ...Object.keys(HUNDREDS),
]);

/**
 * Заменяет последовательности числительных на цифры.
 * «восемьдесят на десять» → «80 на 10», «сто пятьдесят шесть» → «156».
 * Нормализует пробелы до одиночных (downstream всё равно схлопывает).
 */
export function convertNumberWords(input: string): string {
  const tokens = input.split(/\s+/).filter(Boolean);
  const out: string[] = [];
  let acc = 0;         // накопитель текущего числа
  let has = false;     // есть ли что накапливать
  const flush = () => { if (has) { out.push(String(acc)); acc = 0; has = false; } };
  for (const tok of tokens) {
    const w = tok.toLowerCase().replace(/[.,;:]+$/, '');
    if (w in HUNDREDS) { acc += HUNDREDS[w]; has = true; continue; }
    if (w in TENS) { acc += TENS[w]; has = true; continue; }
    if (w in TEENS) { acc += TEENS[w]; has = true; continue; }
    if (w in UNITS) { acc += UNITS[w]; has = true; continue; }
    flush();
    out.push(tok);
  }
  flush();
  return out.join(' ');
}

export interface NumberToken {
  value: number;       // числовое значение (десятичные через точку внутри)
  raw: string;         // как в тексте
  start: number;
  end: number;
  precededBy: string;  // ближайшее слово-тег перед числом (lowercase, без пунктуации)
}

const WORD_RE = /[a-zа-яё]+/i;

/**
 * Извлекает все числа команды с привязкой к предшествующему слову-тегу.
 * «холедох 11 мм камень 6 мм» → [{value:11,precededBy:'холедох'},{value:6,precededBy:'камень'}]
 * Игнорирует слова-числа (они уже переведены convertNumberWords).
 */
export function extractNumbers(text: string): NumberToken[] {
  const result: NumberToken[] = [];
  const numRe = /\d+(?:[.,]\d+)?/g;
  let m: RegExpExecArray | null;
  while ((m = numRe.exec(text)) !== null) {
    const raw = m[0];
    const value = parseFloat(raw.replace(',', '.'));
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
    result.push({ value, raw, start: m.index, end: m.index + raw.length, precededBy });
  }
  return result;
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
