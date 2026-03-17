export interface ExamParameter {
  name: string;
  unit: string;
}

export interface ExamTemplate {
  id: string;
  name: string;
  shortName: string;
  aliases: string[];
  hasDate: boolean;
  parameters: ExamParameter[];
}

export const examTemplates: ExamTemplate[] = [
  {
    id: 'oak',
    name: 'ОАК',
    shortName: 'ОАК',
    aliases: ['оак', 'общий анализ крови', 'клинический анализ крови'],
    hasDate: true,
    parameters: [
      { name: 'Hb', unit: 'г/л' },
      { name: 'Эр', unit: '*10¹²/л' },
      { name: 'Л', unit: '*10⁹/л' },
      { name: 'Тр', unit: '*10⁹/л' },
      { name: 'СОЭ', unit: 'мм/ч' },
    ],
  },
  {
    id: 'biochem',
    name: 'Б/х анализ крови',
    shortName: 'Б/х',
    aliases: ['б/х', 'б х', 'биохимия', 'биохимический анализ крови', 'биохимия крови'],
    hasDate: true,
    parameters: [
      { name: 'общий белок', unit: 'г/л' },
      { name: 'креатинин', unit: 'мкмоль/л' },
      { name: 'СКФ', unit: '(по формуле CKD-EPI) мл/мин/1,73 м²' },
      { name: 'глюкоза', unit: 'ммоль/л' },
      { name: 'АЛТ', unit: 'МЕ/л' },
      { name: 'АСТ', unit: 'МЕ/л' },
      { name: 'общий билирубин', unit: 'мкмоль/л' },
      { name: 'прямой билирубин', unit: 'мкмоль/л' },
      { name: 'ХС - общий', unit: 'ммоль/л' },
      { name: 'ХС - ЛПВП', unit: 'ммоль/л' },
      { name: 'ХС ЛПНП', unit: 'ммоль/л' },
      { name: 'ТГ', unit: 'ммоль/л' },
      { name: 'мочевая кислота', unit: 'мкмоль/л' },
      { name: 'калий', unit: 'ммоль/л' },
      { name: 'натрий', unit: 'ммоль/л' },
      { name: 'железо', unit: 'мкмоль/л' },
      { name: 'ферритин', unit: 'нг/мл' },
      { name: 'КНТЖ', unit: '%' },
    ],
  },
  {
    id: 'oam',
    name: 'ОАМ',
    shortName: 'ОАМ',
    aliases: ['оам', 'общий анализ мочи', 'анализ мочи'],
    hasDate: true,
    parameters: [
      { name: 'отн. плотность', unit: '' },
      { name: 'белок', unit: 'г/л' },
      { name: 'Л', unit: 'в п/з' },
      { name: 'Эр', unit: 'в п/з' },
    ],
  },
  {
    id: 'ecg',
    name: 'ЭКГ',
    shortName: 'ЭКГ',
    aliases: ['экг', 'электрокардиограмма'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'xray_chest',
    name: 'Рентгенография органов грудной клетки',
    shortName: 'Рентген ОГК',
    aliases: ['рентген', 'рентгенография', 'рентген грудной клетки', 'рентген огк', 'рентгенография органов грудной клетки'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'echokg',
    name: 'ЭХОКГ',
    shortName: 'ЭХОКГ',
    aliases: ['эхокг', 'эхокардиография', 'узи сердца'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'holter',
    name: 'Холтеровское мониторирование ЭКГ',
    shortName: 'ХМЭКГ',
    aliases: ['холтер', 'хмэкг', 'холтеровское мониторирование', 'суточное мониторирование экг'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'smad',
    name: 'СМАД',
    shortName: 'СМАД',
    aliases: ['смад', 'суточное мониторирование ад'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'uzdg_bca',
    name: 'УЗДГ БЦА',
    shortName: 'УЗДГ БЦА',
    aliases: ['уздг бца', 'уздг', 'узи сосудов шеи', 'дуплекс бца'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'uzi_obp',
    name: 'УЗИ ОБП',
    shortName: 'УЗИ ОБП',
    aliases: ['узи обп', 'узи брюшной полости', 'узи органов брюшной полости'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'uzi_kidneys',
    name: 'УЗИ почек',
    shortName: 'УЗИ почек',
    aliases: ['узи почек'],
    hasDate: true,
    parameters: [],
  },
];

/**
 * Маппинг русских голосовых названий параметров → имя параметра в шаблоне.
 * Ключ — id шаблона, значение — массив пар [regex для голосового ввода, имя параметра в template.parameters].
 */
export const PARAM_VOICE_ALIASES: Record<string, Array<{ pattern: RegExp; paramName: string }>> = {
  oak: [
    { pattern: /гемоглобин/iu, paramName: 'Hb' },
    { pattern: /эритроцит/iu, paramName: 'Эр' },
    { pattern: /лейкоцит/iu, paramName: 'Л' },
    { pattern: /тромбоцит/iu, paramName: 'Тр' },
    { pattern: /СОЭ/u, paramName: 'СОЭ' },
    { pattern: /соэ/iu, paramName: 'СОЭ' },
  ],
  biochem: [
    { pattern: /общ\S*\s+бел\S*/iu, paramName: 'общий белок' },
    { pattern: /креатинин/iu, paramName: 'креатинин' },
    { pattern: /СКФ/u, paramName: 'СКФ' },
    { pattern: /скф/iu, paramName: 'СКФ' },
    { pattern: /скорость\s+клубочковой/iu, paramName: 'СКФ' },
    { pattern: /глюкоз/iu, paramName: 'глюкоза' },
    { pattern: /сахар\s+кров/iu, paramName: 'глюкоза' },
    { pattern: /АЛТ/u, paramName: 'АЛТ' },
    { pattern: /алт/iu, paramName: 'АЛТ' },
    { pattern: /аланин\S*\s*аминотрансфераз/iu, paramName: 'АЛТ' },
    { pattern: /АСТ/u, paramName: 'АСТ' },
    { pattern: /аст/iu, paramName: 'АСТ' },
    { pattern: /аспартат\S*\s*аминотрансфераз/iu, paramName: 'АСТ' },
    { pattern: /общ\S*\s+билирубин/iu, paramName: 'общий билирубин' },
    { pattern: /прям\S*\s+билирубин/iu, paramName: 'прямой билирубин' },
    { pattern: /билирубин\s+общ/iu, paramName: 'общий билирубин' },
    { pattern: /билирубин\s+прям/iu, paramName: 'прямой билирубин' },
    { pattern: /холестерин\s+общ/iu, paramName: 'ХС - общий' },
    { pattern: /общ\S*\s+холестерин/iu, paramName: 'ХС - общий' },
    { pattern: /ХС\s*[-–—]?\s*общ/iu, paramName: 'ХС - общий' },
    // "холестерин 5,6" без уточнения — считаем общим
    { pattern: /(?<![а-яё])холестерин(?!\s+(?:ЛПВП|ЛПНП|липопротеид))/iu, paramName: 'ХС - общий' },
    { pattern: /ЛПВП/iu, paramName: 'ХС - ЛПВП' },
    { pattern: /ЛПНП/iu, paramName: 'ХС ЛПНП' },
    { pattern: /триглицерид/iu, paramName: 'ТГ' },
    { pattern: /(?<![а-яё])ТГ(?![а-яё])/u, paramName: 'ТГ' },
    { pattern: /мочев\S*\s+кислот/iu, paramName: 'мочевая кислота' },
    { pattern: /калий/iu, paramName: 'калий' },
    { pattern: /натрий/iu, paramName: 'натрий' },
    { pattern: /железо/iu, paramName: 'железо' },
    { pattern: /ферритин/iu, paramName: 'ферритин' },
    { pattern: /КНТЖ/iu, paramName: 'КНТЖ' },
    { pattern: /коэффициент\s+насыщения\s+трансферрина/iu, paramName: 'КНТЖ' },
  ],
  oam: [
    { pattern: /(?:относительн\S*\s+)?плотност/iu, paramName: 'отн. плотность' },
    { pattern: /удельн\S*\s+вес/iu, paramName: 'отн. плотность' },
    { pattern: /белок/iu, paramName: 'белок' },
    { pattern: /лейкоцит/iu, paramName: 'Л' },
    { pattern: /эритроцит/iu, paramName: 'Эр' },
  ],
};

/**
 * Извлекает значения параметров из блока текста, используя голосовые алиасы.
 * Возвращает Record<paramName, value> для использования с formatExamLine.
 */
export function parseExamValuesFromText(
  templateId: string,
  text: string
): Record<string, string> {
  const aliases = PARAM_VOICE_ALIASES[templateId];
  if (!aliases) return {};

  const values: Record<string, string> = {};

  for (const { pattern, paramName } of aliases) {
    if (values[paramName]) continue; // уже найден

    // Ищем паттерн + числовое значение после него
    // Формат: "Гемоглобин 149" или "Гемоглобин - 149" или "гемоглобин — 149,5"
    const fullPattern = new RegExp(
      pattern.source + '\\S*\\s*[-–—:]?\\s*([\\d][\\d.,]*)',
      'iu'
    );
    const match = text.match(fullPattern);
    if (match) {
      values[paramName] = match[1].replace(/,$/g, '');
    }
  }

  return values;
}

/**
 * Извлекает дату из текста обследования.
 * Поддерживает: "от 4 марта 2026 года", "от 04.03.2026", "от 4.03.26"
 */
export function parseExamDate(text: string): string | undefined {
  // "от DD месяц YYYY года"
  const longDate = text.match(
    /от\s+(\d{1,2}\s+(?:январ\S*|феврал\S*|март\S*|апрел\S*|ма[яй]\S*|июн\S*|июл\S*|август\S*|сентябр\S*|октябр\S*|ноябр\S*|декабр\S*)\s+\d{4}\s*(?:год\S*)?)/iu
  );
  if (longDate) return longDate[1].trim().replace(/[.:,;]+$/g, '');

  // "от DD.MM.YYYY" или "от DD.MM.YY"
  const numDate = text.match(/от\s+(\d{1,2}[.,]\d{2}[.,]\d{2,4})\s*(?:г\.?|год\S*)?/iu);
  if (numDate) return numDate[1];

  return undefined;
}

/**
 * Форматирует шаблон обследования в строку.
 */
export function formatExamLine(
  template: ExamTemplate,
  values?: Record<string, string>,
  date?: string
): string {
  const datePart = template.hasDate
    ? ` от ${date || '_.____г.'}`
    : '';

  if (template.parameters.length === 0) {
    const desc = values?.['description'] || '';
    return `${template.name}${datePart}:${desc ? ' ' + desc : ''}`;
  }

  // Если есть значения — выводим только параметры с данными + единицы измерения
  const hasAnyValue = values && Object.keys(values).length > 0;

  const params = hasAnyValue
    ? template.parameters
        .filter((p) => values[p.name]) // только параметры с значениями
        .map((p) => {
          const unitStr = p.unit ? ` ${p.unit}` : '';
          return `${p.name} - ${values[p.name]}${unitStr}`;
        })
    : template.parameters.map((p) => {
        const unitStr = p.unit ? ` ${p.unit}` : '';
        return `${p.name} - ${unitStr}`;
      });

  return `${template.name}${datePart}: ${params.join(', ')}.`;
}

/**
 * Генерирует строку с примером формата для LLM промпта.
 */
export function getExamFormatExample(): string {
  return examTemplates
    .slice(0, 3) // ОАК, Б/х, ОАМ — основные с параметрами
    .map((t, i) => `${i + 1}. ${formatExamLine(t)}`)
    .join('\n');
}

/**
 * Ищет шаблон обследования по тексту (аббревиатуре или полному названию).
 */
export function findExamTemplate(text: string): ExamTemplate | undefined {
  const lower = text.toLowerCase().trim();
  return examTemplates.find((t) =>
    t.aliases.some((a) => lower.includes(a)) ||
    t.name.toLowerCase() === lower ||
    t.shortName.toLowerCase() === lower
  );
}
