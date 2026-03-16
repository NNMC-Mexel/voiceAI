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

  const params = template.parameters.map((p) => {
    const val = values?.[p.name] || '';
    const unitStr = p.unit ? ` ${p.unit}` : '';
    return `${p.name} - ${val}${unitStr}`;
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
