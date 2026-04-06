export interface ExamParameter {
  name: string;       // Название параметра (Hb, Эр, Л, etc.)
  unit: string;       // Единица измерения (г/л, *10¹²/л, etc.)
}

export interface ExamTemplate {
  id: string;
  name: string;                    // Полное название (Общий анализ крови)
  shortName: string;               // Сокращение (ОАК)
  aliases: string[];               // Все варианты названий для распознавания
  hasDate: boolean;                // Нужна ли дата
  parameters: ExamParameter[];     // Параметры с единицами измерения
}

export const examTemplates: ExamTemplate[] = [
  {
    id: 'oak',
    name: 'ОАК',
    shortName: 'ОАК',
    aliases: ['оак', 'общий анализ крови', 'клинический анализ крови', 'кровь общий'],
    hasDate: true,
    parameters: [
      { name: 'Hb', unit: 'г/л' },
      { name: 'Эр', unit: '*10¹²/л' },
      { name: 'Тр', unit: '*10⁹/л' },
      { name: 'Л', unit: '*10⁹/л' },
      { name: 'нейтрофилы', unit: '%' },
      { name: 'эозинофилы', unit: '%' },
      { name: 'базофилы', unit: '%' },
      { name: 'моноциты', unit: '%' },
      { name: 'лимфоциты', unit: '%' },
      { name: 'СОЭ', unit: 'мм/ч' },
    ],
  },
  {
    id: 'biochem',
    name: 'Б/х анализ крови',
    shortName: 'Б/х',
    aliases: ['б/х', 'б х', 'биохимия', 'биохимический анализ крови', 'биохимия крови', 'бх анализ'],
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
      { name: 'ЛЖСС', unit: 'мкмоль/л' },
      { name: 'ОЖСС', unit: 'мкмоль/л' },
      { name: 'трансферрин', unit: 'г/л' },
      { name: 'TSat', unit: '%' },
      { name: 'витамин В12', unit: 'пг/мл' },
      { name: 'витамин D', unit: 'нмоль/л' },
      { name: 'фолиевая кислота', unit: 'нг/мл' },
      { name: 'HbA1c', unit: '%' },
      { name: 'С-реактивный белок', unit: 'мг/л' },
      { name: 'мочевина', unit: 'ммоль/л' },
      { name: 'магний', unit: 'ммоль/л' },
      { name: 'ГГТП', unit: 'МЕ/л' },
      { name: 'ЛДГ', unit: 'МЕ/л' },
      { name: 'ЩФ', unit: 'МЕ/л' },
      { name: 'фибриноген', unit: 'г/л' },
    ],
  },
  {
    id: 'oam',
    name: 'ОАМ',
    shortName: 'ОАМ',
    aliases: ['оам', 'общий анализ мочи', 'моча общий', 'анализ мочи'],
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
    aliases: ['экг', 'электрокардиограмма', 'электрокардиография', 'кардиограмма'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'xray_chest',
    name: 'Рентгенография органов грудной клетки',
    shortName: 'Рентген ОГК',
    aliases: ['рентген', 'рентгенография', 'рентген грудной клетки', 'рентген огк', 'рентгенография органов грудной клетки', 'р огк', 'рентгенография огк'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'echokg',
    name: 'ЭХОКГ',
    shortName: 'ЭХОКГ',
    aliases: ['эхокг', 'эхокардиография', 'эхо кг', 'узи сердца'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'holter',
    name: 'Холтеровское мониторирование ЭКГ',
    shortName: 'ХМЭКГ',
    aliases: ['холтер', 'хмэкг', 'холтеровское мониторирование', 'суточное мониторирование экг', 'суточный холтер', 'холтер экг'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'smad',
    name: 'СМАД',
    shortName: 'СМАД',
    aliases: ['смад', 'суточное мониторирование ад', 'суточный мониторинг давления'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'uzdg_bca',
    name: 'УЗДГ БЦА',
    shortName: 'УЗДГ БЦА',
    aliases: ['уздг бца', 'уздг', 'узи сосудов шеи', 'дуплекс бца', 'дуплексное сканирование бца'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'uzi_obp',
    name: 'УЗИ ОБП',
    shortName: 'УЗИ ОБП',
    aliases: ['узи обп', 'узи брюшной полости', 'узи органов брюшной полости', 'ультразвуковое исследование обп'],
    hasDate: true,
    parameters: [],
  },
  {
    id: 'uzi_kidneys',
    name: 'УЗИ почек',
    shortName: 'УЗИ почек',
    aliases: ['узи почек', 'ультразвуковое исследование почек'],
    hasDate: true,
    parameters: [],
  },
];

/**
 * Формирует строку шаблона обследования.
 * Если есть значения — подставляет их, если нет — оставляет пустые поля с единицами.
 * @param template - шаблон обследования
 * @param values - объект { параметр: значение } (опционально)
 * @param date - дата обследования (опционально)
 */
export function formatExamTemplate(
  template: ExamTemplate,
  values?: Record<string, string>,
  date?: string
): string {
  const datePart = template.hasDate
    ? ` от ${date || '_.____г.'}`
    : '';

  if (template.parameters.length === 0) {
    const description = values?.['description'] || '';
    return `${template.name}${datePart}:${description ? ' ' + description : ''}`;
  }

  const params = template.parameters.map((p) => {
    const val = values?.[p.name] || '';
    const unitStr = p.unit ? ` ${p.unit}` : '';
    return `${p.name} - ${val}${unitStr}`;
  });

  return `${template.name}${datePart}: ${params.join(', ')}.`;
}

/**
 * Формирует полный нумерованный список всех обследований (пустой шаблон).
 */
export function formatAllExamTemplates(): string {
  return examTemplates
    .map((t, i) => `${i + 1}. ${formatExamTemplate(t)}`)
    .join('\n');
}
