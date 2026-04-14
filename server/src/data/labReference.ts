/**
 * Справочник лабораторных показателей.
 *
 * Используется в system-промпте Claude для структурирования диктовки врача
 * в канонический формат: "Полное название (ИНДЕКС) значение единица (норма)".
 *
 * ПРАВИЛА:
 * - Claude включает в outpatientExams ТОЛЬКО те показатели, которые врач реально
 *   озвучил в диктовке. Не подставляет шаблон с прочерками для отсутствующих.
 * - Формат: "Лейкоциты (WBC) 7,42 10⁹/л" — с индексом в скобках и единицей.
 * - Диапазон нормы (normalRange) показывается ТОЛЬКО когда значение вне нормы
 *   или когда врач явно озвучил норму.
 * - Для показателей с пол-зависимыми нормами (Hb, Hct, КФК и т.д.) указаны оба
 *   диапазона; Claude выбирает по полу пациента из контекста диктовки.
 *
 * Список — ~70 ходовых показателей, покрывающих амбулаторную кардиологию,
 * терапию, эндокринологию. Не полный — добавлять по мере надобности.
 */

export interface LabParameter {
  /** Полное русское название (каноническая форма для вывода) */
  fullName: string;
  /** Общепринятый индекс (лат. или греч., как в бланках лабораторий) */
  index: string;
  /** Единица измерения (каноническая форма для вывода) */
  unit: string;
  /** Нормальный диапазон (строка для прямой подстановки в вывод) */
  normalRange?: string;
  /** Отдельные нормы для мужчин/женщин если отличаются */
  normalRangeByGender?: { male: string; female: string };
  /** Алиасы для распознавания — все варианты произношения / написания */
  aliases: string[];
  /** Категория (для группировки в промпте) */
  category: 'ОАК' | 'Б/Х' | 'Коагулограмма' | 'Гормоны' | 'Кардиомаркеры' | 'ОАМ' | 'Электролиты';
}

export const labReference: LabParameter[] = [
  // ─── ОАК ────────────────────────────────────────────────────────────────
  {
    fullName: 'Лейкоциты',
    index: 'WBC',
    unit: '10⁹/л',
    normalRange: '4.00 - 10.50',
    aliases: ['лейкоциты', 'л', 'wbc', 'белые кровяные тельца'],
    category: 'ОАК',
  },
  {
    fullName: 'Эритроциты',
    index: 'RBC',
    unit: '10¹²/л',
    normalRangeByGender: { male: '4.50 - 5.90', female: '4.10 - 5.70' },
    aliases: ['эритроциты', 'эр', 'rbc', 'красные кровяные тельца'],
    category: 'ОАК',
  },
  {
    fullName: 'Гемоглобин',
    index: 'HGB',
    unit: 'г/л',
    normalRangeByGender: { male: '130 - 169', female: '120 - 150' },
    aliases: ['гемоглобин', 'hb', 'hgb', 'гб', 'нв'],
    category: 'ОАК',
  },
  {
    fullName: 'Гематокрит',
    index: 'HCT',
    unit: '%',
    normalRangeByGender: { male: '40 - 50', female: '36 - 46' },
    aliases: ['гематокрит', 'ht', 'hct'],
    category: 'ОАК',
  },
  {
    fullName: 'Тромбоциты',
    index: 'PLT',
    unit: '10⁹/л',
    normalRange: '152 - 361',
    aliases: ['тромбоциты', 'тр', 'plt'],
    category: 'ОАК',
  },
  {
    fullName: 'Средний объём эритроцита',
    index: 'MCV',
    unit: 'фл',
    normalRange: '80 - 100',
    aliases: ['mcv', 'средний объём эритроцита'],
    category: 'ОАК',
  },
  {
    fullName: 'Средн. содержание Hb в эритроците',
    index: 'MCH',
    unit: 'пг',
    normalRange: '27 - 34',
    aliases: ['mch'],
    category: 'ОАК',
  },
  {
    fullName: 'Средн. концентрация Hb в эритроците',
    index: 'MCHC',
    unit: 'г/л',
    normalRange: '320 - 360',
    aliases: ['mchc'],
    category: 'ОАК',
  },
  {
    fullName: 'Нейтрофилы',
    index: 'NEUT',
    unit: '%',
    normalRange: '47 - 72',
    aliases: ['нейтрофилы', 'neut'],
    category: 'ОАК',
  },
  {
    fullName: 'Лимфоциты',
    index: 'LYMPH',
    unit: '%',
    normalRange: '19 - 37',
    aliases: ['лимфоциты', 'лф', 'lymph'],
    category: 'ОАК',
  },
  {
    fullName: 'Моноциты',
    index: 'MONO',
    unit: '%',
    normalRange: '3 - 11',
    aliases: ['моноциты', 'мон', 'mono'],
    category: 'ОАК',
  },
  {
    fullName: 'Эозинофилы',
    index: 'EO',
    unit: '%',
    normalRange: '0.5 - 5',
    aliases: ['эозинофилы', 'эоз', 'eo'],
    category: 'ОАК',
  },
  {
    fullName: 'Базофилы',
    index: 'BASO',
    unit: '%',
    normalRange: '0 - 1',
    aliases: ['базофилы', 'баз', 'baso'],
    category: 'ОАК',
  },
  {
    fullName: 'СОЭ',
    index: 'ESR',
    unit: 'мм/ч',
    normalRangeByGender: { male: '0 - 15', female: '0 - 20' },
    aliases: ['соэ', 'esr'],
    category: 'ОАК',
  },

  // ─── Биохимия ───────────────────────────────────────────────────────────
  {
    fullName: 'Аланинаминотрансфераза',
    index: 'ALT',
    unit: 'МЕ/л',
    normalRange: '0.00 - 0.52',
    aliases: ['алт', 'alt', 'аланинаминотрансфераза'],
    category: 'Б/Х',
  },
  {
    fullName: 'Аспартатаминотрансфераза',
    index: 'AST',
    unit: 'МЕ/л',
    normalRange: '0.00 - 0.53',
    aliases: ['аст', 'ast', 'аспартатаминотрансфераза'],
    category: 'Б/Х',
  },
  {
    fullName: 'Щелочная фосфатаза',
    index: 'ALP',
    unit: 'МЕ/л',
    normalRange: '40 - 150',
    aliases: ['щф', 'alp', 'щелочная фосфатаза'],
    category: 'Б/Х',
  },
  {
    fullName: 'Гамма-ГТ',
    index: 'GGT',
    unit: 'МЕ/л',
    normalRangeByGender: { male: '0 - 55', female: '0 - 38' },
    aliases: ['ггт', 'ggt', 'гамма-гт', 'гамма глутамилтрансфераза'],
    category: 'Б/Х',
  },
  {
    fullName: 'Креатинкиназа',
    index: 'CK',
    unit: 'МЕ/л',
    normalRangeByGender: { male: '30 - 200', female: '20 - 180' },
    aliases: ['кк', 'кфк', 'ck', 'креатинкиназа'],
    category: 'Б/Х',
  },
  {
    fullName: 'КФК-МВ',
    index: 'CK-MB',
    unit: 'МЕ/л',
    normalRange: '0 - 24',
    aliases: ['кфк-мв', 'ck-mb', 'кфк мв'],
    category: 'Кардиомаркеры',
  },
  {
    fullName: 'Лактатдегидрогеназа',
    index: 'LDH',
    unit: 'МЕ/л',
    normalRange: '135 - 225',
    aliases: ['лдг', 'ldh', 'лактатдегидрогеназа'],
    category: 'Б/Х',
  },
  {
    fullName: 'Креатинин',
    index: 'CREA',
    unit: 'мкмоль/л',
    normalRangeByGender: { male: '62 - 106', female: '44 - 80' },
    aliases: ['креатинин', 'crea', 'creatinine'],
    category: 'Б/Х',
  },
  {
    fullName: 'Мочевина',
    index: 'UREA',
    unit: 'ммоль/л',
    normalRange: '2.5 - 8.3',
    aliases: ['мочевина', 'urea'],
    category: 'Б/Х',
  },
  {
    fullName: 'Мочевая кислота',
    index: 'UA',
    unit: 'мкмоль/л',
    normalRangeByGender: { male: '202 - 416', female: '142 - 339' },
    aliases: ['мочевая кислота', 'ua'],
    category: 'Б/Х',
  },
  {
    fullName: 'Общий белок',
    index: 'TP',
    unit: 'г/л',
    normalRange: '64 - 83',
    aliases: ['общий белок', 'белок', 'tp', 'total protein'],
    category: 'Б/Х',
  },
  {
    fullName: 'Альбумин',
    index: 'ALB',
    unit: 'г/л',
    normalRange: '35 - 52',
    aliases: ['альбумин', 'alb'],
    category: 'Б/Х',
  },
  {
    fullName: 'Глюкоза',
    index: 'GLU',
    unit: 'ммоль/л',
    normalRange: '4.11 - 5.89',
    aliases: ['глюкоза', 'сахар', 'glu', 'glucose'],
    category: 'Б/Х',
  },
  {
    fullName: 'Общий холестерин',
    index: 'CHOL',
    unit: 'ммоль/л',
    normalRange: '3.10 - 5.20',
    aliases: ['общий холестерин', 'холестерин', 'хс', 'ох', 'chol', 'tc'],
    category: 'Б/Х',
  },
  {
    fullName: 'ЛПВП',
    index: 'HDL',
    unit: 'ммоль/л',
    normalRange: '> 1.45',
    aliases: ['лпвп', 'hdl', 'липопротеиды высокой плотности'],
    category: 'Б/Х',
  },
  {
    fullName: 'ЛПНП',
    index: 'LDL',
    unit: 'ммоль/л',
    normalRange: '< 3.37',
    aliases: ['лпнп', 'ldl', 'липопротеиды низкой плотности'],
    category: 'Б/Х',
  },
  {
    fullName: 'Триглицериды',
    index: 'TG',
    unit: 'ммоль/л',
    normalRange: '0.68 - 2.30',
    aliases: ['триглицериды', 'тг', 'tg'],
    category: 'Б/Х',
  },
  {
    fullName: 'Общий билирубин',
    index: 'TBIL',
    unit: 'мкмоль/л',
    normalRange: '0 - 22.2',
    aliases: ['общий билирубин', 'билирубин общий', 'tbil'],
    category: 'Б/Х',
  },
  {
    fullName: 'Прямой билирубин',
    index: 'DBIL',
    unit: 'мкмоль/л',
    normalRange: '0 - 5.1',
    aliases: ['прямой билирубин', 'билирубин прямой', 'dbil'],
    category: 'Б/Х',
  },
  {
    fullName: 'С-реактивный белок',
    index: 'CRP',
    unit: 'мг/л',
    normalRange: '0 - 5',
    aliases: ['срб', 'с-реактивный белок', 'c-реактивный белок', 'crp'],
    category: 'Б/Х',
  },

  // ─── Электролиты ────────────────────────────────────────────────────────
  {
    fullName: 'Натрий',
    index: 'Na',
    unit: 'ммоль/л',
    normalRange: '136 - 145',
    aliases: ['натрий', 'na'],
    category: 'Электролиты',
  },
  {
    fullName: 'Калий',
    index: 'K',
    unit: 'ммоль/л',
    normalRange: '3.5 - 5.1',
    aliases: ['калий', 'k'],
    category: 'Электролиты',
  },
  {
    fullName: 'Хлор',
    index: 'Cl',
    unit: 'ммоль/л',
    normalRange: '98 - 107',
    aliases: ['хлор', 'cl'],
    category: 'Электролиты',
  },
  {
    fullName: 'Кальций общий',
    index: 'Ca',
    unit: 'ммоль/л',
    normalRange: '2.15 - 2.55',
    aliases: ['кальций', 'кальций общий', 'ca'],
    category: 'Электролиты',
  },
  {
    fullName: 'Кальций ионизированный',
    index: 'iCa',
    unit: 'ммоль/л',
    normalRange: '1.15 - 1.33',
    aliases: ['кальций ионизированный', 'ionized calcium', 'ica'],
    category: 'Электролиты',
  },
  {
    fullName: 'Магний',
    index: 'Mg',
    unit: 'ммоль/л',
    normalRange: '0.66 - 1.07',
    aliases: ['магний', 'mg'],
    category: 'Электролиты',
  },
  {
    fullName: 'Фосфор',
    index: 'P',
    unit: 'ммоль/л',
    normalRange: '0.81 - 1.45',
    aliases: ['фосфор', 'p'],
    category: 'Электролиты',
  },
  {
    fullName: 'Железо',
    index: 'Fe',
    unit: 'мкмоль/л',
    normalRangeByGender: { male: '11.6 - 31.3', female: '9.0 - 30.4' },
    aliases: ['железо', 'fe'],
    category: 'Б/Х',
  },
  {
    fullName: 'Ферритин',
    index: 'FERR',
    unit: 'нг/мл',
    normalRangeByGender: { male: '30 - 400', female: '13 - 150' },
    aliases: ['ферритин', 'ferr'],
    category: 'Б/Х',
  },

  // ─── Коагулограмма ──────────────────────────────────────────────────────
  {
    fullName: 'Протромбиновое время',
    index: 'PT',
    unit: 'сек',
    normalRange: '11 - 15',
    aliases: ['пв', 'протромбиновое время', 'pt', 'prothrombin time'],
    category: 'Коагулограмма',
  },
  {
    fullName: 'Протромбиновый индекс (ПВ по Квику)',
    index: 'PT%',
    unit: '%',
    normalRange: '78 - 142',
    aliases: ['пти', 'протромбиновый индекс', 'пв по квику'],
    category: 'Коагулограмма',
  },
  {
    fullName: 'МНО',
    index: 'INR',
    unit: 'ед',
    normalRange: '0.8 - 1.2',
    aliases: ['мно', 'inr'],
    category: 'Коагулограмма',
  },
  {
    fullName: 'АЧТВ',
    index: 'APTT',
    unit: 'сек',
    normalRange: '25.1 - 36.5',
    aliases: ['ачтв', 'aptt'],
    category: 'Коагулограмма',
  },
  {
    fullName: 'Фибриноген',
    index: 'FIB',
    unit: 'г/л',
    normalRange: '2.0 - 4.0',
    aliases: ['фибриноген', 'fib'],
    category: 'Коагулограмма',
  },
  {
    fullName: 'Д-димер',
    index: 'D-dimer',
    unit: 'мкг/мл',
    normalRange: '< 0.5',
    aliases: ['д-димер', 'д димер', 'd-dimer', 'ddimer'],
    category: 'Коагулограмма',
  },

  // ─── Гормоны ────────────────────────────────────────────────────────────
  {
    fullName: 'Тиреотропный гормон',
    index: 'TSH',
    unit: 'мкМЕ/мл',
    normalRange: '0.35 - 5.10',
    aliases: ['ттг', 'tsh'],
    category: 'Гормоны',
  },
  {
    fullName: 'Тироксин свободный',
    index: 'FT4',
    unit: 'пмоль/л',
    normalRange: '9 - 22',
    aliases: ['т4 свободный', 'т4 св', 'ft4', 'free t4'],
    category: 'Гормоны',
  },
  {
    fullName: 'Трийодтиронин свободный',
    index: 'FT3',
    unit: 'пмоль/л',
    normalRange: '2.6 - 5.7',
    aliases: ['т3 свободный', 'т3 св', 'ft3', 'free t3'],
    category: 'Гормоны',
  },
  {
    fullName: 'Антитела к ТПО',
    index: 'anti-TPO',
    unit: 'МЕ/мл',
    normalRange: '0 - 34',
    aliases: ['ат тпо', 'антитела к тпо', 'anti-tpo'],
    category: 'Гормоны',
  },
  {
    fullName: 'Гликированный гемоглобин',
    index: 'HbA1c',
    unit: '%',
    normalRange: '< 7.0 (цель)',
    aliases: ['гликированный гемоглобин', 'гликозилированный гемоглобин', 'hba1c', 'hba 1c', 'нв а1с'],
    category: 'Гормоны',
  },
  {
    fullName: 'Инсулин',
    index: 'INS',
    unit: 'мкЕд/мл',
    normalRange: '2.6 - 24.9',
    aliases: ['инсулин', 'ins'],
    category: 'Гормоны',
  },
  {
    fullName: 'С-пептид',
    index: 'C-peptide',
    unit: 'нг/мл',
    normalRange: '0.9 - 7.1',
    aliases: ['с-пептид', 'c-peptide'],
    category: 'Гормоны',
  },
  {
    fullName: 'Кортизол',
    index: 'CORT',
    unit: 'нмоль/л',
    normalRange: '171 - 536',
    aliases: ['кортизол', 'cort'],
    category: 'Гормоны',
  },

  // ─── Кардиомаркеры ──────────────────────────────────────────────────────
  {
    fullName: 'Тропонин I',
    index: 'TnI',
    unit: 'нг/мл',
    normalRange: '< 0.04',
    aliases: ['тропонин i', 'тропонин', 'troponin i', 'tni'],
    category: 'Кардиомаркеры',
  },
  {
    fullName: 'Тропонин T',
    index: 'TnT',
    unit: 'нг/мл',
    normalRange: '< 0.014',
    aliases: ['тропонин t', 'troponin t', 'tnt'],
    category: 'Кардиомаркеры',
  },
  {
    fullName: 'BNP',
    index: 'BNP',
    unit: 'пг/мл',
    normalRange: '< 100',
    aliases: ['bnp'],
    category: 'Кардиомаркеры',
  },
  {
    fullName: 'NT-proBNP',
    index: 'NT-proBNP',
    unit: 'пг/мл',
    normalRange: '< 125',
    aliases: ['nt-probnp', 'про-бнп', 'нт про бнп'],
    category: 'Кардиомаркеры',
  },
];

/**
 * Возвращает компактный список справочника для подстановки в system-промпт
 * Claude. Формат: "Полное название (ИНДЕКС) — единица (норма)"
 */
export function buildLabReferenceForPrompt(): string {
  const byCategory: Record<string, LabParameter[]> = {};
  for (const p of labReference) {
    if (!byCategory[p.category]) byCategory[p.category] = [];
    byCategory[p.category].push(p);
  }

  const lines: string[] = [];
  for (const [category, params] of Object.entries(byCategory)) {
    lines.push(`### ${category}`);
    for (const p of params) {
      const norm = p.normalRangeByGender
        ? `М: ${p.normalRangeByGender.male} / Ж: ${p.normalRangeByGender.female}`
        : (p.normalRange || '—');
      lines.push(`- ${p.fullName} (${p.index}) ${p.unit} — норма ${norm}`);
    }
  }
  return lines.join('\n');
}
