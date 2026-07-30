// КТ ОБП — рабочий шаблон врача Михайлова (fill-in модель).
// Текст нормы дословно из PDF; врач диктует по нему, меняются только значения.
// Слоты: КВР печени, плотность, холедох, поджелудочная, размеры селезёнки (+авто-СИ),
// аорта/чревный/портальная. Свитчи: содержимое ЖП, конкременты почек. Дописки: лёгкие, заключение.

import type { DocTemplate, SlotValues } from '../doc-model.js';

const t = (text: string) => ({ kind: 'text' as const, text });

// Селезёночный индекс ≈ произведение трёх размеров (как диктует врач).
function splenicIndex(v: SlotValues): string {
  const d = v.spleen;
  if (d && d.length === 3) return String(Math.round(d[0] * d[1] * d[2]));
  return '___';
}

export const abdomenMikhailov: DocTemplate = {
  id: 'CT_ABDOMEN_MIKHAILOV',
  name: 'КТ ОБП (шаблон Михайлова)',
  modality: 'CT',
  title: 'Компьютерная томография органов брюшной полости и забрюшинного пространства с внутривенным контрастным усилением',
  aliases: ['обп', 'кт обп', 'брюшная полость', 'живот'],
  fieldRoutingVersion: 'ct-abdomen-field-routing-v1',
  conclusionBlockId: 'conclusion',
  blocks: [
    {
      id: 'liver',
      label: 'Печень',
      anchors: ['печень'],
      nodes: [
        t('не увеличена в объеме (КВР '),
        {
          kind: 'slot',
          slot: {
            name: 'kvr',
            fieldId: 'liver.kvr',
            keywords: ['квр', 'к в р', 'ка вэ эр', 'кранио-вертебральн'],
            routingAliases: ['квр', 'к в р', 'ка вэ эр'],
            unit: 'mm',
            allowImplicitUnit: true,
            defaultKind: 'placeholder',
            requiredForApproval: true,
            validation: {
              ruleId: 'ct-abdomen-liver-kvr-v1',
              minExclusive: 0,
              maxInclusive: 400,
              templateClaimMinInclusive: 80,
              templateClaimMaxInclusive: 160,
            },
            default: '___',
          },
        },
        t(' мм), соотношение сегментов не нарушено, паренхима однородная, плотность в пределах нормы — средние значения '),
        {
          kind: 'slot',
          slot: {
            name: 'density',
            fieldId: 'liver.density',
            keywords: ['плотность'],
            unit: 'HU',
            allowImplicitUnit: true,
            signMode: 'always',
            defaultKind: 'clinical_default',
            validation: {
              ruleId: 'ct-liver-density-hu-v1',
              minInclusive: -1000,
              maxInclusive: 3000,
              templateClaimMinInclusive: 40,
              templateClaimMaxInclusive: 80,
            },
            default: '+60',
          },
        },
        t(' HU. Внутри- и внепеченочные желчные протоки не расширены.'),
      ],
    },
    {
      id: 'gallbladder',
      label: 'Желчный пузырь',
      anchors: ['желчный пузырь', 'желчный'],
      nodes: [
        {
          kind: 'switch',
          sw: {
            name: 'gb', fieldId: 'gallbladder.content', default: 'norm',
            options: [
              {
                id: 'norm',
                triggers: [
                  'конкрементов не выявлено',
                  'конкременты не визуализируются',
                  'конкремент не выявлен',
                  'конкременты не выявлены',
                  'конкремент отсутствует',
                  'конкременты отсутствуют',
                  'без конкрементов',
                  'камней нет',
                ],
                nodes: [t('с гомогенным содержимым, стенки не утолщены.')],
              },
              {
                id: 'stones',
                triggers: ['конкремент', 'камни', 'камень'],
                excludes: [
                  'конкрементов не выявлено',
                  'конкременты не визуализируются',
                  'конкремент не выявлен',
                  'конкременты не выявлены',
                  'конкремент отсутствует',
                  'конкременты отсутствуют',
                  'без конкрементов',
                  'камней нет',
                ],
                nodes: [t('в просвете определяются конкременты, стенки не утолщены, паравезикальная клетчатка не изменена.')],
              },
            ],
          },
        },
      ],
    },
    {
      id: 'choledoch',
      label: 'Холедох',
      anchors: ['холедох'],
      nodes: [
        t('до '),
        {
          kind: 'slot',
          slot: {
            name: 'choledoch',
            fieldId: 'choledoch.diameter',
            keywords: ['холедох'],
            decimals: 1,
            unit: 'mm',
            allowImplicitUnit: true,
            defaultKind: 'clinical_default',
            validation: {
              ruleId: 'ct-diameter-mm-positive-v1',
              minExclusive: 0,
              maxInclusive: 100,
              templateClaimMaxInclusive: 10,
            },
            default: '5,5',
          },
        },
        t(' мм — стенки не утолщены, конкрементов не выявлено.'),
      ],
    },
    {
      id: 'pancreas',
      label: 'Поджелудочная железа',
      anchors: ['поджелудочная', 'поджелудочной'],
      nodes: [
        t('структура однородная, частично дольчатая, объем не уменьшен (поперечник головки до '),
        {
          kind: 'slot',
          slot: {
            name: 'head',
            fieldId: 'pancreas.head',
            keywords: ['головк'],
            decimals: 1,
            unit: 'mm',
            allowImplicitUnit: true,
            defaultKind: 'clinical_default',
            validation: {
              ruleId: 'ct-diameter-mm-positive-v1',
              minExclusive: 0,
              maxInclusive: 200,
              templateClaimMaxInclusive: 40,
            },
            default: '24,0',
          },
        },
        t(' мм, тело и хвост до '),
        {
          kind: 'slot',
          slot: {
            name: 'tail',
            fieldId: 'pancreas.tail',
            keywords: ['хвост', 'тело'],
            decimals: 1,
            unit: 'mm',
            allowImplicitUnit: true,
            defaultKind: 'clinical_default',
            validation: {
              ruleId: 'ct-diameter-mm-positive-v1',
              minExclusive: 0,
              maxInclusive: 200,
              templateClaimMaxInclusive: 35,
            },
            default: '16,5',
          },
        },
        t(' мм), панкреатический проток не расширен — патологических образований не выявлено.'),
      ],
    },
    {
      id: 'spleen',
      label: 'Селезёнка',
      anchors: ['селезёнка', 'селезенка'],
      nodes: [
        t('не увеличена ('),
        {
          kind: 'slot',
          slot: {
            name: 'spleen',
            fieldId: 'spleen.dimensions',
            keywords: ['селезенка', 'размеры'],
            arity: 3,
            decimals: 1,
            join: 'х',
            unit: 'mm',
            allowImplicitUnit: true,
            implicitUnit: 'cm',
            defaultKind: 'placeholder',
            requiredForApproval: true,
            validation: {
              ruleId: 'ct-dimensions-mm-positive-v1',
              minExclusive: 0,
              maxInclusive: 1000,
              aggregate: {
                ruleId: 'splenic-index-normal-claim-v1',
                operation: 'product',
                divisor: 1000,
                maxInclusive: 480,
              },
            },
            default: '__х__х__',
          },
        },
        t(' мм – СИ ≈ '),
        {
          kind: 'derived',
          name: 'si',
          fieldId: 'spleen.index',
          dependsOn: ['spleen.dimensions'],
          formulaVersion: 'splenic-index-mm-to-cm3-v2',
          outputDivisor: 1000,
          compute: splenicIndex,
        },
        t(') — структура в пределах нормы, без очаговых изменений. Селезеночные сосуды типичного строения без кальциноза и аневризмальных расширений.'),
      ],
    },
    {
      id: 'adrenals',
      label: 'Надпочечники',
      anchors: ['надпочечник'],
      nodes: [t('форма и размеры не изменены, объёмных образований нет.')],
    },
    {
      id: 'kidneys',
      label: 'Почки',
      anchors: ['почки', 'почка', 'почек'],
      nodes: [
        t('расположены типично, однородной структуры, без истончения паренхимы и снижения кортико-медуллярной дифференциации. Строение сосудистой ножки почки классическое. Чашечно-лоханочная система не расширена, '),
        {
          kind: 'switch',
          sw: {
            name: 'kidneyStones', fieldId: 'kidneys.stone_status', default: 'norm',
            options: [
              {
                id: 'norm',
                triggers: [
                  'конкрементов не выявлено',
                  'конкременты не визуализируются',
                  'конкремент не выявлен',
                  'конкременты не выявлены',
                  'конкремент отсутствует',
                  'конкременты отсутствуют',
                  'без конкрементов',
                  'камней нет',
                ],
                nodes: [t('конкременты не визуализируются.')],
              },
              {
                id: 'stones',
                triggers: ['конкремент', 'камень', 'камни'],
                excludes: [
                  'конкрементов не выявлено',
                  'конкременты не визуализируются',
                  'конкремент не выявлен',
                  'конкременты не выявлены',
                  'конкремент отсутствует',
                  'конкременты отсутствуют',
                  'без конкрементов',
                  'камней нет',
                ],
                nodes: [
                  t('определяется конкремент до '),
                  {
                    kind: 'slot',
                    slot: {
                      name: 'kidneyStoneSize',
                      fieldId: 'kidneys.stone_size',
                      keywords: ['конкремент', 'камень', 'размер'],
                      unit: 'mm',
                      allowImplicitUnit: true,
                      defaultKind: 'placeholder',
                      requiredForApproval: true,
                      validation: {
                        ruleId: 'ct-stone-size-mm-positive-v1',
                        minExclusive: 0,
                        maxInclusive: 100,
                      },
                      default: '___',
                    },
                  },
                  t(' мм.'),
                ],
              },
            ],
          },
        },
      ],
    },
    {
      id: 'lymph_hilum',
      label: 'Лимфатические узлы ворот почек',
      // Одиночное «в воротах» не является достаточным lymph-якорем: это может
      // быть добавочная долька селезёнки, образование печени и т. п.
      anchors: ['ворот почек', 'лимфатические узлы', 'лимфоузлы'],
      nodes: [t('не увеличены.')],
    },
    {
      id: 'stomach',
      label: 'Желудок',
      anchors: ['желудок'],
      nodes: [t('расположен типично, стенки равномерной толщины (≤10 мм), без локальных утолщений и деформаций. Перистальтика сохранена. Перигастральная клетчатка не инфильтрирована. Лимфатические узлы парагастральные и чревные не увеличены.')],
    },
    {
      id: 'bowel',
      label: 'Кишечник',
      anchors: [
        'кишечник',
        'кишка',
        'ободочн',
        'сигмовидн',
        'нисходящ',
        'восходящ',
        'поперечн',
        'тонкая кишка',
      ],
      routingRules: [
        {
          id: 'bowel-wall-finding',
          phrases: ['утолщение стенок', 'утолщени'],
        },
      ],
      nodes: [t('петли тонкого и толстого кишечника не расширены, патологических утолщений стенок не выявлено, признаки кишечной непроходимости отсутствуют. Свободная жидкость и газ в брюшной полости не выявлены.')],
    },
    {
      id: 'celiac_trunk',
      label: 'Чревный ствол',
      anchors: ['чревный ствол', 'чревного ствола'],
      routingRules: [
        {
          id: 'celiac-compression-study',
          sticky: true,
          phrases: [
            'в артериальную фазу',
            'чредного ствола',
            'члевного ствола',
            'членного стола',
            'экстравазальн',
            'срединной дугообразной связки',
            'средины дугообразной связки',
            'аортального устья',
            'аартального устья',
          ],
        },
      ],
      nodes: [t('без признаков гемодинамически значимого стеноза и экстравазальной компрессии.')],
    },
    {
      id: 'vessels',
      label: 'Сосудистые структуры',
      anchors: ['сосуд', 'аорта', 'портальн', 'воротн'],
      nodes: [
        t('на уровне исследования без признаков тромбоза, патологических сужений и/или расширений. Брюшная аорта до '),
        {
          kind: 'slot',
          slot: {
            name: 'aorta',
            fieldId: 'vessels.aorta',
            keywords: ['аорта'],
            decimals: 1,
            unit: 'mm',
            allowImplicitUnit: true,
            defaultKind: 'clinical_default',
            validation: {
              ruleId: 'ct-abdominal-aorta-normal-claim-v1',
              minExclusive: 0,
              maxInclusive: 100,
              templateClaimMaxInclusive: 30,
            },
            default: '16,0',
          },
        },
        t(' мм. Чревный ствол до '),
        {
          kind: 'slot',
          slot: {
            name: 'celiac',
            fieldId: 'vessels.celiac',
            keywords: ['чревн'],
            decimals: 1,
            unit: 'mm',
            allowImplicitUnit: true,
            defaultKind: 'clinical_default',
            validation: {
              ruleId: 'ct-celiac-diameter-normal-claim-v1',
              minExclusive: 0,
              maxInclusive: 100,
              templateClaimMaxInclusive: 12,
            },
            default: '8,0',
          },
        },
        t(' мм. Портальная вена не расширена (до '),
        {
          kind: 'slot',
          slot: {
            name: 'portal',
            fieldId: 'vessels.portal',
            keywords: ['портальн', 'воротн'],
            decimals: 1,
            unit: 'mm',
            allowImplicitUnit: true,
            defaultKind: 'clinical_default',
            validation: {
              ruleId: 'ct-portal-vein-normal-claim-v1',
              minExclusive: 0,
              maxInclusive: 100,
              templateClaimMaxInclusive: 16,
            },
            default: '12,0',
          },
        },
        t(' мм), без повышенного числа желудочных коллатералей.'),
      ],
    },
    {
      id: 'skeleton',
      label: 'Скелет',
      anchors: ['скелет', 'кости', 'позвоночн'],
      routingRules: [
        {
          id: 'degenerative-spine',
          phrases: ['дегенеративно дистрофические', 'дегенеративн'],
        },
      ],
      nodes: [t('остеодеструктивных изменений на уровне исследования не выявлено.')],
    },
    {
      id: 'pelvis',
      label: 'Органы малого таза',
      anchors: ['органы малого таза', 'малый таз', 'малого таза', 'малом тазу', 'мочевой пузырь', 'матка'],
      routingRules: [
        {
          id: 'pelvic-location',
          phrases: ['в малом тазу'],
        },
      ],
      nodes: [t('Мочевой пузырь умеренного наполнения без явных экзофитных образований и конкрементов, наружный контур ровный. Матка не увеличена, контуры чёткие, ровные. Придатки чётко дифференцируются, в объеме не изменены.')],
    },
    {
      id: 'lung_bases',
      label: 'Базальные отделы легких',
      anchors: ['базальн', 'легкие', 'лёгкие', 'плевральн'],
      appendable: true,
      nodes: [t('без очаговых и инфильтративных изменений. В обеих плевральных полостях выпота не выявлено.')],
    },
    {
      id: 'conclusion',
      label: 'Заключение',
      anchors: ['заключение'],
      appendable: true,
      nodes: [t('Патологических изменений органов брюшной полости и забрюшинного пространства не выявлено.')],
    },
  ],
};
