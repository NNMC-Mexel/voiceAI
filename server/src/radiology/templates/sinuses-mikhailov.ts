// КТ придаточных пазух носа (ППН) — рабочий шаблон Михайлова (fill-in).
// Слэш-меню PDF → свитчи. Парные пазухи (право/лево) адресуются по стороне.
// Реализовано ядро (лобные, решётчатый, верхнечелюстные Л/П, клиновидная, ОМК Л/П,
// перегородка, раковины, заключение). Анатомические варианты (Keros/Kuhn/клетки Agger/Haller/
// Onodi) — добавляются такими же блоками; ждут команд-формулировок от врача.

import type { DocBlock, DocTemplate } from '../doc-model.js';

const t = (text: string) => ({ kind: 'text' as const, text });

function mucosaSwitch(name: string) {
  return {
    kind: 'switch' as const,
    sw: {
      name, default: 'norm',
      options: [
        { id: 'norm', triggers: [], nodes: [t('Слизистая оболочка не утолщена.')] },
        {
          id: 'thick', triggers: ['утолщ'],
          nodes: [
            t('Слизистая оболочка пристеночно утолщена до '),
            { kind: 'slot' as const, slot: { name: `${name}_mm`, keywords: ['до', 'утолщ'], default: '___' } },
            t(' мм.'),
          ],
        },
        { id: 'polyp', triggers: ['полип'], nodes: [t('Слизистая оболочка полиповидно изменена.')] },
      ],
    },
  };
}

function contentSwitch(name: string) {
  return {
    kind: 'switch' as const,
    sw: {
      name, default: 'norm',
      options: [
        { id: 'norm', triggers: [], nodes: [t(' Просвет свободен.')] },
        {
          id: 'cyst', triggers: ['киста', 'ретенц'],
          nodes: [
            t(' Определяется ретенционная киста диаметром '),
            { kind: 'slot' as const, slot: { name: `${name}_cyst`, keywords: ['киста', 'диаметр', 'до', 'размер'], default: '___' } },
            t(' мм.'),
          ],
        },
        { id: 'fluid', triggers: ['уровень жидкости', 'жидкость'], nodes: [t(' Определяется горизонтальный уровень жидкости.')] },
        { id: 'total', triggers: ['тотальное затемнение', 'затемнение'], nodes: [t(' Определяется тотальное затемнение.')] },
      ],
    },
  };
}

function maxillary(side: 'right' | 'left', id: string, label: string): DocBlock {
  return {
    id, label, side, anchors: ['верхнечелюстная', 'гайморова', 'верхнечелюстной'],
    appendable: true,
    nodes: [
      t('обычных размеров, пневматизация сохранена. '),
      mucosaSwitch(`${id}_mucosa`),
      contentSwitch(`${id}_content`),
      t(' Естественное соустье свободно. Костные стенки интактны.'),
    ],
  };
}

export const sinusesMikhailov: DocTemplate = {
  id: 'CT_SINUSES_MIKHAILOV',
  name: 'КТ придаточных пазух носа (шаблон Михайлова)',
  modality: 'CT',
  title: 'Компьютерная томография придаточных пазух носа',
  aliases: ['ппн', 'пазухи', 'придаточные пазухи', 'носовые пазухи'],
  conclusionBlockId: 'conclusion',
  blocks: [
    {
      id: 'frontal', label: 'Лобные пазухи', anchors: ['лобные', 'лобная'], appendable: true,
      nodes: [
        t('развиты правильно, пневматизация сохранена. '),
        mucosaSwitch('frontal_mucosa'),
        contentSwitch('frontal_content'),
        t(' Лобно-носовые каналы свободны. Костные стенки интактны.'),
      ],
    },
    {
      id: 'ethmoid', label: 'Решётчатый лабиринт', anchors: ['решетчатый', 'решётчатый', 'клетки решетчатого'], appendable: true,
      nodes: [
        t('Передние и задние клетки решётчатого лабиринта хорошо пневматизированы. '),
        mucosaSwitch('ethmoid_mucosa'),
        t(' Костные перегородки сохранены. Признаки остеита отсутствуют.'),
      ],
    },
    maxillary('right', 'maxillary_r', 'Правая верхнечелюстная пазуха'),
    maxillary('left', 'maxillary_l', 'Левая верхнечелюстная пазуха'),
    {
      id: 'sphenoid', label: 'Клиновидная пазуха', anchors: ['клиновидная', 'основная пазуха'], appendable: true,
      nodes: [
        t('Правая и левая половины хорошо пневматизированы. '),
        mucosaSwitch('sphenoid_mucosa'),
        contentSwitch('sphenoid_content'),
        t(' Соустья свободны. Межпазушная перегородка по средней линии.'),
      ],
    },
    {
      id: 'omc_r', label: 'Остиомеатальный комплекс справа', side: 'right', anchors: ['остиомеатальный', 'омк'], appendable: true,
      nodes: [
        {
          kind: 'switch',
          sw: {
            name: 'omc_r', default: 'norm',
            options: [
              { id: 'norm', triggers: [], nodes: [t('свободен.')] },
              { id: 'narrow', triggers: ['сужен', 'сужение'], nodes: [t('сужен.')] },
              { id: 'block', triggers: ['обтурирован', 'блок'], nodes: [t('обтурирован.')] },
            ],
          },
        },
      ],
    },
    {
      id: 'omc_l', label: 'Остиомеатальный комплекс слева', side: 'left', anchors: ['остиомеатальный', 'омк'], appendable: true,
      nodes: [
        {
          kind: 'switch',
          sw: {
            name: 'omc_l', default: 'norm',
            options: [
              { id: 'norm', triggers: [], nodes: [t('свободен.')] },
              { id: 'narrow', triggers: ['сужен', 'сужение'], nodes: [t('сужен.')] },
              { id: 'block', triggers: ['обтурирован', 'блок'], nodes: [t('обтурирован.')] },
            ],
          },
        },
      ],
    },
    {
      id: 'septum', label: 'Носовая перегородка', anchors: ['перегородка', 'носовая перегородка'], appendable: true,
      nodes: [
        {
          kind: 'switch',
          sw: {
            name: 'septum', default: 'norm',
            options: [
              { id: 'norm', triggers: [], nodes: [t('расположена по средней линии.')] },
              { id: 'right', triggers: ['вправо', 'правую'], nodes: [t('искривлена вправо.')] },
              { id: 'left', triggers: ['влево', 'левую'], nodes: [t('искривлена влево.')] },
              { id: 's', triggers: ['s-образн', 'эс-образн', 'с-образн'], nodes: [t('S-образно деформирована.')] },
            ],
          },
        },
      ],
    },
    {
      id: 'turbinates', label: 'Носовые раковины', anchors: ['раковины', 'раковина', 'concha'], appendable: true,
      nodes: [
        t('Нижние и средние носовые раковины обычных размеров. '),
        {
          kind: 'switch',
          sw: {
            name: 'concha', default: 'norm',
            options: [
              { id: 'norm', triggers: [], nodes: [t('Concha bullosa отсутствует.')] },
              { id: 'present', triggers: ['concha bullosa', 'конха буллеза', 'буллез', 'буллёз'], nodes: [t('Определяется concha bullosa.')] },
            ],
          },
        },
      ],
    },
    {
      id: 'conclusion', label: 'ЗАКЛЮЧЕНИЕ', anchors: ['заключение'], appendable: true,
      nodes: [t('КТ-признаки нормы придаточных пазух носа.')],
    },
  ],
};
