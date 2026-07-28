// Модель «шаблон-как-документ» (fill-in) — под рабочие шаблоны врача Михайлова.
// Шаблон = готовый нормальный текст с пропусками (слоты) и слэш-меню (свитчи).
// Врач диктует по шаблону: система меняет ТОЛЬКО значения слотов, переключает
// варианты и дописывает добавленное. Всё остальное остаётся текстом нормы.

export interface SlotDef {
  name: string;
  keywords: string[];        // теги-стемы перед числом: «плотность 62» → 'плотность'
  arity?: number;            // >1 — набор («12 на 6 на 5»); хранится как number[]
  decimals?: number;         // формат вывода (0 — целое, 1 — «14,0»)
  join?: string;             // разделитель для arity>1 (напр. 'х')
  default: string;           // печатается, если врач не задал (норм. значение или '___')
}

export type DocNode =
  | { kind: 'text'; text: string }
  | { kind: 'slot'; slot: SlotDef }
  | { kind: 'derived'; name: string; compute: (v: SlotValues) => string };

export interface SwitchOption {
  id: string;
  triggers: string[];        // фразы выбора отклонения; у нормальной опции — []
  nodes: DocNode[];          // текст опции (может содержать слоты)
}

export interface SwitchDef {
  name: string;
  default: string;           // id нормальной опции
  options: SwitchOption[];
}

export type BlockNode = DocNode | { kind: 'switch'; sw: SwitchDef };

export interface DocBlock {
  id: string;
  label: string;             // «Печень», «1. Полости черепа…»
  anchors: string[];         // якоря команды для адресации блока
  /**
   * Более узкие правила маршрутизации сплошной диктовки. Они не участвуют в
   * fill-in командах DocEngine и поэтому не могут случайно переключить текущий
   * блок при ручном редактировании.
   */
  routingRules?: {
    id: string;
    phrases: string[];
    /** Не разрешать вложенным organ-якорям разорвать этот клинический фрагмент. */
    sticky?: boolean;
  }[];
  nodes: BlockNode[];
  appendable?: boolean;      // может принимать дописанные врачом фразы
  side?: 'right' | 'left';   // для парных структур (правая/левая пазуха)
}

export interface DocTemplate {
  id: string;
  name: string;
  modality: 'CT' | 'US' | 'MRI' | 'XR';
  title: string;             // шапка исследования (печатается сверху)
  aliases: string[];
  gendered?: boolean;        // есть ли муж/жен-варианты секций
  blocks: DocBlock[];
  conclusionBlockId?: string; // какой блок — «Заключение» (для дописывания)
}

export type SlotValues = Record<string, number[] | undefined>;

export interface DocState {
  slots: SlotValues;                 // arity=1 хранится как [v]
  switches: Record<string, string>;  // name → выбранный optionId (иначе default)
  appends: Record<string, string[]>; // blockId → дописанные фразы
}
