// Модель «шаблон-как-документ» (fill-in) — под рабочие шаблоны врача Михайлова.
// Шаблон = готовый нормальный текст с пропусками (слоты) и слэш-меню (свитчи).
// Врач диктует по шаблону: система меняет ТОЛЬКО значения слотов, переключает
// варианты и дописывает добавленное. Всё остальное остаётся текстом нормы.

export interface SlotDef {
  name: string;
  /**
   * Stable clinical field identifier used by the provenance-aware composer.
   * It must not depend on the position of the node inside the template.
   */
  fieldId?: string;
  keywords: string[];        // теги-стемы перед числом: «плотность 62» → 'плотность'
  /**
   * Exact, standalone speech aliases that may deterministically route an atom
   * to this field's section even when the organ anchor is spoken later.
   *
   * These aliases are deliberately separate from `keywords`: keywords may be
   * morphological stems used inside an already-routed section, while routing
   * aliases are matched without fuzzy suffix expansion. An alias is enabled
   * only when it resolves to exactly one field in the whole template.
   */
  routingAliases?: string[];
  arity?: number;            // >1 — набор («12 на 6 на 5»); хранится как number[]
  decimals?: number;         // формат вывода (0 — целое, 1 — «14,0»)
  join?: string;             // разделитель для arity>1 (напр. 'х')
  /** Render an explicit `+` for non-negative values while preserving `-`. */
  signMode?: 'always';
  /** Canonical unit expected by this field (for example `mm` or `HU`). */
  unit?: 'mm' | 'cm' | 'HU' | 'percent';
  /**
   * Whether a value without a spoken unit may inherit the unit from the
   * versioned template schema. Explicit conversions are allowed only through
   * a versioned deterministic conversion rule recorded in the assignment.
   */
  allowImplicitUnit?: boolean;
  /**
   * Spoken values without a unit use this schema unit before deterministic
   * conversion to `unit`. Useful when the physician convention is centimetres
   * while the document renders millimetres.
   */
  implicitUnit?: 'mm' | 'cm' | 'HU' | 'percent';
  /** Distinguishes an empty placeholder from a real clinical default. */
  defaultKind?: 'placeholder' | 'clinical_default';
  /**
   * Versioned deterministic value constraints. `min/max` protect the parser
   * from impossible/non-finite values; `templateClaim*` protects literals
   * such as "не увеличена" or "в пределах нормы" from contradictory input.
   */
  validation?: {
    ruleId: string;
    minExclusive?: number;
    minInclusive?: number;
    maxInclusive?: number;
    templateClaimMinInclusive?: number;
    templateClaimMaxInclusive?: number;
    aggregate?: {
      ruleId: string;
      operation: 'product';
      divisor?: number;
      maxInclusive?: number;
    };
  };
  /**
   * An unresolved placeholder for this field blocks physician approval even
   * when its section was not mentioned. The doctor must dictate/correct the
   * value and recompose the review draft, or use another template.
   */
  requiredForApproval?: boolean;
  default: string;           // печатается, если врач не задал (норм. значение или '___')
}

export type DocNode =
  | { kind: 'text'; text: string }
  | { kind: 'slot'; slot: SlotDef }
  | {
      kind: 'derived';
      name: string;
      fieldId?: string;
      dependsOn?: string[];
      formulaVersion?: string;
      /** Deterministic post-processing for canonical-unit derived values. */
      outputDivisor?: number;
      unit?: 'mm' | 'cm' | 'HU' | 'percent';
      compute: (v: SlotValues) => string;
    };

export interface SwitchOption {
  id: string;
  triggers: string[];        // фразы выбора отклонения; у нормальной опции — []
  /** Versioned phrases that explicitly veto this option. */
  excludes?: string[];
  nodes: DocNode[];          // текст опции (может содержать слоты)
}

export interface SwitchDef {
  name: string;
  /** Stable clinical field identifier used by the composer. */
  fieldId?: string;
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
  /**
   * Version of the exact field-alias routing contract. Templates without this
   * value do not route by field aliases.
   */
  fieldRoutingVersion?: string;
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
