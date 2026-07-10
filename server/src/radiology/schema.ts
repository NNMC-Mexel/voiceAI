// Схема data-driven шаблонов лучевой диагностики.
// Шаблон = данные (+ чистые функции рендера), а не код: добавить исследование =
// добавить конфиг, не трогая движок. См. docs/radiology-ct-spec-v0.1.md.

// ─── Техника исследования (раздел 3.1 ТЗ) ────────────────────────────────────
export type Phase = 'native' | 'arterial' | 'pancreatic' | 'portal' | 'delayed' | 'excretory';
export type StudyType =
  | 'native' | 'contrast' | 'multiphase' | 'angio' | 'lowdose' | 'enterography' | 'urography';
export type Quality =
  | 'diagnostic' | 'limited_artifacts' | 'limited_breathing'
  | 'limited_no_contrast' | 'limited_prep';

export interface TechniqueState {
  studyType?: StudyType;
  phases: Phase[];
  quality?: Quality;
}

// Контекст, доступный рендеру находки — знает про фазность (для gating-правил).
export interface EngineContext {
  technique: TechniqueState;
  hasPhase(p: Phase): boolean;
  /** Многофазное контрастное исследование: артериальная + (венозная|отсроченная). */
  isMultiphase(): boolean;
  /** Пациент группы риска ГЦР (для LI-RADS). Задаётся отдельной командой. */
  hccRisk: boolean;
}

// ─── Слоты: как числа команды ложатся в поля находки ─────────────────────────
export type SlotRole =
  | 'keyword'      // число после ключевого слова: «стенка 5» → keywords:['стенка']
  | 'bareSize'     // «18 мм» без тега — типичный размер
  | 'dimensions'   // «размеры 14 7 5» — набор из N чисел
  | 'category';    // «босняк 1», «li-rads 5» — категория после классификатора

export interface SlotSpec {
  name: string;
  role: SlotRole;
  keywords?: string[];         // для role:'keyword'|'category' — теги перед числом
  count?: number;              // для role:'dimensions' — сколько чисел забрать
  unit?: string;               // 'мм' | 'см' | 'HU'
  physRange?: [number, number]; // диапазон правдоподобия; вне — флаг на проверку
}

export interface FlagSpec {
  name: string;
  phrases: string[];           // любая из фраз включает флаг
}

// Значения, извлечённые из команды и переданные в рендер.
export interface FindingView {
  slots: Record<string, number | undefined>;
  dims: Record<string, number[]>;   // для role:'dimensions'
  flags: Record<string, boolean>;
  segments: string[];               // ['S6','S7'] или ['2','4']
  side?: 'right' | 'left';
  ctx: EngineContext;
}

export interface FindingRender {
  description: string;             // текст в раздел «Данные исследования»
  conclusion?: string;             // строка-кандидат в заключение
  warnings?: string[];             // подсветки/gating-замечания врачу
  /** false — находка не выносится в заключение по умолчанию (переопределяется командой). */
  addToConclusion?: boolean;
  critical?: boolean;              // всегда в заключение (свободный газ и т.п.)
}

export interface Finding {
  id: string;
  triggers: string[];              // фразы, активирующие находку
  slots?: SlotSpec[];
  flags?: FlagSpec[];
  /** Требуемые фазы; если не выполнены — движок передаёт это в render через ctx. */
  requiresPhases?: Phase[];
  render: (view: FindingView) => FindingRender;
}

export interface NormalDef {
  measurements?: SlotSpec[];       // числовые поля, извлекаемые вместе с нормой
  /**
   * Текст нормы; может подставлять измерения и '___' для незаполненных.
   * conclusion — если измерение само по себе даёт находку (индекс селезёнки >480).
   */
  render: (view: FindingView) => {
    description: string;
    warnings?: string[];
    conclusion?: string;
  };
}

export interface Section {
  id: string;
  organ: string;                   // отображаемое имя блока
  anchors: string[];               // слова-якоря, выбирающие секцию в команде
  normal: NormalDef;
  findings?: Finding[];
  repeatable?: boolean;            // несколько находок в одной секции (очаги печени)
  /** Секция удаляется целиком по команде (после холецистэктомии). */
  removable?: { triggers: string[]; render: (view: FindingView) => FindingRender };
}

// ─── Противоречия (раздел 34 ТЗ) ─────────────────────────────────────────────
export interface Conflict {
  code: string;
  message: string;
  sectionId?: string;
}

// Полное состояние протокола — на нём работают проверки противоречий.
export interface ProtocolState {
  technique: TechniqueState;
  hccRisk: boolean;
  sections: Record<string, SectionState>;
}

export interface SectionState {
  status: 'untouched' | 'normal' | 'pathology' | 'removed';
  measurements: Record<string, number>;
  dims: Record<string, number[]>;   // наборы размеров («размеры 14 7 5»)
  instances: FindingInstance[];
}

export interface FindingInstance {
  findingId: string;
  view: FindingView;
  render: FindingRender;
  suppressConclusion?: boolean;    // «не выносить в заключение»
  forceConclusion?: boolean;       // «добавить в заключение»
}

export interface RadiologyTemplate {
  id: string;
  name: string;
  modality: 'CT' | 'US' | 'MRI' | 'XR';
  aliases: string[];               // для выбора шаблона голосом/поиском
  /** Команды переключения техники: фраза → мутатор TechniqueState. */
  technique: TechniqueCommand[];
  techniqueText: (t: TechniqueState) => string;
  sections: Section[];
  /** Глобальные проверки противоречий поверх собранного состояния. */
  conflicts: (state: ProtocolState) => Conflict[];
  /** Заголовок «нет патологии» для автозаключения, если находок нет. */
  emptyConclusion: string;
}

export interface TechniqueCommand {
  triggers: string[];
  apply: (t: TechniqueState) => void;
}
