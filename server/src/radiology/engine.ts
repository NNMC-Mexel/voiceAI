// Движок структурированного протокола: команда → состояние → отчёт.
// Детерминистичен, без LLM. Один движок обслуживает любой шаблон-конфиг.

import { extractNumbers, hasPhrase, normalizeCommand, type NumberToken } from './numbers.js';
import type {
  Conflict, EngineContext, Finding, FindingRender, FindingView, Phase,
  ProtocolState, RadiologyTemplate, Section, SectionState, SlotSpec, TechniqueState,
} from './schema.js';

export interface ApplyResult {
  ok: boolean;
  handled: 'technique' | 'section' | 'global' | 'unknown';
  message?: string;
  warnings?: string[];
}

export interface BuiltReport {
  technique: string;
  sections: { id: string; organ: string; text: string }[];
  description: string;      // склеенное «Данные исследования»
  conclusion: string;       // автозаключение
  highlights: string[];     // подсветки: обязательные пустые поля, gating-замечания
  conflicts: Conflict[];    // противоречия (раздел 34 ТЗ)
}

function emptySection(): SectionState {
  return { status: 'untouched', measurements: {}, dims: {}, instances: [] };
}

// ─── Разбор сущностей команды ────────────────────────────────────────────────
interface Parsed {
  numbers: NumberToken[];
  segments: string[];
  side?: 'right' | 'left';
  text: string;            // очищенный от сегментов текст
}

function parseEntities(normalized: string): Parsed {
  let work = normalized;
  const segments: string[] = [];

  // «сегменты 2 4 6 8» / «сегмент 4» → S2,S4,S6,S8
  work = work.replace(/сегмент[ыа]?\s+([\d\s]+)/g, (_m, nums: string) => {
    for (const n of nums.trim().split(/\s+/)) if (/^\d+$/.test(n)) segments.push('S' + n);
    return ' ';
  });
  // «s6» / «s 7»
  work = work.replace(/\bs\s?(\d)\b/g, (_m, n: string) => { segments.push('S' + n); return ' '; });

  // \b/\w не работают с кириллицей в JS — матчим по началу слова через [^а-я].
  let side: 'right' | 'left' | undefined;
  if (/(^|[^а-я])прав/.test(work) || /справа/.test(work)) side = 'right';
  else if (/(^|[^а-я])лев/.test(work) || /слева/.test(work)) side = 'left';

  return { numbers: extractNumbers(work), segments, side, text: work };
}

// Наполнение view слотами находки/нормы из разобранных сущностей.
function fillView(parsed: Parsed, ctx: EngineContext, specs: SlotSpec[], flagSpecs: {
  name: string; phrases: string[];
}[] = []): FindingView {
  const slots: Record<string, number | undefined> = {};
  const dims: Record<string, number[]> = {};
  const flags: Record<string, boolean> = {};
  const consumed = new Set<number>(); // индексы numbers, уже разобранные

  const take = (pred: (t: NumberToken) => boolean): NumberToken | undefined => {
    for (let i = 0; i < parsed.numbers.length; i++) {
      if (consumed.has(i) && !pred(parsed.numbers[i])) continue;
      if (consumed.has(i)) continue;
      if (pred(parsed.numbers[i])) { consumed.add(i); return parsed.numbers[i]; }
    }
    return undefined;
  };

  const norm = (k: string) => k.toLowerCase().replace(/ё/g, 'е');
  // Порядок важен: специфичные (dimensions/keyword/category) раньше bareSize.
  for (const s of specs.filter((x) => x.role === 'dimensions')) {
    const kws = (s.keywords ?? ['размеры', 'размер']).map(norm);
    const vals: number[] = [];
    for (let i = 0; i < parsed.numbers.length && vals.length < (s.count ?? 3); i++) {
      if (consumed.has(i)) continue;
      if (kws.includes(parsed.numbers[i].precededBy)) { vals.push(parsed.numbers[i].value); consumed.add(i); }
    }
    dims[s.name] = vals;
  }
  for (const s of specs.filter((x) => x.role === 'keyword' || x.role === 'category')) {
    const kws = (s.keywords ?? []).map(norm);
    const tok = take((t) => kws.includes(t.precededBy));
    if (tok) slots[s.name] = tok.value;
  }
  for (const s of specs.filter((x) => x.role === 'bareSize')) {
    const tok = take(() => true);
    if (tok) slots[s.name] = tok.value;
  }

  for (const f of flagSpecs) flags[f.name] = f.phrases.some((p) => hasPhrase(parsed.text, p));

  return { slots, dims, flags, segments: parsed.segments, side: parsed.side, ctx };
}

// ─── Движок ──────────────────────────────────────────────────────────────────
export class RadiologyEngine {
  private tpl: RadiologyTemplate;
  private technique: TechniqueState = { phases: [] };
  private hccRisk = false;
  private sections: Record<string, SectionState> = {};
  private history: string[] = [];   // снапшоты JSON для undo
  private lastTouched?: string;      // id секции последней команды (для «не выносить»)

  constructor(tpl: RadiologyTemplate) {
    this.tpl = tpl;
    for (const s of tpl.sections) this.sections[s.id] = emptySection();
  }

  private ctx(): EngineContext {
    const t = this.technique;
    return {
      technique: t,
      hccRisk: this.hccRisk,
      hasPhase: (p: Phase) => t.phases.includes(p),
      isMultiphase: () => t.phases.includes('arterial') && (t.phases.includes('portal') || t.phases.includes('delayed')),
    };
  }

  private snapshot(): void {
    this.history.push(JSON.stringify({
      technique: this.technique, hccRisk: this.hccRisk, sections: this.sections, lastTouched: this.lastTouched,
    }));
    if (this.history.length > 100) this.history.shift();
  }

  /** Применить одну голосовую команду. */
  apply(raw: string): ApplyResult {
    const normalized = normalizeCommand(raw);
    if (!normalized) return { ok: false, handled: 'unknown' };

    // 1. Глобальные команды управления (раздел 4 ТЗ)
    const g = this.applyGlobal(normalized);
    if (g) return g;

    // 2. Секции по якорю органа — раньше техники, т.к. органные команды содержат
    //    фазовые слова («натив», «венозная») как теги измерений, а не как технику.
    const section = this.matchSection(normalized);
    if (section) {
      this.snapshot();
      return this.applySection(section, normalized);
    }

    // 3. Команды техники (применяем ВСЕ совпавшие: «артериальная венозная отсроченная»)
    let techApplied = false;
    for (const tc of this.tpl.technique) {
      if (tc.triggers.some((p) => hasPhrase(normalized, p))) {
        if (!techApplied) { this.snapshot(); techApplied = true; }
        tc.apply(this.technique);
      }
    }
    if (techApplied) return { ok: true, handled: 'technique', message: this.tpl.techniqueText(this.technique) };

    return { ok: false, handled: 'unknown', message: 'Не распознан орган/команда' };
  }

  private applyGlobal(normalized: string): ApplyResult | null {
    if (hasPhrase(normalized, 'удалить последнее') || hasPhrase(normalized, 'отменить')) {
      const prev = this.history.pop();
      if (!prev) return { ok: false, handled: 'global', message: 'Нет команд для отмены' };
      const st = JSON.parse(prev);
      this.technique = st.technique; this.hccRisk = st.hccRisk;
      this.sections = st.sections; this.lastTouched = st.lastTouched;
      return { ok: true, handled: 'global', message: 'Отменена последняя команда' };
    }
    if (hasPhrase(normalized, 'остальное норма')) {
      this.snapshot();
      for (const s of this.tpl.sections) {
        if (this.sections[s.id].status === 'untouched') this.sections[s.id].status = 'normal';
      }
      return { ok: true, handled: 'global', message: 'Незаполненные блоки → норма' };
    }
    if (hasPhrase(normalized, 'не выносить в заключение') && this.lastTouched) {
      this.snapshot();
      const inst = this.sections[this.lastTouched].instances;
      if (inst.length) inst[inst.length - 1].suppressConclusion = true;
      return { ok: true, handled: 'global', message: 'Находка не будет в заключении' };
    }
    if (hasPhrase(normalized, 'добавить в заключение') && this.lastTouched) {
      this.snapshot();
      const inst = this.sections[this.lastTouched].instances;
      if (inst.length) inst[inst.length - 1].forceConclusion = true;
      return { ok: true, handled: 'global', message: 'Находка вынесена в заключение' };
    }
    if (hasPhrase(normalized, 'группа риска') || hasPhrase(normalized, 'риск гцр')) {
      this.snapshot(); this.hccRisk = true;
      return { ok: true, handled: 'global', message: 'Пациент отмечен как группа риска ГЦР' };
    }
    // «заключение автоматически» — сборка происходит в build(); команда безоперационна.
    if (hasPhrase(normalized, 'заключение автоматически')) {
      return { ok: true, handled: 'global', message: 'Заключение будет собрано автоматически' };
    }
    return null;
  }

  private matchSection(normalized: string): Section | undefined {
    // самый длинный якорь-совпадение выигрывает (ч.л.с. vs с)
    let best: Section | undefined; let bestLen = 0;
    for (const s of this.tpl.sections) {
      for (const a of s.anchors) {
        if (hasPhrase(normalized, a) && a.length > bestLen) { best = s; bestLen = a.length; }
      }
    }
    return best;
  }

  private applySection(section: Section, normalized: string): ApplyResult {
    const st = this.sections[section.id];
    const ctx = this.ctx();
    const parsed = parseEntities(normalized);
    this.lastTouched = section.id;

    // Удаление секции (после холецистэктомии)
    if (section.removable && section.removable.triggers.some((t) => hasPhrase(normalized, t))) {
      const view = fillView(parsed, ctx, []);
      st.status = 'removed';
      st.instances = [{ findingId: `${section.id}_removed`, view, render: section.removable.render(view) }];
      return { ok: true, handled: 'section' };
    }

    // Патологические находки
    const matched: Finding[] = (section.findings ?? []).filter(
      (f) => f.triggers.some((t) => hasPhrase(normalized, t)),
    );
    if (matched.length) {
      if (!section.repeatable) st.instances = [];
      const warnings: string[] = [];
      for (const f of matched) {
        const view = fillView(parsed, ctx, f.slots ?? [], f.flags ?? []);
        const render = f.render(view);
        st.instances.push({ findingId: f.id, view, render });
        if (render.warnings) warnings.push(...render.warnings);
      }
      st.status = 'pathology';
      return { ok: true, handled: 'section', warnings: warnings.length ? warnings : undefined };
    }

    // Иначе — норма (возможно с измерениями: «печень норма плотность 56»)
    const view = fillView(parsed, ctx, section.normal.measurements ?? []);
    for (const [k, v] of Object.entries(view.slots)) if (v !== undefined) st.measurements[k] = v;
    for (const [k, v] of Object.entries(view.dims)) if (v.length) st.dims[k] = v;
    st.status = 'normal';
    st.instances = [];
    const r = section.normal.render({ ...view, slots: { ...st.measurements }, dims: { ...st.dims } });
    return { ok: true, handled: 'section', warnings: r.warnings };
  }

  private state(): ProtocolState {
    return { technique: this.technique, hccRisk: this.hccRisk, sections: this.sections };
  }

  /** Собрать полный протокол: описание + заключение + подсветки + противоречия. */
  build(): BuiltReport {
    const ctx = this.ctx();
    const sectionsOut: { id: string; organ: string; text: string }[] = [];
    const conclusionLines: string[] = [];
    const highlights: string[] = [];

    for (const section of this.tpl.sections) {
      const st = this.sections[section.id];
      let text = '';

      if (st.status === 'removed') {
        text = st.instances[0]?.render.description ?? '';
      } else if (st.status === 'pathology') {
        text = st.instances.map((i) => i.render.description).join(' ');
        for (const inst of st.instances) {
          if (inst.render.warnings) highlights.push(...inst.render.warnings);
          const include = inst.forceConclusion
            || (!inst.suppressConclusion && (inst.render.critical || inst.render.addToConclusion !== false));
          if (include && inst.render.conclusion) conclusionLines.push(inst.render.conclusion);
        }
      } else {
        // normal или untouched → базовая норма (untouched показываем как шаблонную норму)
        const view: FindingView = {
          slots: { ...st.measurements }, dims: { ...st.dims }, flags: {}, segments: [], ctx,
        };
        const r = section.normal.render(view);
        text = r.description;
        if (r.warnings && st.status === 'normal') highlights.push(...r.warnings);
        if (r.conclusion) conclusionLines.push(r.conclusion);
      }

      sectionsOut.push({ id: section.id, organ: section.organ, text });
    }

    const conclusion = conclusionLines.length
      ? [...new Set(conclusionLines)].join('\n')
      : this.tpl.emptyConclusion;

    const description = [
      this.tpl.techniqueText(this.technique),
      ...sectionsOut.map((s) => s.text).filter(Boolean),
    ].join('\n');

    return {
      technique: this.tpl.techniqueText(this.technique),
      sections: sectionsOut,
      description,
      conclusion,
      highlights: [...new Set(highlights)],
      conflicts: this.tpl.conflicts(this.state()),
    };
  }
}
