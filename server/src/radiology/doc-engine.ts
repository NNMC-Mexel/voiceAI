// Движок fill-in: команда врача → правка значений/свитчей/дописок → готовый документ.
// Детерминистичен, без LLM. Переиспользует парсинг чисел из numbers.ts.

import {
  assignNumbersToKeywordGroups,
  extractNumbers,
  hasPhrase,
  normalizeCommand,
} from './numbers.js';
import type {
  BlockNode, DocBlock, DocNode, DocState, DocTemplate, SlotDef, SlotValues,
} from './doc-model.js';

export interface DocApplyResult {
  ok: boolean;
  action: 'slot' | 'switch' | 'append' | 'undo' | 'unknown';
  blockId?: string;
  detail?: string;
}

export interface DocReport {
  title: string;
  blocks: { id: string; label: string; text: string }[];
  text: string;              // весь документ одним куском
  conclusion: string;
}

const APPEND_TRIGGERS = ['добавь', 'добавить', 'дополнительно', 'примечание', 'также'];

function fmt(values: number[], slot: SlotDef): string {
  const one = (n: number): string => {
    const s = slot.decimals !== undefined ? n.toFixed(slot.decimals) : String(n);
    const normalized = s.replace('.', ',');
    return slot.signMode === 'always' && n >= 0 ? `+${normalized}` : normalized;
  };
  const parts = values.map(one);
  // Продиктовали не все размеры («12 на 6» вместо трёх) — недостающие остаются пустыми.
  const arity = slot.arity ?? 1;
  while (parts.length < arity) parts.push('__');
  return parts.join(slot.join ?? ' ');
}

function collectSlots(nodes: BlockNode[]): SlotDef[] {
  const out: SlotDef[] = [];
  const walk = (ns: (BlockNode | DocNode)[]) => {
    for (const n of ns) {
      if (n.kind === 'slot') out.push(n.slot);
      else if (n.kind === 'switch') for (const o of n.sw.options) walk(o.nodes);
    }
  };
  walk(nodes);
  return out;
}

export class DocEngine {
  private tpl: DocTemplate;
  private state: DocState = { slots: {}, switches: {}, appends: {} };
  private history: string[] = [];
  private lastBlock?: string;

  constructor(tpl: DocTemplate) { this.tpl = tpl; }

  private snapshot(): void {
    this.history.push(JSON.stringify(this.state));
    if (this.history.length > 200) this.history.shift();
  }

  apply(raw: string): DocApplyResult {
    const n = normalizeCommand(raw);
    if (!n) return { ok: false, action: 'unknown' };

    if (hasPhrase(n, 'удалить последнее') || hasPhrase(n, 'отменить')) {
      const prev = this.history.pop();
      if (!prev) return { ok: false, action: 'undo', detail: 'нет команд для отмены' };
      this.state = JSON.parse(prev);
      return { ok: true, action: 'undo' };
    }

    const block = this.matchBlock(n) ?? (this.lastBlock ? this.blockById(this.lastBlock) : undefined);

    // 1) дописывание
    const appTrig = APPEND_TRIGGERS.find((t) => hasPhrase(n, t));
    if (appTrig) {
      const target = block ?? this.blockById(this.tpl.conclusionBlockId ?? '');
      if (!target?.appendable) return { ok: false, action: 'unknown', detail: 'блок не поддерживает дописывание' };
      // дописку берём из СЫРОГО текста — сохраняем регистр («S6», аббревиатуры)
      const rawLower = raw.toLowerCase().replace(/ё/g, 'е');
      const pos = rawLower.indexOf(appTrig);
      let extra = (pos >= 0 ? raw.slice(pos + appTrig.length) : n.slice(n.indexOf(appTrig) + appTrig.length)).trim();
      // убрать имя якоря-блока, если попало в начало
      extra = extra.replace(/^[,\s—-]+/, '');
      if (!extra) return { ok: false, action: 'append', blockId: target.id, detail: 'пустая дописка' };
      extra = extra.charAt(0).toUpperCase() + extra.slice(1);
      this.snapshot();
      (this.state.appends[target.id] ??= []).push(extra.endsWith('.') ? extra : extra + '.');
      this.lastBlock = target.id;
      return { ok: true, action: 'append', blockId: target.id };
    }

    if (!block) {
      return { ok: false, action: 'unknown', detail: 'не понял, к какому органу относится — начните с названия (например: «печень плотность 60»)' };
    }
    this.lastBlock = block.id;

    const nums = extractNumbers(this.stripSegments(n));

    // 2) свитчи (слэш-меню): выбор отклонения
    for (const node of block.nodes) {
      if (node.kind !== 'switch') continue;
      for (const opt of node.sw.options) {
        if (opt.triggers.length && opt.triggers.some((t) => hasPhrase(n, t))) {
          this.snapshot();
          this.state.switches[node.sw.name] = opt.id;
          this.fillSlots(collectSlots(opt.nodes), nums, block, n); // слоты выбранной опции
          return { ok: true, action: 'switch', blockId: block.id, detail: opt.id };
        }
      }
    }

    // 3) слоты (значения) вне свитчей
    this.snapshot();
    const filled = this.fillSlots(collectSlots(block.nodes).filter((s) => !this.inSwitch(block, s)), nums, block, n);
    if (filled) return { ok: true, action: 'slot', blockId: block.id };
    this.history.pop(); // ничего не изменилось — снимаем лишний снапшот
    const detail = nums.length
      ? `не понял, какое это значение для блока «${block.label}» — назовите параметр (например: «холедох 6»)`
      : `для блока «${block.label}» не названо ни значения, ни отклонения`;
    return { ok: false, action: 'unknown', blockId: block.id, detail };
  }

  private inSwitch(block: DocBlock, slot: SlotDef): boolean {
    for (const node of block.nodes) {
      if (node.kind === 'switch') {
        for (const o of node.sw.options) if (collectSlots(o.nodes).includes(slot)) return true;
      }
    }
    return false;
  }

  // Мутирует state.slots; снапшот делает вызывающий. Возвращает, было ли заполнение.
  private fillSlots(slots: SlotDef[], nums: ReturnType<typeof extractNumbers>, block: DocBlock, n: string): boolean {
    let any = false;
    const used = new Set<number>();
    // сначала скалярные слоты по ключевому слову
    const scalar = slots.filter((slot) => (slot.arity ?? 1) <= 1);
    const assignments = assignNumbersToKeywordGroups(nums, scalar.map((slot) => slot.keywords), used);
    for (let slotIndex = 0; slotIndex < scalar.length; slotIndex++) {
      const numberIndex = assignments[slotIndex];
      if (numberIndex === undefined) continue;
      this.state.slots[scalar[slotIndex].name] = [nums[numberIndex].value];
      used.add(numberIndex);
      any = true;
    }
    // затем dims-слот: до arity незанятых чисел, если блок адресован якорем/ключом.
    // Принимаем и неполный набор («селезёнка 120 на 130») — недостающее покажем как «__».
    for (const s of slots) {
      const arity = s.arity ?? 1;
      if (arity <= 1) continue;
      const anchored = block.anchors.some((a) => hasPhrase(n, a)) || s.keywords.some((k) => hasPhrase(n, k));
      const free = nums.map((_tk, i) => i).filter((i) => !used.has(i));
      if (anchored && free.length >= 1) {
        const take = free.slice(0, arity);
        this.state.slots[s.name] = take.map((i) => nums[i].value);
        for (const i of take) used.add(i);
        any = true;
      }
    }
    return any;
  }

  private stripSegments(n: string): string {
    return n; // в fill-модели сегменты не нужны; заглушка для симметрии
  }

  private detectSide(n: string): 'right' | 'left' | undefined {
    if (/(^|[^а-я])прав/.test(n) || /справа/.test(n)) return 'right';
    if (/(^|[^а-я])лев/.test(n) || /слева/.test(n)) return 'left';
    return undefined;
  }

  private matchBlock(n: string): DocBlock | undefined {
    // все блоки с совпавшим якорем; при парных структурах уточняем по стороне
    const side = this.detectSide(n);
    const hits: { block: DocBlock; len: number }[] = [];
    for (const b of this.tpl.blocks) {
      let len = 0;
      for (const a of b.anchors) if (hasPhrase(n, a) && a.length > len) len = a.length;
      if (len > 0) hits.push({ block: b, len });
    }
    if (!hits.length) return undefined;
    if (side) {
      const sided = hits.filter((h) => h.block.side === side);
      if (sided.length) return sided.sort((a, b) => b.len - a.len)[0].block;
    }
    return hits.sort((a, b) => b.len - a.len)[0].block;
  }

  private blockById(id: string): DocBlock | undefined {
    return this.tpl.blocks.find((b) => b.id === id);
  }

  // ─── Рендер документа ──────────────────────────────────────────────────────
  private renderNode(node: BlockNode): string {
    if (node.kind === 'text') return node.text;
    if (node.kind === 'slot') {
      const v = this.state.slots[node.slot.name];
      return v && v.length ? fmt(v, node.slot) : node.slot.default;
    }
    if (node.kind === 'derived') return node.compute(this.state.slots);
    // switch
    const sel = this.state.switches[node.sw.name] ?? node.sw.default;
    const opt = node.sw.options.find((o) => o.id === sel) ?? node.sw.options.find((o) => o.id === node.sw.default);
    return (opt?.nodes ?? []).map((c) => this.renderNode(c)).join('');
  }

  build(): DocReport {
    const blocks = this.tpl.blocks.map((b) => {
      let body = b.nodes.map((node) => this.renderNode(node)).join('');
      const extra = this.state.appends[b.id];
      if (extra?.length) body += ' ' + extra.join(' ');
      return { id: b.id, label: b.label, body: body.trim(), text: `${b.label}: ${body}`.trim() };
    });
    const conclusionBlock = blocks.find((b) => b.id === this.tpl.conclusionBlockId);
    return {
      title: this.tpl.title,
      blocks: blocks.map(({ id, label, text }) => ({ id, label, text })),
      text: [this.tpl.title, ...blocks.map((b) => b.text)].join('\n'),
      conclusion: conclusionBlock?.body ?? '', // без префикса-лейбла «Заключение:»
    };
  }

  getState(): DocState { return this.state; }
}
