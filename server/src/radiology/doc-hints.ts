// Подсказки «что можно диктовать» — извлекаются из конфига шаблона.
// Для каждого блока: готовые примеры команд (якорь + ключевое слово + значение),
// чтобы врач не угадывал грамматику. Примеры — рабочие: их можно сразу выполнить.

import type { BlockNode, DocNode, DocTemplate, SlotDef } from './doc-model.js';

export interface BlockHint {
  blockId: string;
  label: string;
  examples: string[];
}

// Читаемое слово-тег для примера (по стему-ключу).
const READABLE: Record<string, string> = {
  квр: 'КВР', плотность: 'плотность', холедох: 'холедох', головк: 'головка',
  хвост: 'хвост', аорта: 'аорта', чревн: 'чревный', портальн: 'портальная',
  воротн: 'воротная', конкремент: 'конкремент', смещен: 'смещение', утолщ: 'утолщение',
};
// Правдоподобные значения-заглушки для слотов без числового дефолта.
const SAMPLE: Record<string, string> = { kvr: '150' };
function readable(keyword: string): string {
  return READABLE[keyword] ?? keyword;
}
function sample(slot: SlotDef): string {
  if (/^\d/.test(slot.default)) return slot.default;   // нормальное значение как заглушка
  return SAMPLE[slot.name] ?? '10';
}

function collectSlots(nodes: (BlockNode | DocNode)[]): SlotDef[] {
  const out: SlotDef[] = [];
  for (const n of nodes) {
    if (n.kind === 'slot') out.push(n.slot);
    else if (n.kind === 'switch') for (const o of n.sw.options) out.push(...collectSlots(o.nodes));
  }
  return out;
}

export function buildHints(tpl: DocTemplate): BlockHint[] {
  const hints: BlockHint[] = [];
  for (const b of tpl.blocks) {
    const anchor = b.anchors[0] ?? '';
    const side = b.side === 'right' ? 'справа' : b.side === 'left' ? 'слева' : '';
    const prefix = [anchor, side].filter(Boolean).join(' ');
    const examples: string[] = [];

    for (const node of b.nodes) {
      if (node.kind === 'slot') {
        const s = node.slot;
        if ((s.arity ?? 1) > 1) { examples.push(`${prefix} 12 6 5`); continue; }
        const kw = readable(s.keywords[0]);
        // не дублируем слово, если ключ совпадает с якорем блока («холедох холедох» → «холедох»)
        const kwDup = anchor.startsWith(s.keywords[0]) || s.keywords[0].startsWith(anchor);
        examples.push(kwDup ? `${prefix} ${sample(s)}` : `${prefix} ${kw} ${sample(s)}`);
      } else if (node.kind === 'switch') {
        for (const opt of node.sw.options) {
          if (!opt.triggers.length) continue;
          const hasSlot = collectSlots(opt.nodes).length > 0;
          examples.push(`${prefix} ${opt.triggers[0]}${hasSlot ? ' 6' : ''}`);
        }
      }
    }
    if (b.appendable) examples.push(`${prefix} добавь …`);
    if (examples.length) hints.push({ blockId: b.id, label: b.label, examples });
  }
  return hints;
}
