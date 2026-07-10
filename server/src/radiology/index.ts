// Реестр шаблонов лучевой диагностики + публичный API движка.
// Добавить исследование = добавить конфиг в templates/ и зарегистрировать здесь.

import { RadiologyEngine } from './engine.js';
import type { RadiologyTemplate } from './schema.js';
import { ctAbdomen } from './templates/ct-abdomen.js';

export * from './schema.js';
export { RadiologyEngine } from './engine.js';
export type { ApplyResult, BuiltReport } from './engine.js';

// Очередь 1 (ТЗ). Пока реализован эталонный шаблон КТ ОБП; остальные добавляются
// как data-конфиги без изменения движка.
export const templates: RadiologyTemplate[] = [
  ctAbdomen,
];

export function getTemplate(id: string): RadiologyTemplate | undefined {
  return templates.find((t) => t.id === id);
}

export function createEngine(templateId: string): RadiologyEngine {
  const tpl = getTemplate(templateId);
  if (!tpl) throw new Error(`Неизвестный шаблон лучевой диагностики: ${templateId}`);
  return new RadiologyEngine(tpl);
}
