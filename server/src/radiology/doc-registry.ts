// Реестр рабочих шаблонов Михайлова (fill-in модель). Добавить исследование =
// добавить конфиг в templates/ и зарегистрировать здесь; движок не меняется.

import { DocEngine } from './doc-engine.js';
import type { DocTemplate } from './doc-model.js';
import { abdomenMikhailov } from './templates/abdomen-mikhailov.js';
import { chestFemale, chestMale } from './templates/chest-mikhailov.js';
import { urinaryMikhailov } from './templates/urinary-mikhailov.js';
import { brainMikhailov } from './templates/brain-mikhailov.js';
import { sinusesMikhailov } from './templates/sinuses-mikhailov.js';

export const docTemplates: DocTemplate[] = [
  abdomenMikhailov,   // КТ ОБП
  chestMale,          // КТ ОГК (муж)
  chestFemale,        // КТ ОГК (жен)
  urinaryMikhailov,   // КТ мочевыделительной системы (жен)
  brainMikhailov,     // КТ головного мозга
  sinusesMikhailov,   // КТ ППН
];

export function getDocTemplate(id: string): DocTemplate | undefined {
  return docTemplates.find((tpl) => tpl.id === id);
}

export function createDocEngine(id: string): DocEngine {
  const tpl = getDocTemplate(id);
  if (!tpl) throw new Error(`Неизвестный шаблон: ${id}`);
  return new DocEngine(tpl);
}
