// Тесты 4 новых fill-шаблонов Михайлова (ОГК муж/жен, мочевыделительная, мозг, ППН).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDocEngine, docTemplates, getDocTemplate } from './radiology/doc-registry.js';
import { DocEngine } from './radiology/doc-engine.js';
import type { DocTemplate } from './radiology/doc-model.js';

function run(tpl: DocTemplate, cmds: string[]) {
  const e = new DocEngine(tpl);
  for (const c of cmds) e.apply(c);
  return e.build();
}
const blk = (r: ReturnType<DocEngine['build']>, id: string) => r.blocks.find((b) => b.id === id)?.text ?? '';

test('реестр содержит все 6 шаблонов', () => {
  assert.equal(docTemplates.length, 6);
  for (const id of ['CT_ABDOMEN_MIKHAILOV', 'CT_CHEST_M', 'CT_CHEST_F', 'CT_UROGRAPHY_F', 'CT_BRAIN_MIKHAILOV', 'CT_SINUSES_MIKHAILOV']) {
    assert.ok(getDocTemplate(id), `нет шаблона ${id}`);
  }
});

// ─── ОГК ──────────────────────────────────────────────────────────────────────
test('ОГК муж: норма + различие пола', () => {
  const r = run(getDocTemplate('CT_CHEST_M')!, []);
  assert.match(blk(r, 'lungs'), /Легочные поля с обеих сторон полностью расправлены/);
  assert.ok(r.blocks.some((b) => b.id === 'soft_tissue'));
  assert.ok(!r.blocks.some((b) => b.id === 'breast'));
});

test('ОГК жен: молочные железы + заключение', () => {
  const r = run(getDocTemplate('CT_CHEST_F')!, []);
  assert.ok(r.blocks.some((b) => b.id === 'breast'));
  assert.ok(!r.blocks.some((b) => b.id === 'soft_tissue'));
  assert.match(r.conclusion, /молочных желез соответствует возрастной норме/);
});

test('ОГК: свитч плеврального выпота со слотом', () => {
  const r = run(getDocTemplate('CT_CHEST_M')!, ['плевра выпот 15']);
  assert.match(blk(r, 'pleura'), /определяется свободная жидкость толщиной слоя до 15 мм/);
});

test('ОГК: дописывание очага в лёгкие', () => {
  const r = run(getDocTemplate('CT_CHEST_M')!, ['лёгкие добавь в S6 справа очаг до 8 мм']);
  assert.match(blk(r, 'lungs'), /S6 справа очаг до 8 мм/);
  assert.match(blk(r, 'lungs'), /однородной пневматизации/); // норма сохранилась
});

// ─── Мочевыделительная ────────────────────────────────────────────────────────
test('урография: норма +62HU и слоты', () => {
  const r = run(getDocTemplate('CT_UROGRAPHY_F')!, ['печень КВР 140 плотность 60', 'селезёнка 10 на 5 на 4']);
  assert.match(blk(r, 'liver'), /КВР 140 мм/);
  assert.match(blk(r, 'liver'), /\+60 HU/);
  assert.match(blk(r, 'spleen'), /СИ ≈ 200/);
});

test('урография: конкремент ЧЛС со слотами размера и плотности', () => {
  const r = run(getDocTemplate('CT_UROGRAPHY_F')!, ['члс конкремент 7 плотность 900']);
  assert.match(blk(r, 'pcs'), /конкремент до 7 мм плотностью до 900 HU/);
});

// ─── Мозг ─────────────────────────────────────────────────────────────────────
test('мозг: норма', () => {
  const r = run(getDocTemplate('CT_BRAIN_MIKHAILOV')!, []);
  assert.match(blk(r, 'csf'), /Срединные структуры не смещены/);
  assert.match(r.conclusion, /без признаков острой или очаговой патологии/);
});

test('мозг: смещение срединных структур со слотом', () => {
  const r = run(getDocTemplate('CT_BRAIN_MIKHAILOV')!, ['срединные структуры смещение 6']);
  assert.match(blk(r, 'csf'), /смещены на 6 мм/);
});

test('мозг: кровоизлияние', () => {
  const r = run(getDocTemplate('CT_BRAIN_MIKHAILOV')!, ['вещество кровоизлияние']);
  assert.match(blk(r, 'brain_matter'), /соответствующая кровоизлиянию/);
  assert.doesNotMatch(blk(r, 'brain_matter'), /кровоизлияния, ишемии, отека не определяется/);
});

// ─── ППН ──────────────────────────────────────────────────────────────────────
test('ППН: норма', () => {
  const r = run(getDocTemplate('CT_SINUSES_MIKHAILOV')!, []);
  assert.match(blk(r, 'frontal'), /Слизистая оболочка не утолщена/);
  assert.match(blk(r, 'maxillary_r'), /Просвет свободен/);
});

test('ППН: правая верхнечелюстная — утолщение слизистой по стороне', () => {
  const r = run(getDocTemplate('CT_SINUSES_MIKHAILOV')!, ['верхнечелюстная справа утолщение до 8']);
  assert.match(blk(r, 'maxillary_r'), /пристеночно утолщена до 8 мм/);
  assert.match(blk(r, 'maxillary_l'), /Слизистая оболочка не утолщена/); // левая осталась нормой
});

test('ППН: ретенционная киста левой верхнечелюстной', () => {
  const r = run(getDocTemplate('CT_SINUSES_MIKHAILOV')!, ['верхнечелюстная слева киста 12']);
  assert.match(blk(r, 'maxillary_l'), /ретенционная киста диаметром 12 мм/);
});

test('ППН: искривление перегородки', () => {
  const r = run(getDocTemplate('CT_SINUSES_MIKHAILOV')!, ['перегородка искривлена вправо']);
  assert.match(blk(r, 'septum'), /искривлена вправо/);
});

test('createDocEngine бросает на неизвестном шаблоне', () => {
  assert.throws(() => createDocEngine('NOPE'));
});
