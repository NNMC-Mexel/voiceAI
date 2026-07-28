// Тесты fill-in движка на рабочем шаблоне ОБП врача Михайлова.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DocEngine } from './radiology/doc-engine.js';
import { abdomenMikhailov } from './radiology/templates/abdomen-mikhailov.js';

const eng = () => new DocEngine(abdomenMikhailov);
function run(cmds: string[]) {
  const e = eng();
  for (const c of cmds) e.apply(c);
  return e.build();
}
const blk = (r: ReturnType<DocEngine['build']>, id: string) => r.blocks.find((b) => b.id === id)?.text ?? '';

test('пустая сессия → полный нормальный документ с дефолтами', () => {
  const r = run([]);
  assert.match(blk(r, 'liver'), /средние значения \+60 HU/);
  assert.match(blk(r, 'choledoch'), /до 5,5 мм/);
  assert.match(blk(r, 'spleen'), /__х__х__ мм – СИ ≈ ___/);
  assert.match(blk(r, 'vessels'), /аорта до 16,0 мм/);
  assert.match(r.conclusion, /Патологических изменений органов брюшной полости.*не выявлено/);
});

test('слоты: печень КВР и плотность', () => {
  const r = run(['печень КВР 145 плотность 62']);
  assert.match(blk(r, 'liver'), /КВР 145 мм/);
  assert.match(blk(r, 'liver'), /\+62 HU/);
});

test('слот холедох', () => {
  const r = run(['холедох 7']);
  assert.match(blk(r, 'choledoch'), /до 7,0 мм/);
});

test('размеры селезёнки + авто-СИ', () => {
  const r = run(['селезёнка 12 на 6 на 5']);
  assert.match(blk(r, 'spleen'), /12,0х6,0х5,0 мм – СИ ≈ 360/);
});

test('селезёнка: неполные размеры принимаются, СИ пока не считается', () => {
  const r = run(['селезёнка 120 на 130 мм не увеличена']);
  assert.match(blk(r, 'spleen'), /120,0х130,0х__ мм – СИ ≈ ___/);
});

test('селезёнка: третий размер дополняет и включает СИ', () => {
  const e = eng();
  e.apply('селезёнка 12 на 6 на 5');
  assert.match(e.build().blocks.find((b) => b.id === 'spleen')!.text, /СИ ≈ 360/);
});

test('внятная причина отказа: число без параметра', () => {
  const e = eng();
  const r = e.apply('печень 20');
  assert.equal(r.ok, false);
  assert.match(r.detail ?? '', /какое это значение/);
});

test('внятная причина отказа: нет органа', () => {
  const e = eng();
  const r = e.apply('20');
  assert.equal(r.ok, false);
  assert.match(r.detail ?? '', /к какому органу/);
});

test('свитч: конкременты почек с размером', () => {
  const r = run(['почки конкремент 6']);
  assert.match(blk(r, 'kidneys'), /определяется конкремент до 6 мм/);
  assert.doesNotMatch(blk(r, 'kidneys'), /конкременты не визуализируются/);
});

test('свитч: содержимое желчного пузыря', () => {
  const r = run(['желчный конкременты']);
  assert.match(blk(r, 'gallbladder'), /в просвете определяются конкременты/);
});

test('дописывание в блок лёгких', () => {
  const r = run(['базальные добавь единичные плевропульмональные спайки справа']);
  assert.match(blk(r, 'lung_bases'), /Единичные плевропульмональные спайки справа\./);
  assert.match(blk(r, 'lung_bases'), /выпота не выявлено/); // норма сохранилась
});

test('дописывание в заключение', () => {
  const r = run(['заключение добавь рекомендована консультация уролога']);
  assert.match(blk(r, 'conclusion'), /Рекомендована консультация уролога\./);
});

test('сосуды: аорта и чревный ствол', () => {
  const r = run(['аорта 18 чревный 9']);
  assert.match(blk(r, 'vessels'), /аорта до 18,0 мм/);
  assert.match(blk(r, 'vessels'), /Чревный ствол до 9,0 мм/);
});

test('undo откатывает слот', () => {
  const e = eng();
  e.apply('печень плотность 40');
  assert.match(e.build().blocks.find((b) => b.id === 'liver')!.text, /\+40 HU/);
  e.apply('удалить последнее');
  assert.match(e.build().blocks.find((b) => b.id === 'liver')!.text, /\+60 HU/);
});

test('undo откатывает свитч+слот целиком', () => {
  const e = eng();
  e.apply('почки конкремент 6');
  e.apply('удалить последнее');
  assert.match(e.build().blocks.find((b) => b.id === 'kidneys')!.text, /конкременты не визуализируются/);
});

test('полная сессия диктовки собирает документ', () => {
  const r = run([
    'печень КВР 152 плотность 58',
    'холедох 6',
    'поджелудочная головка 26',
    'селезёнка 11 на 6 на 4',
    'почки конкремент 5',
    'базальные добавь единичные плевропульмональные спайки справа',
  ]);
  assert.match(blk(r, 'liver'), /КВР 152 мм.*\+58 HU/);
  assert.match(blk(r, 'pancreas'), /головки до 26/);
  assert.match(blk(r, 'spleen'), /СИ ≈ 264/); // 11*6*4
  assert.match(blk(r, 'kidneys'), /конкремент до 5 мм/);
  assert.match(blk(r, 'lung_bases'), /плевропульмональные спайки/);
});
