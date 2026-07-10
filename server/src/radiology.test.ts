// Тесты движка лучевой диагностики на примерах команд из ТЗ (docs/radiology-ct-spec-v0.1.md).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RadiologyEngine } from './radiology/engine.js';
import { ctAbdomen } from './radiology/templates/ct-abdomen.js';
import { convertNumberWords, normalizeCommand } from './radiology/numbers.js';

const eng = () => new RadiologyEngine(ctAbdomen);
// Прогнать серию команд и собрать отчёт.
function run(cmds: string[]) {
  const e = eng();
  for (const c of cmds) e.apply(c);
  return e.build();
}
const sec = (r: ReturnType<RadiologyEngine['build']>, id: string) =>
  r.sections.find((s) => s.id === id)?.text ?? '';

// ─── Числа ───────────────────────────────────────────────────────────────────
test('convertNumberWords: слова → цифры', () => {
  assert.equal(convertNumberWords('восемьдесят на десять'), '80 на 10');
  assert.equal(convertNumberWords('сто пятьдесят шесть'), '156');
  assert.equal(convertNumberWords('двадцать четыре'), '24');
});

test('normalizeCommand: ё→е, lowercase, схлопывание', () => {
  assert.equal(normalizeCommand('Селезёнка  РАЗМЕРЫ 14 7 5'), 'селезенка размеры 14 7 5');
});

// ─── Техника ─────────────────────────────────────────────────────────────────
test('техника: три фазы даёт многофазный автотекст', () => {
  const r = run(['ОБП три фазы', 'ОБП качество хорошее']);
  assert.match(r.technique, /внутривенным болюсным контрастированием/);
  assert.match(r.technique, /артериальной, портально-венозной, отсроченной/);
  assert.match(r.technique, /диагностическое/);
});

test('техника: качество ограничено дыханием', () => {
  const r = run(['ОБП контраст', 'ОБП качество ограничено дыханием']);
  assert.match(r.technique, /дыхательными артефактами/);
});

// ─── Печень ──────────────────────────────────────────────────────────────────
test('печень норма с плотностью', () => {
  const r = run(['печень норма плотность 56']);
  assert.match(sec(r, 'liver'), /56 HU/);
  assert.match(sec(r, 'liver'), /структура паренхимы однородная/);
});

test('печень плотность <40 → авто-стеатоз в заключении', () => {
  const r = run(['печень плотность 38']);
  assert.match(r.conclusion, /жировой инфильтрации/);
});

test('печень норма без плотности → подсветка обязательного поля', () => {
  const r = run(['печень норма']);
  assert.match(sec(r, 'liver'), /___ HU/);
  assert.ok(r.highlights.some((h) => /не указана средняя плотность/.test(h)));
});

test('цирроз без ГЦР при многофазном → LR-4/LR-5', () => {
  const r = run(['ОБП три фазы', 'печень цирроз без ГЦР']);
  assert.match(sec(r, 'liver'), /цирротическому типу/);
  assert.match(r.conclusion, /LR-4\/LR-5 не выявлено/);
});

test('цирроз без ГЦР без многофазного → gating-предупреждение', () => {
  const r = run(['ОБП натив', 'печень цирроз без ГЦР']);
  assert.match(sec(r, 'liver'), /Оценка по LI-RADS ограничена/);
  assert.ok(r.highlights.some((h) => /LR-4\/LR-5/.test(h)));
});

// ─── Очаги печени ─────────────────────────────────────────────────────────────
test('киста печени S6', () => {
  const r = run(['печень киста S6 12 мм']);
  assert.match(sec(r, 'liver_lesions'), /В S6 печени определяется простая киста диаметром 12 мм/);
  assert.match(r.conclusion, /Простая киста S6 печени/);
});

test('гемангиома: типичная только при многофазном', () => {
  const typical = run(['ОБП три фазы', 'печень гемангиома S7 22 мм']);
  assert.match(typical.conclusion, /типичной гемангиомы S7 печени/);

  const limited = run(['ОБП натив', 'печень гемангиома S7 22 мм']);
  assert.match(sec(limited, 'liver_lesions'), /вероятно соответствующими гемангиоме/);
  assert.ok(limited.highlights.some((h) => /типичная.*недоступна/.test(h)));
});

test('несколько очагов накапливаются (repeatable)', () => {
  const r = run(['ОБП три фазы', 'печень киста S6 12 мм', 'печень гемангиома S7 22 мм']);
  assert.match(sec(r, 'liver_lesions'), /киста диаметром 12 мм/);
  assert.match(sec(r, 'liver_lesions'), /гемангиом/);
});

test('гиповаскулярные метастазы с сегментами', () => {
  const r = run(['печень метастазы гиповаскулярные множественные от 6 до 24 мм сегменты 2 4 6 8']);
  assert.match(sec(r, 'liver_lesions'), /от 6 до 24 мм/);
  assert.match(sec(r, 'liver_lesions'), /S2, S4, S6, S8/);
  assert.match(r.conclusion, /метастатического поражения печени/);
});

// ─── Желчные протоки / пузырь ─────────────────────────────────────────────────
test('холедохолитиаз', () => {
  const r = run(['холедохолитиаз камень 6 мм холедох 11 мм']);
  assert.match(sec(r, 'bile_ducts'), /Холедох расширен до 11 мм/);
  assert.match(sec(r, 'bile_ducts'), /конкремент до 6 мм/);
  assert.match(r.conclusion, /холедохолитиаза с билиарной гипертензией/);
});

test('ЖКБ без воспаления', () => {
  const r = run(['желчный камни без воспаления']);
  assert.match(r.conclusion, /желчнокаменной болезни без признаков острого холецистита/);
});

test('острый холецистит', () => {
  const r = run(['острый холецистит стенка 5 мм камень шейка перивезикальная жидкость']);
  assert.match(sec(r, 'gallbladder'), /Стенка утолщена до 5 мм/);
  assert.match(sec(r, 'gallbladder'), /шейки желчного пузыря определяется конкремент/);
  assert.match(sec(r, 'gallbladder'), /перивезикальной жидкости/);
  assert.match(r.conclusion, /острого калькулёзного холецистита/);
});

test('после холецистэктомии: норма ЖП запрещена', () => {
  const r = run(['желчный удалён']);
  assert.match(sec(r, 'gallbladder'), /Желчный пузырь удалён/);
  assert.doesNotMatch(sec(r, 'gallbladder'), /обычных размеров/);
});

// ─── Поджелудочная ────────────────────────────────────────────────────────────
test('хронический панкреатит', () => {
  const r = run(['хронический панкреатит кальцинаты проток 5 мм атрофия']);
  assert.match(sec(r, 'pancreas'), /уменьшена в объёме/);
  assert.match(sec(r, 'pancreas'), /кальцинаты/);
  assert.match(sec(r, 'pancreas'), /расширен до 5 мм/);
  assert.match(r.conclusion, /хронического кальцифицирующего панкреатита/);
});

// ─── Селезёнка ────────────────────────────────────────────────────────────────
test('селезёнка: индекс и спленомегалия', () => {
  const r = run(['селезёнка размеры 14 7 5']);
  assert.match(sec(r, 'spleen'), /14,0 × 7,0 × 5,0 см/);
  assert.match(sec(r, 'spleen'), /селезёночный индекс — 490/);
  assert.match(r.conclusion, /спленомегалии/);
});

test('селезёнка: маленькая — без спленомегалии', () => {
  const r = run(['селезёнка размеры 10 5 4']); // индекс 200
  assert.doesNotMatch(r.conclusion, /спленомегали/);
});

// ─── Портальная система ───────────────────────────────────────────────────────
test('портальная гипертензия', () => {
  const r = run(['портальная гипертензия воротная 16 ВБВ 14 селезёночная 12 вариксы пищевода желудка асцит малый']);
  assert.match(sec(r, 'portal'), /Воротная вена расширена до 16 мм/);
  assert.match(sec(r, 'portal'), /брыжеечная вена — до 14 мм/);
  assert.match(sec(r, 'portal'), /селезёночная вена — до 12 мм/);
  assert.match(r.conclusion, /портальной гипертензии/);
});

// ─── Надпочечники ─────────────────────────────────────────────────────────────
test('надпочечник: липид-содержащая аденома по нативу', () => {
  const r = run(['правый надпочечник аденома 18 мм натив 4']);
  assert.match(sec(r, 'adrenals'), /В правом надпочечнике/);
  assert.match(sec(r, 'adrenals'), /нативной плотностью 4 HU/);
  assert.match(sec(r, 'adrenals'), /липид-содержащей аденоме/);
  assert.match(r.conclusion, /Липид-содержащая аденома правого надпочечника/);
});

test('надпочечник: расчёт washout', () => {
  const r = run(['левый надпочечник образование 24 мм натив 18 венозная 92 отсрочка 34']);
  assert.match(sec(r, 'adrenals'), /Абсолютный washout составляет 78%/);
  assert.match(sec(r, 'adrenals'), /относительный washout — 63%/);
  assert.match(r.conclusion, /аденомы левого надпочечника/);
});

// ─── Почки / Bosniak ──────────────────────────────────────────────────────────
test('почка Bosniak I', () => {
  const r = run(['правая почка Босняк 1 18 мм']);
  assert.match(sec(r, 'kidneys'), /В правой почке определяется простая кортикальная киста диаметром 18 мм/);
  assert.match(sec(r, 'kidneys'), /Bosniak I/);
  assert.match(r.conclusion, /Простая киста правой почки Bosniak I/);
});

test('противоречие: Bosniak I + перегородки', () => {
  const r = run(['правая почка Босняк 1 18 мм перегородки']);
  assert.ok(r.conflicts.some((c) => c.code === 'bosniak1_conflict'));
});

// ─── Свободная жидкость / газ ──────────────────────────────────────────────────
test('свободный газ — критическая находка, всегда в заключении', () => {
  const r = run(['пневмоперитонеум']);
  assert.match(r.conclusion, /Свободный газ в брюшной полости/);
});

test('газа нет → норма брюшины', () => {
  const r = run(['газа нет']);
  assert.match(sec(r, 'peritoneum'), /Свободной жидкости и свободного газа.*не определяется/);
});

// ─── Управление ───────────────────────────────────────────────────────────────
test('удалить последнее (undo)', () => {
  const e = eng();
  e.apply('печень цирроз');
  assert.match(e.build().sections.find((s) => s.id === 'liver')!.text, /цирротическому типу/);
  e.apply('удалить последнее');
  assert.doesNotMatch(e.build().sections.find((s) => s.id === 'liver')!.text, /цирротическому типу/);
});

test('не выносить в заключение', () => {
  const e = eng();
  e.apply('ОБП три фазы');
  e.apply('печень киста S6 12 мм');
  e.apply('не выносить в заключение');
  assert.doesNotMatch(e.build().conclusion, /Простая киста S6/);
});

test('полностью нормальный протокол → пустое заключение', () => {
  const r = run(['ОБП контраст', 'остальное норма']);
  assert.equal(r.conclusion, ctAbdomen.emptyConclusion);
});

test('описание включает все секции нормой по умолчанию', () => {
  const r = run(['ОБП контраст']);
  assert.match(r.description, /Печень обычных размеров/);
  assert.match(r.description, /Надпочечники обычной формы/);
  assert.match(r.description, /Свободной жидкости и свободного газа/);
});
