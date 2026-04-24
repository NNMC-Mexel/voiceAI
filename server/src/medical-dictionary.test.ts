import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyMedicalDictionary } from './services/medical-dictionary.ts';

// Регрессия: Whisper-артефакты единиц, подмеченные на Нургалиевой
// (server/temp/nurgalieva_whisper2.txt и раньше). После словаря текст должен
// содержать канонические формы, чтобы P5 fp-check и quality_score их видели.

test('applyMedicalDictionary normalizes гэслчэль → г/л', () => {
  assert.match(applyMedicalDictionary('Гемоглобин 128 гэслчэль.'), /128\s+г\/л/);
  assert.match(applyMedicalDictionary('Альбумин 40 гсслч.'), /40\s+г\/л/);
  assert.match(applyMedicalDictionary('Белок 0,3 гэслшэль.'), /0,3\s+г\/л/);
});

test('applyMedicalDictionary normalizes коэслч → КОЕ/мл', () => {
  assert.match(applyMedicalDictionary('Рост 10⁵ коэслч.'), /КОЕ\/мл/);
  assert.match(applyMedicalDictionary('10 в 5 степени коэсслч'), /КОЕ\/мл/);
});

test('applyMedicalDictionary normalizes мольслчель → моль/л', () => {
  assert.match(applyMedicalDictionary('Глюкоза 5,9 мольслчель.'), /5,9\s+моль\/л/);
  assert.match(applyMedicalDictionary('Натрий 138 МОЛЬСЛЧЭЛЬ'), /138\s+моль\/л/i);
});

test('applyMedicalDictionary normalizes ецлшэль → Ед/л', () => {
  assert.match(applyMedicalDictionary('АЛТ 49 ЕЦЛШЭЛЬ.'), /49\s+Ед\/л/);
  assert.match(applyMedicalDictionary('ГГТ 62 еслшэль'), /62\s+Ед\/л/);
});

test('applyMedicalDictionary normalizes уцлотшмин → уд/мин', () => {
  assert.match(applyMedicalDictionary('ЧСС 84 уцлотшмин.'), /84\s+уд\/мин/);
});

test('applyMedicalDictionary normalizes ртутного статья → мм рт.ст.', () => {
  const out = applyMedicalDictionary('АД 152/96 мм ртутного статья.');
  assert.match(out, /152\/96\s+мм\s+рт\.ст\.?/);
});

test('applyMedicalDictionary restores X/Y from «165 сотых миллиметра»', () => {
  const out = applyMedicalDictionary('АД до 165 сотых миллиметра ртутного статья.');
  assert.match(out, /165\/100\s+мм\s+рт\.ст\.?/);
});

test('applyMedicalDictionary normalizes эсреактивный → С-реактивный', () => {
  assert.match(applyMedicalDictionary('эсреактивный белок 16,8'), /С-реактивный/i);
  assert.match(applyMedicalDictionary('Эс-реактивного белка'), /С-реактивн/i);
});

test('applyMedicalDictionary restores нитриты положительно', () => {
  assert.match(
    applyMedicalDictionary('кетоновые тела следы, нетритоположительны.'),
    /нитриты\s+положительно/,
  );
});

test('applyMedicalDictionary normalizes trailing мга → мг', () => {
  assert.match(applyMedicalDictionary('Ко-Диован 80 на 12,5 мга.'), /12,5\s+мг/);
  // Trigger only after digit — не трогаем слова
  const untouched = applyMedicalDictionary('Пациентка на Камчатке.');
  assert.match(untouched, /Камчатке/);
});

test('applyMedicalDictionary normalizes ПХ → pH when followed by digit', () => {
  assert.match(applyMedicalDictionary('относительная плотность 1,026 ПХ 5,5'), /pH\s+5,5/);
  // Не трогаем ПХ в других контекстах
  const untouched = applyMedicalDictionary('Компания ПХ выпустила отчёт.');
  assert.match(untouched, /ПХ выпустила/);
});

test('applyMedicalDictionary normalizes мгслчэль → мг/л', () => {
  assert.match(applyMedicalDictionary('СРБ 16,8 мгслчэль.'), /16,8\s+мг\/л/);
  assert.match(applyMedicalDictionary('СРБ 5,1 мг слчэль.'), /5,1\s+мг\/л/);
});

// Интегральный кейс: кусок raw-Whisper на Нургалиевой превращается в читаемую
// лабораторную строку после применения словаря.
test('applyMedicalDictionary cleans up real Nurgalieva ОАК fragment', () => {
  const raw = 'Гемоглобин 128 гэслчэль, эритроциты 4,32 умножений 10 в I степени во II степени слчэль, лейкоциты 10,8 умножения 1,0 слчэль, СОЭ 22 ММС ЛЧЕ.';
  const clean = applyMedicalDictionary(raw);
  assert.match(clean, /Гемоглобин\s+128\s+г\/л/);
  // «слчэль» после числа/единицы → «/л»
  assert.match(clean, /\/л/);
  assert.ok(!/гэслчэль/i.test(clean), 'raw artefacts must be gone');
  assert.ok(!/слчэль/i.test(clean), 'raw artefacts must be gone');
});

// Регрессия: расширенные варианты Whisper-артефактов из Нургалиевой 2-го прогона.
test('applyMedicalDictionary handles G-prefix and Whisper consonant variants', () => {
  // «128G СЛЧЭЛЬ» — латинский G → г/л
  assert.match(
    applyMedicalDictionary('Гемоглобин 128G СЛЧЭЛЬ, эритроциты 4,32'),
    /128g\/л|128\s*г\/л/i,
  );
  // «МгМолСЛЧЭль» → мкмоль/л (мг+мол склеивается Whisper-ом)
  assert.match(
    applyMedicalDictionary('Креатинин 95,1 МгМолСЛЧЭль'),
    /95,1\s+мкмоль\/л/i,
  );
  // «МОЛЬ СЛЦАЛЬ» — отделённое пробелом, с буквой «ц»
  assert.match(applyMedicalDictionary('мочевина 7,1 МОЛЬ СЛЦАЛЬ'), /7,1\s+моль\/л/i);
  // «геслшель» после дефиса
  assert.match(applyMedicalDictionary('Альбумин 40-геслшель'), /40-?\s*г\/л/);
  // «мольселчэль» — 2 буквы между «с» и «чшц»
  assert.match(applyMedicalDictionary('натрий 138 мольселчэль'), /138\s+моль\/л/);
  // «коэслчмиллилитров» — слипание с «миллилитров» хвостом
  assert.match(
    applyMedicalDictionary('рост эшеричиркали 1,0 коэслчмиллилитров'),
    /1,0\s+КОЕ\/мл/i,
  );
});
