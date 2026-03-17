/**
 * Медицинский словарь постобработки для Whisper STT.
 *
 * Whisper при language='ru' транскрибирует всё кириллицей, включая
 * латинские аббревиатуры. Этот модуль исправляет типичные ошибки
 * после транскрипции, до отправки текста в LLM.
 *
 * Словарь организован по категориям для удобства расширения.
 * Замены применяются с учётом границ слов (case-insensitive).
 *
 * Поддержка пользовательских замен:
 * Врач может добавлять свои замены через UI (правый клик → попап).
 * Пользовательские замены хранятся в JSON-файле и загружаются при старте.
 */

import { readFile, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Типы ────────────────────────────────────────────────────────────────────

interface ReplacementRule {
  /** Паттерн для поиска (регулярное выражение или строка) */
  pattern: RegExp;
  /** Строка замены или функция замены */
  replacement: string | ((substring: string, ...args: string[]) => string);
}

export interface UserCorrection {
  id: string;
  wrong: string;
  correct: string;
  createdAt: string;
}

interface UserCorrectionsData {
  corrections: UserCorrection[];
}

// ─── Утилита для создания правил ─────────────────────────────────────────────

/** Создаёт правило замены с границами слов, case-insensitive */
function wordRule(from: string, to: string): ReplacementRule {
  // Экранируем спецсимволы regex
  const escaped = from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return {
    pattern: new RegExp(`(?<![а-яёa-z])${escaped}(?![а-яёa-z])`, 'giu'),
    replacement: to,
  };
}

/** Создаёт правило замены с произвольным regex */
function regexRule(pattern: RegExp, replacement: string | ((substring: string, ...args: string[]) => string)): ReplacementRule {
  return { pattern, replacement };
}

// ═══════════════════════════════════════════════════════════════════════════════
// СЛОВАРЬ ЗАМЕН
// ═══════════════════════════════════════════════════════════════════════════════

// ─── 1. Латинские аббревиатуры (кардиология) ─────────────────────────────────
// Whisper часто транскрибирует их как фонетическую кириллицу.

const LATIN_CARDIOLOGY_ABBREVIATIONS: ReplacementRule[] = [
  // Шкалы и классификации
  wordRule('чедс васк', 'CHA₂DS₂-VASc'),
  wordRule('чадс васк', 'CHA₂DS₂-VASc'),
  wordRule('чадсваск', 'CHA₂DS₂-VASc'),
  wordRule('чедсваск', 'CHA₂DS₂-VASc'),
  wordRule('ЧАДС2-ВАСК', 'CHA₂DS₂-VASc'),
  wordRule('хас блед', 'HAS-BLED'),
  wordRule('хасблед', 'HAS-BLED'),
  wordRule('ХАС-БЛЕД', 'HAS-BLED'),
  wordRule('найха', 'NYHA'),
  wordRule('НАЙХА', 'NYHA'),
  wordRule('энвайэйч', 'NYHA'),
  wordRule('цэцэс', 'CCS'),
  wordRule('ЦЦС', 'CCS'),
  wordRule('эхра', 'EHRA'),
  wordRule('ЭХРА', 'EHRA'),
  wordRule('ера', 'EHRA'),

  // Процедуры / анатомия
  wordRule('РАПТ', 'RA-ПТ'),
  wordRule('рапт', 'RA-ПТ'),
  wordRule('эр эй', 'RA'),
  wordRule('пи си ай', 'PCI'),
  wordRule('ПСИ', 'PCI'),
  wordRule('кабг', 'КАБГ'),
  wordRule('ТАВИ', 'TAVI'),
  wordRule('тави', 'TAVI'),

  // SpO2 / сатурация
  wordRule('спио два', 'SpO₂'),
  wordRule('СПИО2', 'SpO₂'),
  wordRule('спио2', 'SpO₂'),
  wordRule('эспиотудва', 'SpO₂'),
  wordRule('эспио два', 'SpO₂'),
  wordRule('сатурация кислорода', 'SpO₂'),
];

// ─── 2. Общие медицинские аббревиатуры ───────────────────────────────────────

const GENERAL_MEDICAL_ABBREVIATIONS: ReplacementRule[] = [
  // Кардиология — диагностика
  wordRule('экг', 'ЭКГ'),
  wordRule('эхокг', 'ЭхоКГ'),
  wordRule('эхо кг', 'ЭхоКГ'),
  wordRule('хмэкг', 'ХМЭКГ'),
  wordRule('холтер экг', 'ХМЭКГ'),
  wordRule('чпэхокг', 'ЧПЭхоКГ'),
  wordRule('чп эхокг', 'ЧПЭхоКГ'),
  wordRule('чреспищеводная эхокг', 'ЧПЭхоКГ'),
  wordRule('чреспищеводная эхокардиография', 'ЧПЭхоКГ'),
  wordRule('каг', 'КАГ'),
  wordRule('коронарография', 'КАГ'),
  wordRule('коронароангиография', 'КАГ'),
  wordRule('экс', 'ЭКС'),
  wordRule('электрокардиостимулятор', 'ЭКС'),

  // Артерии коронарные
  wordRule('пмжв', 'ПМЖВ'),
  wordRule('пка', 'ПКА'),
  wordRule('ов лка', 'ОВ ЛКА'),
  wordRule('лка', 'ЛКА'),
  wordRule('ствол лка', 'ствол ЛКА'),

  // Показатели гемодинамики
  wordRule('фв', 'ФВ'),
  wordRule('фракция выброса', 'ФВ'),
  wordRule('рсдла', 'РСДЛА'),
  wordRule('кдр', 'КДР'),
  wordRule('кср', 'КСР'),
  wordRule('кдо', 'КДО'),
  wordRule('ксо', 'КСО'),
  wordRule('тмжп', 'ТМЖП'),
  wordRule('тзслж', 'ТЗСЛЖ'),
  wordRule('ммлж', 'ММЛЖ'),
  wordRule('иммлж', 'ИММЛЖ'),
  wordRule('ла', 'ЛА'),
  wordRule('лп', 'ЛП'),
  wordRule('лж', 'ЛЖ'),
  wordRule('пж', 'ПЖ'),
  wordRule('пп', 'ПП'),

  // Жизненные показатели
  wordRule('артериальное давление', 'АД'),
  wordRule('ад', 'АД'),
  wordRule('чсс', 'ЧСС'),
  wordRule('частота сердечных сокращений', 'ЧСС'),
  wordRule('чдд', 'ЧДД'),
  wordRule('имт', 'ИМТ'),

  // Лаборатория
  wordRule('оак', 'ОАК'),
  wordRule('оам', 'ОАМ'),
  wordRule('бак', 'БАК'),
  wordRule('коагулограмма', 'коагулограмма'),
  wordRule('мно', 'МНО'),
  wordRule('ачтв', 'АЧТВ'),
  wordRule('алт', 'АЛТ'),
  wordRule('аст', 'АСТ'),
  wordRule('ттг', 'ТТГ'),
  wordRule('т3', 'Т3'),
  wordRule('т4', 'Т4'),
  wordRule('свт4', 'свТ4'),
  wordRule('лдг', 'ЛДГ'),
  wordRule('кфк', 'КФК'),
  wordRule('кфк мб', 'КФК-МБ'),
  wordRule('bnp', 'BNP'),
  wordRule('бнп', 'BNP'),
  wordRule('нт про бнп', 'NT-proBNP'),
  wordRule('энти про бнп', 'NT-proBNP'),
  wordRule('протромбин', 'протромбин'),
  wordRule('креатинин', 'креатинин'),
  wordRule('скф', 'СКФ'),
  wordRule('срб', 'СРБ'),
  wordRule('с реактивный белок', 'С-реактивный белок'),
  wordRule('гликированный гемоглобин', 'гликированный гемоглобин (HbA1c)'),
  wordRule('hba1c', 'HbA1c'),

  // Диагнозы — аббревиатуры
  wordRule('ибс', 'ИБС'),
  wordRule('хсн', 'ХСН'),
  wordRule('сссу', 'СССУ'),
  wordRule('нрс', 'НРС'),
  wordRule('онмк', 'ОНМК'),
  wordRule('тиа', 'ТИА'),
  wordRule('дгпж', 'ДГПЖ'),
  wordRule('аит', 'АИТ'),
  wordRule('дн', 'ДН'),
  wordRule('хобл', 'ХОБЛ'),
  wordRule('бронхиальная астма', 'БА'),
  wordRule('сд', 'СД'),
  wordRule('сд 2 типа', 'СД 2 типа'),
  wordRule('аг', 'АГ'),
  wordRule('гб', 'ГБ'),
  wordRule('гипертоническая болезнь', 'ГБ'),
  wordRule('окс', 'ОКС'),
  wordRule('оим', 'ОИМ'),
  wordRule('тэла', 'ТЭЛА'),
  wordRule('двс', 'ДВС'),
  wordRule('жкб', 'ЖКБ'),
  wordRule('мкб', 'МКБ'),
  wordRule('хпн', 'ХПН'),
  wordRule('хбп', 'ХБП'),
  wordRule('цвб', 'ЦВБ'),
  wordRule('дэп', 'ДЭП'),
  wordRule('пикс', 'ПИКС'),
  wordRule('блнпг', 'БЛНПГ'),
  wordRule('бпнпг', 'БПНПГ'),

  // МКБ-10
  wordRule('мкб 10', 'МКБ-10'),
  wordRule('мкб10', 'МКБ-10'),
  wordRule('мкб-10', 'МКБ-10'),

  // Визуализация / диагностика (дополнительные)
  wordRule('узи', 'УЗИ'),
  wordRule('мрт', 'МРТ'),
  wordRule('кт', 'КТ'),
  wordRule('мскт', 'МСКТ'),
  wordRule('уздс бца', 'УЗДС БЦА'),
  wordRule('уздг', 'УЗДГ'),
  wordRule('фгдс', 'ФГДС'),
  wordRule('ээг', 'ЭЭГ'),
  wordRule('соэ', 'СОЭ'),
  wordRule('лпнп', 'ЛПНП'),
  wordRule('лпвп', 'ЛПВП'),
  wordRule('гпод', 'ГПОД'),
  wordRule('вбн', 'ВБН'),
  wordRule('мтс', 'МТС'),

  // Способ введения
  wordRule('в в', 'в/в'),
  wordRule('в м', 'в/м'),
  wordRule('п к', 'п/к'),
];

// ─── 3. Лекарственные препараты ──────────────────────────────────────────────
// Исправляем типичные ошибки транскрипции названий препаратов.

const DRUG_NAME_CORRECTIONS: ReplacementRule[] = [
  // Антикоагулянты
  wordRule('ксорелто', 'ксарелто'),
  wordRule('ксарелта', 'ксарелто'),
  wordRule('ривароксобан', 'ривароксабан'),
  wordRule('дабигатрон', 'дабигатран'),
  wordRule('прадакс', 'прадакса'),
  wordRule('эликвис', 'эликвис'),
  wordRule('апиксобан', 'апиксабан'),
  wordRule('варфорин', 'варфарин'),

  // Антиагреганты
  wordRule('кардиомагнил', 'кардиомагнил'),
  wordRule('тромбопол', 'тромбопол'),
  wordRule('тромбоасс', 'ТромбоАСС'),
  wordRule('тромбо асс', 'ТромбоАСС'),
  wordRule('клопидогрел', 'клопидогрел'),
  wordRule('клопидогрель', 'клопидогрел'),
  wordRule('плавикс', 'плавикс'),
  wordRule('брилинта', 'брилинта'),
  wordRule('тикагрелол', 'тикагрелор'),
  wordRule('тикагрелор', 'тикагрелор'),

  // Бета-блокаторы
  wordRule('бисопралол', 'бисопролол'),
  wordRule('бисопролол', 'бисопролол'),
  wordRule('конкор', 'конкор'),
  wordRule('метопралол', 'метопролол'),
  wordRule('метопролол', 'метопролол'),
  wordRule('беталок', 'беталок'),
  wordRule('небиволол', 'небиволол'),
  wordRule('карведелол', 'карведилол'),
  wordRule('карведилол', 'карведилол'),
  wordRule('небилет', 'небилет'),

  // Блокаторы кальциевых каналов
  wordRule('амлодепин', 'амлодипин'),
  wordRule('нифедепин', 'нифедипин'),
  wordRule('верапомил', 'верапамил'),
  wordRule('дилтиазем', 'дилтиазем'),
  wordRule('лерканидепин', 'лерканидипин'),

  // ИАПФ / сартаны
  wordRule('эналоприл', 'эналаприл'),
  wordRule('лизиноприл', 'лизиноприл'),
  wordRule('переноприл', 'периндоприл'),
  wordRule('периндоприл', 'периндоприл'),
  wordRule('рамиприл', 'рамиприл'),
  wordRule('каптоприл', 'каптоприл'),
  wordRule('валсортан', 'валсартан'),
  wordRule('валсартан', 'валсартан'),
  wordRule('лозортан', 'лозартан'),
  wordRule('лозартан', 'лозартан'),
  wordRule('кандесортан', 'кандесартан'),
  wordRule('кандесартан', 'кандесартан'),
  wordRule('телмисортан', 'телмисартан'),
  wordRule('телмисартан', 'телмисартан'),

  // Диуретики
  wordRule('индапомид', 'индапамид'),
  wordRule('индапамид', 'индапамид'),
  wordRule('торасимид', 'торасемид'),
  wordRule('торасемид', 'торасемид'),
  wordRule('фуросимид', 'фуросемид'),
  wordRule('фуросемид', 'фуросемид'),
  wordRule('гидрохлоротиазид', 'гидрохлоротиазид'),
  wordRule('верошпирон', 'верошпирон'),
  wordRule('спиронолактон', 'спиронолактон'),
  wordRule('эплеринон', 'эплеренон'),
  wordRule('эспиро', 'эспиро'),

  // Статины
  wordRule('розувостатин', 'розувастатин'),
  wordRule('розовостатин', 'розувастатин'),
  wordRule('аторвостатин', 'аторвастатин'),
  wordRule('симвостатин', 'симвастатин'),
  wordRule('правостатин', 'правастатин'),

  // Антиаритмики
  wordRule('этацизин', 'этацизин'),
  wordRule('кордорон', 'кордарон'),
  wordRule('амиодарон', 'амиодарон'),
  wordRule('соталол', 'соталол'),
  wordRule('пропафинон', 'пропафенон'),
  wordRule('пропафенон', 'пропафенон'),
  wordRule('аллапинин', 'аллапинин'),

  // Комбинированные препараты
  wordRule('нолипрел', 'нолипрел'),
  wordRule('валодип', 'валодип'),
  wordRule('кантаб', 'кантаб'),
  wordRule('илпио', 'илпио'),

  // Нитраты
  wordRule('нитроглецирин', 'нитроглицерин'),
  wordRule('нитросорбит', 'нитросорбид'),
  wordRule('моночинкве', 'моночинкве'),

  // SGLT2-ингибиторы
  wordRule('джардинз', 'джардинс'),
  wordRule('джардинс', 'джардинс'),
  wordRule('эмпаглифлозин', 'эмпаглифлозин'),
  wordRule('форсиго', 'форсига'),
  wordRule('дапаглефлозин', 'дапаглифлозин'),

  // Центральные антигипертензивные
  wordRule('физиотенс', 'физиотенз'),
  wordRule('моксонидин', 'моксонидин'),

  // Щитовидная железа
  wordRule('л тироксин', 'L-тироксин'),
  wordRule('эль тироксин', 'L-тироксин'),
  wordRule('левотироксин', 'левотироксин'),
  wordRule('эутирокс', 'эутирокс'),

  // Другие часто используемые
  wordRule('омепрозол', 'омепразол'),
  wordRule('пантопрозол', 'пантопразол'),
  wordRule('аспаркам', 'аспаркам'),
  wordRule('панангин', 'панангин'),
  wordRule('милдронат', 'милдронат'),
  wordRule('мельдоний', 'мельдоний'),
  wordRule('мексидол', 'мексидол'),
  wordRule('триметазидин', 'триметазидин'),
  wordRule('предуктал', 'предуктал'),
];

// ─── 4. Единицы измерения ────────────────────────────────────────────────────

const UNITS_CORRECTIONS: ReplacementRule[] = [
  // мм рт.ст. — все возможные вариации Whisper
  regexRule(/мм\s*рт\.?\s*ст\.?/giu, 'мм рт.ст.'),
  regexRule(/миллиметров?\s+ртутного\s+столба/giu, 'мм рт.ст.'),
  regexRule(/мм\s*ртутного\s*столба/giu, 'мм рт.ст.'),
  regexRule(/(?<![а-яёa-z])мм\s*р\s*т\s*с\s*т(?![а-яёa-z])/giu, 'мм рт.ст.'),
  regexRule(/мм\.\s*рт\.\s*ст\.\s*\./giu, 'мм рт.ст.'),
  regexRule(/(?<![а-яёa-z])ммртст(?![а-яёa-z])/giu, 'мм рт.ст.'),
  regexRule(/мм\s+рт\s+ст/giu, 'мм рт.ст.'),
  regexRule(/миллиметры?\s+ртутного/giu, 'мм рт.ст.'),
  regexRule(/уд(?:аров)?\s*(?:в|\/)\s*мин(?:уту)?/giu, 'уд/мин'),
  regexRule(/(?<![а-яёa-z])мкг(?![а-яёa-z])/giu, 'мкг'),
  regexRule(/микрограмм/giu, 'мкг'),
  regexRule(/миллиграмм/giu, 'мг'),
  regexRule(/(?<![а-яёa-z])ммоль\s*(?:на|\/)\s*л(?:итр)?/giu, 'ммоль/л'),
  regexRule(/(?<![а-яёa-z])мкмоль\s*(?:на|\/)\s*л(?:итр)?/giu, 'мкмоль/л'),
  regexRule(/(?<![а-яёa-z])пмоль\s*(?:на|\/)\s*л(?:итр)?/giu, 'пмоль/л'),
  regexRule(/(?<![а-яёa-z])г\s*(?:на|\/)\s*л(?:итр)?(?![а-яёa-z])/giu, 'г/л'),
  regexRule(/мк\s*ме\s*(?:на|\/)\s*мл/giu, 'мкМЕ/мл'),
  regexRule(/(?<![а-яёa-z])мл\s*(?:на|\/)\s*мин/giu, 'мл/мин'),
  regexRule(/килограмм(?:ов)?\s*(?:на|\/)\s*м(?:етр)?\s*(?:квадратн|²|2)/giu, 'кг/м²'),
];

// ─── 5. Фонетические замены (Whisper слышит не то) ───────────────────────────
// Специфичные случаи, когда Whisper стабильно ошибается.

const PHONETIC_CORRECTIONS: ReplacementRule[] = [
  // "эн уай эйч эй" → NYHA (если проскользнуло)
  wordRule('эн уай эйч эй', 'NYHA'),
  wordRule('си эйч эй ди эс', 'CHA₂DS₂-VASc'),

  // Классы ХСН
  regexRule(/функциональный\s+класс\s+(\d)/giu, 'ФК $1'),
  regexRule(/(?<![а-яёa-z])фк\s*(\d)/giu, 'ФК $1'),

  // Степени
  regexRule(/(\d)\s*(?:ой|ый|ая|я)\s+степен/giu, '$1 степен'),

  // Стадии
  regexRule(/(\d)\s*(?:ой|ый|ая|я)\s+стади/giu, '$1 стади'),

  // ── Частые ошибки Whisper в медицинских терминах ──
  // "визикулярное" → "везикулярное"
  regexRule(/визикулярн/giu, 'везикулярн'),
  // "косно-суставная" → "костно-суставная"
  regexRule(/косно[- ]?суставн/giu, 'костно-суставн'),
  // "розовостатин" → "розувастатин"
  wordRule('розовостатин', 'розувастатин'),
  // "урозы" / "уразы" → "оразы" (Ураза/пост)
  wordRule('урозы', 'оразы'),
  wordRule('уразы', 'оразы'),
  // "слизистобледные" → "слизистые бледно" (склеенные слова)
  regexRule(/слизистобледн/giu, 'слизистые бледно-'),
  // "слизисто бледные" → "слизистые бледно-"
  regexRule(/слизисто\s*бледн/giu, 'слизистые бледно-'),
  // "на квадратный метр" → "кг/м²" (в контексте ИМТ)
  regexRule(/кг\s+на\s+квадратный\s+метр/giu, 'кг/м²'),
  // "миллиметров ртутного столба" → "мм рт.ст."
  regexRule(/миллиметр\S*\s+ртутного\s+столба/giu, 'мм рт.ст.'),
  // "эутериоз" → "эутиреоз"
  wordRule('эутериоз', 'эутиреоз'),
  // "лоббильность" / "лобильность" → "лабильность"
  wordRule('лоббильность', 'лабильность'),
  wordRule('лобильность', 'лабильность'),
  // "дислепидемия" → "дислипидемия"
  wordRule('дислепидемия', 'дислипидемия'),
  // "миомоматки" → "миома матки" (склеенные слова)
  regexRule(/миомоматки/giu, 'миома матки'),
  // "СМАТ" → "СМАД" (суточное мониторирование АД)
  wordRule('СМАТ', 'СМАД'),
  wordRule('смат', 'СМАД'),
  // "аускультативное дыхание" → "Аускультативно: дыхание"
  regexRule(/аускультативное\s+дыхание/giu, 'Аускультативно: дыхание'),
  // Мусор от Whisper — титры видео/субтитров
  regexRule(/\s*(?:Р|р)едактор\s+субтитров[\s\S]*$/giu, ''),
];

// ─── 5a. Форматирование жизненных показателей ────────────────────────────────
// АД "144 на 97" → "144/97 мм рт.ст.", ЧСС "78 ударов в минуту" → "78 уд/мин"

const VITALS_FORMATTING: ReplacementRule[] = [
  // АД: "АД 144 на 97" → "АД – 144/97 мм рт.ст."
  // Также захватываем случай когда "мм рт ст" уже стоит после чисел
  regexRule(
    /(?<![а-яёa-z])АД\s*[-–—]?\s*(\d{2,3})\s+на\s+(\d{2,3})\s*(?:мм\s*рт\.?\s*ст\.?)?/giu,
    'АД – $1/$2 мм рт.ст.'
  ),
  // "АД 144/97" с/без единиц → нормализуем
  regexRule(
    /(?<![а-яёa-z])АД\s*[-–—]?\s*(\d{2,3})\/(\d{2,3})\s*(?:мм\s*рт\.?\s*ст\.?)?/giu,
    'АД – $1/$2 мм рт.ст.'
  ),
  // "давление 144 на 97" (если "артериальное давление" уже заменено на АД, но текст другой)
  regexRule(
    /давлени\S*\s+(\d{2,3})\s+на\s+(\d{2,3})\s*(?:мм\s*рт\.?\s*ст\.?)?/giu,
    'АД – $1/$2 мм рт.ст.'
  ),

  // ЧСС: "ЧСС 78 ударов в минуту" → "ЧСС – 78 уд/мин"
  regexRule(
    /(?<![а-яёa-z])ЧСС\s*[-–—]?\s*(\d{2,3})\s*(?:ударов?\s*в\s*минуту|уд\s*(?:в|\/)\s*мин\S*)/giu,
    'ЧСС – $1 уд/мин'
  ),
  // "ЧСС 78" без единиц → добавляем
  regexRule(
    /(?<![а-яёa-z])ЧСС\s*[-–—]?\s*(\d{2,3})(?!\s*[-–—/]?\s*\d)(?!\s*уд)/giu,
    'ЧСС – $1 уд/мин'
  ),
  // "пульс 78" → "ЧСС – 78 уд/мин"
  regexRule(
    /(?<![а-яёa-z])пульс\s+(\d{2,3})(?!\s*[-–—/]?\s*\d)/giu,
    'ЧСС – $1 уд/мин'
  ),
];

// ─── 6. Нормализация пунктуации / пробелов ───────────────────────────────────

// ─── 6a. Голосовые команды пунктуации ────────────────────────────────────────
// Врач может произносить знаки пунктуации голосом — заменяем на символы.

const VOICE_PUNCTUATION: ReplacementRule[] = [
  // Скобки — все варианты произношения:
  // "открыть скобку" / "открытая скобка" / "скобка открывается" / "скобку открыть"
  regexRule(/\s*(?:открыть|открытая|открытую|открой)\s+скобк[уиа]\s*/giu, ' ('),
  regexRule(/\s*скобк[аи]\s+открывается\s*/giu, ' ('),
  regexRule(/\s*скобк[уа]\s+(?:открыть|открой)\s*/giu, ' ('),
  // "закрыть скобку" / "закрытая скобка" / "скобка закрывается"
  regexRule(/\s*(?:закрыть|закрытая|закрытую|закрой)\s+скобк[уиа]\s*/giu, ') '),
  regexRule(/\s*,?\s*скобк[аи]\s+закрывается\s*/giu, ') '),
  regexRule(/\s*скобк[уа]\s+(?:закрыть|закрой)\s*/giu, ') '),
  // "в скобках ... конец скобок"
  regexRule(/\s*(?:в\s+)?скобк(?:ах|и)\s*/giu, ' ('),
  regexRule(/\s*конец\s+скобо?к\s*/giu, ') '),

  // Двоеточие — с/без запятой перед ним
  regexRule(/\s*,?\s*двоеточие\s*/giu, ': '),

  // Точка (как голосовая команда — в конце предложения или перед названием секции)
  regexRule(/\s*,?\s*точка\s*\.\s*/giu, '. '),
  regexRule(/\s*,?\s*точка\s+(?=[А-ЯA-Z])/gu, '. '),

  // Вопросительный знак
  regexRule(/\s+(?:вопросительный\s+знак|знак\s+вопроса)\s*/giu, '? '),

  // Тире
  regexRule(/\s+тире\s+/giu, ' — '),

  // Восклицательный знак
  regexRule(/\s+(?:восклицательный\s+знак|знак\s+восклицания)\s*/giu, '! '),

  // Точка с запятой
  regexRule(/\s+точка\s+с\s+запятой\s*/giu, '; '),

  // Запятая (только если произнесено как команда, а не как пауза в речи)
  regexRule(/\s+(?:поставь|ставлю|ставим)\s+запятую\s*/giu, ', '),

  // Новая строка / новый абзац
  regexRule(/\s+(?:новая\s+строка|новый\s+абзац|с\s+новой\s+строки)\s*/giu, '\n'),
];

const PUNCTUATION_FIXES: ReplacementRule[] = [
  // Множественные пробелы → один
  regexRule(/\s{2,}/g, ' '),
  // Пробел перед точкой/запятой
  regexRule(/\s+([.,;:!?])/g, '$1'),
  // Пробел после открывающей скобки и перед закрывающей — убираем
  regexRule(/\(\s+/g, '('),
  regexRule(/\s+\)/g, ')'),
  // Заглавная после точки
  regexRule(/\.\s+([а-яё])/gu, (match) => match.toUpperCase()),
];

// ═══════════════════════════════════════════════════════════════════════════════
// ОСНОВНАЯ ФУНКЦИЯ
// ═══════════════════════════════════════════════════════════════════════════════

/** Все правила замены в порядке применения */
const ALL_RULES: ReplacementRule[] = [
  ...LATIN_CARDIOLOGY_ABBREVIATIONS,
  ...GENERAL_MEDICAL_ABBREVIATIONS,
  ...DRUG_NAME_CORRECTIONS,
  ...UNITS_CORRECTIONS,
  ...PHONETIC_CORRECTIONS,
  ...VITALS_FORMATTING,
  ...VOICE_PUNCTUATION,
  ...PUNCTUATION_FIXES,
];

/**
 * Применяет медицинский словарь постобработки к тексту Whisper.
 * Исправляет типичные ошибки транскрипции медицинских терминов.
 * Включает базовые правила + пользовательские замены.
 */
export function applyMedicalDictionary(text: string): string {
  let result = text;
  // Базовые правила
  for (const rule of ALL_RULES) {
    if (typeof rule.replacement === 'string') {
      result = result.replace(rule.pattern, rule.replacement);
    } else {
      result = result.replace(rule.pattern, rule.replacement as (...args: string[]) => string);
    }
  }
  // Пользовательские замены (применяются после базовых, имеют приоритет)
  for (const rule of userRules) {
    if (typeof rule.replacement === 'string') {
      result = result.replace(rule.pattern, rule.replacement);
    } else {
      result = result.replace(rule.pattern, rule.replacement as (...args: string[]) => string);
    }
  }
  return result.trim();
}

/** Количество правил в словаре (для логирования) */
export const DICTIONARY_RULE_COUNT = ALL_RULES.length;

// ═══════════════════════════════════════════════════════════════════════════════
// ПОЛЬЗОВАТЕЛЬСКИЕ ЗАМЕНЫ
// ═══════════════════════════════════════════════════════════════════════════════

const DATA_DIR = path.join(__dirname, '..', 'data');
const USER_CORRECTIONS_PATH = path.join(DATA_DIR, 'user-corrections.json');

/** Текущие пользовательские замены (загружаются при старте) */
let userCorrections: UserCorrection[] = [];
/** Скомпилированные правила из пользовательских замен */
let userRules: ReplacementRule[] = [];

function compileUserRules(): void {
  userRules = userCorrections.map((c) => wordRule(c.wrong, c.correct));
}

/** Загружает пользовательские замены из JSON-файла */
export async function loadUserCorrections(): Promise<UserCorrection[]> {
  try {
    if (!existsSync(USER_CORRECTIONS_PATH)) {
      userCorrections = [];
      userRules = [];
      return [];
    }
    const raw = await readFile(USER_CORRECTIONS_PATH, 'utf-8');
    const data: UserCorrectionsData = JSON.parse(raw);
    userCorrections = Array.isArray(data.corrections) ? data.corrections : [];
    compileUserRules();
    console.log(`[medical-dictionary] Loaded ${userCorrections.length} user corrections`);
    return userCorrections;
  } catch (err) {
    console.warn('[medical-dictionary] Failed to load user corrections:', err);
    userCorrections = [];
    userRules = [];
    return [];
  }
}

async function saveUserCorrections(): Promise<void> {
  if (!existsSync(DATA_DIR)) {
    await mkdir(DATA_DIR, { recursive: true });
  }
  const data: UserCorrectionsData = { corrections: userCorrections };
  await writeFile(USER_CORRECTIONS_PATH, JSON.stringify(data, null, 2), 'utf-8');
}

/** Добавляет или обновляет пользовательскую замену */
export async function addUserCorrection(wrong: string, correct: string): Promise<UserCorrection> {
  const trimWrong = wrong.trim();
  const trimCorrect = correct.trim();

  // Обновляем существующую, если есть
  const existing = userCorrections.find(
    (c) => c.wrong.toLowerCase() === trimWrong.toLowerCase()
  );
  if (existing) {
    existing.correct = trimCorrect;
    existing.createdAt = new Date().toISOString();
    compileUserRules();
    await saveUserCorrections();
    return existing;
  }

  const correction: UserCorrection = {
    id: `corr_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    wrong: trimWrong,
    correct: trimCorrect,
    createdAt: new Date().toISOString(),
  };
  userCorrections.push(correction);
  compileUserRules();
  await saveUserCorrections();
  return correction;
}

/** Удаляет пользовательскую замену по ID */
export async function deleteUserCorrection(id: string): Promise<boolean> {
  const index = userCorrections.findIndex((c) => c.id === id);
  if (index === -1) return false;
  userCorrections.splice(index, 1);
  compileUserRules();
  await saveUserCorrections();
  return true;
}

/** Возвращает все пользовательские замены */
export function getUserCorrections(): UserCorrection[] {
  return [...userCorrections];
}

