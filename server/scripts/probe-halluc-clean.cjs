#!/usr/bin/env node
/**
 * Прогоняет _debug_*_raw.txt через обновлённую cleanWhisperHallucinations
 * и печатает потерю символов. Если потеря >20% — паттерны всё ещё слишком агрессивны.
 */
const fs = require('fs');
const path = require('path');

function cleanWhisperHallucinations(text) {
  let cleaned = text.replace(
    /((?:[\wа-яёА-ЯЁ][\wа-яёА-ЯЁ.,\-]*\s+){2,11}[\wа-яёА-ЯЁ][\wа-яёА-ЯЁ.,\-]*)\s+(?:\1\s+){2,}/giu,
    '$1 '
  );
  cleaned = cleaned.replace(
    /((?:[а-яёА-ЯЁ]+\s+){2,6}\d[\d.\-]+(?:\s+года?)?)\s+\d+(?:\s+\1\s+\d+){2,}/giu,
    (match) => {
      const firstEnd = match.indexOf(match.trim().split(/\s+/).slice(0, 4).join(' '), 1);
      if (firstEnd > 0) return match.substring(0, firstEnd).trim();
      return match.split(/(?=Общий|ОАК|Б\/х|БХ|ЭКГ|ЭхоКГ|УЗДГ|Ферритин|ОАМ)/iu)[0].trim();
    }
  );
  cleaned = cleaned.replace(
    /((?:[а-яёА-ЯЁa-zA-Z]+\s+){1,4}[а-яёА-ЯЁa-zA-Z]+)[,.\s]+(?:\1[,.\s]+){2,}/giu,
    '$1, '
  );
  // Pattern 2 — UPDATED
  cleaned = cleaned.replace(
    /(?:\s+(?:Дождь|Осторожно|Всё хорошо|не знаю|Тогда мы)[.,!?]?\s*){3,}[\s\S]*$/gi,
    ''
  );
  cleaned = cleaned.replace(
    /(\s+[\wа-яёА-ЯЁ]+[.,!?]?)\1{2,}\s*$/giu,
    ''
  );
  cleaned = cleaned.replace(
    /(?:\s*([а-яёА-ЯЁ]{3,})\s*[.!?,]\s*){3,}/giu,
    (match, word) => {
      const w = word.toLowerCase();
      const repeats = match.toLowerCase().split(w).length - 1;
      return repeats >= 3 ? '. ' : match;
    }
  );
  cleaned = cleaned.replace(
    /\s*(?:Субтитры\s+создавал?\s+\S+|Продолжение\s+следует\.{0,3}|Спасибо\s+за\s+(?:просмотр|внимание)\.?|Подписывайтесь\s+на\s+канал\.?)\s*/giu,
    ' '
  );
  cleaned = cleaned.replace(
    /,\s+[а-яёА-ЯЁa-zA-Z]{2,4}\s+(?=[А-ЯЁA-Z])/gu,
    '. '
  );
  const medicalLatin = /^(?:CRTD?|CRT-D|MRI|NYHA|EHRA|VVIR|DDDR|SpO2|ProBNP|Medtronic|Quadra|Assura|Brava|Compi|EssentioDR|Sphera|FV|HIV|HbA1c|NT|BNP|COPD|COVID|TAVI|PCI|AV|ECG|CT|BMI|GFR|INR|LMWH|UFH|ACE|ARB|SGLT2|DPP-4|GLP-1|TSH|T[34]|CRP|ESR|WBC|RBC|PLT|Hb|Ht|MCH|MCV|MCHC|ALP|ALT|AST|GGT|LDH|CPK|CK|Fe|TIBC|HBsAg|Anti|IgG|IgM|EF|LVEF|RVSP|LA|LV|RV|RA|IVS|LVPW|CO|CI|SV|EDV|ESV|EDD|ESD)$/i;
  cleaned = cleaned.replace(
    /(?<=[.!?]\s|^)[^.!?]*?(?=[.!?]|$)/gu,
    (sentence) => {
      const latinWords = sentence.match(/\b[a-zA-Z]{3,}\b/g) || [];
      const nonMedical = latinWords.filter(w => !medicalLatin.test(w));
      if (nonMedical.length >= 2 && nonMedical.length > latinWords.length * 0.4) {
        return '';
      }
      return sentence;
    }
  );
  const latinGarbageMatch = cleaned.match(
    /(?:[a-zA-Z]{3,}\s+){2,}[^.]*(?:\.\s*(?:[а-яёА-ЯЁ]+\s+){0,3}(?:[a-zA-Z]{2,}|[&@#$%])\s*){2,}[\s\S]*$/u
  );
  if (latinGarbageMatch && latinGarbageMatch.index !== undefined) {
    if (latinGarbageMatch.index > cleaned.length * 0.7) {
      cleaned = cleaned.substring(0, latinGarbageMatch.index).trim();
    }
  }
  cleaned = cleaned.replace(/(?:\s*\d+\.\s*){5,}\s*$/gu, '');
  cleaned = cleaned.replace(
    /\s*(?:cycling|following|next)\s*[.\s]*(?:\d+\.\s*){2,}[\s\S]*$/giu,
    ''
  );
  cleaned = cleaned.replace(/[\u0590-\u05FF\u0600-\u06FF\u3000-\u9FFF\uAC00-\uD7AF]+\S*/gu, '');
  cleaned = cleaned.replace(/(?:\s+[A-Z][a-z]+){2,}\s*\.?\s*$/g, '');
  cleaned = cleaned.replace(
    /\s*АД\s+мм\s+рт\.?\s*ст\.?,?\s*ЧСС\s+уд[\s/\-]*мин,?\s*SpO2\.?\s*/giu,
    ' '
  );
  cleaned = cleaned.replace(
    /\s*Жалобы\s+пациента,?\s+анамнез\s+заболевания,?\s+объективный\s+осмотр[^.]*\.?\s*/giu,
    ' '
  );
  return cleaned.trim();
}

const rawPath = process.argv[2] || path.resolve(__dirname, '..', 'temp', '_debug_1776676730008_raw.txt');
const raw = fs.readFileSync(rawPath, 'utf-8');
const cleaned = cleanWhisperHallucinations(raw);
const lossPct = ((raw.length - cleaned.length) / raw.length * 100).toFixed(1);

console.log(`raw:      ${raw.length} chars`);
console.log(`cleaned:  ${cleaned.length} chars`);
console.log(`loss:     ${raw.length - cleaned.length} chars (${lossPct}%)`);
console.log(`\n--- LAST 400 CHARS OF CLEANED ---\n${cleaned.slice(-400)}`);
