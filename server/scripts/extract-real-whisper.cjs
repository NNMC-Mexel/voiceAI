#!/usr/bin/env node
/**
 * Извлекает RAW WHISPER TEXT из structure-logs/*.log для создания
 * тест-набора на реальных транскриптах (а не на «причёсанной» диктовке).
 *
 * Берём САМЫЙ СВЕЖИЙ лог для каждого уникального префикса (первые 80 символов),
 * чтобы не дублировать одинаковые кейсы.
 */
const fs = require('fs');
const path = require('path');

const LOGS = path.resolve(__dirname, '..', 'temp', 'structure-logs');
const OUT = path.resolve(__dirname, '..', 'temp', '_real_whisper');

function extractRaw(content) {
  const m = content.match(/--- RAW WHISPER TEXT \((\d+) chars\) ---\s*\n([\s\S]*?)\n\s*\n---\s/u);
  if (!m) return null;
  return { chars: parseInt(m[1], 10), text: m[2].trim() };
}

function slug(text) {
  const first = text.slice(0, 300).toLowerCase();
  // Определяем кейс по ключевым маркерам
  if (/еженов|есжанов/i.test(first)) return 'eszhanov_whisper';
  if (/ксталасова|кстала|кунаржая|куныржай/i.test(first)) return 'kstalasova_whisper';
  if (/дюсенов|дю.енов/i.test(first)) return 'dyusenov_whisper';
  if (/каппаева|каппаев/i.test(first)) return 'kappaeva_whisper';
  if (/сердцебиен|купируемые/i.test(first)) return 'cardio_whisper';
  if (/выпадение волос|сонливость|тиреотокс/i.test(first)) return 'endo_whisper';
  if (/гликозилирован|гликирован/i.test(first) && !/жалобы/i.test(first)) return 'labs_only_whisper';
  // fallback: первые 20 слов-нормализованных
  return 'other_' + first.replace(/[^a-zа-я0-9]+/giu, '_').slice(0, 40);
}

function main() {
  if (!fs.existsSync(OUT)) fs.mkdirSync(OUT, { recursive: true });
  const files = fs.readdirSync(LOGS)
    .filter((f) => f.endsWith('.log'))
    .sort()
    .reverse(); // новейшие первыми

  const bySlug = new Map();
  for (const f of files) {
    const raw = extractRaw(fs.readFileSync(path.join(LOGS, f), 'utf-8'));
    if (!raw) continue;
    // Пропускаем синтетические входы: их признак — формат «Ф.И.О.:» с колонкой и заглавными
    if (/^Ф\.И\.О\.:\s+[А-ЯЁ]/u.test(raw.text)) continue;
    const s = slug(raw.text);
    // Берём самый свежий (первый, т.к. сортировка reverse по имени)
    if (!bySlug.has(s)) bySlug.set(s, { log: f, raw });
  }

  console.log(`Found ${bySlug.size} unique real Whisper cases:\n`);
  for (const [s, { log, raw }] of bySlug) {
    const outFile = path.join(OUT, `${s}.txt`);
    fs.writeFileSync(outFile, raw.text, 'utf-8');
    console.log(`  ${s.padEnd(25)} ${raw.chars.toString().padStart(5)} chars  ← ${log}`);
  }
  console.log(`\nSaved to: ${OUT}`);
}

main();
