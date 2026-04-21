#!/usr/bin/env node
/**
 * Batch: рендерит PDF для всех raw Whisper-кейсов:
 *  - temp/*_input.txt (одиночные inputs, напр. kstalasova_input.txt)
 *  - temp/qa_with_raw.txt (кэш нескольких записей с секциями RAW WHISPER TEXT)
 */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const TEMP = path.resolve(__dirname, '..', 'temp');
const CACHE = path.join(TEMP, 'qa_with_raw.txt');
const SELF_INPUT = path.join(__dirname, 'qa-to-pdf.cjs');

function extractRawSections(text) {
  const out = [];
  const labelRe = /^\[(.+?)\] \(\d+ KB\)/mu;
  const parts = text.split(/^={40,}$/mu).map((p) => p.trim()).filter(Boolean);
  for (let i = 0; i < parts.length - 1; i++) {
    const m = parts[i].match(labelRe);
    if (!m) continue;
    const rawMatch = parts[i + 1].match(/--- RAW WHISPER TEXT \[\d+ chars\] ---\s*\n([\s\S]*?)(?=\n---\s|\n={40,}|$)/u);
    if (!rawMatch) continue;
    out.push({ label: m[1], raw: rawMatch[1].trim() });
  }
  return out;
}

function slugFromLabel(label) {
  // "ии аудио Дюсенов 06-04-2026.mp3" → "dyusenov"
  // Берём первое «фамилию»-подобное слово после "аудио"
  const m = label.match(/аудио\s+([А-Яа-яЁё]+)/u);
  const lat = { а:'a',б:'b',в:'v',г:'g',д:'d',е:'e',ё:'e',ж:'zh',з:'z',и:'i',й:'y',к:'k',л:'l',м:'m',н:'n',о:'o',п:'p',р:'r',с:'s',т:'t',у:'u',ф:'f',х:'h',ц:'ts',ч:'ch',ш:'sh',щ:'sch',ъ:'',ы:'y',ь:'',э:'e',ю:'yu',я:'ya' };
  const name = (m ? m[1] : label).toLowerCase();
  return name.replace(/./gu, (c) => lat[c] ?? (/[a-z0-9]/i.test(c) ? c : '_'));
}

async function main() {
  const tasks = [];

  // 1) *_input.txt
  for (const f of fs.readdirSync(TEMP)) {
    if (!/_input\.txt$/.test(f)) continue;
    tasks.push({ input: path.join(TEMP, f), output: path.join(TEMP, f.replace(/\.txt$/, '.pdf')) });
  }

  // 2) qa_with_raw.txt → temp_raw/<slug>.txt → PDF
  if (fs.existsSync(CACHE)) {
    const rawDir = path.join(TEMP, '_raw_extracts');
    fs.mkdirSync(rawDir, { recursive: true });
    const sections = extractRawSections(fs.readFileSync(CACHE, 'utf-8'));
    for (const sec of sections) {
      const slug = slugFromLabel(sec.label);
      const inputPath = path.join(rawDir, `${slug}.txt`);
      fs.writeFileSync(inputPath, sec.raw, 'utf-8');
      tasks.push({ input: inputPath, output: path.join(TEMP, `${slug}.pdf`), label: sec.label });
    }
  }

  console.log(`Queued ${tasks.length} PDF(s):`);
  tasks.forEach((t, i) => console.log(`  ${i + 1}. ${path.relative(TEMP, t.input)} → ${path.relative(TEMP, t.output)}${t.label ? `  (${t.label})` : ''}`));

  for (const t of tasks) {
    console.log(`\n--- Rendering ${path.basename(t.output)} ---`);
    try {
      execFileSync(process.execPath, [SELF_INPUT, t.input, t.output], { stdio: 'inherit', cwd: path.resolve(__dirname, '..') });
    } catch (e) {
      console.error(`FAILED: ${t.input}: ${e.message}`);
    }
  }

  console.log(`\nDone. Output dir: ${TEMP}`);
}

main().catch((e) => { console.error('Fatal:', e); process.exit(1); });
