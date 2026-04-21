import { spawn } from 'child_process';
import { readFile, writeFile, unlink, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import type { TranscriptionResult, WhisperConfig } from '../types.js';
import { applyMedicalDictionary, DICTIONARY_RULE_COUNT } from './medical-dictionary.js';

interface WhisperServerResponse {
  text: string;
  language: string;
  elapsed: number;
  chunks?: number;
  chunk_details?: Array<{ chunk: number; duration: number; chars: number; elapsed: number; avg_logprob?: number }>;
  avg_logprob?: number;
  low_confidence?: boolean;
}

// КОРОТКИЙ initial_prompt — только структура консультации.
// Whisper hard limit: 224 токена. Всё что сверху — молча обрезается с НАЧАЛА.
// Предметную лексику передаём через hotwords (faster-whisper ≥1.0) — нет лимита.
// КОРОТКИЙ initial_prompt — только структура консультации.
// Whisper hard limit: 224 токена. Всё что сверху — молча обрезается с НАЧАЛА.
// НЕ включать единицы измерения (мм рт.ст., уд/мин) и аббревиатуры —
// они «протекают» в транскрипцию как мусорные фрагменты.
const MEDICAL_INITIAL_PROMPT =
  'Медицинская консультация. Жалобы, анамнез, осмотр, диагноз, рекомендации.';

// HOTWORDS: ОТКЛЮЧЕНЫ.
// Тестирование показало, что любой список hotwords в faster-whisper вызывает:
// 1) Галлюцинации-циклы (одна фраза повторяется 10-15+ раз)
// 2) «Протекание» hotwords в текст транскрипции как мусор
// Все ошибки распознавания медтерминов исправляются в medical-dictionary.ts
// (постобработка), что надёжнее и не вызывает побочных эффектов.
const MEDICAL_HOTWORDS = '';

export class WhisperService {
  private config: WhisperConfig;
  private tempDir: string;

  constructor(config: WhisperConfig) {
    this.config = config;
    this.tempDir = './temp';
  }

  async init(): Promise<void> {
    if (!existsSync(this.tempDir)) {
      await mkdir(this.tempDir, { recursive: true });
    }
  }

  async healthCheck(): Promise<boolean> {
    // Если настроен persistent HTTP-сервер — проверяем его
    if (this.config.serverUrl) {
      try {
        const response = await fetch(`${this.config.serverUrl}/health`);
        return response.ok;
      } catch {
        return false;
      }
    }
    // Fallback: проверяем наличие temp-директории
    return existsSync(this.tempDir);
  }

  async transcribe(audioBuffer: Buffer, filename: string): Promise<TranscriptionResult> {
    const safeFilename = path.basename(filename);
    const tempAudioPath = path.join(this.tempDir, `${Date.now()}_${safeFilename}`);
    await writeFile(tempAudioPath, audioBuffer);

    try {
      // transcribeFile already applies applyMedicalDictionary internally
      return await this.transcribeFile(tempAudioPath, filename);
    } finally {
      try {
        await unlink(tempAudioPath);
      } catch {
        // Ignore cleanup errors
      }
    }
  }

  /**
   * Remove Whisper hallucination loops: repeated phrases like
   * "снижение памяти, снижение памяти, снижение памяти" → "снижение памяти"
   * Also removes trailing garbage ("Дождь. Дождь. Дождь..." etc.)
   */
  private cleanWhisperHallucinations(text: string): string {
    // 1a. Remove long repeated phrases (3+ repetitions of 3-12 token phrases including numbers/dates)
    // e.g. "Общий анализ крови от 05-10-2025 года 295 Общий анализ крови от 05-10-2025 года 295..."
    let cleaned = text.replace(
      /((?:[\wа-яёА-ЯЁ][\wа-яёА-ЯЁ.,\-]*\s+){2,11}[\wа-яёА-ЯЁ][\wа-яёА-ЯЁ.,\-]*)\s+(?:\1\s+){2,}/giu,
      '$1 '
    );

    // 1b. Remove near-identical repeated phrases where only trailing number changes
    // e.g. "X от DATE года 295 X от DATE года 296 X от DATE года 297" → "X от DATE года 295"
    cleaned = cleaned.replace(
      /((?:[а-яёА-ЯЁ]+\s+){2,6}\d[\d.\-]+(?:\s+года?)?)\s+\d+(?:\s+\1\s+\d+){2,}/giu,
      (match) => {
        // Keep only the first occurrence
        const firstEnd = match.indexOf(match.trim().split(/\s+/).slice(0, 4).join(' '), 1);
        if (firstEnd > 0) return match.substring(0, firstEnd).trim();
        return match.split(/(?=Общий|ОАК|Б\/х|БХ|ЭКГ|ЭхоКГ|УЗДГ|Ферритин|ОАМ)/iu)[0].trim();
      }
    );

    // 1c. Remove repeated phrases (3+ consecutive repetitions of 1-5 word phrases)
    // e.g. "снижение памяти, снижение памяти, снижение памяти" → "снижение памяти"
    cleaned = cleaned.replace(
      /((?:[а-яёА-ЯЁa-zA-Z]+\s+){1,4}[а-яёА-ЯЁa-zA-Z]+)[,.\s]+(?:\1[,.\s]+){2,}/giu,
      '$1, '
    );

    // 2. Remove trailing hallucination garbage — non-medical babble at the end.
    // NB: слова «Раз», «Или», «Вы» убраны — в медицинской диктовке «раз в день»,
    // «или в течение», «вы принимаете» встречаются часто; на двойном повторе
    // («одну раз раз в день») regex уничтожал весь хвост документа.
    // Ужесточено до {3,} — три повтора точно галлюцинация, два могут быть речью.
    cleaned = cleaned.replace(
      /(?:\s+(?:Дождь|Осторожно|Всё хорошо|не знаю|Тогда мы)[.,!?]?\s*){3,}[\s\S]*$/gi,
      ''
    );

    // 3. Remove generic trailing garbage: any single word/phrase repeated 3+ times at end
    cleaned = cleaned.replace(
      /(\s+[\wа-яёА-ЯЁ]+[.,!?]?)\1{2,}\s*$/giu,
      ''
    );

    // 3a. Remove repeated words/phrases separated by periods: "Холостой. Холостой. Холостой." → ""
    cleaned = cleaned.replace(
      /(?:\s*([а-яёА-ЯЁ]{3,})\s*[.!?,]\s*){3,}/giu,
      (match, word) => {
        // Only remove if the same word repeats 3+ times
        const w = word.toLowerCase();
        const repeats = match.toLowerCase().split(w).length - 1;
        return repeats >= 3 ? '. ' : match;
      }
    );

    // 3b. Remove Whisper "subtitle" hallucinations that appear at chunk boundaries
    cleaned = cleaned.replace(
      /\s*(?:Субтитры\s+создавал?\s+\S+|Продолжение\s+следует\.{0,3}|Спасибо\s+за\s+(?:просмотр|внимание)\.?|Подписывайтесь\s+на\s+канал\.?)\s*/giu,
      ' '
    );

    // 4. Remove truncated words left after loop removal (e.g. "снижение пам" after removing "снижение памяти" repeats)
    // A truncated fragment is 2-4 chars of a word followed by a non-letter
    cleaned = cleaned.replace(
      /,\s+[а-яёА-ЯЁa-zA-Z]{2,4}\s+(?=[А-ЯЁA-Z])/gu,
      '. '
    );

    // 5. Remove inline Latin garbage sentences — any sentence with 2+ non-medical Latin words
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

    // 5b. Remove trailing hallucination with mixed Latin nonsense words
    // Catches: "Гипотония Fay Riom Induca &i 6 5. Медицинская об 10 loosenhet..."
    // Heuristic: if we see 3+ non-medical Latin words in a short span near the end, truncate from there
    const latinGarbageMatch = cleaned.match(
      /(?:[a-zA-Z]{3,}\s+){2,}[^.]*(?:\.\s*(?:[а-яёА-ЯЁ]+\s+){0,3}(?:[a-zA-Z]{2,}|[&@#$%])\s*){2,}[\s\S]*$/u
    );
    if (latinGarbageMatch && latinGarbageMatch.index !== undefined) {
      // Only truncate if the garbage is in the last 30% of the text
      if (latinGarbageMatch.index > cleaned.length * 0.7) {
        cleaned = cleaned.substring(0, latinGarbageMatch.index).trim();
      }
    }

    // 5c. Remove trailing number sequence garbage: "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    cleaned = cleaned.replace(
      /(?:\s*\d+\.\s*){5,}\s*$/gu,
      ''
    );

    // 5d. Remove trailing "cycling/following" + number garbage
    cleaned = cleaned.replace(
      /\s*(?:cycling|following|next)\s*[.\s]*(?:\d+\.\s*){2,}[\s\S]*$/giu,
      ''
    );

    // 6. Remove multilingual garbage: Hebrew, Arabic, CJK, Turkish special chars
    cleaned = cleaned.replace(
      /[\u0590-\u05FF\u0600-\u06FF\u3000-\u9FFF\uAC00-\uD7AF]+\S*/gu,
      ''
    );

    // 6. Remove trailing multilingual garbage at end of text
    // Catches patterns like "Ally выпарии", "Master maneuver" at the end
    cleaned = cleaned.replace(
      /(?:\s+[A-Z][a-z]+){2,}\s*\.?\s*$/g,
      ''
    );

    // 7. Remove leaked initial_prompt / hotword fragments
    // These appear when Whisper injects prompt/hotword text into transcription
    // e.g. "АД мм рт.ст., ЧСС уд/мин, SpO2." or "Жалобы пациента, анамнез заболевания"
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

  async transcribeFile(audioPath: string, filename: string): Promise<TranscriptionResult> {
    const startTime = Date.now();

    const applyDict = (result: TranscriptionResult): TranscriptionResult => {
      // Log raw Whisper output BEFORE any processing
      console.log('\n\x1b[33m━━━ [WHISPER RAW] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m');
      console.log(result.text);
      console.log('\x1b[33m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\n');

      // First clean hallucinations, then apply dictionary
      const dehalucinated = this.cleanWhisperHallucinations(result.text);
      const corrected = applyMedicalDictionary(dehalucinated);
      if (corrected !== result.text) {
        console.log(`[medical-dictionary] Applied corrections (${DICTIONARY_RULE_COUNT} rules)`);
      }
      if (dehalucinated !== result.text) {
        console.log(`[whisper] Cleaned hallucinations: ${result.text.length} → ${dehalucinated.length} chars`);
      }

      // Log corrected Whisper text AFTER dictionary
      if (corrected !== result.text) {
        console.log('\n\x1b[36m━━━ [WHISPER + DICT] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m');
        console.log(corrected);
        console.log('\x1b[36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\n');
      }

      return { ...result, text: corrected };
    };

    // Шаг 1: Persistent HTTP whisper-сервер (нет перезагрузки модели → быстро)
    if (this.config.serverUrl) {
      try {
        const result = await this.transcribeWithServer(audioPath);
        return applyDict({
          ...result,
          duration: (Date.now() - startTime) / 1000,
        });
      } catch (error) {
        console.warn(`Whisper HTTP server unavailable, falling back to subprocess... (${error})`);
      }
    }

    // Шаг 2: whisper.cpp subprocess
    try {
      const result = await this.transcribeWithWhisperCpp(audioPath);
      return applyDict({
        ...result,
        duration: (Date.now() - startTime) / 1000,
      });
    } catch (error) {
      console.log(`whisper.cpp not available, trying faster-whisper... (${error})`);

      // Шаг 3: faster-whisper CUDA subprocess
      try {
        const result = await this.transcribeWithFasterWhisper(audioPath, this.config.device);
        return applyDict({
          ...result,
          duration: (Date.now() - startTime) / 1000,
        });
      } catch (fallbackError) {
        // Шаг 4: faster-whisper CPU (если CUDA не сработал)
        if (this.config.device === 'cuda') {
          try {
            console.log(`faster-whisper CUDA failed, retrying on CPU... (${fallbackError})`);
            const result = await this.transcribeWithFasterWhisper(audioPath, 'cpu');
            return applyDict({
              ...result,
              duration: (Date.now() - startTime) / 1000,
            });
          } catch (cpuError) {
            fallbackError = cpuError;
          }
        }

        // Шаг 5: Mock (только если явно разрешено)
        const allowMock = process.env.ALLOW_MOCK_TRANSCRIPTION === 'true';
        if (allowMock) {
          console.log(`faster-whisper not available, using mock... (${fallbackError})`);
          return this.getMockTranscription();
        }

        throw new Error(`Whisper transcription failed for ${filename}: ${fallbackError}`);
      }
    }
  }

  // ─── Persistent HTTP whisper-сервер ─────────────────────────────────────────

  private async transcribeWithServer(audioPath: string): Promise<Omit<TranscriptionResult, 'duration'>> {
    const controller = new AbortController();
    // 10 минут — с chunking каждый кусок транскрибируется отдельно
    const timeout = setTimeout(() => controller.abort(), 600_000);

    try {
      // Читаем файл и кодируем в base64 — это позволяет вызвать Whisper-сервер
      // на другой машине (например ПК с RTX 5090 в campus-сети)
      const audioBytes = await readFile(audioPath);
      const audioBase64 = audioBytes.toString('base64');

      const response = await fetch(`${this.config.serverUrl}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // НЕ передаём initial_prompt и hotwords — сервер использует свои встроенные.
        // Тесты показали, что:
        // 1) hotwords вызывают галлюцинации-циклы (одна фраза x15)
        // 2) клиентский initial_prompt хуже серверного (протекает в текст)
        // Серверный INITIAL_PROMPT содержит медтермины и работает стабильнее.
        body: JSON.stringify({
          audio_base64: audioBase64,
          language: this.config.language,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(`Whisper server error ${response.status}: ${err.error ?? response.statusText}`);
      }

      const data = (await response.json()) as WhisperServerResponse;
      if (data.chunks && data.chunks > 1) {
        console.log(`[whisper] Chunked transcription: ${data.chunks} chunks, ${data.elapsed?.toFixed(1)}s total`);
        if (data.chunk_details) {
          for (const c of data.chunk_details) {
            console.log(`  chunk ${c.chunk}: ${c.duration}s → ${c.chars} chars (${c.elapsed}s) logprob=${c.avg_logprob ?? 'n/a'}`);
          }
        }
      }
      if (typeof data.avg_logprob === 'number') {
        console.log(`[whisper] avg_logprob=${data.avg_logprob.toFixed(2)}${data.low_confidence ? ' [LOW CONFIDENCE — возможны галлюцинации]' : ''}`);
      }
      return { text: data.text, language: data.language };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Whisper server request timeout');
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  // ─── whisper.cpp subprocess ──────────────────────────────────────────────────

  private async transcribeWithWhisperCpp(audioPath: string): Promise<Omit<TranscriptionResult, 'duration'>> {
    return new Promise((resolve, reject) => {
      const args = [
        '-m', this.config.modelPath,
        '-f', audioPath,
        '-l', this.config.language,
        '--output-txt',
        '-otxt',
        ...(this.config.device === 'cuda' ? ['--gpu'] : []),
      ];

      const whisper = spawn('whisper-cpp', args);
      let stdout = '';
      let stderr = '';

      whisper.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      whisper.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      whisper.on('close', (code) => {
        if (code === 0) {
          resolve({
            text: this.parseWhisperCppOutput(stdout),
            language: this.config.language,
          });
        } else {
          reject(new Error(`whisper.cpp exited with code ${code}: ${stderr}`));
        }
      });

      whisper.on('error', (error) => {
        reject(error);
      });
    });
  }

  // ─── faster-whisper subprocess ───────────────────────────────────────────────

  private async transcribeWithFasterWhisper(
    audioPath: string,
    device: 'cuda' | 'cpu'
  ): Promise<Omit<TranscriptionResult, 'duration'>> {
    return new Promise((resolve, reject) => {
      const audioPathLiteral = JSON.stringify(audioPath);
      const computeType =
        device === 'cpu'
          ? process.env.WHISPER_CPU_COMPUTE_TYPE || 'int8'
          : process.env.WHISPER_COMPUTE_TYPE || 'float16';
      const beamSize = parseInt(process.env.WHISPER_BEAM_SIZE || '5', 10);
      const deviceIndex = parseInt(process.env.WHISPER_DEVICE_INDEX || '0', 10);

      const modelName = JSON.stringify(this.config.modelPath);
      const pythonScript = `
import sys
import json
import time
from faster_whisper import WhisperModel

start = time.time()
model = WhisperModel(${modelName}, device="${device}", compute_type="${computeType}", device_index=${deviceIndex})
load_time = time.time() - start
print(json.dumps({"event": "whisper_model_loaded", "device": "${device}", "compute_type": "${computeType}", "load_time_sec": load_time}), file=sys.stderr)
segments, info = model.transcribe(${audioPathLiteral}, language="${this.config.language}", beam_size=${beamSize}, initial_prompt=${JSON.stringify(MEDICAL_INITIAL_PROMPT)}, temperature=0, no_speech_threshold=0.6, condition_on_previous_text=False, word_timestamps=True, repetition_penalty=1.2)
text = " ".join([segment.text for segment in segments])
print(json.dumps({"text": text, "language": info.language}))
`;

      const pythonCmd = process.env.WHISPER_PYTHON || 'python';
      const python = spawn(pythonCmd, ['-c', pythonScript]);
      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code === 0) {
          try {
            const lastLine = stdout.trim().split('\n').filter(Boolean).pop() ?? '';
            const result = JSON.parse(lastLine);
            resolve({
              text: result.text,
              language: result.language,
            });
          } catch {
            reject(new Error('Failed to parse faster-whisper output'));
          }
        } else {
          reject(new Error(`faster-whisper exited with code ${code}: ${stderr}`));
        }
      });

      python.on('error', (error) => {
        reject(error);
      });
    });
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────────

  private parseWhisperCppOutput(output: string): string {
    const lines = output.split('\n');
    const textLines = lines
      .filter((line) => line.includes(']'))
      .map((line) => {
        const match = line.match(/\]\s*(.+)$/);
        return match ? match[1].trim() : '';
      })
      .filter(Boolean);

    return textLines.join(' ');
  }

  private getMockTranscription(): TranscriptionResult {
    return {
      text: 'Пациент Иванов Иван Петрович, 45 лет. Жалобы на головные боли во второй половине дня. История заболевания: ухудшение в течение двух месяцев. Объективно: АД 145/90 мм рт.ст., ЧСС 78/мин. Диагноз: артериальная гипертензия II стадии. Рекомендации: коррекция терапии, контроль АД, консультация невролога.',
      duration: 2.5,
      language: 'ru',
    };
  }
}
