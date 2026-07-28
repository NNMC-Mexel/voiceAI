import { spawn } from 'child_process';
import { createHash } from 'crypto';
import { readFile, writeFile, unlink, mkdir, readdir, rm, stat } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import type { TranscriptionResult, WhisperConfig } from '../types.js';
import { applyMedicalDictionary, DICTIONARY_RULE_COUNT } from './medical-dictionary.js';
import {
  denormalizeDetailed as denormalizeGigaAMDetailed,
} from './gigaam-denormalize.js';

interface GigaAMCtcDecoderMetadata {
  mode?: string;
  active?: boolean;
  implementation?: string;
  implementationMetadata?: Record<string, unknown>;
  beamWidth?: number;
  languageModel?: {
    active?: boolean;
    file?: string | null;
    sha256?: string | null;
    alpha?: number | null;
    beta?: number | null;
  };
  contexts?: {
    configured?: boolean;
    file?: string;
    sha256?: string | null;
    scopes?: Record<string, number>;
  };
  hotwordWeight?: number | null;
}

interface WhisperServerResponse {
  schema_version?: string;
  text: string;
  raw_text?: string;
  source?: 'gigaam' | 'whisper';
  language: string;
  elapsed: number;
  method?: string;
  runtime_id?: string;
  model?: {
    name?: string;
    requested?: string;
    decoder?: string | null;
    acousticDecoder?: string | null;
    ctcDecoder?: GigaAMCtcDecoderMetadata;
    checkpoint?: {
      expectedChecksum?: string | null;
      verified?: boolean;
      hashes?: Record<string, string | undefined>;
    };
  };
  context_bias?: {
    scope?: string | null;
    active?: boolean;
    terms?: number;
  };
  hashes?: {
    audio_sha256?: string;
    normalized_audio_sha256?: string;
    raw_text_sha256?: string;
    text_sha256?: string;
    output_text_sha256?: string;
    final_text_sha256?: string;
  };
  words?: Array<{
    text?: string;
    word?: string;
    start?: number;
    end?: number;
    confidence?: number | null;
  }>;
  confidence?: {
    available?: boolean;
    avg_logprob?: number;
    low_confidence?: boolean;
    reason?: string;
  };
  chunks?: number;
  chunk_details?: Array<{
    chunk: number;
    duration: number;
    chars: number;
    elapsed: number;
    avg_logprob?: number;
    suspicious?: string[];
    fallback_used?: boolean;
    selected_beam?: number;
  }>;
  avg_logprob?: number;
  low_confidence?: boolean;
  integrity?: {
    degraded?: boolean;
    degradation_reasons?: string[];
    seam_conflicts?: Array<{
      seam?: number;
      start?: number;
      end?: number;
      left_text?: string;
      right_text?: string;
      critical?: boolean;
      critical_classes?: string[];
    }>;
    critical_seam_conflict?: boolean;
    approval_blocked?: boolean;
    benchmark_eligible?: boolean;
    training_eligible?: boolean;
  };
  configuration?: {
    longform_mode?: string;
    longform_strict?: boolean;
    vad?: {
      modelId?: string;
      requestedRevision?: string | null;
      snapshotRevision?: string;
      available?: boolean;
      loaded?: boolean;
      localOnly?: boolean;
      errorType?: string;
    };
    vad_target_seconds?: number;
    vad_hard_max_seconds?: number;
    vad_padding_seconds?: number;
    fallback_chunk_seconds?: number;
    fallback_overlap_seconds?: number;
  };
}

interface ASRContextBiasMetadata {
  scope: string | null;
  active: boolean;
  terms: number;
}

interface ASRHashMetadata {
  audioSha256?: string;
  normalizedAudioSha256?: string;
  rawTextSha256?: string;
  outputTextSha256?: string;
  finalTextSha256?: string;
  normalizedTextSha256: string;
}

export interface DetailedStreamingTranscription {
  schemaVersion: string;
  runtimeId: string;
  checkpointVerified: boolean;
  rawText: string;
  normalizedText: string;
  normalization: {
    version: string;
    transformations: Array<{
      kind: string;
      source: { start: number; end: number; text: string };
      normalized: { start: number; end: number; text: string };
    }>;
    issues: Array<{
      id: string;
      code: string;
      severity: 'critical' | 'warning';
      message: string;
      source?: { start: number; end: number; text: string };
      normalized?: { start: number; end: number; text: string };
      values?: number[];
    }>;
  };
  rawAvailable: boolean;
  language: string;
  source: 'gigaam' | 'whisper' | 'unknown';
  words: Array<{
    text: string;
    startMs: number;
    endMs: number;
    confidence: number | null;
  }>;
  model: {
    asr: { name: string; version: string; checksum?: string };
    vad: { name: string; version: string; checksum?: string } | null;
    decoder: { name: string; version: string; checksum?: string };
    languageModel: { name: string; version: string; checksum?: string } | null;
    contextVocabulary: { name: string; version: string; checksum?: string } | null;
    dictionary: { name: string; version: string };
    normalizer: { name: string; version: string };
  };
  contextBias: ASRContextBiasMetadata;
  hashes: ASRHashMetadata;
  provenance: {
    schemaVersion: string;
    runtimeId: string;
    acousticDecoder: string | null;
    ctcDecoder: GigaAMCtcDecoderMetadata | null;
    contextBias: ASRContextBiasMetadata;
    hashes: ASRHashMetadata;
    checkpointVerified: boolean;
  };
  longform: {
    mode: 'vad' | 'emission_stitch' | 'text_fallback' | 'single';
    degraded: boolean;
    vad: { name: string; version: string; checksum?: string } | null;
    seams: Array<{
      startMs: number;
      endMs: number;
      conflict: boolean;
      critical: boolean;
      leftText?: string;
      rightText?: string;
    }>;
  };
}

const REMOTE_CHUNK_SECONDS = Number.parseInt(process.env.WHISPER_REMOTE_CHUNK_SECONDS || '600', 10);
/**
 * A 32 MiB binary payload becomes roughly 42.7 MiB after base64 encoding.
 * This leaves room for JSON metadata under a 48 MiB reverse-proxy/runtime
 * request limit. Never allow configuration to raise the single-payload limit.
 */
export const REMOTE_SINGLE_PAYLOAD_MAX_BYTES =
  Math.min(
    32,
    Math.max(1, Number.parseInt(process.env.WHISPER_REMOTE_CHUNK_MIN_MB || '32', 10) || 32),
  ) * 1024 * 1024;
const REMOTE_CHUNK_MIN_SIZE_BYTES = REMOTE_SINGLE_PAYLOAD_MAX_BYTES;
const REMOTE_CHUNK_MIN_DURATION_SECONDS = Number.parseInt(process.env.WHISPER_REMOTE_CHUNK_MIN_SECONDS || '1200', 10);

export interface StreamingTranscriptionContext {
  /**
   * Trusted canonical template id supplied by the server-side session. It is
   * mapped to a configured decoder scope; callers cannot supply hotwords.
   */
  templateId?: string;
}

const APPROVED_CONTEXT_SCOPE_ENV_BY_TEMPLATE: Readonly<Record<string, string>> = {
  CT_ABDOMEN_MIKHAILOV: 'GIGAAM_CT_ABDOMEN_CONTEXT_SCOPE',
};

function approvedContextScope(templateId: string | undefined): string | undefined {
  if (!templateId) return undefined;
  const envName = APPROVED_CONTEXT_SCOPE_ENV_BY_TEMPLATE[templateId];
  if (!envName) return undefined;
  const scope = process.env[envName]?.trim();
  if (!scope) return undefined;
  if (!/^[a-z0-9_.-]{1,80}$/u.test(scope)) {
    throw new Error(`${envName} contains an invalid GigaAM context scope`);
  }
  return scope;
}

function sha256Text(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function decodedBase64Bytes(value: string): number {
  const compact = value.replace(/\s+/gu, '');
  if (!compact) return 0;
  const padding = compact.endsWith('==') ? 2 : compact.endsWith('=') ? 1 : 0;
  return Math.floor((compact.length * 3) / 4) - padding;
}

// КОРОТКИЙ initial_prompt — только структура консультации.
// Whisper hard limit: 224 токена. Всё что сверху — молча обрезается с НАЧАЛА.
// Предметную лексику передаём через hotwords (faster-whisper ≥1.0) — нет лимита.
// КОРОТКИЙ initial_prompt — только структура консультации.
// Whisper hard limit: 224 токена. Всё что сверху — молча обрезается с НАЧАЛА.
// НЕ включать единицы измерения (мм рт.ст., уд/мин) и аббревиатуры —
// они «протекают» в транскрипцию как мусорные фрагменты.
const MEDICAL_INITIAL_PROMPT =
  'Медицинский протокол. АД – 150/90 мм рт.ст. Максимальные цифры АД до 180/110 мм рт.ст. ' +
  'ЧСС – 76 уд/мин. SpO₂ 96%. СОЭ 12 мм/ч. HbA1c 7,8%. ЭКГ. ЭхоКГ. Тредмил-тест. ' +
  'Менингеальных знаков нет. Парезов нет. ' +
  'АЛТ 31 МЕ/л, АСТ 27 МЕ/л, ЛПНП 4,2 ммоль/л, триглицериды 2,4 ммоль/л. ' +
  'валсартан, амлодипин, метформин, розувастатин, эмпаглифлозин, ацетилсалициловая кислота.';

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
    // Groq hosted Whisper: если задан ключ — подсистема считается доступной
    // (Groq + локальный fallback). Не дёргаем Groq API на каждый /api/health,
    // чтобы не жечь rate-limit; реальные ошибки уходят в fallback при транскрипции.
    if (this.config.groq) {
      return true;
    }
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

    // Шаг 0: Groq hosted Whisper (OpenAI-совместимый) — если задан GROQ_API_KEY.
    // Приоритетный путь; при любой ошибке падаем в локальную цепочку ниже.
    if (this.config.groq) {
      const attempts = this.getRetryCount('GROQ_WHISPER_RETRIES', 3);
      for (let attempt = 1; attempt <= attempts; attempt++) {
        try {
          const result = await this.transcribeWithGroqSmart(audioPath, filename);
          return applyDict({
            ...result,
            duration: (Date.now() - startTime) / 1000,
          });
        } catch (error) {
          if (attempt >= attempts) {
            console.warn(`[whisper] Groq unavailable after ${attempts} attempts, falling back to server/local... (${error})`);
            break;
          }
          const delayMs = 1500 * attempt;
          console.warn(`[whisper] Groq failed on attempt ${attempt}/${attempts}; retrying in ${delayMs}ms (${error})`);
          await this.sleep(delayMs);
        }
      }
    }

    // Шаг 1: Persistent HTTP whisper-сервер (нет перезагрузки модели → быстро)
    if (this.config.serverUrl) {
      const attempts = this.getRetryCount('WHISPER_SERVER_RETRIES', 2);
      let lastServerError: unknown;
      for (let attempt = 1; attempt <= attempts; attempt++) {
        try {
          const result = await this.transcribeWithServer(audioPath);
          return applyDict({
            ...result,
            duration: (Date.now() - startTime) / 1000,
          });
        } catch (error) {
          lastServerError = error;
          if (attempt >= attempts) {
            console.warn(`Whisper HTTP server unavailable after ${attempts} attempts, falling back to subprocess... (${error})`);
            break;
          }
          const delayMs = 1500 * attempt;
          console.warn(`[whisper] HTTP server failed on attempt ${attempt}/${attempts}; retrying in ${delayMs}ms (${error})`);
          await this.sleep(delayMs);
        }
      }

      if (this.config.remoteOnly) {
        const reason = lastServerError instanceof Error ? lastServerError.message : String(lastServerError ?? 'unknown error');
        throw new Error(
          `Whisper remote-only mode is enabled, and WHISPER_SERVER_URL failed for ${filename}. ` +
          `Groq and local subprocess fallbacks are disabled by configuration. Last remote error: ${reason}`
        );
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
      const localPythonIssue = this.localPythonConfigIssue();
      if (localPythonIssue) {
        throw new Error(`Whisper transcription failed for ${filename}: ${localPythonIssue}`);
      }

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

  // ─── Streaming chunk transcription (base64 → text) ──────────────────────────

  async transcribeBase64(audioBase64: string): Promise<string> {
    const result = await this.transcribeBase64Detailed(audioBase64);
    return result.normalizedText;
  }

  /**
   * Returns both the immutable model output and the post-processed text.
   * The distinction is required for honest ASR feedback and fine-tuning.
   */
  async transcribeBase64Detailed(
    audioBase64: string,
    context: StreamingTranscriptionContext = {},
  ): Promise<DetailedStreamingTranscription> {
    if (!this.config.serverUrl) {
      throw new Error('Whisper HTTP server not configured — streaming unavailable');
    }
    const decodedBytes = decodedBase64Bytes(audioBase64);
    if (decodedBytes > REMOTE_SINGLE_PAYLOAD_MAX_BYTES) {
      throw new Error(
        `Remote ASR audio payload is ${decodedBytes} bytes; maximum single payload is `
        + `${REMOTE_SINGLE_PAYLOAD_MAX_BYTES} bytes. Split the recording into valid audio chunks.`,
      );
    }
    const contextScope = approvedContextScope(context.templateId);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000);

    try {
      const response = await fetch(`${this.config.serverUrl}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: audioBase64,
          language: this.config.language,
          beam_size: this.config.beamSize,
          ...(contextScope ? { context_scope: contextScope } : {}),
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(`Whisper chunk error ${response.status}: ${err.error ?? response.statusText}`);
      }

      const data = (await response.json()) as WhisperServerResponse;
      const hasExplicitRaw = typeof data.raw_text === 'string' && data.raw_text.trim().length > 0;
      // Keep the explicit runtime output byte-for-byte so the supplied hash can
      // be verified. `text` is only a display fallback, never raw training truth.
      const rawText = hasExplicitRaw ? data.raw_text! : (data.text ?? '');
      const cleaned = this.cleanWhisperHallucinations(rawText);
      // The global dictionary contains cross-specialty replacements and is not
      // confidence/evidence aware. Canonical radiology sessions skip it.
      const dictionaryApplied = !context.templateId;
      const corrected = dictionaryApplied ? applyMedicalDictionary(cleaned) : cleaned;
      // In canonical radiology mode every transformation starts from the
      // immutable raw ASR. Generic hallucination cleaning is intentionally not
      // allowed to remove clinical words before the raw→normalized safety gate.
      const normalizationInput = context.templateId ? rawText : corrected;
      const detailedNormalization = denormalizeGigaAMDetailed(normalizationInput);
      const normalizedText = detailedNormalization.text;
      const normalization = {
        version: detailedNormalization.version,
        transformations: detailedNormalization.transformations.map((transformation) => ({
          kind: transformation.type,
          source: {
            start: transformation.start,
            end: transformation.end,
            text: transformation.sourceText,
          },
          normalized: {
            start: transformation.start,
            end: transformation.start + transformation.normalizedText.length,
            text: transformation.normalizedText,
          },
        })),
        issues: detailedNormalization.issues.map((issue) => ({
          id: sha256Text(JSON.stringify({
            code: issue.code,
            start: issue.start,
            end: issue.end,
            sourceText: issue.sourceText,
            values: issue.values,
          })).slice(0, 24),
          code: issue.code,
          severity: issue.severity,
          message: issue.message,
          source: {
            start: issue.start,
            end: issue.end,
            text: issue.sourceText,
          },
          normalized: {
            start: issue.start,
            end: issue.start + issue.normalizedText.length,
            text: issue.normalizedText,
          },
          values: issue.values,
        })),
      };
      const runtimeVersion = data.runtime_id?.trim() || 'unknown';
      const checkpointHashes = data.model?.checkpoint?.hashes;
      const checkpointChecksum = checkpointHashes?.sha256 || checkpointHashes?.md5;
      const ctcDecoder = data.model?.ctcDecoder;
      const acousticDecoder = data.model?.acousticDecoder?.trim()
        || data.model?.decoder?.trim()
        || null;
      const decoderImplementation = ctcDecoder?.implementation?.trim()
        || acousticDecoder
        || (ctcDecoder?.mode ? `gigaam_ctc_${ctcDecoder.mode}` : 'unknown');
      const responseContextScope = data.context_bias?.scope?.trim() || null;
      if (responseContextScope !== (contextScope ?? null)) {
        throw new Error(
          `GigaAM context scope mismatch: requested ${contextScope ?? 'none'}, `
          + `received ${responseContextScope ?? 'none'}`,
        );
      }
      const contextBias: ASRContextBiasMetadata = {
        scope: responseContextScope,
        active: data.context_bias?.active === true,
        terms: Number.isSafeInteger(data.context_bias?.terms)
          ? Number(data.context_bias?.terms)
          : 0,
      };
      const decoderVersion = [
        runtimeVersion,
        acousticDecoder ? `acoustic=${acousticDecoder}` : '',
        ctcDecoder?.mode ? `mode=${ctcDecoder.mode}` : '',
        Number.isFinite(ctcDecoder?.beamWidth) ? `beam=${ctcDecoder?.beamWidth}` : '',
        contextBias.scope ? `context=${contextBias.scope}` : '',
        contextBias.active ? `terms=${contextBias.terms}` : '',
      ].filter(Boolean).join(';');
      const languageModelMetadata = ctcDecoder?.languageModel;
      const languageModel = languageModelMetadata?.active === true
        ? {
            name: languageModelMetadata.file?.trim() || 'ctc-language-model',
            version: [
              runtimeVersion,
              typeof languageModelMetadata.alpha === 'number'
                ? `alpha=${languageModelMetadata.alpha}`
                : '',
              typeof languageModelMetadata.beta === 'number'
                ? `beta=${languageModelMetadata.beta}`
                : '',
            ].filter(Boolean).join(';'),
            ...(languageModelMetadata.sha256
              ? { checksum: languageModelMetadata.sha256 }
              : {}),
          }
        : null;
      const normalizedTextSha256 = sha256Text(normalizedText);
      const computedRawTextSha256 = hasExplicitRaw ? sha256Text(rawText) : undefined;
      if (
        data.hashes?.raw_text_sha256
        && computedRawTextSha256
        && data.hashes.raw_text_sha256 !== computedRawTextSha256
      ) {
        throw new Error('GigaAM raw_text SHA-256 does not match the explicit raw_text payload');
      }
      const hashes: ASRHashMetadata = {
        ...(data.hashes?.audio_sha256 ? { audioSha256: data.hashes.audio_sha256 } : {}),
        ...(data.hashes?.normalized_audio_sha256
          ? { normalizedAudioSha256: data.hashes.normalized_audio_sha256 }
          : {}),
        ...((data.hashes?.raw_text_sha256 || computedRawTextSha256)
          ? { rawTextSha256: data.hashes?.raw_text_sha256 || computedRawTextSha256 }
          : {}),
        ...((data.hashes?.output_text_sha256 || data.hashes?.text_sha256)
          ? { outputTextSha256: data.hashes.output_text_sha256 || data.hashes.text_sha256 }
          : { outputTextSha256: sha256Text(data.text ?? '') }),
        ...(data.hashes?.final_text_sha256
          ? { finalTextSha256: data.hashes.final_text_sha256 }
          : { finalTextSha256: normalizedTextSha256 }),
        normalizedTextSha256,
      };
      const source = data.source === 'gigaam'
        ? 'gigaam'
        : data.source === 'whisper'
          ? 'whisper'
          : 'unknown';
      const words = (data.words ?? [])
        .map((word) => {
          const text = (word.text ?? word.word ?? '').trim();
          const start = Number(word.start);
          const end = Number(word.end);
          if (!text || !Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
          return {
            text,
            startMs: Math.round(start * 1000),
            endMs: Math.round(end * 1000),
            confidence: typeof word.confidence === 'number' && Number.isFinite(word.confidence)
              ? word.confidence
              : null,
          };
        })
        .filter((word): word is NonNullable<typeof word> => word !== null);
      const vadConfiguration = data.configuration?.vad;
      const vadRevision = vadConfiguration?.snapshotRevision
        || vadConfiguration?.requestedRevision
        || 'unknown';
      const vad = vadConfiguration?.modelId
        ? {
            name: vadConfiguration.modelId,
            version: String(vadRevision),
            ...(/^[a-f0-9]{40,64}$/iu.test(String(vadRevision))
              ? { checksum: String(vadRevision).toLowerCase() }
              : {}),
          }
        : null;
      const integrityAvailable = data.integrity !== undefined;
      const method = data.method ?? '';
      const longformMode: DetailedStreamingTranscription['longform']['mode'] =
        method.includes('vad_')
          ? 'vad'
          : method.includes('stitched')
            ? 'emission_stitch'
            : (data.integrity?.degraded === true || !integrityAvailable)
              ? 'text_fallback'
              : 'single';
      const longform: DetailedStreamingTranscription['longform'] = {
        mode: longformMode,
        degraded: data.integrity?.degraded === true || !integrityAvailable,
        vad,
        seams: (data.integrity?.seam_conflicts ?? []).map((seam) => ({
          startMs: Math.round(Number(seam.start ?? 0) * 1000),
          endMs: Math.round(Number(seam.end ?? seam.start ?? 0) * 1000),
          conflict: true,
          critical: seam.critical === true,
          ...(seam.left_text ? { leftText: seam.left_text } : {}),
          ...(seam.right_text ? { rightText: seam.right_text } : {}),
        })),
      };

      return {
        schemaVersion: data.schema_version?.trim() || 'unknown',
        runtimeId: runtimeVersion,
        checkpointVerified: data.model?.checkpoint?.verified === true,
        rawText,
        normalizedText,
        normalization,
        rawAvailable: hasExplicitRaw,
        language: data.language || this.config.language,
        source,
        words,
        model: {
          asr: {
            name: data.model?.name || data.model?.requested || 'remote-asr',
            version: runtimeVersion,
            ...(checkpointChecksum ? { checksum: checkpointChecksum } : {}),
          },
          vad,
          decoder: {
            name: decoderImplementation,
            version: decoderVersion || runtimeVersion,
            ...(ctcDecoder?.contexts?.sha256
              ? { checksum: ctcDecoder.contexts.sha256 }
              : {}),
          },
          languageModel,
          contextVocabulary: contextBias.scope
            ? {
                name: contextBias.scope,
                version: `terms=${contextBias.terms}`,
                ...(ctcDecoder?.contexts?.sha256
                  ? { checksum: ctcDecoder.contexts.sha256 }
                  : {}),
              }
            : null,
          dictionary: {
            name: dictionaryApplied ? 'medical-dictionary' : 'medical-dictionary-disabled',
            version: dictionaryApplied ? `rules-${DICTIONARY_RULE_COUNT}` : 'not-applied-radiology',
          },
          normalizer: {
            name: 'gigaam-radiology-normalizer',
            version: detailedNormalization.version,
          },
        },
        contextBias,
        hashes,
        provenance: {
          schemaVersion: data.schema_version?.trim() || 'unknown',
          runtimeId: runtimeVersion,
          acousticDecoder,
          ctcDecoder: ctcDecoder ?? null,
          contextBias,
          hashes,
          checkpointVerified: data.model?.checkpoint?.verified === true,
        },
        longform,
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Whisper chunk request timeout');
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  // ─── Groq hosted Whisper (OpenAI-совместимый) ───────────────────────────────

  private guessAudioMime(name: string): string {
    const ext = name.toLowerCase().split('.').pop() || '';
    switch (ext) {
      case 'webm': return 'audio/webm';
      case 'wav': return 'audio/wav';
      case 'mp3':
      case 'mpga': return 'audio/mpeg';
      case 'mp4':
      case 'm4a': return 'audio/mp4';
      case 'ogg':
      case 'oga': return 'audio/ogg';
      case 'flac': return 'audio/flac';
      default: return 'application/octet-stream';
    }
  }

  private async transcribeWithGroq(
    audioPath: string,
    filename: string
  ): Promise<Omit<TranscriptionResult, 'duration'>> {
    const groq = this.config.groq;
    if (!groq) throw new Error('Groq config missing');

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000);

    try {
      const audioBytes = await readFile(audioPath);
      const safeName = path.basename(filename) || 'audio.webm';
      const blob = new Blob([audioBytes], { type: this.guessAudioMime(safeName) });

      const form = new FormData();
      form.append('file', blob, safeName);
      form.append('model', groq.model);
      form.append('language', this.config.language);
      // verbose_json — чтобы получить распознанный language; на text fallback ниже
      form.append('response_format', 'verbose_json');
      form.append('temperature', '0');
      // initial_prompt → prompt: помогает с медицинской лексикой.
      // Whisper hard limit 224 токена — тот же, что и у локального пути.
      form.append('prompt', MEDICAL_INITIAL_PROMPT);

      const response = await fetch(`${groq.baseUrl}/audio/transcriptions`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${groq.apiKey}` },
        body: form,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errText = await response.text().catch(() => response.statusText);
        throw new Error(`Groq transcription error ${response.status}: ${errText.slice(0, 300)}`);
      }

      const data = (await response.json()) as { text?: string; language?: string };
      if (typeof data.text !== 'string') {
        throw new Error('Groq response missing "text" field');
      }

      console.log(`[whisper] Groq ${groq.model} → ${data.text.length} chars (lang=${data.language ?? this.config.language})`);

      return {
        text: data.text,
        language: data.language || this.config.language,
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Groq transcription request timeout');
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  // ─── Persistent HTTP whisper-сервер ─────────────────────────────────────────

  private async transcribeWithGroqSmart(
    audioPath: string,
    filename: string
  ): Promise<Omit<TranscriptionResult, 'duration'>> {
    const maxBytes = this.getByteLimit('GROQ_WHISPER_MAX_BYTES', 20 * 1024 * 1024);
    const audioStat = await stat(audioPath);
    if (audioStat.size <= maxBytes) {
      return this.transcribeWithGroq(audioPath, filename);
    }

    console.log(`[whisper] Groq chunking large audio ${(audioStat.size / 1024 / 1024).toFixed(1)} MB: ${filename}`);
    return this.transcribeWithGroqChunked(audioPath, filename);
  }

  private async transcribeWithGroqChunked(
    audioPath: string,
    filename: string
  ): Promise<Omit<TranscriptionResult, 'duration'>> {
    const chunkDir = path.join(this.tempDir, `groq_chunks_${Date.now()}_${process.pid}`);
    await mkdir(chunkDir, { recursive: true });

    try {
      const segmentSeconds = Math.max(120, Math.min(
        Number.parseInt(process.env.GROQ_WHISPER_SEGMENT_SECONDS || '300', 10) || 300,
        900,
      ));
      const outputPattern = path.join(chunkDir, 'chunk_%03d.mp3');
      await this.runFfmpeg([
        '-hide_banner',
        '-loglevel', 'error',
        '-i', audioPath,
        '-vn',
        '-ac', '1',
        '-ar', '16000',
        '-b:a', process.env.GROQ_WHISPER_CHUNK_BITRATE || '48k',
        '-f', 'segment',
        '-segment_time', String(segmentSeconds),
        '-reset_timestamps', '1',
        outputPattern,
      ]);

      const chunks = (await readdir(chunkDir))
        .filter((name) => /\.mp3$/i.test(name))
        .sort()
        .map((name) => path.join(chunkDir, name));

      if (chunks.length === 0) {
        throw new Error('ffmpeg produced no Groq audio chunks');
      }

      const texts: string[] = [];
      const failedChunks: string[] = [];
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const chunkSize = (await stat(chunk)).size;
        console.log(`[whisper] Groq chunk ${i + 1}/${chunks.length}: ${(chunkSize / 1024 / 1024).toFixed(1)} MB`);
        try {
          const result = await this.transcribeGroqChunkWithRetry(chunk, `${path.basename(filename)}.part${i + 1}.mp3`);
          if (result.text.trim()) texts.push(result.text.trim());
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          failedChunks.push(`${i + 1}/${chunks.length}: ${message.slice(0, 160)}`);
          console.warn(`[whisper] Groq chunk ${i + 1}/${chunks.length} failed: ${message}`);
        }
      }

      if (texts.length === 0) {
        throw new Error(`all Groq chunks failed: ${failedChunks.join('; ')}`);
      }

      if (failedChunks.length > 0) {
        texts.push(`[Внимание: не удалось распознать ${failedChunks.length} фрагмент(ов) аудио.]`);
      }

      return {
        text: texts.join('\n').trim(),
        language: this.config.language,
      };
    } finally {
      await rm(chunkDir, { recursive: true, force: true }).catch(() => undefined);
    }
  }

  private async transcribeGroqChunkWithRetry(
    audioPath: string,
    filename: string
  ): Promise<Omit<TranscriptionResult, 'duration'>> {
    const attempts = this.getRetryCount('GROQ_WHISPER_CHUNK_RETRIES', 2);
    let lastError: unknown;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await this.transcribeWithGroq(audioPath, filename);
      } catch (error) {
        lastError = error;
        if (attempt >= attempts) break;
        const delayMs = 1000 * attempt;
        console.warn(`[whisper] Groq chunk failed on attempt ${attempt}/${attempts}; retrying in ${delayMs}ms (${error})`);
        await this.sleep(delayMs);
      }
    }
    throw lastError;
  }

  private async transcribeWithServer(audioPath: string): Promise<Omit<TranscriptionResult, 'duration'>> {
    const fileStat = await stat(audioPath);
    const durationSeconds = await this.getAudioDurationSeconds(audioPath);
    if (
      fileStat.size >= REMOTE_CHUNK_MIN_SIZE_BYTES ||
      (durationSeconds > 0 && durationSeconds >= REMOTE_CHUNK_MIN_DURATION_SECONDS)
    ) {
      return this.transcribeWithServerChunkedFile(audioPath, durationSeconds);
    }

    return this.transcribeWithServerChunk(audioPath);
  }

  private async transcribeWithServerChunk(audioPath: string): Promise<Omit<TranscriptionResult, 'duration'>> {
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
          // Передаём beam_size явно, чтобы фиксировать конфигурацию между клиентом
          // и Python-сервером. Без этого сервер уходит в свой env-дефолт (5) —
          // даже если у нас WHISPER_BEAM_SIZE=1, live-прогон шёл на beam=5.
          beam_size: this.config.beamSize,
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
            const flags = c.suspicious?.length ? ` [${c.suspicious.join(',')}]` : '';
            const fb = c.fallback_used ? ' [fallback→beam' + (c.selected_beam ?? '?') + ']' : '';
            console.log(`  chunk ${c.chunk}: ${c.duration}s → ${c.chars} chars (${c.elapsed}s) logprob=${c.avg_logprob ?? 'n/a'}${flags}${fb}`);
          }
        }
      }
      if (typeof data.avg_logprob === 'number') {
        console.log(`[whisper] avg_logprob=${data.avg_logprob.toFixed(2)}${data.low_confidence ? ' [LOW CONFIDENCE — возможны галлюцинации]' : ''}`);
      }

      // Собираем warnings из chunk_details — только для flagged чанков
      // (suspicious.length > 0). Клиент использует для UX-бейджей/подсветки.
      const warnings = (data.chunk_details ?? [])
        .filter((c) => c.suspicious && c.suspicious.length > 0)
        .map((c) => ({
          chunk: c.chunk,
          reasons: c.suspicious ?? [],
          avgLogprob: typeof c.avg_logprob === 'number' ? c.avg_logprob : 0,
          fallbackUsed: Boolean(c.fallback_used),
          selectedBeam: typeof c.selected_beam === 'number' ? c.selected_beam : 0,
        }));

      return {
        text: data.text,
        language: data.language,
        ...(warnings.length > 0 ? { warnings } : {}),
      };
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

  private async transcribeWithServerChunkedFile(audioPath: string, durationSeconds: number): Promise<Omit<TranscriptionResult, 'duration'>> {
    const base = path.basename(audioPath, path.extname(audioPath)).replace(/[^\w.-]+/g, '_');
    const chunkDir = path.join(this.tempDir, `remote_chunks_${Date.now()}_${base}`);
    await mkdir(chunkDir, { recursive: true });

    try {
      const fileStat = await stat(audioPath);
      const configuredSeconds = Math.max(60, REMOTE_CHUNK_SECONDS);
      const segmentSeconds = fileStat.size >= 100 * 1024 * 1024
        ? Math.min(configuredSeconds, Number.parseInt(process.env.WHISPER_REMOTE_LARGE_FILE_CHUNK_SECONDS || '180', 10) || 180)
        : configuredSeconds;
      const chunkPattern = path.join(chunkDir, 'chunk_%03d.wav');
      await this.runProcess('ffmpeg', [
        '-hide_banner',
        '-loglevel', 'error',
        '-i', audioPath,
        '-f', 'segment',
        '-segment_time', String(Math.max(60, segmentSeconds)),
        '-reset_timestamps', '1',
        '-ar', '16000',
        '-ac', '1',
        chunkPattern,
      ], 120_000);

      const chunks = (await readdir(chunkDir))
        .filter((name) => /^chunk_\d+\.wav$/i.test(name))
        .sort()
        .map((name) => path.join(chunkDir, name));

      if (chunks.length === 0) {
        throw new Error('ffmpeg produced no remote Whisper chunks');
      }

      const parts: string[] = [];
      const warnings: NonNullable<TranscriptionResult['warnings']> = [];
      let language = this.config.language;

      console.log(
        `[whisper] Remote client-side chunking: ${chunks.length} chunks` +
        (durationSeconds > 0 ? `, source=${Math.round(durationSeconds)}s` : '') +
        `, segment=${Math.max(60, segmentSeconds)}s`
      );

      for (let i = 0; i < chunks.length; i++) {
        try {
          const result = await this.transcribeServerChunkWithRetry(chunks[i], i + 1, chunks.length);
          const text = result.text.trim();
          if (text) parts.push(text);
          if (result.language) language = result.language;
          for (const warning of result.warnings ?? []) {
            warnings.push({ ...warning, chunk: i + 1 });
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          warnings.push({
            chunk: i + 1,
            reasons: ['remote_chunk_failed'],
            avgLogprob: 0,
            fallbackUsed: false,
            selectedBeam: 0,
          });
          console.warn(`[whisper] remote chunk ${i + 1}/${chunks.length} failed after retries; continuing with partial transcription (${message})`);
        }
      }

      if (parts.length === 0) {
        throw new Error('all remote Whisper chunks failed');
      }

      return {
        text: parts.join('\n\n'),
        language,
        ...(warnings.length > 0 ? { warnings } : {}),
      };
    } finally {
      await rm(chunkDir, { recursive: true, force: true });
    }
  }

  private async transcribeServerChunkWithRetry(
    audioPath: string,
    chunkIndex: number,
    totalChunks: number,
  ): Promise<Omit<TranscriptionResult, 'duration'>> {
    const attempts = this.getRetryCount('WHISPER_SERVER_CHUNK_RETRIES', 3);
    let lastError: unknown;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await this.transcribeWithServerChunk(audioPath);
      } catch (error) {
        lastError = error;
        if (attempt >= attempts) break;
        const delayMs = 1500 * attempt;
        console.warn(
          `[whisper] remote chunk ${chunkIndex}/${totalChunks} failed on attempt ${attempt}/${attempts}; ` +
          `retrying in ${delayMs}ms (${error})`
        );
        await this.sleep(delayMs);
      }
    }
    throw lastError;
  }

  private async getAudioDurationSeconds(audioPath: string): Promise<number> {
    try {
      const output = await this.runProcess('ffprobe', [
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=nw=1:nk=1',
        audioPath,
      ], 30_000);
      const duration = Number.parseFloat(output.trim());
      return Number.isFinite(duration) ? duration : 0;
    } catch {
      return 0;
    }
  }

  private runProcess(command: string, args: string[], timeoutMs: number): Promise<string> {
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, { windowsHide: true });
      let stdout = '';
      let stderr = '';
      let done = false;
      const timer = setTimeout(() => {
        if (!done) {
          done = true;
          child.kill('SIGKILL');
          reject(new Error(`${command} timed out after ${timeoutMs}ms`));
        }
      }, timeoutMs);

      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      child.on('error', (error) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        reject(error);
      });
      child.on('close', (code) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        if (code === 0) resolve(stdout);
        else reject(new Error(`${command} exited with code ${code}: ${stderr.trim()}`));
      });
    });
  }

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
segments, info = model.transcribe(${audioPathLiteral}, language="${this.config.language}", beam_size=${beamSize}, initial_prompt=${JSON.stringify(MEDICAL_INITIAL_PROMPT)}, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), no_speech_threshold=0.6, condition_on_previous_text=False, word_timestamps=True, repetition_penalty=1.2, compression_ratio_threshold=2.4, log_prob_threshold=-1.0)
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

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private getRetryCount(envName: string, fallback: number): number {
    const value = Number.parseInt(process.env[envName] || '', 10);
    if (!Number.isFinite(value)) return fallback;
    return Math.max(1, Math.min(value, 5));
  }

  private getByteLimit(envName: string, fallback: number): number {
    const value = Number.parseInt(process.env[envName] || '', 10);
    if (!Number.isFinite(value) || value <= 0) return fallback;
    return value;
  }

  private runFfmpeg(args: string[]): Promise<void> {
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn('ffmpeg', args, { windowsHide: true });
      let stderr = '';

      ffmpeg.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      ffmpeg.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`ffmpeg exited with code ${code}: ${stderr.slice(-1000)}`));
        }
      });

      ffmpeg.on('error', (error) => {
        reject(error);
      });
    });
  }

  private localPythonConfigIssue(): string | null {
    const pythonCmd = (process.env.WHISPER_PYTHON || '').trim();
    if (!pythonCmd) return null;

    const looksLikePath =
      path.isAbsolute(pythonCmd) ||
      pythonCmd.includes('/') ||
      pythonCmd.includes('\\') ||
      /\.exe$/i.test(pythonCmd);

    if (looksLikePath && !existsSync(pythonCmd)) {
      return `local faster-whisper Python path does not exist: ${pythonCmd}`;
    }

    return null;
  }

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
