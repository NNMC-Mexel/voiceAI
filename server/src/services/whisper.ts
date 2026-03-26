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
}

const MEDICAL_INITIAL_PROMPT = `Консультация кардиолога. Риск падения по шкале Морзе. Оценка боли. ` +
  // Диагностика и аббревиатуры
  `АД, ЧСС, ИМТ, SpO2, ЭКГ, ЭхоКГ, ХМЭКГ, КАГ, СМАД, УЗДГ БЦА, ОАК, ОАМ, ProBNP, ` +
  `ПМЖВ, ПКА, ОВ ЛКА, РСДЛА, ФВ, КДО, КСО, КДР, КСР, МЖП, ` +
  `NYHA, CCS, EHRA, CHADS2 VASc, МКБ-10, ` +
  // Диагнозы
  `ИБС, ХСН, СССУ, НРС, ОНМК, ДГПЖ, АИТ, ДН, ОКС, ТЭЛА, ` +
  `фибрилляция предсердий, стенокардия напряжения, артериальная гипертензия, ` +
  `дилатационная кардиомиопатия, трикуспидальная регургитация, митральная регургитация, ` +
  `аортальная недостаточность, диффузный гипокинез, дилатация, атеросклероз аорты, ` +
  `тахи-брадикардитический вариант, тахисистолический вариант, ` +
  `пароксизмальная форма, перманентная форма, ` +
  // Процедуры
  `стентирование, имплантация ЭКС, электрокардиостимулятор, ` +
  `РЧА, ВСЭФИ, ЭФИ, криоаблация, кардиоресинхронизирующее устройство, CRT-D, ` +
  `кардиовертер-дефибриллятор, репозиция электрода, истощение батареи, ` +
  // Устройства (латиница)
  `Medtronic, Quadra Assura, Brava Quad, EssentioDR, Sphera DR, DDDR, VVIR, ` +
  // ЖЭС / НЖТ / АВ
  `ЖЭС, НЖЭС, ЖТ, НЖТ, АВ блокада, АВ узловая, риентри, slow-fast, ` +
  `предсердная экстрасистолия, желудочковая экстрасистолия, предсердная тахикардия, ` +
  // Лекарства — антикоагулянты / антиагреганты
  `ксарелто, ривароксабан, дабигатран, прадакса, каптоприл, нитроглицерин, ` +
  `тромбопол, тромбоАСС, кардиомагнил, ` +
  // Лекарства — бета-блокаторы / ингибиторы / сартаны
  `бисопролол, конкор, сотогексал, стопресс, периндоприл, рамиприл, хартил, ` +
  `кантаб, кандесартан, физиотенз, моксонидин, престилол, ` +
  // Лекарства — статины
  `розувастатин, аторвастатин, аторис, розулип, роксера, ` +
  // Лекарства — антиаритмики / прочие кардио
  `этацизин, кордарон, амиодарон, дилтиазем, дигоксин, ` +
  `джардинс, форсига, эплеренон, эспиро, альдарон, верошпирон, ` +
  `индапамид, торасемид, тригрим, нолипрел, валодип, илпио, ` +
  `короним, амлодипин, L-тироксин, ` +
  // Лекарства — ЖКТ / диабет
  `омепразол, пантопразол, нольпаза, антарис, глюконил, ` +
  // Единицы
  `мм.рт.ст., уд/мин, мкг, мг, мкмоль/л, ммоль/л, г/л, пмоль/л, мкМЕ/мл, ` +
  // Прочие термины
  `гипотиреоз, гипокалиемия, гемотрансфузия, бронхиальная астма, сахарный диабет, ` +
  `варикозное расширение вен, пастозность, иррадиация, аускультативно, везикулярное дыхание`;

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

  async transcribeFile(audioPath: string, filename: string): Promise<TranscriptionResult> {
    const startTime = Date.now();

    const applyDict = (result: TranscriptionResult): TranscriptionResult => {
      const corrected = applyMedicalDictionary(result.text);
      if (corrected !== result.text) {
        console.log(`[medical-dictionary] Applied corrections (${DICTIONARY_RULE_COUNT} rules)`);
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
    // 5 минут — достаточно даже для длинных записей (до 30 мин аудио)
    const timeout = setTimeout(() => controller.abort(), 300_000);

    try {
      // Читаем файл и кодируем в base64 — это позволяет вызвать Whisper-сервер
      // на другой машине (например ПК с RTX 5090 в campus-сети)
      const audioBytes = await readFile(audioPath);
      const audioBase64 = audioBytes.toString('base64');

      const response = await fetch(`${this.config.serverUrl}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: audioBase64,
          language: this.config.language,
          initial_prompt: MEDICAL_INITIAL_PROMPT,
          temperature: 0,
          no_speech_threshold: 0.6,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(`Whisper server error ${response.status}: ${err.error ?? response.statusText}`);
      }

      const data = (await response.json()) as WhisperServerResponse;
      return { text: data.text, language: data.language };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Whisper server request timeout (300s)');
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
segments, info = model.transcribe(${audioPathLiteral}, language="${this.config.language}", beam_size=${beamSize}, initial_prompt=${JSON.stringify(MEDICAL_INITIAL_PROMPT)}, temperature=0, no_speech_threshold=0.6)
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
