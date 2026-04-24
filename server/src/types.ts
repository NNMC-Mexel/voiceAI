export interface PatientInfo {
  fullName: string;
  age: string;
  gender: string;
  complaintDate: string;
}

export interface RiskAssessment {
  fallInLast3Months: string;   // "да" | "нет"
  dizzinessOrWeakness: string; // "да" | "нет"
  needsEscort: string;         // "да" | "нет"
  painScore: string;           // "0", "1", ... "10"
}

export interface MedicalDocument {
  patient: PatientInfo;
  riskAssessment: RiskAssessment;
  complaints: string;
  anamnesis: string;
  outpatientExams: string;      // Амбулаторные обследования
  clinicalCourse: string;       // Анамнез жизни
  allergyHistory: string;       // Аллергологический анамнез
  objectiveStatus: string;      // Объективные данные
  neurologicalStatus: string;   // Неврологический статус
  diagnosis: string;            // Предварительный диагноз
  finalDiagnosis: string;       // Заключительный диагноз
  conclusion: string;           // Сопутствующий диагноз
  doctorNotes: string;          // План обследования
  recommendations: string;      // План лечения (включая диету пунктом списка)
}

export interface TranscriptionWarning {
  chunk: number;         // 1-indexed
  reasons: string[];     // ['bp_without_format', 'garbage_unit_tokens', …]
  avgLogprob: number;
  fallbackUsed: boolean; // был ли применён beam-retry
  selectedBeam: number;  // финальный beam у этого чанка
}

export interface TranscriptionResult {
  text: string;
  duration: number;
  language: string;
  // Если хоть один чанк flagged — клиент может показать UX-warning. Пусто =
  // всё ок. Дефолтом undefined для обратной совместимости с whisper.cpp и
  // faster-whisper subprocess путями (они этих данных не знают).
  warnings?: TranscriptionWarning[];
}

export interface StructureResult {
  document: MedicalDocument;
  rawText: string;
  processingTime: number;
}

export interface WhisperConfig {
  modelPath: string;
  language: string;
  device: 'cuda' | 'cpu';
  serverUrl?: string; // Если задан — используется persistent HTTP whisper-сервер
  beamSize: number;   // Передаётся в payload; 1 = greedy, 5 = beam search.
                      // Фиксирует конфигурацию между клиентом и сервером —
                      // без этого Python-сервер уходит в свой env-дефолт.
}

export type LLMProviderKind = 'llama' | 'anthropic';

export interface LLMConfig {
  provider: LLMProviderKind;
  serverUrl: string;
  model: string;
  maxTokens: number;
  temperature: number;
  parallelSlots: number;
  requestTimeoutMs: number;
  allowMockOnFailure: boolean;
  anthropic?: {
    apiKey: string;
    model: string;
    maxTokens: number;
    maxRetries: number;
  };
}

export interface SecurityConfig {
  corsOrigins: string[];
  apiKey?: string;
  authPassword?: string;
  rateLimitWindowMs: number;
  rateLimitMaxRequests: number;
}

export interface TtsConfig {
  serverUrl: string;
  enabled: boolean;
}

export interface ServerConfig {
  port: number;
  host: string;
  uploadDir: string;
  whisper: WhisperConfig;
  llm: LLMConfig;
  tts: TtsConfig;
  security: SecurityConfig;
}

export const defaultConfig: ServerConfig = {
  port: 3001,
  host: '0.0.0.0',
  uploadDir: './uploads',
  whisper: {
    modelPath: './models/whisper-large-v3',
    language: 'ru',
    device: 'cuda',
    beamSize: 1,
  },
  llm: {
    provider: 'llama',
    serverUrl: 'http://localhost:8080',
    model: 'qwen3-8b',
    maxTokens: 512,
    temperature: 0,
    parallelSlots: 15,
    requestTimeoutMs: 120000,
    allowMockOnFailure: false,
  },
  tts: {
    serverUrl: '',
    enabled: false,
  },
  security: {
    corsOrigins: ['https://localhost:5173', 'http://localhost:5173', 'http://127.0.0.1:5173'],
    rateLimitWindowMs: 60000,
    rateLimitMaxRequests: 120,
  },
};
