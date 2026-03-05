export interface PatientInfo {
  fullName: string;
  age: string;
  gender: string;
  complaintDate: string;
}

export interface MedicalDocument {
  patient: PatientInfo;
  complaints: string;
  anamnesis: string;
  clinicalCourse: string;       // Анамнез жизни
  allergyHistory: string;       // Аллергологический анамнез
  objectiveStatus: string;      // Объективные данные
  neurologicalStatus: string;   // Неврологический статус
  diagnosis: string;            // Предварительный диагноз (основной)
  conclusion: string;           // Сопутствующий диагноз
  recommendations: string;      // План лечения
  doctorNotes: string;          // Прочее
}

export interface TranscriptionResult {
  text: string;
  duration: number;
  language: string;
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
}

export interface LLMConfig {
  serverUrl: string;
  model: string;
  maxTokens: number;
  temperature: number;
  parallelSlots: number;
  requestTimeoutMs: number;
  allowMockOnFailure: boolean;
}

export interface SecurityConfig {
  corsOrigins: string[];
  apiKey?: string;
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
  },
  llm: {
    serverUrl: 'http://localhost:8080',
    model: 'qwen3-8b',
    maxTokens: 512,
    temperature: 0.1,
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
