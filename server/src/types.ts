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
  recommendations: string;      // План лечения
  diet: string;                 // Диета
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
