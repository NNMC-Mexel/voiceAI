import type { MedicalDocument } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';
const API_TIMEOUT_MS = Number.parseInt(import.meta.env.VITE_API_TIMEOUT_MS || '120000', 10);

interface UploadResponse {
  success: boolean;
  filename: string;
  mimetype: string;
}

interface TranscriptionResponse {
  success: boolean;
  text: string;
  duration: number;
  language: string;
}

interface StructureResponse {
  success: boolean;
  document: MedicalDocument;
  rawText: string;
  processingTime: number;
}

interface AugmentResponse {
  success: boolean;
  document: MedicalDocument;
}

interface ProcessResponse {
  success: boolean;
  transcription: {
    text: string;
    duration: number;
    language: string;
  };
  document: MedicalDocument;
  processingTime: number;
}

interface RecommendationsResponse {
  success: boolean;
  recommendations: string;
}

interface ChatResponse {
  success: boolean;
  answer: string;
}

type RewriteableField = keyof Omit<MedicalDocument, 'patient'>;

interface InstructResponse {
  success: boolean;
  document: MedicalDocument;
  changedFields: RewriteableField[];
  patientChanged: boolean;
}

interface ServerConfig {
  maxRecordingDuration: number;
  supportedAudioFormats: string[];
  language: string;
  llmModel?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), API_TIMEOUT_MS);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
      });

      if (!response.ok) {
        let errorMessage = `Request failed: ${response.status}`;
        try {
          const err = await response.json();
          if (err?.error) errorMessage = String(err.error);
          else if (err?.message) errorMessage = String(err.message);
        } catch {
          const text = await response.text();
          if (text) errorMessage = text;
        }
        throw new Error(errorMessage);
      }

      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Запрос превысил таймаут ${Math.round(API_TIMEOUT_MS / 1000)}с`);
      }
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request('/api/health');
  }

  async getConfig(): Promise<ServerConfig> {
    return this.request('/api/config');
  }

  async uploadAudio(audioBlob: Blob, filename: string = 'recording.webm'): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', audioBlob, filename);

    return this.request('/api/upload', {
      method: 'POST',
      body: formData,
    });
  }

  async transcribe(filename: string): Promise<TranscriptionResponse> {
    return this.request('/api/transcribe', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ filename }),
    });
  }

  async structureText(text: string): Promise<StructureResponse> {
    return this.request('/api/structure', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
  }

  async augmentDocument(document: MedicalDocument, text: string): Promise<AugmentResponse> {
    return this.request('/api/augment', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ document, text }),
    });
  }

  async processAudio(audioBlob: Blob, filename: string = 'recording.webm'): Promise<ProcessResponse> {
    const formData = new FormData();
    formData.append('file', audioBlob, filename);

    return this.request('/api/process', {
      method: 'POST',
      body: formData,
    });
  }

  async processAddendum(
    audioBlob: Blob,
    document: MedicalDocument,
    filename: string = 'addendum.webm'
  ): Promise<{ transcription: TranscriptionResponse; document: MedicalDocument }> {
    const upload = await this.uploadAudio(audioBlob, filename);
    const transcription = await this.transcribe(upload.filename);
    const augmented = await this.augmentDocument(document, transcription.text);

    return { transcription, document: augmented.document };
  }

  async saveDocument(document: MedicalDocument): Promise<{ success: boolean; id: string; savedAt: string }> {
    return this.request('/api/documents', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(document),
    });
  }

  async getRecommendations(document: MedicalDocument): Promise<RecommendationsResponse> {
    return this.request('/api/recommendations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ document }),
    });
  }

  async chat(
    question: string,
    history: Array<{ role: 'user' | 'assistant'; text: string }>,
    document?: MedicalDocument
  ): Promise<ChatResponse> {
    return this.request('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ document, question, history }),
    });
  }

  async rewriteField(field: RewriteableField, text: string): Promise<{ success: boolean; field: string; text: string }> {
    return this.request('/api/rewrite-field', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ field, text }),
    });
  }

  async instructDocument(document: MedicalDocument, instruction: string): Promise<InstructResponse> {
    return this.request('/api/instruct', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ document, instruction }),
    });
  }

  async tts(text: string): Promise<string> {
    const result = await this.request<{ success: boolean; audio_base64: string }>('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    return result.audio_base64;
  }
}

export const apiClient = new ApiClient();
export default apiClient;
