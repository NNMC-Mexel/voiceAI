import type { MedicalDocument } from '../types';
import type { QualityWarning } from '../types';

// ─── Patient / Visit types ────────────────────────────────────────────────────

export interface PatientSummary {
  id: number;
  fullName: string;
  birthDate: string;
  gender: string;
  phone: string;
  iin: string;
  notes?: string;
  updatedAt: string;
  createdAt?: string;
}

export interface PatientForm {
  fullName: string;
  birthDate: string;
  gender: string;
  phone: string;
  iin: string;
  notes: string;
}

export interface VisitSummary {
  id: number;
  visitDate: string;
  createdAt: string;
  diagnosisPreview: string;
}

export interface VisitDetail {
  id: number;
  patientId: number;
  visitDate: string;
  createdAt: string;
  rawTranscription: string;
  document: MedicalDocument | null;
}

export interface DoctorInfo {
  id: number;
  name: string;
  email: string;
  specialty: string;
  role: 'admin' | 'doctor';
}

export interface AdminDoctorInfo extends DoctorInfo {
  isActive: boolean;
  createdAt: string;
}

// ─────────────────────────────────────────────────────────────────────────────

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
  warnings?: string[];
  qualityWarnings?: QualityWarning[];
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
  warnings?: string[];
  qualityWarnings?: QualityWarning[];
}

interface RecommendationsResponse {
  success: boolean;
  recommendations: string;
}

interface ChatResponse {
  success: boolean;
  answer: string;
}

type RewriteableField = Exclude<keyof Omit<MedicalDocument, 'patient' | 'riskAssessment'>, 'manualCheck'>;

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

  getToken(): string | null {
    return localStorage.getItem('auth_token');
  }

  setToken(token: string): void {
    localStorage.setItem('auth_token', token);
  }

  clearToken(): void {
    localStorage.removeItem('auth_token');
  }

  /** Legacy single-password login — kept for backward compat */
  async login(password: string): Promise<boolean> {
    const response = await fetch(`${this.baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    });
    if (!response.ok) return false;
    const data = (await response.json()) as { success: boolean; token: string };
    if (data.success && data.token) {
      this.setToken(data.token);
      return true;
    }
    return false;
  }

  async loginDoctor(email: string, password: string): Promise<{
    success: boolean;
    token?: string;
    doctor?: DoctorInfo;
  }> {
    const response = await fetch(`${this.baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await response.json() as { success?: boolean; token?: string; doctor?: DoctorInfo; error?: string };
    if (response.ok && data.token) {
      this.setToken(data.token);
    }
    if (!response.ok) throw new Error(data.error || `Login failed: ${response.status}`);
    return data as { success: boolean; token: string; doctor: DoctorInfo };
  }

  async registerDoctor(name: string, email: string, password: string, specialty?: string): Promise<{
    success: boolean;
    token?: string;
    doctor?: DoctorInfo;
  }> {
    const response = await fetch(`${this.baseUrl}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, specialty }),
    });
    const data = await response.json() as { success?: boolean; token?: string; doctor?: DoctorInfo; error?: string };
    if (response.ok && data.token) {
      this.setToken(data.token);
    }
    if (!response.ok) throw new Error(data.error || `Register failed: ${response.status}`);
    return data as { success: boolean; token: string; doctor: DoctorInfo };
  }

  async getMe(): Promise<DoctorInfo | null> {
    try {
      const result = await this.request<{ doctor: DoctorInfo }>('/api/auth/me');
      return result.doctor;
    } catch {
      return null;
    }
  }

  async checkAuth(): Promise<boolean> {
    const token = this.getToken();
    try {
      const headers = new Headers();
      if (token) {
        headers.set('Authorization', `Bearer ${token}`);
      }
      const response = await fetch(`${this.baseUrl}/api/auth/check`, {
        headers,
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async logout(): Promise<void> {
    const token = this.getToken();
    if (token) {
      await fetch(`${this.baseUrl}/api/auth/logout`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {});
    }
    this.clearToken();
  }

  async getAdminDoctors(): Promise<{ doctors: AdminDoctorInfo[] }> {
    return this.request('/api/admin/doctors');
  }

  async createAdminDoctor(data: {
    name: string;
    email: string;
    password: string;
    specialty?: string;
    role: 'admin' | 'doctor';
  }): Promise<{ success: boolean; doctor: AdminDoctorInfo }> {
    return this.request('/api/admin/doctors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  }

  async updateAdminDoctor(
    id: number,
    data: { name?: string; specialty?: string; role?: 'admin' | 'doctor'; isActive?: boolean },
  ): Promise<{ success: boolean; doctor: AdminDoctorInfo }> {
    return this.request(`/api/admin/doctors/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  }

  async deleteAdminDoctor(id: number): Promise<{ success: boolean }> {
    return this.request(`/api/admin/doctors/${id}`, { method: 'DELETE' });
  }

  async updateProfile(data: { name?: string; specialty?: string }): Promise<{ success: boolean; doctor: DoctorInfo }> {
    return this.request('/api/settings/profile', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<{ success: boolean }> {
    return this.request('/api/settings/password', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ currentPassword, newPassword }),
    });
  }

  private async request<T>(path: string, init?: RequestInit, timeoutMs?: number): Promise<T> {
    // Retry только для идемпотентных GET без тела. POST с большими файлами/аудио
    // НЕ ретраим — это опасно (двойная транскрипция, двойной LLM вызов).
    const method = (init?.method ?? 'GET').toUpperCase();
    const isIdempotent = method === 'GET';
    const maxAttempts = isIdempotent ? 3 : 1;

    let lastError: unknown;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      const controller = new AbortController();
      const effectiveTimeout = timeoutMs ?? API_TIMEOUT_MS;
      const timeout = effectiveTimeout > 0
        ? window.setTimeout(() => controller.abort(), effectiveTimeout)
        : null;

      const token = this.getToken();
      const headers = new Headers(init?.headers);
      if (token) {
        headers.set('Authorization', `Bearer ${token}`);
      }

      try {
        const response = await fetch(`${this.baseUrl}${path}`, {
          ...init,
          headers,
          signal: controller.signal,
        });

        if (!response.ok) {
          if (response.status === 401) {
            this.clearToken();
            window.dispatchEvent(new Event('auth:logout'));
          }
          let errorMessage = `Request failed: ${response.status}`;
          try {
            const text = await response.text();
            if (text) {
              try {
                const err = JSON.parse(text) as { error?: unknown; message?: unknown };
                if (err?.error) errorMessage = String(err.error);
                else if (err?.message) errorMessage = String(err.message);
              } catch {
                errorMessage = text;
              }
            }
          } catch {
            // ignore body read errors
          }
          // 4xx (кроме 408/429) — ошибка клиента, retry бесполезен
          const transient = response.status === 408 || response.status === 429 || response.status >= 500;
          if (!transient || attempt === maxAttempts) {
            throw new Error(errorMessage);
          }
          lastError = new Error(errorMessage);
        } else {
          return (await response.json()) as T;
        }
      } catch (error) {
        lastError = error;
        if (error instanceof Error && error.name === 'AbortError') {
          if (attempt === maxAttempts) {
            throw new Error(`Запрос превысил таймаут ${Math.round(effectiveTimeout / 1000)}с`);
          }
        } else if (attempt === maxAttempts) {
          throw error;
        }
      } finally {
        if (timeout !== null) window.clearTimeout(timeout);
      }

      // Exponential backoff: 300ms, 900ms
      if (attempt < maxAttempts) {
        await new Promise((r) => setTimeout(r, 300 * Math.pow(3, attempt - 1)));
      }
    }
    throw lastError instanceof Error ? lastError : new Error('Request failed');
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

    // 3-8МБ webm через LAN обычно <1с, но если backend занят другим запросом
    // (одновременная транскрипция/LLM) — default 120с может не хватать.
    return this.request('/api/upload', {
      method: 'POST',
      body: formData,
    }, 300_000);
  }

  async transcribe(filename: string): Promise<TranscriptionResponse> {
    // Транскрипция длинного аудио (3-5 мин) на загруженном Whisper может
    // превышать 120с — даём 10 мин, как у processAudio.
    return this.request('/api/transcribe', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ filename }),
    }, 600_000);
  }

  async structureText(text: string): Promise<StructureResponse> {
    // Структурирование 8k+ символов Claude Haiku занимает 30-50с;
    // с retry/сетью возможны 2+ минуты.
    return this.request('/api/structure', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    }, 300_000);
  }

  async augmentDocument(document: MedicalDocument, text: string): Promise<AugmentResponse> {
    return this.request('/api/augment', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ document, text }),
    }, 300_000);
  }

  async processAudio(audioBlob: Blob, filename: string = 'recording.webm'): Promise<ProcessResponse> {
    const formData = new FormData();
    formData.append('file', audioBlob, filename);

    return this.request('/api/process', {
      method: 'POST',
      body: formData,
    }, 600_000);
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
    }, 300_000);
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
    }, 300_000);
  }

  async tts(text: string): Promise<string> {
    const result = await this.request<{ success: boolean; audio_base64: string }>('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    return result.audio_base64;
  }

  // ─── Patients ────────────────────────────────────────────────────────────────

  async getPatients(opts?: { q?: string; page?: number }): Promise<{
    patients: PatientSummary[];
    page: number;
    hasMore: boolean;
  }> {
    const params = new URLSearchParams();
    if (opts?.q) params.set('q', opts.q);
    if (opts?.page) params.set('page', String(opts.page));
    const qs = params.toString() ? `?${params.toString()}` : '';
    return this.request(`/api/patients${qs}`);
  }

  async createPatient(data: Partial<PatientForm>): Promise<{ success: boolean; patient: PatientSummary }> {
    return this.request('/api/patients', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  }

  async getPatient(id: number): Promise<{ patient: PatientSummary; visits: VisitSummary[] }> {
    return this.request(`/api/patients/${id}`);
  }

  async updatePatient(id: number, data: Partial<PatientForm>): Promise<{ success: boolean; patient: PatientSummary }> {
    return this.request(`/api/patients/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  }

  async saveVisit(patientId: number, document: MedicalDocument, rawTranscription?: string): Promise<{ success: boolean; visitId: number; visitDate: string }> {
    return this.request(`/api/patients/${patientId}/visits`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document, rawTranscription }),
    }, 30_000);
  }

  async getVisit(visitId: number): Promise<{ visit: VisitDetail }> {
    return this.request(`/api/visits/${visitId}`);
  }

  // ─── Mobile ↔ Desktop sync ───────────────────────────────────────────────────

  async syncUpload(file: File): Promise<{ success: boolean; syncId: string }> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return this.request('/api/sync/upload', { method: 'POST', body: formData }, 300_000);
  }

  async syncStatus(syncId: string): Promise<{ session: { id: string; status: string; errorMessage: string; expiresAt: string } }> {
    return this.request(`/api/sync/${encodeURIComponent(syncId)}/status`);
  }

  async syncPending(): Promise<{ sessions: Array<{ id: string; filename: string; createdAt: string; expiresAt: string }> }> {
    return this.request('/api/sync/pending');
  }

  async syncClaim(syncId: string): Promise<{ success: boolean; document: MedicalDocument; rawTranscription: string; filename: string }> {
    return this.request(`/api/sync/${encodeURIComponent(syncId)}/claim`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
  }

  async syncDelete(syncId: string): Promise<{ success: boolean }> {
    return this.request(`/api/sync/${encodeURIComponent(syncId)}`, { method: 'DELETE' });
  }

  // ─── Document processing ─────────────────────────────────────────────────────

  async getDocumentCapabilities(): Promise<{ pdf: boolean; word: boolean; image: boolean }> {
    return this.request('/api/document-capabilities');
  }

  async processDocument(file: File): Promise<ProcessResponse & {
    transcription: { text: string; language: string; extractionMethod: string; pageCount?: number; warning?: string };
  }> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return this.request('/api/process-document', {
      method: 'POST',
      body: formData,
    }, 300_000);
  }

  // ─── Streaming session API ───────────────────────────────────────────────────

  async startSession(): Promise<{ sessionId: string }> {
    return this.request('/api/session/start', { method: 'POST' }, 10_000);
  }

  async sendChunk(sessionId: string, audioBase64: string, chunkIndex: number): Promise<{ ok: boolean; chunkIndex: number }> {
    return this.request(`/api/session/${encodeURIComponent(sessionId)}/chunk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_base64: audioBase64, chunk_index: chunkIndex }),
    }, 30_000);
  }

  async finishSession(sessionId: string): Promise<ProcessResponse & { transcription: { text: string; language: string } }> {
    return this.request(`/api/session/${encodeURIComponent(sessionId)}/finish`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    }, 300_000);
  }

  // ─── Corrections API ────────────────────────────────────────────────────────

  async addCorrection(
    wrong: string,
    correct: string,
    options?: { scope?: string; requireDose?: boolean }
  ): Promise<{ success: boolean; id: string; totalCorrections: number }> {
    return this.request('/api/corrections', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wrong, correct, ...options }),
    });
  }

  async getCorrections(): Promise<{ corrections: Array<{ id: string; wrong: string; correct: string; createdAt: string; scope?: string; requireDose?: boolean }>; total: number }> {
    return this.request('/api/corrections');
  }

  async deleteCorrection(id: string): Promise<{ success: boolean }> {
    return this.request(`/api/corrections/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  }
}

export const apiClient = new ApiClient();
export default apiClient;
