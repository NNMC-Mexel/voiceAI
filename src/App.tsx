import { lazy, Suspense, useState, useCallback, useRef, useEffect, useMemo } from 'react';
import type { VoiceRecorderStreamOptions } from './hooks/useVoiceRecorder';
import type { AppStep, MedicalDocument, QualityWarning } from './types';
import { emptyDocument } from './types';
import { LoginScreen } from './components/LoginScreen';
import { RecordingScreen } from './components/RecordingScreen';
import { ProcessingScreen } from './components/ProcessingScreen';
import type { ProcessingPhase } from './components/ProcessingScreen';
import { EditingScreen } from './components/EditingScreen';
import { PatientListScreen } from './components/PatientListScreen';
import { PatientScreen } from './components/PatientScreen';
import { SyncUploadScreen } from './components/SyncUploadScreen';
import { SettingsScreen } from './components/SettingsScreen';
import { AdminPanelScreen } from './components/AdminPanelScreen';
import { ProtocolWorkspaceScreen } from './components/ProtocolWorkspaceScreen';
import { RadiologyWorkspaceScreen } from './components/RadiologyWorkspaceScreen';
import { apiClient } from './api/client';
import type { DoctorInfo, PatientSummary } from './api/client';

type MedicalDocumentTextField = Exclude<keyof MedicalDocument, 'patient' | 'riskAssessment'>;
type QualityWarningInput = QualityWarning | string;

const PreviewScreen = lazy(() =>
  import('./components/PreviewScreen').then((module) => ({ default: module.PreviewScreen }))
);

function filenameForBlob(blob: Blob, baseName: string): string {
  const type = blob.type.toLowerCase();
  if (type.includes('mp4')) return `${baseName}.mp4`;
  if (type.includes('ogg')) return `${baseName}.ogg`;
  if (type.includes('wav')) return `${baseName}.wav`;
  return `${baseName}.webm`;
}

const SESSION_STEP_KEY = 'voicemed_step';
const SESSION_DOC_KEY = 'voicemed_document';
const SESSION_RAW_TEXT_KEY = 'voicemed_raw_text';
const SESSION_WARNINGS_KEY = 'voicemed_quality_warnings';
const AUDIO_JOB_POLL_INTERVAL_MS = 2_000;
const AUDIO_JOB_TIMEOUT_MS = 15 * 60 * 1000;
const MERGE_TEXT_FIELDS: MedicalDocumentTextField[] = [
  'complaints',
  'anamnesis',
  'outpatientExams',
  'clinicalCourse',
  'allergyHistory',
  'objectiveStatus',
  'neurologicalStatus',
  'diagnosis',
  'finalDiagnosis',
  'conclusion',
  'doctorNotes',
  'recommendations',
  'manualCheck',
];

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizePatientName(name: string): string {
  return name.trim().replace(/\s+/g, ' ').toLowerCase();
}

function normalizePatientGender(gender: string): string {
  const value = gender.trim().toLowerCase();
  if (value.startsWith('м')) return 'male';
  if (value.startsWith('ж')) return 'female';
  return gender.trim();
}

function normalizeTextBlock(value: string): string {
  return value.trim().replace(/\s+/g, ' ').toLowerCase();
}

function appendUniqueBlock(current: string | undefined, incoming: string | undefined): string {
  const prev = (current || '').trim();
  const next = (incoming || '').trim();
  if (!next) return prev;
  if (!prev) return next;

  const normalizedPrev = normalizeTextBlock(prev);
  const normalizedNext = normalizeTextBlock(next);
  if (normalizedPrev === normalizedNext || normalizedPrev.includes(normalizedNext)) return prev;
  if (normalizedNext.includes(normalizedPrev) && normalizedPrev.length > 20) return next;
  return `${prev}\n\n${next}`;
}

function mergeSupplementDocument(base: MedicalDocument, supplement: MedicalDocument): MedicalDocument {
  const next: MedicalDocument = {
    ...base,
    patient: { ...base.patient },
    riskAssessment: { ...base.riskAssessment },
  };

  for (const key of ['fullName', 'age', 'gender', 'complaintDate', 'birthDate'] as const) {
    const value = supplement.patient[key]?.trim();
    if (value && !next.patient[key]?.trim()) next.patient[key] = value;
  }

  for (const key of ['fallInLast3Months', 'dizzinessOrWeakness', 'needsEscort', 'painScore'] as const) {
    const value = supplement.riskAssessment[key]?.trim();
    if (value && !next.riskAssessment[key]?.trim()) next.riskAssessment[key] = value;
  }

  for (const key of MERGE_TEXT_FIELDS) {
    next[key] = appendUniqueBlock(next[key], supplement[key]);
  }

  return next;
}

function appendRawSource(current: string, label: string, text: string): string {
  const clean = text.trim();
  if (!clean) return current;
  const header = `--- ${label} ---`;
  return current.trim()
    ? `${current.trim()}\n\n${header}\n${clean}`
    : clean;
}

function formatDocumentProcessingError(err: unknown): string {
  const message = err instanceof Error ? err.message : 'Ошибка обработки документа';
  if (/таймаут|timeout|aborted/i.test(message)) {
    return 'Обработка документа заняла слишком много времени. Попробуйте текстовый PDF/Word или загрузите скан как отдельное фото.';
  }
  return message;
}

function normalizeQualityWarnings(warnings: QualityWarningInput[] | undefined): QualityWarning[] {
  if (!Array.isArray(warnings)) return [];
  return warnings
    .map((warning): QualityWarning | null => {
      if (typeof warning !== 'string') return warning;
      const [code, evidence] = warning.split(':');
      return {
        code: (code || 'possibleLostLabValue') as QualityWarning['code'],
        severity: code === 'important_number_missing' ? 'critical' : 'warning',
        message: warning,
        evidence,
      };
    })
    .filter((warning): warning is QualityWarning => Boolean(warning));
}

function mergeQualityWarnings(current: QualityWarning[], incoming: QualityWarningInput[] | undefined): QualityWarning[] {
  const map = new Map<string, QualityWarning>();
  for (const warning of [...current, ...normalizeQualityWarnings(incoming)]) {
    const key = `${warning.code}:${warning.field || ''}:${warning.evidence || ''}:${warning.message}`;
    if (!map.has(key)) map.set(key, warning);
  }
  return Array.from(map.values());
}

function inferBirthDateFromAge(age: string, referenceDate: string): string {
  const years = Number.parseInt(age, 10);
  if (!Number.isFinite(years) || years <= 0 || years > 130) return '';

  const ref = referenceDate ? new Date(`${referenceDate}T00:00:00`) : new Date();
  if (Number.isNaN(ref.getTime())) return '';
  return `${ref.getFullYear() - years}-01-01`;
}

// Врач лучевой диагностики — стартовый экран для него это выбор шаблона.
function isRadiologyDoctor(d: DoctorInfo | null): boolean {
  return !!d && /лучев|радиолог|рентген|кт|мрт|узи/i.test(d.specialty);
}

function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [doctor, setDoctor] = useState<DoctorInfo | null>(null);
  const [activePatient, setActivePatient] = useState<PatientSummary | null>(null);
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null);
  const [pendingSyncs, setPendingSyncs] = useState<Array<{ id: string; filename: string; createdAt: string }>>([]);
  const [step, setStep] = useState<AppStep>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_STEP_KEY) as AppStep | null;
      return saved && ['recording', 'processing', 'editing', 'preview', 'patients', 'patient', 'sync-upload', 'settings', 'admin', 'protocols', 'radiology'].includes(saved) ? saved : 'recording';
    } catch {
      return 'recording';
    }
  });

  const handleLoginDoctor = useCallback((d: DoctorInfo) => {
    setDoctor(d);
    setAuthenticated(true);
    if (isRadiologyDoctor(d)) setStep('radiology');
  }, []);
  const [document, setDocument] = useState<MedicalDocument>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_DOC_KEY);
      return saved ? (JSON.parse(saved) as MedicalDocument) : emptyDocument;
    } catch {
      return emptyDocument;
    }
  });
  const [error, setError] = useState<string | null>(null);
  const [rawTranscription, setRawTranscription] = useState<string>(() => {
    try {
      return sessionStorage.getItem(SESSION_RAW_TEXT_KEY) || '';
    } catch {
      return '';
    }
  });
  const [qualityWarnings, setQualityWarnings] = useState<QualityWarning[]>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_WARNINGS_KEY);
      return normalizeQualityWarnings(saved ? (JSON.parse(saved) as QualityWarningInput[]) : []);
    } catch {
      return [];
    }
  });
  const [processingPhase, setProcessingPhase] = useState<ProcessingPhase>('uploading');
  const [processingDetail, setProcessingDetail] = useState('');
  const [isRestructuring, setIsRestructuring] = useState(false);
  const audioBlobRef = useRef<Blob | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const pendingChunksRef = useRef<Promise<void>[]>([]);
  const batchCountRef = useRef(0);
  const browserFinalTranscriptRef = useRef<string[]>([]);
  const browserInterimTranscriptRef = useRef('');

  useEffect(() => {
    try {
      sessionStorage.setItem(SESSION_STEP_KEY, step);
    } catch (err) {
      console.warn('[session] не удалось сохранить step:', err);
    }
  }, [step]);

  useEffect(() => {
    try {
      sessionStorage.setItem(SESSION_DOC_KEY, JSON.stringify(document));
    } catch (err) {
      // QuotaExceededError при очень большом документе — диагностический сигнал
      console.warn('[session] не удалось сохранить документ:', err);
    }
  }, [document]);

  useEffect(() => {
    try {
      sessionStorage.setItem(SESSION_RAW_TEXT_KEY, rawTranscription);
    } catch (err) {
      console.warn('[session] не удалось сохранить raw text:', err);
    }
  }, [rawTranscription]);

  useEffect(() => {
    try {
      sessionStorage.setItem(SESSION_WARNINGS_KEY, JSON.stringify(qualityWarnings));
    } catch (err) {
      console.warn('[session] не удалось сохранить warnings:', err);
    }
  }, [qualityWarnings]);

  useEffect(() => {
    apiClient.checkAuth().then(async (ok) => {
      setAuthenticated(ok);
      if (ok && !doctor) {
        const me = await apiClient.getMe();
        if (me) {
          setDoctor(me);
          // Врач лучевой — на его стартовый экран, если не выбран другой раздел.
          if (isRadiologyDoctor(me)) setStep((s) => (s === 'recording' ? 'radiology' : s));
        }
      }
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onLogout = () => {
      setDoctor(null);
      setAuthenticated(false);
      setStep('recording');
    };
    window.addEventListener('auth:logout', onLogout);
    return () => window.removeEventListener('auth:logout', onLogout);
  }, []);

  useEffect(() => {
    if (step === 'admin' && doctor?.role !== 'admin') {
      setStep('recording');
    }
  }, [doctor?.role, step]);

  // Polling pending syncs (каждые 10 сек когда авторизован)
  useEffect(() => {
    if (!authenticated) return;
    const poll = async () => {
      try {
        const res = await apiClient.syncPending();
        setPendingSyncs(res.sessions);
      } catch { /* ignore */ }
    };
    void poll();
    const timer = setInterval(poll, 10_000);
    return () => clearInterval(timer);
  }, [authenticated]);

  const handleRecordingStart = useCallback(() => {
    pendingChunksRef.current = [];
    batchCountRef.current = 0;
    browserFinalTranscriptRef.current = [];
    browserInterimTranscriptRef.current = '';
    apiClient.startSession().then(({ sessionId }) => {
      sessionIdRef.current = sessionId;
    }).catch((err) => {
      console.warn('[session] failed to start, will use legacy flow:', err);
      sessionIdRef.current = null;
    });
  }, []);

  const handleBrowserTranscript = useCallback((text: string, isFinal: boolean) => {
    const clean = text.trim();
    if (!clean) return;

    if (!isFinal) {
      browserInterimTranscriptRef.current = clean;
      return;
    }

    browserInterimTranscriptRef.current = '';
    const finalParts = browserFinalTranscriptRef.current;
    const last = finalParts[finalParts.length - 1] || '';
    if (clean === last || last.includes(clean)) return;

    if (last && clean.includes(last)) {
      finalParts[finalParts.length - 1] = clean;
    } else {
      finalParts.push(clean);
    }
  }, []);

  const getBrowserTranscript = useCallback((): string => {
    const finalText = browserFinalTranscriptRef.current.join(' ').trim();
    const interimText = browserInterimTranscriptRef.current.trim();
    if (!interimText) return finalText;
    if (!finalText) return interimText;
    if (finalText.includes(interimText)) return finalText;
    return `${finalText} ${interimText}`.trim();
  }, []);

  const handleBatch = useCallback((blob: Blob, _mimeType: string, batchIndex: number) => {
    const sessionId = sessionIdRef.current;
    if (!sessionId) return;
    batchCountRef.current++;
    const p = new Promise<void>((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const dataUrl = reader.result as string;
        const base64 = dataUrl.split(',')[1];
        apiClient.sendChunk(sessionId, base64, batchIndex)
          .then(() => resolve())
          .catch((err) => {
            console.warn(`[session] chunk ${batchIndex} failed:`, err);
            resolve();
          });
      };
      reader.onerror = () => resolve();
      reader.readAsDataURL(blob);
    });
    pendingChunksRef.current.push(p);
  }, []);

  const streamOptions = useMemo<VoiceRecorderStreamOptions>(() => ({
    batchIntervalSeconds: 20,
    onBatch: handleBatch,
  }), [handleBatch]);

  const applyProcessResult = useCallback((transcriptionText: string, doc: MedicalDocument, warnings: QualityWarningInput[]) => {
    const today = new Date().toISOString().slice(0, 10);
    setRawTranscription(transcriptionText);
    setQualityWarnings(normalizeQualityWarnings(warnings));
    setDocument({
      ...doc,
      patient: {
        ...doc.patient,
        complaintDate: doc.patient.complaintDate || today,
      },
    });
    setStep('editing');
  }, []);

  const legacyProcess = useCallback(async (blob: Blob) => {
    const filename = filenameForBlob(blob, 'recording');
    let audioJobLogOpen = false;

    try {
      setProcessingPhase('uploading');
      setProcessingDetail('Загружаем запись и создаем задачу обработки.');
      console.group('%c[AUDIO JOB] Асинхронная обработка', 'color: #4caf50; font-weight: bold; font-size: 13px');
      audioJobLogOpen = true;
      const started = await apiClient.startAudioJob(blob, filename);
      console.log('Job:', started.jobId, started.statusUrl);
      setProcessingPhase('queued');
      setProcessingDetail('Задача принята сервером, ожидаем старт обработки.');

      const deadline = Date.now() + AUDIO_JOB_TIMEOUT_MS;
      let lastStatus = started.status;

      while (Date.now() < deadline) {
        const job = await apiClient.getAudioJob(started.jobId);
        if (job.status !== lastStatus) {
          lastStatus = job.status;
          console.log('Статус:', job.status);
          if (job.status === 'queued') {
            setProcessingPhase('queued');
            setProcessingDetail('Задача находится в очереди обработки.');
          } else if (job.status === 'transcribing') {
            setProcessingPhase('transcribing');
            setProcessingDetail('Whisper распознает речь из аудио.');
          } else if (job.status === 'structuring') {
            setProcessingPhase('structuring');
            setProcessingDetail('LLM распределяет распознанный текст по разделам.');
          }
        }

        if (job.status === 'done' && job.result) {
          setProcessingPhase('finalizing');
          setProcessingDetail('Документ готов, открываем редактор.');
          console.log('Текст транскрипции:', job.result.transcription.text);
          console.log('Документ:', job.result.document);
          console.groupEnd();
          audioJobLogOpen = false;
          applyProcessResult(
            job.result.transcription.text,
            job.result.document,
            job.result.qualityWarnings || job.result.warnings || [],
          );
          return;
        }

        if (job.status === 'failed') {
          console.groupEnd();
          audioJobLogOpen = false;
          if (job.transcription?.text) {
            setRawTranscription(job.transcription.text);
          }
          throw new Error(job.message || job.error || 'Audio job failed');
        }

        await delay(AUDIO_JOB_POLL_INTERVAL_MS);
      }

      console.groupEnd();
      audioJobLogOpen = false;
      throw new Error('Audio job timeout');
    } catch (jobErr) {
      if (audioJobLogOpen) {
        console.groupEnd();
      }
      setProcessingPhase('fallback');
      setProcessingDetail('Основной поток не завершился, запускаем резервный маршрут.');
      console.warn('[audio-job] failed, falling back to upload/transcribe/structure:', jobErr);
    }

    let transcriptionText = '';

    try {
      setProcessingPhase('uploading');
      setProcessingDetail('Загружаем запись на сервер распознавания.');
      const upload = await apiClient.uploadAudio(blob, filename);

      console.group('%c[WHISPER] Транскрипция', 'color: #00bcd4; font-weight: bold; font-size: 13px');
      console.log('Файл:', upload.filename);
      setProcessingPhase('transcribing');
      setProcessingDetail('Whisper распознает речь из аудио.');
      const transcription = await apiClient.transcribe(upload.filename);
      console.log('Язык:', transcription.language);
      console.log('Длительность:', transcription.duration, 'с');
      console.log('%cТекст Whisper:\n' + transcription.text, 'color: #00bcd4; white-space: pre-wrap');
      console.groupEnd();
      transcriptionText = transcription.text;
    } catch (transcriptionErr) {
      console.warn('[transcription] server transcription failed:', transcriptionErr);
      const browserTranscript = getBrowserTranscript();
      if (!browserTranscript) throw transcriptionErr;
      console.warn('[transcription] using browser SpeechRecognition fallback');
      transcriptionText = browserTranscript;
    }

    console.group('%c[LLM] Структурирование', 'color: #ff9800; font-weight: bold; font-size: 13px');
    setProcessingPhase('structuring');
    setProcessingDetail('LLM распределяет распознанный текст по разделам.');
    const structured = await apiClient.structureText(transcriptionText);
    console.log('Время обработки:', structured.processingTime, 'мс');
    console.log('%cДокумент от LLM:', 'color: #ff9800; font-weight: bold');
    console.log(structured.document);
    console.groupEnd();

    if (!structured.success || !structured.document) throw new Error('Processing failed');
    setProcessingPhase('finalizing');
    setProcessingDetail('Документ готов, открываем редактор.');
    applyProcessResult(transcriptionText, structured.document, structured.qualityWarnings || structured.warnings || []);
  }, [applyProcessResult, getBrowserTranscript]);

  const handleRecordingComplete = useCallback(async (blob: Blob) => {
    audioBlobRef.current = blob;
    setStep('processing');
    setError(null);
    setProcessingPhase('uploading');
    setProcessingDetail('Подготавливаем запись к обработке.');

    const sessionId = sessionIdRef.current;
    sessionIdRef.current = null;
    const hadBatches = batchCountRef.current > 0;
    batchCountRef.current = 0;

    if (sessionId && hadBatches) {
      try {
        // Ждём завершения всех HTTP-запросов отправки чанков
        setProcessingPhase('queued');
        setProcessingDetail('Завершаем потоковую сессию и собираем фрагменты распознавания.');
        await Promise.allSettled(pendingChunksRef.current);
        pendingChunksRef.current = [];

        console.group('%c[SESSION] Финализация', 'color: #4caf50; font-weight: bold; font-size: 13px');
        setProcessingPhase('structuring');
        setProcessingDetail('LLM структурирует текст из потоковой сессии.');
        const result = await apiClient.finishSession(sessionId);
        console.log('Текст транскрипции:', result.transcription.text);
        console.log('Документ:', result.document);
        console.groupEnd();

        if (result.success && result.document) {
          setProcessingPhase('finalizing');
          setProcessingDetail('Документ готов, открываем редактор.');
          applyProcessResult(result.transcription.text, result.document, result.qualityWarnings || result.warnings || []);
        } else {
          throw new Error('Session processing failed');
        }
      } catch (err) {
        console.warn('[session] failed, falling back to legacy flow:', err);
        pendingChunksRef.current = [];
        try {
          await legacyProcess(blob);
        } catch (legacyErr) {
          console.error('Legacy fallback error:', legacyErr);
          setError(legacyErr instanceof Error ? legacyErr.message : 'Ошибка обработки');
          setStep('recording');
        }
      }
    } else {
      pendingChunksRef.current = [];
      try {
        await legacyProcess(blob);
      } catch (err) {
        console.error('Processing error:', err);
        setError(err instanceof Error ? err.message : 'Ошибка обработки');
        setStep('recording');
      }
    }
  }, [applyProcessResult, legacyProcess]);

  const handleDocumentChange = useCallback((newDocument: MedicalDocument) => {
    setDocument(newDocument);
  }, []);

  const applyStructuredDocument = useCallback((structuredDocument: MedicalDocument) => {
    const today = new Date().toISOString().slice(0, 10);
    setDocument({
      ...structuredDocument,
      patient: {
        ...structuredDocument.patient,
        complaintDate: structuredDocument.patient.complaintDate || today,
      },
    });
  }, []);

  const handleRestructure = useCallback(async () => {
    if (!rawTranscription.trim() || isRestructuring) return;
    setIsRestructuring(true);
    setError(null);
    try {
      const structured = await apiClient.structureText(rawTranscription);
      if (!structured.success || !structured.document) {
        throw new Error('Повторное структурирование не вернуло документ');
      }
      applyStructuredDocument(structured.document);
      setQualityWarnings(normalizeQualityWarnings(structured.qualityWarnings || structured.warnings || []));
    } catch (err) {
      console.error('Restructure error:', err);
      setError(err instanceof Error ? err.message : 'Ошибка повторного структурирования');
    } finally {
      setIsRestructuring(false);
    }
  }, [applyStructuredDocument, isRestructuring, rawTranscription]);

  const handlePreview = useCallback(() => {
    setStep('preview');
  }, []);

  const handleEdit = useCallback(() => {
    setStep('editing');
  }, []);

  const handleBackToRecording = useCallback(() => {
    setStep('recording');
    setError(null);
    setProcessingPhase('uploading');
    setProcessingDetail('');
  }, []);

  const handleDocumentComplete = useCallback(async (file: File) => {
    setStep('processing');
    setError(null);
    setProcessingPhase('document');
    setProcessingDetail('Извлекаем текст из загруженного документа.');
    try {
      const result = await apiClient.processDocument(file);
      if (result.success && result.document) {
        setProcessingPhase('finalizing');
        setProcessingDetail('Документ готов, открываем редактор.');
        const documentPatientName = normalizePatientName(result.document.patient.fullName || '');
        if (
          activePatient &&
          documentPatientName &&
          documentPatientName !== normalizePatientName(activePatient.fullName)
        ) {
          setActivePatient(null);
        }
        applyProcessResult(result.transcription.text, result.document, result.qualityWarnings || result.warnings || []);
        if (result.transcription.warning) {
          console.warn('[document]', result.transcription.warning);
        }
      } else {
        throw new Error('Document processing failed');
      }
    } catch (err) {
      console.error('Document processing error:', err);
      setError(formatDocumentProcessingError(err));
      setStep('recording');
    }
  }, [activePatient, applyProcessResult]);

  const handleDocumentSupplement = useCallback(async (file: File): Promise<string> => {
    setError(null);
    let result: Awaited<ReturnType<typeof apiClient.processDocument>>;
    try {
      result = await apiClient.processDocument(file);
    } catch (err) {
      throw new Error(formatDocumentProcessingError(err));
    }
    if (!result.success || !result.document) {
      throw new Error('Document processing failed');
    }

    const supplementPatientName = normalizePatientName(result.document.patient.fullName || '');
    const currentPatientName = normalizePatientName(document.patient.fullName || activePatient?.fullName || '');
    if (supplementPatientName && currentPatientName && supplementPatientName !== currentPatientName) {
      throw new Error('Файл похож на документ другого пациента. Проверьте ФИО перед добавлением.');
    }

    setDocument((prev) => mergeSupplementDocument(prev, result.document));
    setRawTranscription((prev) => appendRawSource(prev, `Документ: ${file.name}`, result.transcription.text || ''));
    setQualityWarnings((prev) => mergeQualityWarnings(prev, result.qualityWarnings || result.warnings || []));

    if (result.transcription.warning) {
      console.warn('[document supplement]', result.transcription.warning);
    }

    return result.transcription.text || '';
  }, [activePatient?.fullName, document.patient.fullName]);

  const handleNewDocument = useCallback(() => {
    setDocument(emptyDocument);
    setRawTranscription('');
    setQualityWarnings([]);
    setActivePatient(null);
    audioBlobRef.current = null;
    browserFinalTranscriptRef.current = [];
    browserInterimTranscriptRef.current = '';
    setError(null);
    setProcessingPhase('uploading');
    setProcessingDetail('');
    setStep('recording');
    try {
      sessionStorage.removeItem(SESSION_STEP_KEY);
      sessionStorage.removeItem(SESSION_DOC_KEY);
      sessionStorage.removeItem(SESSION_RAW_TEXT_KEY);
      sessionStorage.removeItem(SESSION_WARNINGS_KEY);
    } catch { /* ignore */ }
  }, []);

  // ─── Sync handlers ───────────────────────────────────────────────────────

  const handleClaimSync = useCallback(async (syncId: string) => {
    try {
      const res = await apiClient.syncClaim(syncId);
      if (res.document) {
        applyProcessResult(res.rawTranscription, res.document, []);
        setPendingSyncs(prev => prev.filter(s => s.id !== syncId));
      }
    } catch (err) {
      console.error('Claim sync error:', err);
    }
  }, [applyProcessResult]);

  const handleDismissSync = useCallback(async (syncId: string) => {
    try {
      await apiClient.syncDelete(syncId);
      setPendingSyncs(prev => prev.filter(s => s.id !== syncId));
    } catch { /* ignore */ }
  }, []);

  // ─── Patient card handlers ────────────────────────────────────────────────

  const handleOpenPatients = useCallback(() => setStep('patients'), []);
  const handleOpenSettings = useCallback(() => setStep('settings'), []);
  const handleOpenAdmin = useCallback(() => setStep('admin'), []);
  const handleOpenProtocols = useCallback(() => setStep('protocols'), []);
  const handleOpenRadiology = useCallback(() => setStep('radiology'), []);
  // «Домашний» экран врача: лучевая — на выбор шаблона, остальные — на запись.
  const homeStep: AppStep = isRadiologyDoctor(doctor) ? 'radiology' : 'recording';
  const goHome = useCallback(() => setStep(homeStep), [homeStep]);

  const handleLogout = useCallback(async () => {
    await apiClient.logout();
    setDoctor(null);
    setAuthenticated(false);
    setStep('recording');
  }, []);

  const handleSelectPatient = useCallback((p: PatientSummary) => {
    setSelectedPatientId(p.id);
    setStep('patient');
  }, []);

  const handleNewRecordingForPatient = useCallback((p: PatientSummary) => {
    setActivePatient(p);
    setStep('recording');
  }, []);

  const handleViewVisitDocument = useCallback((doc: MedicalDocument, rawText: string) => {
    setDocument(doc);
    setRawTranscription(rawText);
    setQualityWarnings([]);
    setStep('editing');
  }, []);

  const handleSaveToPatient = useCallback(async (patientId: number) => {
    await apiClient.saveVisit(patientId, document, rawTranscription);
  }, [document, rawTranscription]);

  const handleCreatePatientAndSaveVisit = useCallback(async (): Promise<PatientSummary> => {
    const fullName = document.patient.fullName.trim().replace(/\s+/g, ' ');
    if (!fullName) {
      throw new Error('Укажите ФИО пациента перед сохранением карточки');
    }

    const birthDate =
      document.patient.birthDate ||
      inferBirthDateFromAge(document.patient.age, document.patient.complaintDate);
    const gender = normalizePatientGender(document.patient.gender);

    const existing = await apiClient.getPatients({ q: fullName, page: 1 });
    const normalizedName = normalizePatientName(fullName);
    const matched = existing.patients.find((patient) => {
      if (normalizePatientName(patient.fullName) !== normalizedName) return false;
      if (birthDate) return patient.birthDate === birthDate;
      return !patient.birthDate;
    });

    const patient = matched || (await apiClient.createPatient({
      fullName,
      birthDate,
      gender,
      phone: '',
      iin: '',
      notes: '',
    })).patient;

    setActivePatient(patient);
    setSelectedPatientId(patient.id);
    await apiClient.saveVisit(patient.id, document, rawTranscription);
    return patient;
  }, [document, rawTranscription]);

  // ─── Render ───────────────────────────────────────────────────────────────

  if (authenticated === null) {
    return <div className="min-h-screen flex items-center justify-center"><p className="text-text-muted">Загрузка...</p></div>;
  }

  if (!authenticated) {
    return <LoginScreen onLogin={handleLoginDoctor} />;
  }

  return (
    <div className="min-h-screen">
      {step === 'patients' && doctor && (
        <PatientListScreen
          doctor={doctor}
          onSelectPatient={handleSelectPatient}
          onNewRecording={() => setStep('recording')}
        />
      )}

      {step === 'patient' && selectedPatientId !== null && (
        <PatientScreen
          patientId={selectedPatientId}
          onBack={() => setStep('patients')}
          onNewRecording={handleNewRecordingForPatient}
          onViewDocument={handleViewVisitDocument}
        />
      )}

      {step === 'sync-upload' && (
        <SyncUploadScreen onBack={() => setStep('recording')} />
      )}

      {step === 'settings' && doctor && (
        <SettingsScreen
          doctor={doctor}
          onBack={goHome}
          onLogout={handleLogout}
          onDoctorUpdate={setDoctor}
        />
      )}

      {step === 'admin' && doctor?.role === 'admin' && (
        <AdminPanelScreen
          doctor={doctor}
          onBack={goHome}
        />
      )}

      {step === 'protocols' && doctor && (
        <ProtocolWorkspaceScreen
          doctor={doctor}
          onBack={() => setStep('recording')}
        />
      )}

      {step === 'radiology' && doctor && (
        <RadiologyWorkspaceScreen
          doctor={doctor}
          onOpenSettings={handleOpenSettings}
          onOpenAdmin={doctor.role === 'admin' ? handleOpenAdmin : undefined}
          onOpenTherapy={doctor.role === 'admin' ? () => setStep('recording') : undefined}
          onLogout={handleLogout}
        />
      )}

      {step === 'recording' && (
        <RecordingScreen
          onRecordingComplete={handleRecordingComplete}
          onRecordingStart={handleRecordingStart}
          onRecordingTranscript={handleBrowserTranscript}
          streamOptions={streamOptions}
          onDocumentUpload={handleDocumentComplete}
          activePatient={activePatient}
          onOpenPatients={doctor ? handleOpenPatients : undefined}
          onOpenSettings={doctor ? handleOpenSettings : undefined}
          onOpenAdmin={doctor?.role === 'admin' ? handleOpenAdmin : undefined}
          onOpenProtocols={doctor ? handleOpenProtocols : undefined}
          onOpenRadiology={isRadiologyDoctor(doctor) || doctor?.role === 'admin' ? handleOpenRadiology : undefined}
          onSyncUpload={doctor ? () => setStep('sync-upload') : undefined}
          pendingSyncs={pendingSyncs}
          onClaimSync={handleClaimSync}
          onDismissSync={handleDismissSync}
          error={error}
        />
      )}

      {step === 'processing' && <ProcessingScreen phase={processingPhase} detail={processingDetail} />}

      {step === 'editing' && (
        <EditingScreen
          document={document}
          onDocumentChange={handleDocumentChange}
          onPreview={handlePreview}
          onBack={handleBackToRecording}
          rawTranscription={rawTranscription}
          qualityWarnings={qualityWarnings}
          onRestructure={handleRestructure}
          isRestructuring={isRestructuring}
          activePatient={activePatient}
          onSaveToPatient={handleSaveToPatient}
          onCreatePatientFromDocument={handleCreatePatientAndSaveVisit}
          onOpenPatients={doctor ? handleOpenPatients : undefined}
          onDocumentSupplementUpload={handleDocumentSupplement}
        />
      )}

      {step === 'preview' && (
        <Suspense fallback={<div className="min-h-screen flex items-center justify-center"><p className="text-text-muted">Загрузка предпросмотра...</p></div>}>
          <PreviewScreen
            document={document}
            audioBlob={audioBlobRef.current}
            onEdit={handleEdit}
            onNewDocument={handleNewDocument}
          />
        </Suspense>
      )}
    </div>
  );
}

export default App;
