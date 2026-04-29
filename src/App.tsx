import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import type { VoiceRecorderStreamOptions } from './hooks/useVoiceRecorder';
import type { AppStep, MedicalDocument } from './types';
import { emptyDocument } from './types';
import { LoginScreen } from './components/LoginScreen';
import { RecordingScreen } from './components/RecordingScreen';
import { ProcessingScreen } from './components/ProcessingScreen';
import { EditingScreen } from './components/EditingScreen';
import { PreviewScreen } from './components/PreviewScreen';
import { PatientListScreen } from './components/PatientListScreen';
import { PatientScreen } from './components/PatientScreen';
import { SyncUploadScreen } from './components/SyncUploadScreen';
import { SettingsScreen } from './components/SettingsScreen';
import { apiClient } from './api/client';
import type { DoctorInfo, PatientSummary } from './api/client';

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

function normalizePatientName(name: string): string {
  return name.trim().replace(/\s+/g, ' ').toLowerCase();
}

function normalizePatientGender(gender: string): string {
  const value = gender.trim().toLowerCase();
  if (value.startsWith('м')) return 'male';
  if (value.startsWith('ж')) return 'female';
  return gender.trim();
}

function inferBirthDateFromAge(age: string, referenceDate: string): string {
  const years = Number.parseInt(age, 10);
  if (!Number.isFinite(years) || years <= 0 || years > 130) return '';

  const ref = referenceDate ? new Date(`${referenceDate}T00:00:00`) : new Date();
  if (Number.isNaN(ref.getTime())) return '';
  return `${ref.getFullYear() - years}-01-01`;
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
      return saved && ['recording', 'processing', 'editing', 'preview'].includes(saved) ? saved : 'recording';
    } catch {
      return 'recording';
    }
  });

  const handleLoginDoctor = useCallback((d: DoctorInfo) => {
    setDoctor(d);
    setAuthenticated(true);
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
  const [qualityWarnings, setQualityWarnings] = useState<string[]>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_WARNINGS_KEY);
      return saved ? (JSON.parse(saved) as string[]) : [];
    } catch {
      return [];
    }
  });
  const [isRestructuring, setIsRestructuring] = useState(false);
  const audioBlobRef = useRef<Blob | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const pendingChunksRef = useRef<Promise<void>[]>([]);
  const batchCountRef = useRef(0);

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
        if (me) setDoctor(me);
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
    apiClient.startSession().then(({ sessionId }) => {
      sessionIdRef.current = sessionId;
    }).catch((err) => {
      console.warn('[session] failed to start, will use legacy flow:', err);
      sessionIdRef.current = null;
    });
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

  const applyProcessResult = useCallback((transcriptionText: string, doc: MedicalDocument, warnings: string[]) => {
    const today = new Date().toISOString().slice(0, 10);
    setRawTranscription(transcriptionText);
    setQualityWarnings(warnings);
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

    const upload = await apiClient.uploadAudio(blob, filename);

    console.group('%c[WHISPER] Транскрипция', 'color: #00bcd4; font-weight: bold; font-size: 13px');
    console.log('Файл:', upload.filename);
    const transcription = await apiClient.transcribe(upload.filename);
    console.log('Язык:', transcription.language);
    console.log('Длительность:', transcription.duration, 'с');
    console.log('%cТекст Whisper:\n' + transcription.text, 'color: #00bcd4; white-space: pre-wrap');
    console.groupEnd();

    console.group('%c[LLM] Структурирование', 'color: #ff9800; font-weight: bold; font-size: 13px');
    const structured = await apiClient.structureText(transcription.text);
    console.log('Время обработки:', structured.processingTime, 'мс');
    console.log('%cДокумент от LLM:', 'color: #ff9800; font-weight: bold');
    console.log(structured.document);
    console.groupEnd();

    if (!structured.success || !structured.document) throw new Error('Processing failed');
    applyProcessResult(transcription.text, structured.document, structured.warnings || []);
  }, [applyProcessResult]);

  const handleRecordingComplete = useCallback(async (blob: Blob) => {
    audioBlobRef.current = blob;
    setStep('processing');
    setError(null);

    const sessionId = sessionIdRef.current;
    sessionIdRef.current = null;
    const hadBatches = batchCountRef.current > 0;
    batchCountRef.current = 0;

    if (sessionId && hadBatches) {
      try {
        // Ждём завершения всех HTTP-запросов отправки чанков
        await Promise.allSettled(pendingChunksRef.current);
        pendingChunksRef.current = [];

        console.group('%c[SESSION] Финализация', 'color: #4caf50; font-weight: bold; font-size: 13px');
        const result = await apiClient.finishSession(sessionId);
        console.log('Текст транскрипции:', result.transcription.text);
        console.log('Документ:', result.document);
        console.groupEnd();

        if (result.success && result.document) {
          applyProcessResult(result.transcription.text, result.document, result.warnings || []);
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
      setQualityWarnings(structured.warnings || []);
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
  }, []);

  const handleDocumentComplete = useCallback(async (file: File) => {
    setStep('processing');
    setError(null);
    try {
      const result = await apiClient.processDocument(file);
      if (result.success && result.document) {
        const documentPatientName = normalizePatientName(result.document.patient.fullName || '');
        if (
          activePatient &&
          documentPatientName &&
          documentPatientName !== normalizePatientName(activePatient.fullName)
        ) {
          setActivePatient(null);
        }
        applyProcessResult(result.transcription.text, result.document, result.warnings || []);
        if (result.transcription.warning) {
          console.warn('[document]', result.transcription.warning);
        }
      } else {
        throw new Error('Document processing failed');
      }
    } catch (err) {
      console.error('Document processing error:', err);
      setError(err instanceof Error ? err.message : 'Ошибка обработки документа');
      setStep('recording');
    }
  }, [activePatient, applyProcessResult]);

  const handleNewDocument = useCallback(() => {
    setDocument(emptyDocument);
    setRawTranscription('');
    setQualityWarnings([]);
    setActivePatient(null);
    audioBlobRef.current = null;
    setError(null);
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
          onBack={() => setStep('recording')}
          onLogout={handleLogout}
          onDoctorUpdate={setDoctor}
        />
      )}

      {step === 'recording' && (
        <RecordingScreen
          onRecordingComplete={handleRecordingComplete}
          onRecordingStart={handleRecordingStart}
          streamOptions={streamOptions}
          onDocumentUpload={handleDocumentComplete}
          activePatient={activePatient}
          onOpenPatients={doctor ? handleOpenPatients : undefined}
          onOpenSettings={doctor ? handleOpenSettings : undefined}
          onSyncUpload={doctor ? () => setStep('sync-upload') : undefined}
          pendingSyncs={pendingSyncs}
          onClaimSync={handleClaimSync}
          onDismissSync={handleDismissSync}
          error={error}
        />
      )}

      {step === 'processing' && <ProcessingScreen />}

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
        />
      )}

      {step === 'preview' && (
        <PreviewScreen
          document={document}
          audioBlob={audioBlobRef.current}
          onEdit={handleEdit}
          onNewDocument={handleNewDocument}
        />
      )}
    </div>
  );
}

export default App;
