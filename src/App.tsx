import { useState, useCallback, useRef, useEffect } from 'react';
import type { AppStep, MedicalDocument } from './types';
import { emptyDocument } from './types';
import { LoginScreen } from './components/LoginScreen';
import { RecordingScreen } from './components/RecordingScreen';
import { ProcessingScreen } from './components/ProcessingScreen';
import { EditingScreen } from './components/EditingScreen';
import { PreviewScreen } from './components/PreviewScreen';
import { apiClient } from './api/client';

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

function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [step, setStep] = useState<AppStep>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_STEP_KEY) as AppStep | null;
      return saved && ['recording', 'processing', 'editing', 'preview'].includes(saved) ? saved : 'recording';
    } catch {
      return 'recording';
    }
  });
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
    apiClient.checkAuth().then(setAuthenticated);
  }, []);

  useEffect(() => {
    const onLogout = () => setAuthenticated(false);
    window.addEventListener('auth:logout', onLogout);
    return () => window.removeEventListener('auth:logout', onLogout);
  }, []);

  const handleRecordingComplete = useCallback(async (blob: Blob) => {
    audioBlobRef.current = blob;
    setStep('processing');
    setError(null);

    try {
      const filename = filenameForBlob(blob, 'recording');

      // Step 1: upload audio
      const upload = await apiClient.uploadAudio(blob, filename);

      // Step 2: Whisper transcription
      console.group('%c[WHISPER] Транскрипция', 'color: #00bcd4; font-weight: bold; font-size: 13px');
      console.log('Файл:', upload.filename);
      const transcription = await apiClient.transcribe(upload.filename);
      console.log('Язык:', transcription.language);
      console.log('Длительность:', transcription.duration, 'с');
      console.log('%cТекст Whisper:\n' + transcription.text, 'color: #00bcd4; white-space: pre-wrap');
      console.groupEnd();

      // Step 3: LLM structuring
      console.group('%c[LLM] Структурирование', 'color: #ff9800; font-weight: bold; font-size: 13px');
      console.log('Входной текст:', transcription.text);
      const structured = await apiClient.structureText(transcription.text);
      console.log('Время обработки:', structured.processingTime, 'мс');
      console.log('%cДокумент от LLM:', 'color: #ff9800; font-weight: bold');
      console.log(structured.document);
      console.groupEnd();

      if (structured.success && structured.document) {
        const today = new Date().toISOString().slice(0, 10);
        setRawTranscription(transcription.text);
        setQualityWarnings(structured.warnings || []);
        setDocument({
          ...structured.document,
          patient: {
            ...structured.document.patient,
            complaintDate: structured.document.patient.complaintDate || today,
          },
        });
        setStep('editing');
      } else {
        throw new Error('Processing failed');
      }
    } catch (err) {
      console.error('Processing error:', err);
      setError(err instanceof Error ? err.message : 'Ошибка обработки');
      setStep('recording');
    }
  }, []);

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

  const handleNewDocument = useCallback(() => {
    setDocument(emptyDocument);
    setRawTranscription('');
    setQualityWarnings([]);
    audioBlobRef.current = null;
    setError(null);
    setStep('recording');
    try {
      sessionStorage.removeItem(SESSION_STEP_KEY);
      sessionStorage.removeItem(SESSION_DOC_KEY);
      sessionStorage.removeItem(SESSION_RAW_TEXT_KEY);
      sessionStorage.removeItem(SESSION_WARNINGS_KEY);
    } catch {
      // ignore
    }
  }, []);

  if (authenticated === null) {
    return <div className="min-h-screen flex items-center justify-center"><p className="text-text-muted">Загрузка...</p></div>;
  }

  if (!authenticated) {
    return <LoginScreen onLogin={() => setAuthenticated(true)} />;
  }

  return (
    <div className="min-h-screen">
      {step === 'recording' && <RecordingScreen onRecordingComplete={handleRecordingComplete} error={error} />}

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
        />
      )}

      {step === 'preview' && <PreviewScreen document={document} audioBlob={audioBlobRef.current} onEdit={handleEdit} onNewDocument={handleNewDocument} />}
    </div>
  );
}

export default App;
