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

function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [step, setStep] = useState<AppStep>('recording');
  const [document, setDocument] = useState<MedicalDocument>(emptyDocument);
  const [error, setError] = useState<string | null>(null);
  const audioBlobRef = useRef<Blob | null>(null);

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
      const result = await apiClient.processAudio(blob, filenameForBlob(blob, 'recording'));

      if (result.success && result.document) {
        const today = new Date().toISOString().slice(0, 10);
        setDocument({
          ...result.document,
          patient: {
            ...result.document.patient,
            complaintDate: result.document.patient.complaintDate || today,
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
    audioBlobRef.current = null;
    setError(null);
    setStep('recording');
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
        />
      )}

      {step === 'preview' && <PreviewScreen document={document} audioBlob={audioBlobRef.current} onEdit={handleEdit} onNewDocument={handleNewDocument} />}
    </div>
  );
}

export default App;
