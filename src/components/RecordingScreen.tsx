import { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, Pause, Play, Square, AlertCircle, CheckCircle, RotateCcw, Upload } from 'lucide-react';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';
import { useWakeWord } from '../hooks/useWakeWord';
import { WaveformVisualizer } from './WaveformVisualizer';


interface RecordingScreenProps {
  onRecordingComplete: (audioBlob: Blob) => void;
  error?: string | null;
}

export function RecordingScreen({ onRecordingComplete, error: externalError }: RecordingScreenProps) {
  const {
    isRecording,
    isPaused,
    audioBlob,
    formattedDuration,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording,
    resetRecording,
  } = useVoiceRecorder();

  const [error, setError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const autoSubmitRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Wake word: "Джарвис" starts recording, "стоп" stops and auto-submits
  const handleWakeWord = useCallback(() => {
    if (isRecording || audioBlob || hasPermission === false) return;
    setError(null);
    autoSubmitRef.current = true;
    void startRecording().catch((err) => {
      setError(err instanceof Error ? err.message : 'Ошибка записи');
      autoSubmitRef.current = false;
    });
  }, [isRecording, audioBlob, hasPermission, startRecording]);

  const handleStopWord = useCallback(() => {
    if (!isRecording) return;
    stopRecording(2); // trim last 2 seconds to cut off "Стоп Нави"
  }, [isRecording, stopRecording]);

  useWakeWord({
    enabled: true,
    isRecording,
    onWakeWord: handleWakeWord,
    onStopWord: handleStopWord,
  });

  // Auto-submit after stop word: when audioBlob appears and autoSubmit flag is set
  useEffect(() => {
    if (audioBlob && autoSubmitRef.current) {
      autoSubmitRef.current = false;
      onRecordingComplete(audioBlob);
    }
  }, [audioBlob, onRecordingComplete]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    onRecordingComplete(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  useEffect(() => {
    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices?.getUserMedia) {
      setHasPermission(false);
      setError('Микрофон недоступен: откройте приложение по HTTPS или localhost.');
      return;
    }

    mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        stream.getTracks().forEach((track) => track.stop());
        setHasPermission(true);
      })
      .catch(() => {
        setHasPermission(false);
      });
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      const key = event.key.toLowerCase();
      if (key !== 'm' && key !== 'ь') return;

      const target = event.target as HTMLElement | null;
      const isEditableTarget =
        !!target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable);
      if (isEditableTarget) return;

      if (hasPermission === false) return;

      if (isRecording) {
        if (isPaused) {
          resumeRecording();
        } else {
          pauseRecording();
        }
        return;
      }

      if (!!audioBlob) return;

      setError(null);
      void startRecording().catch((err) => {
        setError(err instanceof Error ? err.message : 'Ошибка записи');
      });
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    audioBlob,
    hasPermission,
    isPaused,
    isRecording,
    pauseRecording,
    resumeRecording,
    startRecording,
  ]);

  const handleStart = async () => {
    setError(null);
    try {
      await startRecording();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка записи');
    }
  };

  const handleSubmit = () => {
    if (audioBlob) {
      onRecordingComplete(audioBlob);
    }
  };

  const MAX_DURATION_SECONDS = 30 * 60;
  const durationParts = formattedDuration.split(':');
  const currentSeconds = parseInt(durationParts[0], 10) * 60 + parseInt(durationParts[1], 10);
  const progress = (currentSeconds / MAX_DURATION_SECONDS) * 100;

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-12 slide-up">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-medical-100 text-medical-700 rounded-full text-sm font-medium mb-6">
            <Mic className="w-4 h-4" />
            Голосовой ввод
          </div>
          <h1 className="text-4xl font-display font-bold text-medical-900 mb-3">Запись медицинского протокола</h1>
          <p className="text-text-secondary text-lg">Надиктуйте информацию о пациенте, жалобах и диагнозе</p>
        </div>

        <div className="glass-card rounded-3xl p-8 slide-up" style={{ animationDelay: '0.1s' }}>
          {hasPermission === false && (
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-6">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700 text-sm">Для записи нужен доступ к микрофону. Разрешите его в настройках браузера.</p>
            </div>
          )}

          {externalError && (
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-6">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700 text-sm">Ошибка обработки: {externalError}. Попробуйте записать снова.</p>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-6">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          )}

          <div className="bg-slate-50 rounded-2xl p-6 mb-8">
            <WaveformVisualizer isActive={isRecording} isPaused={isPaused} />

            <div className="mt-4 text-center">
              <div
                className={`text-3xl font-mono font-bold ${
                  isRecording ? (isPaused ? 'text-amber-500' : 'text-medical-600') : 'text-text-muted'
                }`}
              >
                {formattedDuration}
              </div>
              {isRecording && (
                <div className="mt-3 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-medical-400 to-medical-600 transition-all duration-1000"
                    style={{ width: `${Math.min(progress, 100)}%` }}
                  />
                </div>
              )}
              <p className="text-text-muted text-sm mt-2">
                {isRecording
                  ? isPaused
                    ? 'Запись на паузе'
                    : 'Идет запись... Скажите "Стоп Нави" для завершения'
                  : audioBlob
                    ? 'Запись завершена'
                    : 'Скажите "Нави" или нажмите кнопку'}
              </p>
            </div>
          </div>

          <div className="flex items-center justify-center gap-4">
            {!isRecording && !audioBlob && (
              <>
                <button
                  onClick={handleStart}
                  disabled={hasPermission === false}
                  className="btn-primary flex items-center gap-3 text-lg px-8 py-4 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Mic className="w-6 h-6" />
                  Начать запись
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-secondary flex items-center gap-2 text-sm"
                >
                  <Upload className="w-4 h-4" />
                  Загрузить файл
                </button>
              </>
            )}

            {isRecording && (
              <>
                <button onClick={isPaused ? resumeRecording : pauseRecording} className="btn-secondary flex items-center gap-2">
                  {isPaused ? (
                    <>
                      <Play className="w-5 h-5" />
                      Продолжить
                    </>
                  ) : (
                    <>
                      <Pause className="w-5 h-5" />
                      Пауза
                    </>
                  )}
                </button>

                <button
                  onClick={() => stopRecording()}
                  className="btn-primary flex items-center gap-2 bg-gradient-to-r from-red-500 to-red-600 shadow-red-500/30 hover:shadow-red-500/40 hover:from-red-600 hover:to-red-700"
                >
                  <Square className="w-5 h-5" />
                  Завершить
                </button>
              </>
            )}

            {!isRecording && audioBlob && (
              <>
                <button onClick={resetRecording} className="btn-secondary flex items-center gap-2">
                  <RotateCcw className="w-5 h-5" />
                  Записать заново
                </button>

                <button onClick={handleSubmit} className="btn-primary flex items-center gap-2 text-lg px-8 py-4">
                  <CheckCircle className="w-6 h-6" />
                  Продолжить
                </button>
              </>
            )}
          </div>
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 slide-up" style={{ animationDelay: '0.2s' }}>
          {[
            {
              icon: '🎙️',
              title: 'Говорите четко',
              desc: 'Произносите медицинские термины разборчиво',
            },
            {
              icon: '📋',
              title: 'Структура',
              desc: 'Начните с ФИО пациента, затем жалобы и диагноз',
            },
            {
              icon: '✏️',
              title: 'Редактирование',
              desc: 'После записи можно отредактировать текст',
            },
          ].map((tip, index) => (
            <div key={index} className="bg-white/60 backdrop-blur rounded-xl p-4 border border-white/80">
              <div className="text-2xl mb-2">{tip.icon}</div>
              <h3 className="font-semibold text-medical-800 mb-1">{tip.title}</h3>
              <p className="text-text-secondary text-sm">{tip.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
