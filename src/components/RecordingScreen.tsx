import { useState, useEffect, useRef, useCallback } from 'react';
import { Mic, Pause, Play, Square, AlertCircle, CheckCircle, RotateCcw, Upload, FileText, Camera, Users, Smartphone, Bell, Settings, Shield, X as XIcon } from 'lucide-react';
import type { PatientSummary } from '../api/client';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';
import type { VoiceRecorderStreamOptions } from '../hooks/useVoiceRecorder';
import { useWakeWord } from '../hooks/useWakeWord';
import { WaveformVisualizer } from './WaveformVisualizer';


interface PendingSync {
  id: string;
  filename: string;
  createdAt: string;
}

interface RecordingScreenProps {
  onRecordingComplete: (audioBlob: Blob) => void;
  onRecordingStart?: () => void;
  onRecordingTranscript?: (text: string, isFinal: boolean) => void;
  streamOptions?: VoiceRecorderStreamOptions;
  onDocumentUpload?: (file: File) => void;
  activePatient?: PatientSummary | null;
  onOpenPatients?: () => void;
  onOpenSettings?: () => void;
  onOpenAdmin?: () => void;
  onOpenProtocols?: () => void;
  onSyncUpload?: () => void;
  pendingSyncs?: PendingSync[];
  onClaimSync?: (syncId: string) => void;
  onDismissSync?: (syncId: string) => void;
  error?: string | null;
}

function formatExternalErrorMessage(message: string): string {
  const isDocumentError = /документ|pdf|word|фото|изображ|vision|claude|anthropic|ollama|модель|model|распозна|баланс|средств/i.test(message);
  return isDocumentError
    ? `Ошибка обработки: ${message}`
    : `Ошибка обработки: ${message}. Попробуйте записать снова.`;
}

export function RecordingScreen({
  onRecordingComplete, onRecordingStart, streamOptions, onDocumentUpload,
  onRecordingTranscript,
  activePatient, onOpenPatients, onOpenSettings, onOpenAdmin, onOpenProtocols, onSyncUpload,
  pendingSyncs = [], onClaimSync, onDismissSync,
  error: externalError,
}: RecordingScreenProps) {
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
  } = useVoiceRecorder(streamOptions);

  const [error, setError] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const autoSubmitRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);

  // Wake word: "Джарвис" starts recording, "стоп" stops and auto-submits
  const handleWakeWord = useCallback(() => {
    if (isRecording || audioBlob || hasPermission === false) return;
    setError(null);
    autoSubmitRef.current = true;
    onRecordingStart?.();
    void startRecording().catch((err) => {
      setError(err instanceof Error ? err.message : 'Ошибка записи');
      autoSubmitRef.current = false;
    });
  }, [isRecording, audioBlob, hasPermission, startRecording, onRecordingStart]);

  const handleStopWord = useCallback(() => {
    if (!isRecording) return;
    stopRecording(2); // trim last 2 seconds to cut off "Стоп Нави"
  }, [isRecording, stopRecording]);

  useWakeWord({
    enabled: true,
    isRecording,
    onWakeWord: handleWakeWord,
    onStopWord: handleStopWord,
    onTranscript: onRecordingTranscript,
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
    // Лимит размера аудио снят по требованию (ранее был 100 МБ).
    // NB: Groq принимает ≤25 МБ (free) / ≤100 МБ (dev) — файлы крупнее
    // автоматически уходят в fallback на локальный whisper-сервер.
    onRecordingComplete(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDocumentFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const MAX_SIZE = 20 * 1024 * 1024; // 20 МБ
    if (file.size > MAX_SIZE) {
      setError('Документ слишком большой. Максимум 20 МБ.');
      if (e.target) e.target.value = '';
      return;
    }
    setError(null);
    onDocumentUpload?.(file);
    if (e.target) e.target.value = '';
  };

  useEffect(() => {
    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices?.getUserMedia) {
      const timer = window.setTimeout(() => {
        setHasPermission(false);
        setError('Микрофон недоступен: откройте приложение по HTTPS или localhost.');
      }, 0);
      return () => window.clearTimeout(timer);
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

      if (audioBlob) return;

      setError(null);
      onRecordingStart?.();
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
    onRecordingStart,
    pauseRecording,
    resumeRecording,
    startRecording,
  ]);

  const handleStart = async () => {
    setError(null);
    try {
      onRecordingStart?.();
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
    <div className="min-h-screen flex flex-col">
      {/* Top bar */}
      {(activePatient || onOpenPatients || onOpenSettings || onOpenAdmin || onOpenProtocols || onSyncUpload || pendingSyncs.length > 0) && (
        <div className="bg-white border-b border-slate-200 px-4 py-2.5 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            {activePatient && (
              <span className="text-sm font-medium text-medical-900 truncate">
                {activePatient.fullName}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* Pending syncs badge */}
            {pendingSyncs.length > 0 && (
              <div className="relative">
                <button
                  className="flex items-center gap-1.5 text-sm text-amber-600 hover:text-amber-700 bg-amber-50 px-3 py-1.5 rounded-lg border border-amber-200"
                  title="Документы с телефона готовы"
                >
                  <Bell className="w-4 h-4" />
                  <span className="font-medium">{pendingSyncs.length}</span>
                </button>
              </div>
            )}
            {onSyncUpload && (
              <button onClick={onSyncUpload} className="flex items-center gap-1.5 text-sm text-medical-600 hover:text-medical-700 whitespace-nowrap">
                <Smartphone className="w-4 h-4" /> На ПК
              </button>
            )}
            {onOpenPatients && (
              <button onClick={onOpenPatients} className="flex items-center gap-1.5 text-sm text-medical-600 hover:text-medical-700 whitespace-nowrap">
                <Users className="w-4 h-4" /> Пациенты
              </button>
            )}
            {onOpenProtocols && (
              <button onClick={onOpenProtocols} className="flex items-center gap-1.5 text-sm text-medical-600 hover:text-medical-700 whitespace-nowrap">
                <FileText className="w-4 h-4" /> Протоколы
              </button>
            )}
            {onOpenAdmin && (
              <button onClick={onOpenAdmin} className="flex items-center gap-1.5 text-sm text-medical-600 hover:text-medical-700 whitespace-nowrap">
                <Shield className="w-4 h-4" /> Админка
              </button>
            )}
            {onOpenSettings && (
              <button onClick={onOpenSettings} className="flex items-center gap-1.5 text-sm text-medical-600 hover:text-medical-700 whitespace-nowrap">
                <Settings className="w-4 h-4" /> Настройки
              </button>
            )}
          </div>
        </div>
      )}

      {/* Pending syncs dropdown */}
      {pendingSyncs.length > 0 && (
        <div className="bg-amber-50 border-b border-amber-200 px-4 py-3">
          <p className="text-xs font-semibold text-amber-800 mb-2 flex items-center gap-1.5">
            <Bell className="w-3.5 h-3.5" /> Документы с телефона готовы к открытию:
          </p>
          <div className="space-y-1.5">
            {pendingSyncs.map(s => (
              <div key={s.id} className="flex items-center gap-2 bg-white rounded-lg px-3 py-2 border border-amber-200">
                <FileText className="w-4 h-4 text-amber-600 flex-shrink-0" />
                <span className="text-sm text-medical-900 truncate flex-1">{s.filename || 'Документ'}</span>
                <button
                  onClick={() => onClaimSync?.(s.id)}
                  className="text-xs bg-medical-600 hover:bg-medical-700 text-white px-2 py-1 rounded-md whitespace-nowrap"
                >
                  Открыть
                </button>
                <button onClick={() => onDismissSync?.(s.id)} className="p-0.5 hover:bg-amber-100 rounded">
                  <XIcon className="w-3.5 h-3.5 text-amber-500" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

    <div className="flex-1 flex items-center justify-center px-4 py-6 sm:p-6">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-8 sm:mb-12 slide-up">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-medical-100 text-medical-700 rounded-full text-sm font-medium mb-6">
            <Mic className="w-4 h-4" />
            Голосовой ввод
          </div>
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-display font-bold text-medical-900 mb-3">
            Запись медицинского протокола
          </h1>
          <p className="text-text-secondary text-base sm:text-lg">
            Надиктуйте информацию о пациенте, жалобах и диагнозе
          </p>
        </div>

        <div className="glass-card rounded-2xl sm:rounded-3xl p-5 sm:p-8 slide-up" style={{ animationDelay: '0.1s' }}>
          {hasPermission === false && (
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-6">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700 text-sm">Для записи нужен доступ к микрофону. Разрешите его в настройках браузера.</p>
            </div>
          )}

          {externalError && (
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-6">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700 text-sm">{formatExternalErrorMessage(externalError)}</p>
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

          {/* Кнопки: на мобильном — wrap + flex-1 чтобы не вылезали; от sm — обычный inline-row */}
          <div className="flex flex-wrap items-stretch justify-center gap-3 sm:gap-4">
            {!isRecording && !audioBlob && (
              <>
                <button
                  onClick={handleStart}
                  disabled={hasPermission === false}
                  className="btn-primary flex items-center justify-center gap-3 text-base sm:text-lg px-5 sm:px-8 py-3 sm:py-4 flex-1 sm:flex-initial whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Mic className="w-6 h-6" />
                  Начать запись
                </button>

                {/* Скрытые file inputs */}
                <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
                <input
                  ref={docInputRef}
                  type="file"
                  accept=".pdf,.docx,.doc,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  onChange={handleDocumentFileChange}
                  className="hidden"
                />
                <input
                  ref={photoInputRef}
                  type="file"
                  accept="image/*"
                  capture="environment"
                  onChange={handleDocumentFileChange}
                  className="hidden"
                />

                {onDocumentUpload && (
                  <button
                    onClick={() => docInputRef.current?.click()}
                    className="btn-secondary flex items-center justify-center gap-2 text-sm flex-1 sm:flex-initial whitespace-nowrap"
                    title="Загрузить PDF или Word документ"
                  >
                    <FileText className="w-4 h-4" />
                    PDF / Word
                  </button>
                )}

                {onDocumentUpload && (
                  <button
                    onClick={() => photoInputRef.current?.click()}
                    className="btn-secondary flex items-center justify-center gap-2 text-sm flex-1 sm:flex-initial whitespace-nowrap"
                    title="Сфотографировать документ (анализы, выписка)"
                  >
                    <Camera className="w-4 h-4" />
                    Фото
                  </button>
                )}

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-secondary flex items-center justify-center gap-2 text-sm flex-1 sm:flex-initial whitespace-nowrap"
                >
                  <Upload className="w-4 h-4" />
                  Аудио
                </button>
              </>
            )}

            {isRecording && (
              <>
                <button
                  onClick={isPaused ? resumeRecording : pauseRecording}
                  className="btn-secondary flex items-center justify-center gap-2 flex-1 sm:flex-initial whitespace-nowrap"
                >
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
                  className="btn-primary flex items-center justify-center gap-2 flex-1 sm:flex-initial whitespace-nowrap bg-gradient-to-r from-red-500 to-red-600 shadow-red-500/30 hover:shadow-red-500/40 hover:from-red-600 hover:to-red-700"
                >
                  <Square className="w-5 h-5" />
                  Завершить
                </button>
              </>
            )}

            {!isRecording && audioBlob && (
              <>
                <button
                  onClick={resetRecording}
                  className="btn-secondary flex items-center justify-center gap-2 flex-1 sm:flex-initial whitespace-nowrap"
                >
                  <RotateCcw className="w-5 h-5" />
                  Записать заново
                </button>

                <button
                  onClick={handleSubmit}
                  className="btn-primary flex items-center justify-center gap-2 text-base sm:text-lg px-5 sm:px-8 py-3 sm:py-4 flex-1 sm:flex-initial whitespace-nowrap"
                >
                  <CheckCircle className="w-6 h-6" />
                  Продолжить
                </button>
              </>
            )}
          </div>
        </div>

        <div className="mt-6 sm:mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 slide-up" style={{ animationDelay: '0.2s' }}>
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
    </div>
  );
}
