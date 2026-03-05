import {
  User,
  Bot,
  MessageSquare,
  History,
  Stethoscope,
  ClipboardList,
  Activity,
  FileCheck,
  ListTodo,
  StickyNote,
  Eye,
  ArrowLeft,
  Mic,
  Pause,
  Play,
  Square,
  CheckCircle,
  PlusCircle,
  AlertCircle,
  Loader2,
  SendHorizontal,
  Sparkles,
  Brain,
  ShieldAlert,
  Volume2,
  VolumeX,
} from 'lucide-react';
import type { MedicalDocument, PatientInfo } from '../types';
import { fieldLabels, patientFieldLabels } from '../types';
import { CollapsibleSection } from './CollapsibleSection';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';
import { WaveformVisualizer } from './WaveformVisualizer';
import { apiClient } from '../api/client';
import { useState, useEffect, useRef } from 'react';

interface EditingScreenProps {
  document: MedicalDocument;
  onDocumentChange: (document: MedicalDocument) => void;
  onPreview: () => void;
  onBack: () => void;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  kind?: 'recommendations' | 'chat';
}

type RewriteableField = keyof Omit<MedicalDocument, 'patient'>;

function isDocumentEditInstruction(text: string): boolean {
  const normalized = text.toLowerCase();
  const editVerbs = [
    'добав',
    'встав',
    'внес',
    'допол',
    'исправ',
    'перепиш',
    'отредакт',
    'обнов',
    'замен',
    'удал',
    'убер',
  ];
  const hasVerb = editVerbs.some((v) => normalized.includes(v));
  if (!hasVerb) return false;

  const sectionHints = [
    'жалоб',
    'анамн',
    'объектив',
    'диагноз',
    'аллерг',
    'неврол',
    'сопутствующ',
    'план лечен',
    'рекомендац',
    'заметк',
    'примечан',
    'прочее',
    'пациент',
  ];
  return sectionHints.some((h) => normalized.includes(h));
}

function filenameForBlob(blob: Blob, baseName: string): string {
  const type = blob.type.toLowerCase();
  if (type.includes('mp4')) return `${baseName}.mp4`;
  if (type.includes('ogg')) return `${baseName}.ogg`;
  if (type.includes('wav')) return `${baseName}.wav`;
  return `${baseName}.webm`;
}

const sectionIcons: Record<keyof Omit<MedicalDocument, 'patient'>, React.ReactNode> = {
  complaints: <MessageSquare className="w-4 h-4" />,
  anamnesis: <History className="w-4 h-4" />,
  clinicalCourse: <Activity className="w-4 h-4" />,
  allergyHistory: <ShieldAlert className="w-4 h-4" />,
  objectiveStatus: <Stethoscope className="w-4 h-4" />,
  neurologicalStatus: <Brain className="w-4 h-4" />,
  diagnosis: <ClipboardList className="w-4 h-4" />,
  conclusion: <FileCheck className="w-4 h-4" />,
  recommendations: <ListTodo className="w-4 h-4" />,
  doctorNotes: <StickyNote className="w-4 h-4" />,
};

export function EditingScreen({
  document,
  onDocumentChange,
  onPreview,
  onBack,
}: EditingScreenProps) {
  const {
    isRecording: isAddendumRecording,
    isPaused: isAddendumPaused,
    audioBlob: addendumBlob,
    formattedDuration: addendumDuration,
    startRecording: startAddendumRecording,
    pauseRecording: pauseAddendumRecording,
    resumeRecording: resumeAddendumRecording,
    stopRecording: stopAddendumRecording,
    resetRecording: resetAddendumRecording,
  } = useVoiceRecorder();
  const {
    isRecording: isCommandRecording,
    audioBlob: commandBlob,
    formattedDuration: commandDuration,
    startRecording: startCommandRecording,
    stopRecording: stopCommandRecording,
    resetRecording: resetCommandRecording,
  } = useVoiceRecorder();

  const [isAddendumOpen, setIsAddendumOpen] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [addendumError, setAddendumError] = useState<string | null>(null);
  const [lastAddendumText, setLastAddendumText] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [voiceCommandError, setVoiceCommandError] = useState<string | null>(null);
  const [isVoiceCommandProcessing, setIsVoiceCommandProcessing] = useState(false);
  const [rewriteLoadingField, setRewriteLoadingField] = useState<RewriteableField | null>(null);
  const [isRecommendationsLoading, setIsRecommendationsLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [isTtsSpeaking, setIsTtsSpeaking] = useState(false);
  const recommendationsInFlightRef = useRef(false);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  const playTts = async (text: string) => {
    try {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
      const audioBase64 = await apiClient.tts(text);
      const bytes = Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      currentAudioRef.current = audio;
      setIsTtsSpeaking(true);
      audio.onended = () => {
        setIsTtsSpeaking(false);
        URL.revokeObjectURL(url);
        currentAudioRef.current = null;
      };
      audio.onerror = () => {
        setIsTtsSpeaking(false);
        URL.revokeObjectURL(url);
        currentAudioRef.current = null;
      };
      await audio.play();
    } catch {
      setIsTtsSpeaking(false);
    }
  };

  const handlePatientChange = (field: keyof PatientInfo, value: string) => {
    onDocumentChange({
      ...document,
      patient: {
        ...document.patient,
        [field]: value,
      },
    });
  };

  const handleFieldChange = (
    field: keyof Omit<MedicalDocument, 'patient'>,
    value: string
  ) => {
    onDocumentChange({
      ...document,
      [field]: value,
    });
  };

  const refreshRecommendationsInChat = async () => {
    if (recommendationsInFlightRef.current) return;

    recommendationsInFlightRef.current = true;
    setIsRecommendationsLoading(true);

    try {
      const result = await apiClient.getRecommendations(document);
      const recommendationText =
        result.recommendations?.trim() || 'Не удалось получить рекомендации ИИ.';
      setChatMessages((prev) => {
        const withoutRecommendations = prev.filter((m) => m.kind !== 'recommendations');
        return [
          {
            role: 'assistant',
            kind: 'recommendations',
            text: `Рекомендации по текущему документу:\n${recommendationText}`,
          },
          ...withoutRecommendations,
        ];
      });
    } catch {
      setChatMessages((prev) => {
        const withoutRecommendations = prev.filter((m) => m.kind !== 'recommendations');
        return [
          {
            role: 'assistant',
            kind: 'recommendations',
            text: 'Не удалось получить рекомендации ИИ.',
          },
          ...withoutRecommendations,
        ];
      });
    } finally {
      recommendationsInFlightRef.current = false;
      setIsRecommendationsLoading(false);
    }
  };

  const handleAddendumApply = async () => {
    if (!addendumBlob) return;

    setIsUpdating(true);
    setAddendumError(null);

    try {
      const result = await apiClient.processAddendum(
        addendumBlob,
        document,
        filenameForBlob(addendumBlob, 'addendum')
      );

      onDocumentChange(result.document);
      setLastAddendumText(result.transcription.text);
      await refreshRecommendationsInChat();

      resetAddendumRecording();
      setIsAddendumOpen(false);
    } catch (err) {
      setAddendumError(err instanceof Error ? err.message : 'Ошибка дополнения документа');
    } finally {
      setIsUpdating(false);
    }
  };

  const submitQuestion = async (question: string) => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || chatLoading || rewriteLoadingField !== null || isVoiceCommandProcessing) return;

    const nextHistory: ChatMessage[] = [...chatMessages, { role: 'user', text: trimmedQuestion }];
    setChatMessages(nextHistory);

    const normalized = trimmedQuestion.toLowerCase();
    const isRewriteComplaintsCommand = /(перепиши|исправь|отредактируй).*(жалоб|жалобы|жалоба)/.test(normalized);
    if (isRewriteComplaintsCommand) {
      await handleRewriteField('complaints', true);
      return;
    }

    if (isDocumentEditInstruction(trimmedQuestion)) {
      setChatLoading(true);
      try {
        const result = await apiClient.instructDocument(document, trimmedQuestion);
        onDocumentChange(result.document);
        await refreshRecommendationsInChat();

        const changed = result.changedFields.map((f) => fieldLabels[f]).join(', ');
        const where = [changed, result.patientChanged ? 'Данные пациента' : '']
          .filter(Boolean)
          .join(', ');
        setChatMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            kind: 'chat',
            text: where
              ? `Внес изменения в разделы: ${where}.`
              : 'Команду понял, но изменений в документе не потребовалось.',
          },
        ]);
      } catch (err) {
        setChatMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            kind: 'chat',
            text:
              err instanceof Error
                ? `Не удалось применить изменение: ${err.message}`
                : 'Не удалось применить изменение.',
          },
        ]);
      } finally {
        setChatLoading(false);
      }
      return;
    }

    setChatLoading(true);
    try {
      const history = nextHistory
        .slice(-8)
        .map((m) => ({ role: m.role, text: m.text }));
      const result = await apiClient.chat(trimmedQuestion, history, document);
      const answerText = result.answer || 'Нет ответа';
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', kind: 'chat', text: answerText },
      ]);
      if (ttsEnabled) {
        void playTts(answerText);
      }
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', kind: 'chat', text: 'Ошибка запроса к чату.' },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleChatSend = async () => {
    const question = chatInput.trim();
    if (!question) return;
    setChatInput('');
    await submitQuestion(question);
  };

  const handleVoiceCommandToggle = async () => {
    if (chatLoading || isVoiceCommandProcessing || rewriteLoadingField !== null) return;
    setVoiceCommandError(null);

    if (isCommandRecording) {
      stopCommandRecording();
      return;
    }

    if (commandBlob) {
      resetCommandRecording();
    }

    try {
      await startCommandRecording();
    } catch (err) {
      setVoiceCommandError(err instanceof Error ? err.message : 'Не удалось начать запись голосовой команды');
    }
  };

  const handleMainDictationToggle = async () => {
    if (isUpdating) return;

    if (isAddendumRecording) {
      if (isAddendumPaused) {
        resumeAddendumRecording();
      } else {
        pauseAddendumRecording();
      }
      return;
    }

    if (addendumBlob) return;

    if (!isAddendumOpen) {
      setIsAddendumOpen(true);
    }

    setAddendumError(null);
    try {
      await startAddendumRecording();
    } catch (err) {
      setAddendumError(
        err instanceof Error ? err.message : 'Не удалось начать запись дополнения'
      );
    }
  };

  const handleRewriteField = async (field: RewriteableField, announceInChat = false) => {
    const sourceText = document[field].trim();
    if (!sourceText || rewriteLoadingField !== null) return;

    setRewriteLoadingField(field);
    try {
      const result = await apiClient.rewriteField(field, sourceText);
      const rewritten = result.text?.trim() || sourceText;
      handleFieldChange(field, rewritten);

      if (announceInChat) {
        setChatMessages((prev) => [
          ...prev,
          { role: 'assistant', kind: 'chat', text: `Блок "${fieldLabels[field]}" переписан.` },
        ]);
      }
    } catch (err) {
      if (announceInChat) {
        setChatMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            kind: 'chat',
            text: err instanceof Error ? `Не удалось переписать блок: ${err.message}` : 'Не удалось переписать блок.',
          },
        ]);
      }
    } finally {
      setRewriteLoadingField(null);
    }
  };

  useEffect(() => {
    void refreshRecommendationsInChat();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!commandBlob || isVoiceCommandProcessing) return;

    const processVoiceCommand = async () => {
      setIsVoiceCommandProcessing(true);
      setVoiceCommandError(null);

      try {
        const upload = await apiClient.uploadAudio(commandBlob, filenameForBlob(commandBlob, 'voice-command'));
        const transcription = await apiClient.transcribe(upload.filename);
        const spokenCommand = transcription.text.trim();

        if (!spokenCommand) {
          setVoiceCommandError('Не удалось распознать команду');
          return;
        }

        await submitQuestion(spokenCommand);
      } catch (err) {
        setVoiceCommandError(
          err instanceof Error ? err.message : 'Не удалось обработать голосовую команду'
        );
      } finally {
        resetCommandRecording();
        setIsVoiceCommandProcessing(false);
      }
    };

    void processVoiceCommand();
  }, [commandBlob, isVoiceCommandProcessing, resetCommandRecording]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      const key = event.key.toLowerCase();

      const target = event.target as HTMLElement | null;
      const isEditableTarget =
        !!target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable);
      if (isEditableTarget) return;

      if (key === 'm' || key === 'ь') {
        event.preventDefault();
        void handleMainDictationToggle();
        return;
      }

      if (key === 't' || key === 'т') {
        event.preventDefault();
        void handleVoiceCommandToggle();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    chatLoading,
    commandBlob,
    handleMainDictationToggle,
    handleVoiceCommandToggle,
    isAddendumOpen,
    isAddendumPaused,
    isAddendumRecording,
    isCommandRecording,
    isUpdating,
    isVoiceCommandProcessing,
    pauseAddendumRecording,
    resumeAddendumRecording,
    rewriteLoadingField,
    startAddendumRecording,
  ]);

  return (
    <div className="min-h-screen py-6 px-4">
      <div className="max-w-[1600px] mx-auto">
        <div className="flex items-center justify-between mb-8 slide-up">
          <div>
            <button
              onClick={onBack}
              className="flex items-center gap-2 text-text-secondary hover:text-medical-600 transition-colors mb-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm">Назад к записи</span>
            </button>
            <h1 className="text-3xl font-display font-bold text-medical-900">Редактирование протокола</h1>
            <p className="text-text-secondary mt-1">Проверьте и отредактируйте распознанные данные</p>
          </div>

          <button onClick={onPreview} className="btn-primary flex items-center gap-2">
            <Eye className="w-5 h-5" />
            Предпросмотр PDF
          </button>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[minmax(420px,1fr)_minmax(0,2fr)] gap-6">
          <div className="glass-card rounded-2xl p-4 lg:sticky lg:top-4 self-start slide-up lg:max-h-[calc(100vh-180px)] flex flex-col">
            <div className="flex items-center gap-2 mb-3">
              <Bot className="w-4 h-4 text-medical-700" />
              <h2 className="text-sm font-semibold text-medical-900">Чат с ИИ</h2>
              <button
                onClick={() => setTtsEnabled((v) => !v)}
                title={ttsEnabled ? 'Выключить голос ИИ' : 'Включить голос ИИ'}
                className={`ml-auto flex items-center gap-1 px-2 py-1 rounded-lg text-xs transition-colors ${
                  ttsEnabled
                    ? 'bg-medical-100 text-medical-700 hover:bg-medical-200'
                    : 'text-text-muted hover:text-medical-600 hover:bg-slate-100'
                }`}
              >
                {ttsEnabled ? (
                  <Volume2 className={`w-3.5 h-3.5 ${isTtsSpeaking ? 'animate-pulse' : ''}`} />
                ) : (
                  <VolumeX className="w-3.5 h-3.5" />
                )}
                Голос
              </button>
            </div>
            <div className="mb-3">
              <button
                onClick={() => void handleVoiceCommandToggle()}
                disabled={chatLoading || isVoiceCommandProcessing || rewriteLoadingField !== null}
                className="btn-secondary w-full flex items-center justify-center gap-2 py-2"
              >
                {isCommandRecording ? (
                  <>
                    <Square className="w-4 h-4" />
                    Остановить запись ({commandDuration})
                  </>
                ) : isVoiceCommandProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Обрабатываем голосовую команду...
                  </>
                ) : (
                  <>
                    <Mic className="w-4 h-4" />
                    Голосовая команда (T)
                  </>
                )}
              </button>
              {voiceCommandError && (
                <p className="text-xs text-red-600 mt-2">{voiceCommandError}</p>
              )}
            </div>
            <div className="flex-1 overflow-y-auto bg-slate-50 rounded-xl p-3 space-y-3 mb-3 min-h-[260px]">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`text-sm p-2 rounded-lg ${
                    msg.role === 'user'
                      ? 'bg-medical-600 text-white ml-6'
                      : 'bg-white border border-slate-200 mr-6'
                  } break-words whitespace-pre-wrap`}
                >
                  {msg.text}
                </div>
              ))}
              {isRecommendationsLoading && (
                <p className="text-xs text-text-muted">Обновляем рекомендации...</p>
              )}
              {chatLoading && <p className="text-xs text-text-muted">ИИ думает...</p>}
            </div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    void handleChatSend();
                  }
                }}
                className="input-field py-2"
                placeholder="Спросить ИИ..."
              />
              <button onClick={handleChatSend} className="btn-primary px-3 py-2">
                <SendHorizontal className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="space-y-6">
            <div className="mb-6 slide-up" style={{ animationDelay: '0.08s' }}>
              <div className="glass-card rounded-2xl p-6">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <div className="inline-flex items-center gap-2 px-3 py-1 bg-medical-100 text-medical-700 rounded-full text-xs font-medium mb-2">
                      <PlusCircle className="w-4 h-4" />
                      Дополнение
                    </div>
                    <h2 className="text-lg font-semibold text-medical-900">Если забыли важную информацию — добавьте голосом</h2>
                    <p className="text-text-secondary text-sm mt-1">Дополнение будет расшифровано и встроено в текущий документ.</p>
                  </div>
                  <button
                    onClick={() => setIsAddendumOpen((v) => !v)}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <Mic className="w-4 h-4" />
                    {isAddendumOpen ? 'Скрыть' : 'Записать дополнение'}
                  </button>
                </div>

                {isAddendumOpen && (
                  <div className="mt-6">
                    {addendumError && (
                      <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-4">
                        <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                        <p className="text-red-700 text-sm">{addendumError}</p>
                      </div>
                    )}

                    <div className="bg-slate-50 rounded-2xl p-4 mb-4">
                      <WaveformVisualizer isActive={isAddendumRecording} isPaused={isAddendumPaused} />
                      <div className="mt-3 text-center">
                        <div
                          className={`text-2xl font-mono font-bold ${
                            isAddendumRecording
                              ? isAddendumPaused
                                ? 'text-amber-500'
                                : 'text-medical-600'
                              : 'text-text-muted'
                          }`}
                        >
                          {addendumDuration}
                        </div>
                        <p className="text-text-muted text-sm mt-1">
                          {isAddendumRecording
                            ? isAddendumPaused
                              ? 'Пауза'
                              : 'Запись дополнения...'
                            : addendumBlob
                            ? 'Запись готова'
                            : 'Короткое дополнение'}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center justify-center gap-3">
                      {!isAddendumRecording && !addendumBlob && (
                        <button
                          onClick={startAddendumRecording}
                          className="btn-primary flex items-center gap-2"
                        >
                          <Mic className="w-5 h-5" />
                          Начать
                        </button>
                      )}

                      {isAddendumRecording && (
                        <>
                          <button
                            onClick={isAddendumPaused ? resumeAddendumRecording : pauseAddendumRecording}
                            className="btn-secondary flex items-center gap-2"
                          >
                            {isAddendumPaused ? (
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
                            onClick={stopAddendumRecording}
                            className="btn-primary flex items-center gap-2 bg-gradient-to-r from-red-500 to-red-600 shadow-red-500/30 hover:shadow-red-500/40 hover:from-red-600 hover:to-red-700"
                          >
                            <Square className="w-5 h-5" />
                            Завершить
                          </button>
                        </>
                      )}

                      {!isAddendumRecording && addendumBlob && (
                        <>
                          <button
                            onClick={resetAddendumRecording}
                            className="btn-secondary flex items-center gap-2"
                            disabled={isUpdating}
                          >
                            Сбросить
                          </button>
                          <button
                            onClick={handleAddendumApply}
                            className="btn-primary flex items-center gap-2"
                            disabled={isUpdating}
                          >
                            {isUpdating ? (
                              <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Дополняем...
                              </>
                            ) : (
                              <>
                                <CheckCircle className="w-5 h-5" />
                                Встроить в документ
                              </>
                            )}
                          </button>
                        </>
                      )}
                    </div>

                    {lastAddendumText && (
                      <div className="mt-4 text-sm text-text-secondary">
                        <span className="font-medium text-medical-800">Последнее дополнение:</span>{' '}
                        {lastAddendumText}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="mb-6 slide-up" style={{ animationDelay: '0.1s' }}>
              <CollapsibleSection title="Данные пациента" icon={<User className="w-4 h-4" />} defaultOpen={true}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-text-secondary mb-2">{patientFieldLabels.fullName}</label>
                    <input
                      type="text"
                      value={document.patient.fullName}
                      onChange={(e) => handlePatientChange('fullName', e.target.value)}
                      placeholder="Иванов Иван Иванович"
                      className="input-field"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-text-secondary mb-2">{patientFieldLabels.age}</label>
                      <input
                        type="text"
                        value={document.patient.age}
                        onChange={(e) => handlePatientChange('age', e.target.value)}
                        placeholder="45 лет"
                        className="input-field"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-text-secondary mb-2">{patientFieldLabels.gender}</label>
                      <select
                        value={document.patient.gender}
                        onChange={(e) => handlePatientChange('gender', e.target.value)}
                        className="input-field"
                      >
                        <option value="">Выберите</option>
                        <option value="мужской">Мужской</option>
                        <option value="женский">Женский</option>
                      </select>
                    </div>
                  </div>

                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-text-secondary mb-2">{patientFieldLabels.complaintDate}</label>
                    <input
                      type="date"
                      value={document.patient.complaintDate}
                      onChange={(e) => handlePatientChange('complaintDate', e.target.value)}
                      className="input-field max-w-xs"
                    />
                  </div>
                </div>
              </CollapsibleSection>
            </div>

            <div className="space-y-4">
              {(Object.keys(fieldLabels) as Array<keyof typeof fieldLabels>).map((field, index) => (
                <div key={field} className="slide-up" style={{ animationDelay: `${0.15 + index * 0.05}s` }}>
                  <CollapsibleSection title={fieldLabels[field]} icon={sectionIcons[field]} defaultOpen={index < 4}>
                    {field === 'complaints' && (
                      <div className="mb-3 flex justify-end">
                        <button
                          onClick={() => void handleRewriteField(field)}
                          disabled={rewriteLoadingField !== null || !document[field].trim()}
                          className="btn-secondary flex items-center gap-2 text-sm py-2 px-3"
                        >
                          {rewriteLoadingField === field ? (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
                              Исправляем...
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-4 h-4" />
                              Исправить ИИ
                            </>
                          )}
                        </button>
                      </div>
                    )}
                    <textarea
                      value={document[field]}
                      onChange={(e) => handleFieldChange(field, e.target.value)}
                      placeholder={`Введите ${fieldLabels[field].toLowerCase()}...`}
                      className="textarea-field"
                      rows={field === 'recommendations' || field === 'anamnesis' ? 4 : 3}
                    />
                  </CollapsibleSection>
                </div>
              ))}
            </div>

            <div className="mt-8 flex justify-end slide-up" style={{ animationDelay: '0.5s' }}>
              <button onClick={onPreview} className="btn-primary flex items-center gap-2 text-lg px-8 py-4">
                <Eye className="w-6 h-6" />
                Предпросмотр и печать
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
