import { Loader2, Brain, CheckCircle2, FileText, Sparkles, Upload } from 'lucide-react';

export type ProcessingPhase =
  | 'uploading'
  | 'queued'
  | 'transcribing'
  | 'structuring'
  | 'finalizing'
  | 'document'
  | 'fallback';

const phaseSteps: Array<{
  phase: ProcessingPhase;
  icon: typeof Upload;
  label: string;
  description: string;
}> = [
  {
    phase: 'uploading',
    icon: Upload,
    label: 'Загрузка файла',
    description: 'Передаем запись на сервер обработки.',
  },
  {
    phase: 'queued',
    icon: Loader2,
    label: 'Запись в очереди',
    description: 'Задача создана и скоро начнется обработка.',
  },
  {
    phase: 'transcribing',
    icon: Brain,
    label: 'Распознавание речи',
    description: 'Whisper переводит аудио в медицинский текст.',
  },
  {
    phase: 'structuring',
    icon: Sparkles,
    label: 'Структурирование',
    description: 'LLM распределяет текст по разделам протокола.',
  },
  {
    phase: 'finalizing',
    icon: CheckCircle2,
    label: 'Финальная проверка',
    description: 'Проверяем полноту документа и предупреждения качества.',
  },
  {
    phase: 'document',
    icon: FileText,
    label: 'Обработка документа',
    description: 'Извлекаем текст и формируем медицинский документ.',
  },
  {
    phase: 'fallback',
    icon: Loader2,
    label: 'Резервная обработка',
    description: 'Основной поток не ответил, продолжаем через запасной маршрут.',
  },
];

interface ProcessingScreenProps {
  phase?: ProcessingPhase;
  detail?: string;
}

export function ProcessingScreen({ phase = 'uploading', detail }: ProcessingScreenProps) {
  const phaseIndex = phaseSteps.findIndex((step) => step.phase === phase);
  const safePhaseIndex = phaseIndex >= 0 ? phaseIndex : 0;
  const currentStep = phaseSteps[safePhaseIndex] || phaseSteps[0];
  const CurrentIcon = currentStep.icon || Loader2;
  const progress = Math.round(((safePhaseIndex + 1) / phaseSteps.length) * 100);
  const isSpinner = currentStep.phase === 'queued' || currentStep.phase === 'fallback';

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-md text-center">
        <div className="relative mb-8">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-32 h-32 rounded-full bg-medical-100 animate-ping opacity-20" />
          </div>
          <div className="relative flex items-center justify-center">
            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-medical-400 to-medical-600 flex items-center justify-center shadow-lg shadow-medical-500/30">
              <CurrentIcon className={`w-10 h-10 text-white ${isSpinner ? 'animate-spin' : ''}`} />
            </div>
          </div>
        </div>

        <h2 className="text-2xl font-display font-bold text-medical-900 mb-3">{currentStep.label}</h2>
        <p className="min-h-[48px] text-text-secondary mb-6">{detail || currentStep.description}</p>

        <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden mb-8">
          <div
            className="h-full bg-gradient-to-r from-medical-400 to-medical-600 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="flex items-center justify-center gap-2">
          {phaseSteps.map((step, index) => (
            <div
              key={step.phase}
              className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
                index <= safePhaseIndex ? 'bg-medical-400 scale-110' : 'bg-slate-300'
              }`}
            />
          ))}
        </div>

        <p className="text-text-secondary mt-6">Пожалуйста, подождите. Обработка выполняется локально на сервере.</p>
        <p className="text-text-muted text-sm mt-2">Это может занять несколько минут в зависимости от длины записи.</p>
      </div>
    </div>
  );
}
