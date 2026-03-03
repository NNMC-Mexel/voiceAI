import { useEffect, useState } from 'react';
import { Loader2, Brain, FileText, Sparkles, Upload } from 'lucide-react';

const steps = [
  { icon: Upload, label: 'Загрузка аудио на сервер...' },
  { icon: Brain, label: 'Распознавание речи (Whisper)...' },
  { icon: Sparkles, label: 'ИИ-структурирование (Qwen)...' },
  { icon: FileText, label: 'Формирование документа...' },
];

export function ProcessingScreen() {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % steps.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const CurrentIcon = steps[currentStep]?.icon || Loader2;

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-md text-center">
        <div className="relative mb-8">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-32 h-32 rounded-full bg-medical-100 animate-ping opacity-20" />
          </div>
          <div className="relative flex items-center justify-center">
            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-medical-400 to-medical-600 flex items-center justify-center shadow-lg shadow-medical-500/30">
              <CurrentIcon className="w-10 h-10 text-white animate-spin" />
            </div>
          </div>
        </div>

        <h2 className="text-2xl font-display font-bold text-medical-900 mb-3">{steps[currentStep]?.label}</h2>

        <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden mb-8">
          <div
            className="h-full w-1/3 bg-gradient-to-r from-medical-400 to-medical-600 rounded-full animate-[shimmer_1.5s_ease-in-out_infinite]"
            style={{
              animation: 'shimmer 1.5s ease-in-out infinite',
            }}
          />
        </div>

        <div className="flex items-center justify-center gap-2">
          {steps.map((_, index) => (
            <div
              key={index}
              className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
                index === currentStep ? 'bg-medical-400 scale-125' : 'bg-slate-300'
              }`}
            />
          ))}
        </div>

        <p className="text-text-secondary mt-6">Пожалуйста, подождите. Обработка выполняется локально на сервере.</p>
        <p className="text-text-muted text-sm mt-2">Это может занять несколько минут в зависимости от длины записи.</p>
      </div>

      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
      `}</style>
    </div>
  );
}
