/**
 * SyncUploadScreen — мобильный экран для отправки документов на десктоп.
 * Врач фотографирует или выбирает документ → он обрабатывается на сервере →
 * появляется в очереди на десктопе.
 */

import { useState, useRef } from 'react';
import { Camera, FileText, CheckCircle2, ArrowLeft, Loader2, AlertCircle } from 'lucide-react';
import { apiClient } from '../api/client';

interface SyncUploadScreenProps {
  onBack: () => void;
}

type UploadState = 'idle' | 'uploading' | 'processing' | 'done' | 'error';

export function SyncUploadScreen({ onBack }: SyncUploadScreenProps) {
  const [state, setState]   = useState<UploadState>('idle');
  const [error, setError]   = useState('');
  const [filename, setFilename] = useState('');
  const [syncId, setSyncId] = useState('');

  const photoRef = useRef<HTMLInputElement>(null);
  const fileRef  = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setFilename(file.name);
    setState('uploading');
    setError('');

    try {
      const res = await apiClient.syncUpload(file);
      setSyncId(res.syncId);
      setState('processing');

      // Опрашиваем статус пока не готово
      await pollStatus(res.syncId);
      setState('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка загрузки');
      setState('error');
    }
  };

  const pollStatus = async (id: string): Promise<void> => {
    for (let i = 0; i < 60; i++) {
      await new Promise(r => setTimeout(r, 3000));
      try {
        const res = await apiClient.syncStatus(id);
        if (res.session.status === 'ready') return;
        if (res.session.status === 'error') {
          throw new Error(res.session.errorMessage || 'Ошибка обработки');
        }
      } catch (err) {
        if (err instanceof Error && err.message !== 'Ошибка обработки') continue;
        throw err;
      }
    }
    throw new Error('Превышено время ожидания обработки');
  };

  const handleCancel = async () => {
    if (syncId) {
      apiClient.syncDelete(syncId).catch(() => {});
    }
    setState('idle');
    setSyncId('');
    setFilename('');
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-50 to-slate-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-4 py-3 flex items-center gap-3">
        <button onClick={onBack} className="p-2 hover:bg-slate-100 rounded-xl">
          <ArrowLeft className="w-5 h-5 text-slate-600" />
        </button>
        <div>
          <h1 className="font-semibold text-medical-900">Отправить на компьютер</h1>
          <p className="text-xs text-text-muted">Документ появится в очереди на десктопе</p>
        </div>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center px-6 py-8 gap-6">
        {state === 'idle' && (
          <>
            <div className="text-center mb-2">
              <p className="text-text-secondary text-sm">
                Сфотографируйте анализы или выберите PDF/Word
              </p>
            </div>

            {/* Кнопка камеры — большая, для удобства на телефоне */}
            <button
              onClick={() => photoRef.current?.click()}
              className="w-full max-w-xs bg-medical-600 hover:bg-medical-700 text-white rounded-2xl p-8 flex flex-col items-center gap-3 shadow-lg shadow-medical-500/30 active:scale-95 transition-all"
            >
              <Camera className="w-14 h-14" />
              <span className="text-lg font-semibold">Сфотографировать</span>
              <span className="text-xs opacity-75">JPG, PNG, HEIC</span>
            </button>

            <button
              onClick={() => fileRef.current?.click()}
              className="w-full max-w-xs bg-white border-2 border-medical-200 hover:border-medical-400 text-medical-700 rounded-2xl p-5 flex flex-col items-center gap-2 active:scale-95 transition-all"
            >
              <FileText className="w-8 h-8" />
              <span className="font-medium">Выбрать файл</span>
              <span className="text-xs text-text-muted">PDF, Word (.docx)</span>
            </button>

            <input
              ref={photoRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={e => { const f = e.target.files?.[0]; if (f) void handleFile(f); e.target.value = ''; }}
              className="hidden"
            />
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.docx,.doc,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image/*"
              onChange={e => { const f = e.target.files?.[0]; if (f) void handleFile(f); e.target.value = ''; }}
              className="hidden"
            />
          </>
        )}

        {(state === 'uploading' || state === 'processing') && (
          <div className="text-center space-y-4">
            <div className="w-20 h-20 bg-medical-100 rounded-full flex items-center justify-center mx-auto">
              <Loader2 className="w-10 h-10 text-medical-600 animate-spin" />
            </div>
            <div>
              <p className="font-semibold text-medical-900">
                {state === 'uploading' ? 'Загрузка...' : 'Обработка документа...'}
              </p>
              {filename && <p className="text-sm text-text-muted mt-1">{filename}</p>}
              <p className="text-xs text-text-muted mt-2">
                {state === 'processing' ? 'ИИ анализирует документ, подождите' : 'Отправка на сервер'}
              </p>
            </div>
            <button onClick={handleCancel} className="text-sm text-text-muted hover:text-red-500">
              Отменить
            </button>
          </div>
        )}

        {state === 'done' && (
          <div className="text-center space-y-4">
            <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto">
              <CheckCircle2 className="w-10 h-10 text-green-600" />
            </div>
            <div>
              <p className="font-semibold text-medical-900 text-lg">Готово!</p>
              {filename && <p className="text-sm text-text-muted mt-1">{filename}</p>}
              <p className="text-text-secondary mt-2">
                Документ обработан и ожидает на компьютере
              </p>
              <p className="text-xs text-text-muted mt-1">
                Откройте МедДок на десктопе — в правом углу появится уведомление
              </p>
            </div>
            <button
              onClick={() => { setState('idle'); setSyncId(''); setFilename(''); }}
              className="btn-primary py-2.5 px-6 text-sm"
            >
              Отправить ещё
            </button>
          </div>
        )}

        {state === 'error' && (
          <div className="text-center space-y-4">
            <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto">
              <AlertCircle className="w-10 h-10 text-red-500" />
            </div>
            <div>
              <p className="font-semibold text-red-700">Ошибка</p>
              <p className="text-sm text-text-muted mt-1">{error}</p>
            </div>
            <button
              onClick={() => { setState('idle'); setError(''); setFilename(''); }}
              className="btn-secondary py-2.5 px-6 text-sm"
            >
              Попробовать снова
            </button>
          </div>
        )}
      </div>

      {/* Инструкция внизу */}
      {state === 'idle' && (
        <div className="px-6 pb-8">
          <div className="bg-white/70 rounded-xl p-4 border border-slate-200">
            <p className="text-xs text-text-secondary text-center">
              Как это работает: документ обрабатывается ИИ и появляется в очереди на вашем компьютере. Откройте МедДок на десктопе и нажмите на значок уведомления.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
