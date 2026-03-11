import { useState, useEffect, useRef } from 'react';
import { X, Check, BookmarkPlus } from 'lucide-react';

interface TermCorrectionPopupProps {
  position: { x: number; y: number };
  selectedText: string;
  onSave: (wrong: string, correct: string, remember: boolean) => void;
  onClose: () => void;
}

export function TermCorrectionPopup({
  position,
  selectedText,
  onSave,
  onClose,
}: TermCorrectionPopupProps) {
  const [correctValue, setCorrectValue] = useState('');
  const [remember, setRemember] = useState(true);
  const popupRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Фокус на поле ввода при открытии
  useEffect(() => {
    requestAnimationFrame(() => {
      inputRef.current?.focus();
    });
  }, []);

  // Закрытие по клику вне попапа
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  // Закрытие по Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleSubmit = () => {
    const trimmed = correctValue.trim();
    if (!trimmed || trimmed === selectedText) return;
    onSave(selectedText, trimmed, remember);
  };

  // Позиционирование: не вылезать за экран
  const popupStyle: React.CSSProperties = {
    position: 'fixed',
    left: Math.min(position.x, window.innerWidth - 320),
    top: Math.min(position.y, window.innerHeight - 260),
    zIndex: 9999,
  };

  return (
    <div ref={popupRef} style={popupStyle} className="w-[300px] bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden animate-in fade-in zoom-in-95 duration-150">
      <div className="flex items-center justify-between px-4 py-2.5 bg-medical-50 border-b border-slate-200">
        <span className="text-sm font-semibold text-medical-800">Исправить термин</span>
        <button onClick={onClose} className="p-0.5 rounded hover:bg-medical-100 transition-colors">
          <X className="w-4 h-4 text-medical-600" />
        </button>
      </div>

      <div className="p-4 space-y-3">
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">Распознано:</label>
          <div className="px-3 py-2 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800 font-medium">
            {selectedText}
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">Правильно:</label>
          <input
            ref={inputRef}
            type="text"
            value={correctValue}
            onChange={(e) => setCorrectValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Введите правильный вариант..."
            className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-medical-400 focus:border-transparent"
          />
        </div>

        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={remember}
            onChange={(e) => setRemember(e.target.checked)}
            className="w-4 h-4 rounded border-slate-300 text-medical-600 focus:ring-medical-400"
          />
          <BookmarkPlus className="w-3.5 h-3.5 text-medical-500" />
          <span className="text-xs text-text-secondary">Запомнить для будущих диктовок</span>
        </label>

        <div className="flex items-center gap-2 pt-1">
          <button
            onClick={handleSubmit}
            disabled={!correctValue.trim() || correctValue.trim() === selectedText}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 bg-medical-600 text-white rounded-lg text-sm font-medium hover:bg-medical-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Check className="w-4 h-4" />
            Сохранить
          </button>
          <button
            onClick={onClose}
            className="px-3 py-2 text-sm text-text-secondary hover:text-text-primary hover:bg-slate-100 rounded-lg transition-colors"
          >
            Отмена
          </button>
        </div>
      </div>
    </div>
  );
}
