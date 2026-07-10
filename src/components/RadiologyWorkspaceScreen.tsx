import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Mic, Search, Undo2, Trash2, Copy, Check, LogOut, Settings, Shield, Stethoscope, Lightbulb } from 'lucide-react';
import { apiClient } from '../api/client';
import type { DoctorInfo, RadiologyApplied, RadiologyBlockHint, RadiologyReport, RadiologyTemplateSummary } from '../api/client';

interface RadiologyWorkspaceScreenProps {
  doctor: DoctorInfo;
  onOpenSettings?: () => void;
  onOpenAdmin?: () => void;
  onOpenTherapy?: () => void;
  onLogout?: () => void;
}

// Минимальный тип браузерного распознавания речи (нет в стандартных DOM-типах).
interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((e: { resultIndex: number; results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }> }) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start(): void;
  stop(): void;
}

function getSpeechRecognition(): SpeechRecognitionLike | null {
  const w = window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognitionLike; SpeechRecognition?: new () => SpeechRecognitionLike };
  const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
  return Ctor ? new Ctor() : null;
}

export function RadiologyWorkspaceScreen({ doctor, onOpenSettings, onOpenAdmin, onOpenTherapy, onLogout }: RadiologyWorkspaceScreenProps) {
  const [templates, setTemplates] = useState<RadiologyTemplateSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [commands, setCommands] = useState<string[]>([]);
  const [report, setReport] = useState<RadiologyReport | null>(null);
  const [applied, setApplied] = useState<RadiologyApplied[]>([]);
  const [input, setInput] = useState('');
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState('');
  const [copied, setCopied] = useState(false);
  const [hints, setHints] = useState<RadiologyBlockHint[]>([]);
  const [showHints, setShowHints] = useState(true);
  const recogRef = useRef<SpeechRecognitionLike | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const selected = templates.find((t) => t.id === selectedId) || null;

  useEffect(() => {
    let cancelled = false;
    apiClient.getRadiologyTemplates()
      .then((list) => { if (!cancelled) setTemplates(list); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : 'Не удалось загрузить шаблоны'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  // Пересобираем документ при любом изменении списка команд / выбранного шаблона.
  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    apiClient.buildRadiologyDoc(selectedId, commands)
      .then((res) => { if (!cancelled) { setReport(res.report); setApplied(res.applied); } })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : 'Ошибка сборки протокола'); });
    return () => { cancelled = true; };
  }, [selectedId, commands]);

  // Подсказки «что можно диктовать» для выбранного шаблона.
  useEffect(() => {
    if (!selectedId) { setHints([]); return; }
    let cancelled = false;
    apiClient.getRadiologyHints(selectedId)
      .then((h) => { if (!cancelled) setHints(h); })
      .catch(() => { if (!cancelled) setHints([]); });
    return () => { cancelled = true; };
  }, [selectedId]);

  const useExample = useCallback((ex: string) => {
    // Пример с многоточием («… добавь …») — просто в поле для правки; иначе тоже в поле.
    setInput(ex.replace(/\s*…\s*$/, ' '));
    inputRef.current?.focus();
  }, []);

  const addCommand = useCallback((text: string) => {
    const t = text.trim();
    if (t) setCommands((prev) => [...prev, t]);
  }, []);

  const stopListening = useCallback(() => {
    recogRef.current?.stop();
    recogRef.current = null;
    setListening(false);
    setInterim('');
  }, []);

  const startListening = useCallback(() => {
    const recog = getSpeechRecognition();
    if (!recog) { setError('Голосовой ввод недоступен в этом браузере — используйте поле ввода команды ниже.'); return; }
    recog.lang = 'ru-RU';
    recog.continuous = true;
    recog.interimResults = true;
    recog.onresult = (e) => {
      let finalText = '';
      let interimText = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) finalText += r[0].transcript;
        else interimText += r[0].transcript;
      }
      if (finalText) addCommand(finalText);
      setInterim(interimText);
    };
    recog.onend = () => { setListening(false); setInterim(''); };
    recog.onerror = () => { setListening(false); setInterim(''); };
    recogRef.current = recog;
    recog.start();
    setListening(true);
    setError('');
  }, [addCommand]);

  useEffect(() => () => { recogRef.current?.stop(); }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return templates;
    return templates.filter((t) => (t.name + ' ' + t.title).toLowerCase().includes(q));
  }, [query, templates]);

  const handleCopy = useCallback(() => {
    if (!report) return;
    navigator.clipboard.writeText(report.text).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  }, [report]);

  const resetToSelection = useCallback(() => {
    stopListening();
    setSelectedId(null);
    setCommands([]);
    setReport(null);
    setApplied([]);
    setInput('');
  }, [stopListening]);

  // ─── Верхняя панель ────────────────────────────────────────────────────────
  const TopBar = (
    <div className="flex items-center justify-between px-6 py-3 border-b border-slate-200 bg-white">
      <div className="flex items-center gap-2 text-medical-800 font-semibold">
        <Stethoscope size={18} /> Лучевая диагностика
      </div>
      <div className="flex items-center gap-3 text-sm text-text-muted">
        <span className="hidden sm:inline">{doctor.name}</span>
        {onOpenTherapy && <button onClick={onOpenTherapy} className="hover:text-medical-700">Терапия</button>}
        {onOpenAdmin && <button onClick={onOpenAdmin} className="flex items-center gap-1 hover:text-medical-700"><Shield size={16} /> Админка</button>}
        {onOpenSettings && <button onClick={onOpenSettings} className="flex items-center gap-1 hover:text-medical-700"><Settings size={16} /> Настройки</button>}
        {onLogout && <button onClick={onLogout} className="flex items-center gap-1 hover:text-red-600"><LogOut size={16} /> Выйти</button>}
      </div>
    </div>
  );

  // ─── Экран выбора шаблона ────────────────────────────────────────────────────
  if (!selected) {
    return (
      <div className="min-h-screen bg-medical-50">
        {TopBar}
        <div className="max-w-3xl mx-auto px-6 py-10">
          <h1 className="text-2xl font-bold text-medical-900 mb-1">Выберите шаблон исследования</h1>
          <p className="text-text-muted mb-6">После выбора можно начать голосовую диктовку по шаблону.</p>

          <div className="relative mb-6">
            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Поиск шаблона…"
              className="w-full box-border pl-10 pr-4 py-3 rounded-xl border border-slate-200 bg-white focus:outline-none focus:ring-2 focus:ring-medical-400"
            />
          </div>

          {loading && <p className="text-text-muted">Загрузка шаблонов…</p>}
          {error && <p className="text-red-600 mb-4">{error}</p>}

          <div className="grid gap-3 sm:grid-cols-2">
            {filtered.map((t) => (
              <button
                key={t.id}
                onClick={() => { setSelectedId(t.id); setError(''); }}
                className="text-left p-4 rounded-xl border border-slate-200 bg-white hover:border-medical-400 hover:shadow-sm transition"
              >
                <div className="text-xs font-semibold text-medical-600 mb-1">{t.modality}</div>
                <div className="font-semibold text-medical-900">{t.name}</div>
                <div className="text-sm text-text-muted mt-1 line-clamp-2">{t.title}</div>
              </button>
            ))}
          </div>
          {!loading && filtered.length === 0 && <p className="text-text-muted">Шаблоны не найдены.</p>}
        </div>
      </div>
    );
  }

  // ─── Рабочий экран: запись + живой документ ─────────────────────────────────
  return (
    <div className="min-h-screen bg-medical-50">
      {TopBar}
      <div className="max-w-5xl mx-auto px-6 py-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        {/* Документ */}
        <div className="order-2 lg:order-1">
          <div className="flex items-center justify-between mb-3">
            <button onClick={resetToSelection} className="text-sm text-medical-700 hover:underline">← Сменить шаблон</button>
            <button onClick={handleCopy} className="flex items-center gap-1 text-sm text-medical-700 hover:underline">
              {copied ? <><Check size={15} /> Скопировано</> : <><Copy size={15} /> Копировать</>}
            </button>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="text-center font-semibold text-medical-900 mb-4">{report?.title}</div>
            <div className="space-y-2 text-[15px] leading-relaxed text-medical-900">
              {report?.blocks.map((b) => {
                const sep = b.text.indexOf(': ');
                const label = sep > 0 ? b.text.slice(0, sep) : b.label;
                const body = sep > 0 ? b.text.slice(sep + 2) : b.text;
                const isConclusion = b.id === 'conclusion';
                return (
                  <p key={b.id} className={isConclusion ? 'pt-2 mt-2 border-t border-slate-200' : ''}>
                    <span className="font-semibold">{label}:</span> {body}
                  </p>
                );
              })}
            </div>
          </div>
        </div>

        {/* Панель управления диктовкой */}
        <div className="order-1 lg:order-2">
          <div className="bg-white rounded-xl border border-slate-200 p-5 sticky top-6">
            <div className="text-sm font-semibold text-medical-800 mb-1">{selected.name}</div>
            <p className="text-xs text-text-muted mb-4">Диктуйте только значения и отклонения — норма подставится сама.</p>

            <button
              onClick={listening ? stopListening : startListening}
              className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold transition ${
                listening ? 'bg-red-500 text-white animate-pulse' : 'bg-medical-600 text-white hover:bg-medical-700'
              }`}
            >
              <Mic size={18} /> {listening ? 'Остановить запись' : 'Начать запись'}
            </button>
            {interim && <p className="mt-2 text-sm text-text-muted italic">«{interim}»</p>}

            <div className="mt-4">
              <form
                onSubmit={(e) => { e.preventDefault(); addCommand(input); setInput(''); }}
                className="flex gap-2"
              >
                <input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Команда голосом или текстом…"
                  className="flex-1 min-w-0 box-border px-3 py-2 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-medical-400 text-sm"
                />
                <button type="submit" className="px-3 py-2 rounded-lg bg-medical-100 text-medical-800 font-medium text-sm hover:bg-medical-200">Добавить</button>
              </form>
            </div>

            {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

            <div className="flex items-center justify-between mt-4 text-sm">
              <span className="text-text-muted">Команд: {commands.length}</span>
              <div className="flex gap-3">
                <button onClick={() => setCommands((p) => p.slice(0, -1))} disabled={!commands.length} className="flex items-center gap-1 text-medical-700 disabled:text-slate-300"><Undo2 size={15} /> Отменить</button>
                <button onClick={() => setCommands([])} disabled={!commands.length} className="flex items-center gap-1 text-red-600 disabled:text-slate-300"><Trash2 size={15} /> Очистить</button>
              </div>
            </div>

            {commands.length > 0 && (
              <ul className="mt-3 space-y-1 max-h-48 overflow-auto text-sm">
                {commands.map((c, i) => {
                  const a = applied[i];
                  const unknown = a && !a.ok;
                  return (
                    <li key={i} className={`px-2 py-1 rounded ${unknown ? 'bg-amber-50 text-amber-700' : 'bg-slate-50 text-medical-800'}`}>
                      {c}{unknown ? ' — не распознано' : ''}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          {/* Подсказки: что можно диктовать */}
          <div className="bg-white rounded-xl border border-slate-200 mt-4">
            <button
              onClick={() => setShowHints((v) => !v)}
              className="w-full flex items-center gap-2 px-4 py-3 text-sm font-semibold text-medical-800"
            >
              <Lightbulb size={16} className="text-amber-500" /> Что можно диктовать
              <span className="ml-auto text-text-muted">{showHints ? '▾' : '▸'}</span>
            </button>
            {showHints && (
              <div className="px-4 pb-4 max-h-105 overflow-auto">
                <p className="text-xs text-text-muted mb-3">Диктуйте <b>орган + параметр + число</b>. Норма подставится сама. Нажмите пример, чтобы подставить в поле.</p>
                <div className="space-y-3">
                  {hints.map((h) => (
                    <div key={h.blockId}>
                      <div className="text-xs font-semibold text-medical-700 mb-1">{h.label}</div>
                      <div className="flex flex-wrap gap-1.5">
                        {h.examples.map((ex, i) => (
                          <button
                            key={i}
                            onClick={() => useExample(ex)}
                            className="px-2 py-1 rounded-md bg-medical-50 border border-slate-200 text-xs text-medical-800 hover:border-medical-400"
                          >
                            {ex}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
