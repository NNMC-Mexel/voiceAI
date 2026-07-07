import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Clipboard, FileAudio, FileText, Loader2, Search, Wand2 } from 'lucide-react';
import { apiClient } from '../api/client';
import type { DoctorInfo, ProtocolTemplateInfo, SpecialtyInfo } from '../api/client';

interface ProtocolWorkspaceScreenProps {
  doctor: DoctorInfo;
  onBack: () => void;
}

export function ProtocolWorkspaceScreen({ doctor, onBack }: ProtocolWorkspaceScreenProps) {
  const [specialties, setSpecialties] = useState<SpecialtyInfo[]>([]);
  const [templates, setTemplates] = useState<ProtocolTemplateInfo[]>([]);
  const [selectedSpecialtyId, setSelectedSpecialtyId] = useState<number | ''>('');
  const [selectedTemplateId, setSelectedTemplateId] = useState<number | null>(null);
  const [query, setQuery] = useState('');
  const [dictation, setDictation] = useState('');
  const [filledText, setFilledText] = useState('');
  const [loading, setLoading] = useState(true);
  const [processingAudio, setProcessingAudio] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const audioInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError('');
      try {
        const [specialtyRes, templateRes] = await Promise.all([
          apiClient.getSpecialties(),
          apiClient.getProtocolTemplates(),
        ]);
        if (cancelled) return;
        setSpecialties(specialtyRes.specialties);
        setTemplates(templateRes.templates);
        if (doctor.role !== 'admin') {
          setSelectedSpecialtyId(doctor.departmentId || '');
        } else {
          const matched = doctor.departmentId
            ? specialtyRes.specialties.find((item) => item.id === doctor.departmentId)
            : specialtyRes.specialties.find((item) => /лучев|узи|радиолог/i.test(item.name));
          if (matched) setSelectedSpecialtyId(matched.id);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Не удалось загрузить шаблоны');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => { cancelled = true; };
  }, [doctor.departmentId, doctor.role]);

  const filteredTemplates = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return templates.filter((template) => {
      if (selectedSpecialtyId && template.specialtyId !== selectedSpecialtyId) return false;
      if (!needle) return true;
      return [
        template.name,
        template.modality,
        template.bodyPart,
        template.sourceFilename,
        ...template.aliases,
      ].some((value) => value.toLowerCase().includes(needle));
    });
  }, [query, selectedSpecialtyId, templates]);

  const selectedTemplate = templates.find((template) => template.id === selectedTemplateId) || filteredTemplates[0] || null;

  useEffect(() => {
    if (!selectedTemplate && selectedTemplateId !== null) setSelectedTemplateId(null);
    if (selectedTemplate && selectedTemplateId === null) setSelectedTemplateId(selectedTemplate.id);
  }, [selectedTemplate, selectedTemplateId]);

  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setProcessingAudio(true);
    setError('');
    setMessage('');
    try {
      const upload = await apiClient.uploadAudio(file, file.name);
      const transcription = await apiClient.transcribe(upload.filename);
      setDictation((prev) => [prev.trim(), transcription.text.trim()].filter(Boolean).join('\n\n'));
      setMessage('Аудио распознано и добавлено в диктовку');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось распознать аудио');
    } finally {
      setProcessingAudio(false);
      if (audioInputRef.current) audioInputRef.current.value = '';
    }
  };

  const generateProtocol = async () => {
    if (!selectedTemplate) {
      setError('Выберите шаблон');
      return;
    }
    if (!dictation.trim()) {
      setError('Введите диктовку или загрузите аудио');
      return;
    }
    setGenerating(true);
    setError('');
    setMessage('');
    try {
      const result = await apiClient.fillProtocolTemplate(selectedTemplate.id, dictation);
      setFilledText(result.filledText);
      setMessage('Протокол заполнен');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось заполнить протокол');
    } finally {
      setGenerating(false);
    }
  };

  const copyProtocol = async () => {
    if (!filledText.trim()) return;
    await navigator.clipboard.writeText(filledText);
    setMessage('Текст протокола скопирован');
  };

  const downloadProtocol = () => {
    if (!filledText.trim()) return;
    const blob = new Blob([filledText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedTemplate?.name || 'protocol'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-medical-50">
      <div className="bg-white border-b border-slate-200 px-4 py-4 sm:px-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
          <button onClick={onBack} className="btn-secondary flex items-center gap-2 py-2 px-3 text-sm">
            <ArrowLeft className="w-4 h-4" />
            Назад
          </button>
          <div className="text-right min-w-0">
            <h1 className="font-display font-bold text-medical-900 text-lg leading-tight">Протоколы исследований</h1>
            <p className="text-xs text-text-muted truncate">{doctor.name}</p>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 py-5 sm:px-6">
        {(error || message) && (
          <div className={`mb-4 rounded-xl border px-4 py-3 text-sm ${
            error ? 'border-red-200 bg-red-50 text-red-700' : 'border-medical-200 bg-medical-50 text-medical-700'
          }`}>
            {error || message}
          </div>
        )}

        {loading ? (
          <div className="glass-card rounded-2xl p-10 text-center text-text-muted">Загрузка...</div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-5">
            <aside className="glass-card rounded-2xl p-4 space-y-4">
              <div>
                <label className="text-xs font-semibold text-text-muted uppercase">Специальность</label>
                <select
                  value={selectedSpecialtyId}
                  onChange={(e) => {
                    setSelectedSpecialtyId(e.target.value ? Number(e.target.value) : '');
                    setSelectedTemplateId(null);
                  }}
                  disabled={doctor.role !== 'admin'}
                  className="input-field mt-1 disabled:opacity-70"
                >
                  <option value="">{doctor.role === 'admin' ? 'Все отделы' : 'Отдел не назначен'}</option>
                  {specialties.map((item) => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-text-muted uppercase">Поиск шаблона</label>
                <div className="relative mt-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Почки, ОБП, щитовидная..."
                    className="input-field pl-10"
                  />
                </div>
              </div>

              <div className="space-y-2 max-h-[620px] overflow-auto pr-1">
                {filteredTemplates.map((template) => (
                  <button
                    key={template.id}
                    type="button"
                    onClick={() => setSelectedTemplateId(template.id)}
                    className={`w-full text-left rounded-xl border p-3 transition ${
                      selectedTemplate?.id === template.id
                        ? 'border-medical-400 bg-medical-50'
                        : 'border-slate-200 bg-white hover:border-medical-200'
                    }`}
                  >
                    <div className="font-semibold text-medical-900">{template.name}</div>
                    <div className="text-sm text-text-secondary">
                      {[template.modality, template.bodyPart].filter(Boolean).join(' · ') || 'Без категории'}
                    </div>
                  </button>
                ))}
                {filteredTemplates.length === 0 && (
                  <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-text-muted">
                    Нет шаблонов. Импортируйте их в админке.
                  </div>
                )}
              </div>
            </aside>

            <section className="space-y-5">
              <div className="glass-card rounded-2xl p-5">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <h2 className="section-header mb-1">
                      <FileText className="w-5 h-5" />
                      {selectedTemplate?.name || 'Шаблон не выбран'}
                    </h2>
                    <p className="text-sm text-text-secondary">
                      {selectedTemplate
                        ? [selectedTemplate.modality, selectedTemplate.bodyPart, selectedTemplate.sourceFilename].filter(Boolean).join(' · ')
                        : 'Выберите шаблон слева'}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <input ref={audioInputRef} type="file" accept="audio/*" onChange={handleAudioUpload} className="hidden" />
                    <button
                      type="button"
                      onClick={() => audioInputRef.current?.click()}
                      disabled={processingAudio}
                      className="btn-secondary inline-flex items-center justify-center gap-2 py-2 px-3 text-sm disabled:opacity-50"
                    >
                      {processingAudio ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileAudio className="w-4 h-4" />}
                      Аудио
                    </button>
                    <button
                      type="button"
                      onClick={generateProtocol}
                      disabled={generating || !selectedTemplate}
                      className="btn-primary inline-flex items-center justify-center gap-2 py-2 px-3 text-sm disabled:opacity-50"
                    >
                      {generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                      Заполнить
                    </button>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-semibold text-text-muted uppercase">Диктовка врача</label>
                    <textarea
                      value={dictation}
                      onChange={(e) => setDictation(e.target.value)}
                      placeholder="Например: левая почка расположена обычно, размеры..., заключение..."
                      className="textarea-field mt-1 min-h-[360px]"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-text-muted uppercase">Исходный шаблон</label>
                    <textarea
                      value={selectedTemplate?.contentText || ''}
                      readOnly
                      className="textarea-field mt-1 min-h-[360px] bg-slate-50"
                    />
                  </div>
                </div>
              </div>

              <div className="glass-card rounded-2xl p-5">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-3">
                  <h2 className="section-header mb-0">
                    <FileText className="w-5 h-5" />
                    Готовый протокол
                  </h2>
                  <div className="flex gap-2">
                    <button type="button" onClick={copyProtocol} disabled={!filledText.trim()} className="btn-secondary inline-flex items-center gap-2 py-2 px-3 text-sm disabled:opacity-50">
                      <Clipboard className="w-4 h-4" />
                      Копировать
                    </button>
                    <button type="button" onClick={downloadProtocol} disabled={!filledText.trim()} className="btn-secondary py-2 px-3 text-sm disabled:opacity-50">
                      TXT
                    </button>
                  </div>
                </div>
                <textarea
                  value={filledText}
                  onChange={(e) => setFilledText(e.target.value)}
                  placeholder="После заполнения здесь появится протокол."
                  className="textarea-field min-h-[420px] font-mono text-sm"
                />
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
