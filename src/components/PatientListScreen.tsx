import { useState, useEffect, useCallback, useRef } from 'react';
import { Users, Plus, Search, ChevronRight, Phone, Calendar } from 'lucide-react';
import { apiClient } from '../api/client';
import type { PatientSummary, PatientForm, DoctorInfo } from '../api/client';

interface PatientListScreenProps {
  doctor: DoctorInfo;
  onSelectPatient: (patient: PatientSummary) => void;
  onNewRecording: () => void;
}

const GENDER_LABEL: Record<string, string> = {
  male: 'М',
  female: 'Ж',
  '': '—',
};

function calcAge(birthDate: string): string {
  if (!birthDate) return '';
  const birth = new Date(birthDate);
  if (isNaN(birth.getTime())) return '';
  const now = new Date();
  let age = now.getFullYear() - birth.getFullYear();
  const m = now.getMonth() - birth.getMonth();
  if (m < 0 || (m === 0 && now.getDate() < birth.getDate())) age--;
  return `${age} лет`;
}

function formatDate(iso: string): string {
  if (!iso) return '';
  return new Date(iso).toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' });
}

export function PatientListScreen({ doctor, onSelectPatient, onNewRecording }: PatientListScreenProps) {
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [hasMore, setHasMore] = useState(false);
  const [page, setPage] = useState(1);

  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState<PatientForm>({ fullName: '', birthDate: '', gender: '', phone: '', iin: '', notes: '' });
  const [saving, setSaving] = useState(false);
  const [createError, setCreateError] = useState('');

  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = useCallback(async (q: string, p: number, append = false) => {
    setLoading(true);
    try {
      const res = await apiClient.getPatients({ q, page: p });
      setPatients(prev => append ? [...prev, ...res.patients] : res.patients);
      setHasMore(res.hasMore);
      setPage(p);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load('', 1);
  }, [load]);

  const handleSearch = (q: string) => {
    setSearch(q);
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => void load(q, 1), 300);
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.fullName.trim()) return;
    setSaving(true);
    setCreateError('');
    try {
      const res = await apiClient.createPatient(form);
      setPatients(prev => [res.patient, ...prev]);
      setShowCreate(false);
      setForm({ fullName: '', birthDate: '', gender: '', phone: '', iin: '', notes: '' });
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Ошибка');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-medical-50">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-4 py-4 sm:px-6">
        <div className="max-w-2xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-medical-100 flex items-center justify-center">
              <Users className="w-5 h-5 text-medical-600" />
            </div>
            <div>
              <h1 className="font-display font-bold text-medical-900 text-lg leading-tight">Пациенты</h1>
              <p className="text-xs text-text-muted">{doctor.name}{doctor.specialty ? ` · ${doctor.specialty}` : ''}</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={() => setShowCreate(true)} className="btn-primary flex items-center gap-1.5 py-2 px-3 text-sm">
              <Plus className="w-4 h-4" /> Новый пациент
            </button>
            <button onClick={onNewRecording} className="btn-secondary py-2 px-3 text-sm">
              Запись
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-2xl mx-auto px-4 py-4 sm:px-6">
        {/* Search */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            value={search}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Поиск по имени..."
            className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 bg-white focus:outline-none focus:ring-2 focus:ring-medical-400 text-sm"
          />
        </div>

        {/* Create form */}
        {showCreate && (
          <div className="glass-card rounded-2xl p-5 mb-4 slide-up">
            <h2 className="font-semibold text-medical-900 mb-4">Новый пациент</h2>
            <form onSubmit={handleCreate} className="space-y-3">
              <input required value={form.fullName} onChange={e => setForm(f => ({ ...f, fullName: e.target.value }))}
                placeholder="ФИО пациента *" className="input-field" />
              <div className="grid grid-cols-2 gap-3">
                <input type="date" value={form.birthDate} onChange={e => setForm(f => ({ ...f, birthDate: e.target.value }))}
                  placeholder="Дата рождения" className="input-field" />
                <select value={form.gender} onChange={e => setForm(f => ({ ...f, gender: e.target.value }))} className="input-field">
                  <option value="">Пол</option>
                  <option value="male">Мужской</option>
                  <option value="female">Женский</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <input value={form.phone} onChange={e => setForm(f => ({ ...f, phone: e.target.value }))}
                  placeholder="Телефон" className="input-field" />
                <input value={form.iin} onChange={e => setForm(f => ({ ...f, iin: e.target.value }))}
                  placeholder="ИИН" maxLength={12} className="input-field" />
              </div>
              <textarea value={form.notes} onChange={e => setForm(f => ({ ...f, notes: e.target.value }))}
                placeholder="Заметки" rows={2} className="input-field resize-none" />
              {createError && <p className="text-red-600 text-sm">{createError}</p>}
              <div className="flex gap-2">
                <button type="submit" disabled={saving || !form.fullName.trim()} className="btn-primary flex-1 py-2 text-sm disabled:opacity-50">
                  {saving ? 'Сохранение...' : 'Создать'}
                </button>
                <button type="button" onClick={() => setShowCreate(false)} className="btn-secondary flex-1 py-2 text-sm">Отмена</button>
              </div>
            </form>
          </div>
        )}

        {/* List */}
        {loading && patients.length === 0 ? (
          <div className="text-center py-12 text-text-muted">Загрузка...</div>
        ) : patients.length === 0 ? (
          <div className="text-center py-12">
            <Users className="w-12 h-12 text-slate-300 mx-auto mb-3" />
            <p className="text-text-muted">{search ? 'Ничего не найдено' : 'Пациентов пока нет'}</p>
            {!search && (
              <button onClick={() => setShowCreate(true)} className="btn-primary mt-4 py-2 px-4 text-sm">
                Добавить первого пациента
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {patients.map(p => (
              <button
                key={p.id}
                onClick={() => onSelectPatient(p)}
                className="w-full text-left bg-white rounded-xl border border-slate-200 p-4 hover:border-medical-300 hover:shadow-sm transition-all flex items-center gap-3"
              >
                <div className="w-10 h-10 rounded-full bg-medical-100 flex items-center justify-center text-medical-700 font-semibold text-sm flex-shrink-0">
                  {p.fullName.slice(0, 2).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-medical-900 truncate">{p.fullName}</span>
                    {p.gender && <span className="text-xs text-text-muted">{GENDER_LABEL[p.gender]}</span>}
                    {p.birthDate && <span className="text-xs text-text-muted">{calcAge(p.birthDate)}</span>}
                  </div>
                  <div className="flex items-center gap-3 mt-0.5">
                    {p.phone && (
                      <span className="text-xs text-text-secondary flex items-center gap-1">
                        <Phone className="w-3 h-3" />{p.phone}
                      </span>
                    )}
                    <span className="text-xs text-text-muted flex items-center gap-1">
                      <Calendar className="w-3 h-3" />Обновлён {formatDate(p.updatedAt)}
                    </span>
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-slate-400 flex-shrink-0" />
              </button>
            ))}
            {hasMore && (
              <button
                onClick={() => void load(search, page + 1, true)}
                className="w-full py-3 text-sm text-medical-600 hover:text-medical-700"
              >
                Загрузить ещё...
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
