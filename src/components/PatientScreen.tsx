import { useState, useEffect, useCallback } from 'react';
import { ArrowLeft, Mic, FileText, Calendar, Phone, User, Edit2, Check, X, ClipboardList } from 'lucide-react';
import { apiClient } from '../api/client';
import type { PatientSummary, VisitSummary, PatientForm } from '../api/client';
import type { MedicalDocument } from '../types';

interface PatientScreenProps {
  patientId: number;
  onBack: () => void;
  onNewRecording: (patient: PatientSummary) => void;
  onViewDocument: (doc: MedicalDocument, rawText: string, visitId: number) => void;
}

function formatDate(iso: string): string {
  if (!iso) return '';
  return new Date(iso).toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' });
}

function calcAge(birthDate: string): string {
  if (!birthDate) return '';
  const birth = new Date(birthDate);
  if (isNaN(birth.getTime())) return '';
  const now = new Date();
  let age = now.getFullYear() - birth.getFullYear();
  if (now.getMonth() - birth.getMonth() < 0 || (now.getMonth() === birth.getMonth() && now.getDate() < birth.getDate())) age--;
  return `${age} лет`;
}

const GENDER_LABEL: Record<string, string> = { male: 'Мужской', female: 'Женский', '': '' };

export function PatientScreen({ patientId, onBack, onNewRecording, onViewDocument }: PatientScreenProps) {
  const [patient, setPatient]   = useState<PatientSummary | null>(null);
  const [visits,  setVisits]    = useState<VisitSummary[]>([]);
  const [loading, setLoading]   = useState(true);
  const [editing, setEditing]   = useState(false);
  const [form, setForm]         = useState<PatientForm>({ fullName: '', birthDate: '', gender: '', phone: '', iin: '', notes: '' });
  const [saving, setSaving]     = useState(false);
  const [visitLoading, setVisitLoading] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiClient.getPatient(patientId);
      setPatient(res.patient);
      setVisits(res.visits);
      setForm({
        fullName:  res.patient.fullName,
        birthDate: res.patient.birthDate,
        gender:    res.patient.gender,
        phone:     res.patient.phone,
        iin:       res.patient.iin,
        notes:     res.patient.notes || '',
      });
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => { void load(); }, [load]);

  const handleSave = async () => {
    if (!patient) return;
    setSaving(true);
    try {
      const res = await apiClient.updatePatient(patientId, form);
      setPatient(res.patient);
      setEditing(false);
    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  const openVisit = async (visitId: number) => {
    setVisitLoading(visitId);
    try {
      const res = await apiClient.getVisit(visitId);
      if (res.visit.document) {
        onViewDocument(res.visit.document, res.visit.rawTranscription || '', visitId);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setVisitLoading(null);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-text-muted">Загрузка...</p>
      </div>
    );
  }

  if (!patient) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-text-muted mb-4">Пациент не найден</p>
          <button onClick={onBack} className="btn-secondary">Назад</button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-medical-50">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-4 py-4 sm:px-6">
        <div className="max-w-2xl mx-auto flex items-center gap-3">
          <button onClick={onBack} className="p-2 hover:bg-slate-100 rounded-xl transition-colors">
            <ArrowLeft className="w-5 h-5 text-slate-600" />
          </button>
          <div className="flex-1 min-w-0">
            <h1 className="font-display font-bold text-medical-900 text-lg truncate">{patient.fullName}</h1>
            <p className="text-xs text-text-muted">
              {[GENDER_LABEL[patient.gender], calcAge(patient.birthDate)].filter(Boolean).join(' · ')}
            </p>
          </div>
          <button
            onClick={() => onNewRecording(patient)}
            className="btn-primary flex items-center gap-1.5 py-2 px-3 text-sm"
          >
            <Mic className="w-4 h-4" /> Новый осмотр
          </button>
        </div>
      </div>

      <div className="max-w-2xl mx-auto px-4 py-4 sm:px-6 space-y-4">
        {/* Patient card */}
        <div className="glass-card rounded-2xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-medical-900 flex items-center gap-2">
              <User className="w-4 h-4" /> Карточка пациента
            </h2>
            {!editing ? (
              <button onClick={() => setEditing(true)} className="p-1.5 hover:bg-slate-100 rounded-lg transition-colors">
                <Edit2 className="w-4 h-4 text-slate-500" />
              </button>
            ) : (
              <div className="flex gap-1">
                <button onClick={handleSave} disabled={saving} className="p-1.5 bg-medical-100 hover:bg-medical-200 rounded-lg transition-colors">
                  <Check className="w-4 h-4 text-medical-700" />
                </button>
                <button onClick={() => setEditing(false)} className="p-1.5 hover:bg-slate-100 rounded-lg transition-colors">
                  <X className="w-4 h-4 text-slate-500" />
                </button>
              </div>
            )}
          </div>

          {!editing ? (
            <div className="grid grid-cols-2 gap-3 text-sm">
              {[
                { label: 'ФИО',            val: patient.fullName },
                { label: 'Дата рождения',  val: patient.birthDate ? formatDate(patient.birthDate) : '—' },
                { label: 'Пол',            val: GENDER_LABEL[patient.gender] || '—' },
                { label: 'Возраст',        val: calcAge(patient.birthDate) || '—' },
                { label: 'Телефон',        val: patient.phone || '—' },
                { label: 'ИИН',            val: patient.iin || '—' },
              ].map(({ label, val }) => (
                <div key={label}>
                  <p className="text-xs text-text-muted">{label}</p>
                  <p className="text-medical-900 font-medium">{val}</p>
                </div>
              ))}
              {patient.notes && (
                <div className="col-span-2">
                  <p className="text-xs text-text-muted">Заметки</p>
                  <p className="text-medical-900">{patient.notes}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <input value={form.fullName} onChange={e => setForm(f => ({ ...f, fullName: e.target.value }))}
                placeholder="ФИО *" className="input-field text-sm" />
              <div className="grid grid-cols-2 gap-3">
                <input type="date" value={form.birthDate} onChange={e => setForm(f => ({ ...f, birthDate: e.target.value }))} className="input-field text-sm" />
                <select value={form.gender} onChange={e => setForm(f => ({ ...f, gender: e.target.value }))} className="input-field text-sm">
                  <option value="">Пол</option>
                  <option value="male">Мужской</option>
                  <option value="female">Женский</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <input value={form.phone} onChange={e => setForm(f => ({ ...f, phone: e.target.value }))} placeholder="Телефон" className="input-field text-sm" />
                <input value={form.iin} onChange={e => setForm(f => ({ ...f, iin: e.target.value }))} placeholder="ИИН" maxLength={12} className="input-field text-sm" />
              </div>
              <textarea value={form.notes} onChange={e => setForm(f => ({ ...f, notes: e.target.value }))} placeholder="Заметки" rows={2} className="input-field text-sm resize-none" />
            </div>
          )}
        </div>

        {/* Visit history */}
        <div className="glass-card rounded-2xl p-5">
          <h2 className="font-semibold text-medical-900 mb-4 flex items-center gap-2">
            <ClipboardList className="w-4 h-4" /> История осмотров ({visits.length})
          </h2>

          {visits.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="w-10 h-10 text-slate-300 mx-auto mb-2" />
              <p className="text-text-muted text-sm">Осмотров пока нет</p>
              <button onClick={() => onNewRecording(patient)} className="btn-primary mt-3 py-2 px-4 text-sm">
                Провести первый осмотр
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {visits.map(v => (
                <button
                  key={v.id}
                  onClick={() => void openVisit(v.id)}
                  disabled={visitLoading === v.id}
                  className="w-full text-left bg-slate-50 hover:bg-medical-50 rounded-xl p-3.5 transition-colors border border-slate-200 hover:border-medical-200 disabled:opacity-60"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <Calendar className="w-3.5 h-3.5 text-medical-500 flex-shrink-0" />
                        <span className="text-sm font-medium text-medical-900">{formatDate(v.visitDate)}</span>
                      </div>
                      {v.diagnosisPreview && (
                        <p className="text-xs text-text-secondary truncate">{v.diagnosisPreview}</p>
                      )}
                    </div>
                    <FileText className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
                  </div>
                  {visitLoading === v.id && (
                    <p className="text-xs text-text-muted mt-1">Загрузка...</p>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Quick info */}
        <div className="flex gap-3 text-xs text-text-muted">
          {patient.phone && (
            <span className="flex items-center gap-1">
              <Phone className="w-3 h-3" />{patient.phone}
            </span>
          )}
          <span className="flex items-center gap-1">
            <Calendar className="w-3 h-3" />Зарегистрирован {formatDate(patient.createdAt || '')}
          </span>
        </div>
      </div>
    </div>
  );
}
