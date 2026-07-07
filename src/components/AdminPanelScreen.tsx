import { useCallback, useEffect, useState } from 'react';
import {
  ArrowLeft,
  Ban,
  Check,
  FileText,
  FolderDown,
  KeyRound,
  Plus,
  RotateCcw,
  Save,
  Shield,
  UserCog,
  X,
} from 'lucide-react';
import { apiClient } from '../api/client';
import type { AdminDoctorInfo, DoctorInfo, ProtocolTemplateInfo, SpecialtyInfo } from '../api/client';

interface AdminPanelScreenProps {
  doctor: DoctorInfo;
  onBack: () => void;
}

const ROLE_LABEL: Record<DoctorInfo['role'], string> = {
  admin: 'Администратор',
  doctor: 'Врач',
};

const DEFAULT_TEMPLATE_PATH = 'C:\\Users\\AI\\Downloads\\ШАБЛОН ПРОТОКОЛ СТАЦИОНАР (1)\\ПРОТОКОЛ СТАЦИОНАР';

interface DoctorDraft {
  name: string;
  specialty: string;
  departmentId: number | null;
}

interface PasswordDraft {
  password: string;
  confirm: string;
}

export function AdminPanelScreen({ doctor, onBack }: AdminPanelScreenProps) {
  const [doctors, setDoctors] = useState<AdminDoctorInfo[]>([]);
  const [drafts, setDrafts] = useState<Record<number, DoctorDraft>>({});
  const [passwordDrafts, setPasswordDrafts] = useState<Record<number, PasswordDraft>>({});
  const [loading, setLoading] = useState(false);
  const [savingId, setSavingId] = useState<number | null>(null);
  const [passwordSavingId, setPasswordSavingId] = useState<number | null>(null);
  const [statusId, setStatusId] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [creating, setCreating] = useState(false);
  const [specialties, setSpecialties] = useState<SpecialtyInfo[]>([]);
  const [templates, setTemplates] = useState<ProtocolTemplateInfo[]>([]);
  const [loadingTemplates, setLoadingTemplates] = useState(false);
  const [templatePath, setTemplatePath] = useState(DEFAULT_TEMPLATE_PATH);
  const [templateSpecialtyName, setTemplateSpecialtyName] = useState('Лучевая диагностика');
  const [importingTemplates, setImportingTemplates] = useState(false);
  const [importSummary, setImportSummary] = useState('');
  const [newDoctor, setNewDoctor] = useState({
    name: '',
    email: '',
    password: '',
    specialty: '',
    departmentId: null as number | null,
    role: 'doctor' as DoctorInfo['role'],
  });

  const applyDoctors = useCallback((items: AdminDoctorInfo[]) => {
    setDoctors(items);
    setDrafts(Object.fromEntries(
      items.map((item) => [item.id, { name: item.name, specialty: item.specialty || '', departmentId: item.departmentId }]),
    ));
  }, []);

  const loadDoctors = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await apiClient.getAdminDoctors();
      applyDoctors(res.doctors);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось загрузить пользователей');
    } finally {
      setLoading(false);
    }
  }, [applyDoctors]);

  const loadTemplateCatalog = useCallback(async () => {
    setLoadingTemplates(true);
    try {
      const [specialtyRes, templateRes] = await Promise.all([
        apiClient.getSpecialties(),
        apiClient.getProtocolTemplates({ admin: true }),
      ]);
      setSpecialties(specialtyRes.specialties);
      setTemplates(templateRes.templates);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось загрузить шаблоны');
    } finally {
      setLoadingTemplates(false);
    }
  }, []);

  useEffect(() => {
    void loadDoctors();
    void loadTemplateCatalog();
  }, [loadDoctors, loadTemplateCatalog]);

  const patchDoctorInList = (updated: AdminDoctorInfo) => {
    setDoctors((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
    setDrafts((prev) => ({
      ...prev,
      [updated.id]: { name: updated.name, specialty: updated.specialty || '', departmentId: updated.departmentId },
    }));
  };

  const createDoctor = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setError('');
    setMessage('');
    try {
      const res = await apiClient.createAdminDoctor({
        name: newDoctor.name.trim(),
        email: newDoctor.email.trim(),
        password: newDoctor.password,
        specialty: newDoctor.specialty.trim(),
        departmentId: newDoctor.departmentId,
        role: newDoctor.role,
      });
      setDoctors((prev) => [res.doctor, ...prev]);
      setDrafts((prev) => ({
        ...prev,
        [res.doctor.id]: { name: res.doctor.name, specialty: res.doctor.specialty || '', departmentId: res.doctor.departmentId },
      }));
      setNewDoctor({ name: '', email: '', password: '', specialty: '', departmentId: null, role: 'doctor' });
      setShowCreate(false);
      setMessage('Пользователь добавлен');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось создать пользователя');
    } finally {
      setCreating(false);
    }
  };

  const saveDoctor = async (target: AdminDoctorInfo) => {
    const draft = drafts[target.id];
    if (!draft?.name.trim()) {
      setError('ФИО не может быть пустым');
      return;
    }

    setSavingId(target.id);
    setError('');
    setMessage('');
    try {
      const res = await apiClient.updateAdminDoctor(target.id, {
        name: draft.name.trim(),
        specialty: draft.specialty.trim(),
        departmentId: draft.departmentId,
      });
      patchDoctorInList(res.doctor);
      setMessage('Данные пользователя сохранены');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось сохранить пользователя');
    } finally {
      setSavingId(null);
    }
  };

  const updateRole = async (target: AdminDoctorInfo, role: DoctorInfo['role']) => {
    setSavingId(target.id);
    setError('');
    setMessage('');
    try {
      const res = await apiClient.updateAdminDoctor(target.id, { role });
      patchDoctorInList(res.doctor);
      setMessage('Роль обновлена');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось изменить роль');
    } finally {
      setSavingId(null);
    }
  };

  const toggleStatus = async (target: AdminDoctorInfo) => {
    setStatusId(target.id);
    setError('');
    setMessage('');
    try {
      const res = target.isActive
        ? await apiClient.deleteAdminDoctor(target.id)
        : await apiClient.updateAdminDoctor(target.id, { isActive: true });

      if (res.success) {
        setDoctors((prev) => prev.map((item) => (
          item.id === target.id ? { ...item, isActive: !target.isActive } : item
        )));
        setMessage(target.isActive ? 'Пользователь деактивирован' : 'Пользователь восстановлен');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось изменить статус');
    } finally {
      setStatusId(null);
    }
  };

  const resetPassword = async (target: AdminDoctorInfo) => {
    const draft = passwordDrafts[target.id] || { password: '', confirm: '' };
    setError('');
    setMessage('');

    if (draft.password.length < 8) {
      setError('Новый пароль должен быть не менее 8 символов');
      return;
    }
    if (draft.password !== draft.confirm) {
      setError('Пароль и подтверждение не совпадают');
      return;
    }

    setPasswordSavingId(target.id);
    try {
      await apiClient.updateAdminDoctorPassword(target.id, draft.password);
      setPasswordDrafts((prev) => ({
        ...prev,
        [target.id]: { password: '', confirm: '' },
      }));
      setMessage(`Пароль для ${target.name} изменен`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось изменить пароль');
    } finally {
      setPasswordSavingId(null);
    }
  };

  const importTemplates = async (e: React.FormEvent) => {
    e.preventDefault();
    setImportingTemplates(true);
    setError('');
    setMessage('');
    setImportSummary('');
    try {
      const res = await apiClient.importProtocolTemplates(templatePath.trim(), templateSpecialtyName.trim() || 'Лучевая диагностика');
      await loadTemplateCatalog();
      setImportSummary(`Импортировано: ${res.imported.length}. Пропущено: ${res.skipped.length}.`);
      setMessage('Каталог шаблонов обновлен');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось импортировать шаблоны');
    } finally {
      setImportingTemplates(false);
    }
  };

  const toggleTemplate = async (template: ProtocolTemplateInfo) => {
    setError('');
    setMessage('');
    try {
      const res = await apiClient.updateProtocolTemplate(template.id, { isActive: !template.isActive });
      setTemplates((prev) => prev.map((item) => (item.id === template.id ? res.template : item)));
      setMessage(template.isActive ? 'Шаблон отключен' : 'Шаблон включен');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Не удалось изменить шаблон');
    }
  };

  const activeAdmins = doctors.filter((item) => item.role === 'admin' && item.isActive).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-medical-50">
      <div className="bg-white border-b border-slate-200 px-4 py-4 sm:px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between gap-3">
          <button onClick={onBack} className="btn-secondary flex items-center gap-2 py-2 px-3 text-sm">
            <ArrowLeft className="w-4 h-4" />
            Назад
          </button>
          <div className="text-right min-w-0">
            <h1 className="font-display font-bold text-medical-900 text-lg leading-tight">Админ-панель</h1>
            <p className="text-xs text-text-muted truncate">{doctor.email}</p>
          </div>
        </div>
      </div>

      <main className="max-w-6xl mx-auto px-4 py-5 sm:px-6 space-y-5">
        <section className="glass-card rounded-2xl p-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="section-header mb-1">
                <Shield className="w-5 h-5" />
                Пользователи
              </h2>
              <p className="text-sm text-text-secondary">
                Активных администраторов: {activeAdmins}. Администратор может добавлять врачей, менять роли, блокировать доступ и задавать новый пароль.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowCreate((value) => !value)}
              className="btn-primary flex items-center justify-center gap-2 py-2 px-4 text-sm"
            >
              {showCreate ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
              {showCreate ? 'Закрыть' : 'Добавить'}
            </button>
          </div>

          {showCreate && (
            <form onSubmit={createDoctor} className="mt-5 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-7 gap-3">
              <input
                value={newDoctor.name}
                onChange={(e) => setNewDoctor((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="ФИО"
                className="input-field xl:col-span-2"
                required
              />
              <input
                type="email"
                value={newDoctor.email}
                onChange={(e) => setNewDoctor((prev) => ({ ...prev, email: e.target.value }))}
                placeholder="Email"
                className="input-field"
                required
              />
              <input
                type="password"
                value={newDoctor.password}
                onChange={(e) => setNewDoctor((prev) => ({ ...prev, password: e.target.value }))}
                placeholder="Пароль от 8 символов"
                className="input-field"
                required
              />
              <input
                value={newDoctor.specialty}
                onChange={(e) => setNewDoctor((prev) => ({ ...prev, specialty: e.target.value }))}
                placeholder="Должность / профиль"
                className="input-field"
              />
              <select
                value={newDoctor.departmentId ?? ''}
                onChange={(e) => setNewDoctor((prev) => ({ ...prev, departmentId: e.target.value ? Number(e.target.value) : null }))}
                className="input-field"
              >
                <option value="">Без отдела</option>
                {specialties.map((item) => (
                  <option key={item.id} value={item.id}>{item.name}</option>
                ))}
              </select>
              <div className="flex gap-2">
                <select
                  value={newDoctor.role}
                  onChange={(e) => setNewDoctor((prev) => ({ ...prev, role: e.target.value as DoctorInfo['role'] }))}
                  className="input-field"
                >
                  <option value="doctor">Врач</option>
                  <option value="admin">Администратор</option>
                </select>
                <button
                  type="submit"
                  disabled={creating}
                  className="btn-primary inline-flex items-center justify-center px-4 py-2 disabled:opacity-50"
                  title="Создать пользователя"
                >
                  <Check className="w-5 h-5" />
                </button>
              </div>
            </form>
          )}
        </section>

        {(error || message) && (
          <div className={`rounded-xl border px-4 py-3 text-sm ${
            error
              ? 'border-red-200 bg-red-50 text-red-700'
              : 'border-medical-200 bg-medical-50 text-medical-700'
          }`}>
            {error || message}
          </div>
        )}

        {loading ? (
          <div className="glass-card rounded-2xl p-10 text-center text-text-muted">Загрузка...</div>
        ) : (
          <div className="space-y-3">
            {doctors.map((item) => {
              const isSelf = item.id === doctor.id;
              const draft = drafts[item.id] || { name: item.name, specialty: item.specialty || '', departmentId: item.departmentId };
              const passDraft = passwordDrafts[item.id] || { password: '', confirm: '' };
              const isLastActiveAdmin = item.role === 'admin' && item.isActive && activeAdmins <= 1;

              return (
                <article key={item.id} className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
                  <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.5fr)_minmax(420px,1fr)] gap-4">
                    <div className="min-w-0 space-y-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <UserCog className="w-5 h-5 text-medical-600" />
                        <span className="font-semibold text-medical-900 truncate">{item.email}</span>
                        {isSelf && <span className="text-xs text-medical-700 bg-medical-50 border border-medical-200 rounded-full px-2 py-0.5">вы</span>}
                        <span className={`text-xs font-medium rounded-full px-2 py-0.5 ${
                          item.isActive ? 'bg-medical-50 text-medical-700' : 'bg-slate-100 text-slate-500'
                        }`}>
                          {item.isActive ? 'Активен' : 'Деактивирован'}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <input
                          value={draft.name}
                          onChange={(e) => setDrafts((prev) => ({
                            ...prev,
                            [item.id]: { ...draft, name: e.target.value },
                          }))}
                          placeholder="ФИО"
                          className="input-field"
                        />
                        <input
                          value={draft.specialty}
                          onChange={(e) => setDrafts((prev) => ({
                            ...prev,
                            [item.id]: { ...draft, specialty: e.target.value },
                          }))}
                          placeholder="Должность / профиль"
                          className="input-field"
                        />
                        <select
                          value={draft.departmentId ?? ''}
                          onChange={(e) => setDrafts((prev) => ({
                            ...prev,
                            [item.id]: { ...draft, departmentId: e.target.value ? Number(e.target.value) : null },
                          }))}
                          className="input-field"
                        >
                          <option value="">Без отдела</option>
                          {specialties.map((department) => (
                            <option key={department.id} value={department.id}>{department.name}</option>
                          ))}
                        </select>
                      </div>

                      <p className="text-xs text-text-muted">
                        Создан: {new Date(item.createdAt).toLocaleString()}
                      </p>
                    </div>

                    <div className="space-y-3">
                      <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_auto] gap-2">
                        <select
                          value={item.role}
                          onChange={(e) => void updateRole(item, e.target.value as DoctorInfo['role'])}
                          disabled={isSelf || isLastActiveAdmin || savingId === item.id}
                          className="input-field py-2 text-sm disabled:opacity-60"
                        >
                          <option value="doctor">{ROLE_LABEL.doctor}</option>
                          <option value="admin">{ROLE_LABEL.admin}</option>
                        </select>
                        <button
                          type="button"
                          onClick={() => void saveDoctor(item)}
                          disabled={savingId === item.id}
                          className="btn-secondary inline-flex items-center justify-center gap-2 py-2 px-3 text-sm disabled:opacity-50"
                        >
                          <Save className="w-4 h-4" />
                          Сохранить
                        </button>
                        <button
                          type="button"
                          onClick={() => void toggleStatus(item)}
                          disabled={isSelf || isLastActiveAdmin || statusId === item.id}
                          className="btn-secondary inline-flex items-center justify-center gap-2 py-2 px-3 text-sm disabled:opacity-50"
                        >
                          {item.isActive ? <Ban className="w-4 h-4" /> : <RotateCcw className="w-4 h-4" />}
                          {item.isActive ? 'Блокировать' : 'Восстановить'}
                        </button>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-[1fr_1fr_auto] gap-2">
                        <input
                          type="password"
                          value={passDraft.password}
                          onChange={(e) => setPasswordDrafts((prev) => ({
                            ...prev,
                            [item.id]: { ...passDraft, password: e.target.value },
                          }))}
                          placeholder="Новый пароль"
                          className="input-field"
                        />
                        <input
                          type="password"
                          value={passDraft.confirm}
                          onChange={(e) => setPasswordDrafts((prev) => ({
                            ...prev,
                            [item.id]: { ...passDraft, confirm: e.target.value },
                          }))}
                          placeholder="Повторить пароль"
                          className="input-field"
                        />
                        <button
                          type="button"
                          onClick={() => void resetPassword(item)}
                          disabled={passwordSavingId === item.id}
                          className="btn-primary inline-flex items-center justify-center gap-2 py-2 px-3 text-sm disabled:opacity-50"
                        >
                          <KeyRound className="w-4 h-4" />
                          Пароль
                        </button>
                      </div>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}

        <section className="glass-card rounded-2xl p-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <h2 className="section-header mb-1">
                <FileText className="w-5 h-5" />
                Шаблоны протоколов
              </h2>
              <p className="text-sm text-text-secondary">
                Импортируйте `.docx` шаблоны и привяжите их к специальности. Старые `.doc` файлы нужно конвертировать в `.docx`.
              </p>
            </div>
            <div className="text-sm text-text-muted lg:text-right">
              Активных: {templates.filter((item) => item.isActive).length} из {templates.length}
            </div>
          </div>

          <form onSubmit={importTemplates} className="mt-5 grid grid-cols-1 lg:grid-cols-[1fr_240px_auto] gap-3">
            <input
              value={templatePath}
              onChange={(e) => setTemplatePath(e.target.value)}
              placeholder="Путь к папке с шаблонами"
              className="input-field"
              required
            />
            <input
              value={templateSpecialtyName}
              onChange={(e) => setTemplateSpecialtyName(e.target.value)}
              placeholder="Специальность"
              className="input-field"
              required
            />
            <button
              type="submit"
              disabled={importingTemplates}
              className="btn-primary inline-flex items-center justify-center gap-2 py-2 px-4 text-sm disabled:opacity-50"
            >
              <FolderDown className="w-4 h-4" />
              {importingTemplates ? 'Импорт...' : 'Импортировать'}
            </button>
          </form>

          {importSummary && <p className="mt-3 text-sm text-medical-700">{importSummary}</p>}

          <div className="mt-5">
            {loadingTemplates ? (
              <div className="py-8 text-center text-text-muted">Загрузка шаблонов...</div>
            ) : templates.length === 0 ? (
              <div className="rounded-xl border border-slate-200 bg-white p-5 text-sm text-text-muted">
                Шаблоны еще не импортированы.
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {templates.slice(0, 80).map((template) => {
                  const specialty = specialties.find((item) => item.id === template.specialtyId);
                  return (
                    <div key={template.id} className="bg-white rounded-xl border border-slate-200 p-3">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <h3 className="font-semibold text-medical-900 truncate">{template.name}</h3>
                          <p className="text-sm text-text-secondary truncate">
                            {[specialty?.name, template.modality, template.bodyPart].filter(Boolean).join(' · ') || 'Без категории'}
                          </p>
                          <p className="text-xs text-text-muted truncate">{template.sourceFilename}</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => void toggleTemplate(template)}
                          className={`rounded-lg px-3 py-2 text-xs font-medium whitespace-nowrap ${
                            template.isActive
                              ? 'bg-medical-50 text-medical-700 border border-medical-200'
                              : 'bg-slate-100 text-slate-500 border border-slate-200'
                          }`}
                        >
                          {template.isActive ? 'Активен' : 'Выключен'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
