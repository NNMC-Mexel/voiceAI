import { useEffect, useState } from 'react';
import { ArrowLeft, LogOut, Plus, Shield, User, KeyRound, RotateCcw, Ban } from 'lucide-react';
import { apiClient } from '../api/client';
import type { AdminDoctorInfo, DoctorInfo } from '../api/client';

interface SettingsScreenProps {
  doctor: DoctorInfo;
  onBack: () => void;
  onLogout: () => void;
  onDoctorUpdate?: (doctor: DoctorInfo) => void;
}

const ROLE_LABEL: Record<DoctorInfo['role'], string> = {
  admin: 'Администратор',
  doctor: 'Врач',
};

export function SettingsScreen({ doctor, onBack, onLogout, onDoctorUpdate }: SettingsScreenProps) {
  const [profile, setProfile] = useState({ name: doctor.name, specialty: doctor.specialty || '' });
  const [profileMessage, setProfileMessage] = useState('');
  const [profileError, setProfileError] = useState('');
  const [savingProfile, setSavingProfile] = useState(false);

  const [passwords, setPasswords] = useState({ currentPassword: '', newPassword: '', confirmPassword: '' });
  const [passwordMessage, setPasswordMessage] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [savingPassword, setSavingPassword] = useState(false);

  const [doctors, setDoctors] = useState<AdminDoctorInfo[]>([]);
  const [loadingDoctors, setLoadingDoctors] = useState(false);
  const [adminError, setAdminError] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [creatingDoctor, setCreatingDoctor] = useState(false);
  const [newDoctor, setNewDoctor] = useState({
    name: '',
    email: '',
    password: '',
    specialty: '',
    role: 'doctor' as DoctorInfo['role'],
  });

  const loadDoctors = async () => {
    if (doctor.role !== 'admin') return;
    setLoadingDoctors(true);
    setAdminError('');
    try {
      const res = await apiClient.getAdminDoctors();
      setDoctors(res.doctors);
    } catch (err) {
      setAdminError(err instanceof Error ? err.message : 'Не удалось загрузить врачей');
    } finally {
      setLoadingDoctors(false);
    }
  };

  useEffect(() => {
    void loadDoctors();
  }, [doctor.role]);

  const saveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavingProfile(true);
    setProfileMessage('');
    setProfileError('');
    try {
      const res = await apiClient.updateProfile({
        name: profile.name.trim(),
        specialty: profile.specialty.trim(),
      });
      onDoctorUpdate?.(res.doctor);
      setProfileMessage('Профиль сохранен');
    } catch (err) {
      setProfileError(err instanceof Error ? err.message : 'Не удалось сохранить профиль');
    } finally {
      setSavingProfile(false);
    }
  };

  const changePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordMessage('');
    setPasswordError('');
    if (passwords.newPassword !== passwords.confirmPassword) {
      setPasswordError('Новый пароль и подтверждение не совпадают');
      return;
    }
    setSavingPassword(true);
    try {
      await apiClient.changePassword(passwords.currentPassword, passwords.newPassword);
      setPasswords({ currentPassword: '', newPassword: '', confirmPassword: '' });
      setPasswordMessage('Пароль изменен');
    } catch (err) {
      setPasswordError(err instanceof Error ? err.message : 'Не удалось изменить пароль');
    } finally {
      setSavingPassword(false);
    }
  };

  const createDoctor = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreatingDoctor(true);
    setAdminError('');
    try {
      const res = await apiClient.createAdminDoctor({
        name: newDoctor.name.trim(),
        email: newDoctor.email.trim(),
        password: newDoctor.password,
        specialty: newDoctor.specialty.trim(),
        role: newDoctor.role,
      });
      setDoctors(prev => [res.doctor, ...prev]);
      setNewDoctor({ name: '', email: '', password: '', specialty: '', role: 'doctor' });
      setShowCreate(false);
    } catch (err) {
      setAdminError(err instanceof Error ? err.message : 'Не удалось создать врача');
    } finally {
      setCreatingDoctor(false);
    }
  };

  const updateDoctorRole = async (target: AdminDoctorInfo, role: DoctorInfo['role']) => {
    try {
      const res = await apiClient.updateAdminDoctor(target.id, { role });
      setDoctors(prev => prev.map(d => d.id === target.id ? res.doctor : d));
    } catch (err) {
      setAdminError(err instanceof Error ? err.message : 'Не удалось изменить роль');
    }
  };

  const toggleDoctorStatus = async (target: AdminDoctorInfo) => {
    try {
      const res = target.isActive
        ? await apiClient.deleteAdminDoctor(target.id)
        : await apiClient.updateAdminDoctor(target.id, { isActive: true });
      if (res.success) {
        setDoctors(prev => prev.map(d => d.id === target.id ? { ...d, isActive: !target.isActive } : d));
      }
    } catch (err) {
      setAdminError(err instanceof Error ? err.message : 'Не удалось изменить статус');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-medical-50">
      <div className="bg-white border-b border-slate-200 px-4 py-4 sm:px-6">
        <div className="max-w-4xl mx-auto flex items-center justify-between gap-3">
          <button onClick={onBack} className="btn-secondary flex items-center gap-2 py-2 px-3 text-sm">
            <ArrowLeft className="w-4 h-4" />
            Назад
          </button>
          <div className="text-right min-w-0">
            <h1 className="font-display font-bold text-medical-900 text-lg leading-tight">Настройки</h1>
            <p className="text-xs text-text-muted truncate">{doctor.email}</p>
          </div>
        </div>
      </div>

      <main className="max-w-4xl mx-auto px-4 py-5 sm:px-6 space-y-5">
        <section className="glass-card rounded-2xl p-5">
          <h2 className="section-header">
            <User className="w-5 h-5" />
            Мой профиль
          </h2>
          <form onSubmit={saveProfile} className="space-y-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <input
                value={profile.name}
                onChange={e => setProfile(p => ({ ...p, name: e.target.value }))}
                placeholder="ФИО врача"
                className="input-field"
                required
              />
              <input
                value={profile.specialty}
                onChange={e => setProfile(p => ({ ...p, specialty: e.target.value }))}
                placeholder="Специальность"
                className="input-field"
              />
            </div>
            {profileError && <p className="text-sm text-red-600">{profileError}</p>}
            {profileMessage && <p className="text-sm text-medical-700">{profileMessage}</p>}
            <button type="submit" disabled={savingProfile || !profile.name.trim()} className="btn-primary py-2 px-4 text-sm disabled:opacity-50">
              {savingProfile ? 'Сохранение...' : 'Сохранить'}
            </button>
          </form>
        </section>

        <section className="glass-card rounded-2xl p-5">
          <h2 className="section-header">
            <KeyRound className="w-5 h-5" />
            Безопасность
          </h2>
          <form onSubmit={changePassword} className="space-y-3">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <input
                type="password"
                value={passwords.currentPassword}
                onChange={e => setPasswords(p => ({ ...p, currentPassword: e.target.value }))}
                placeholder="Текущий пароль"
                className="input-field"
                required
              />
              <input
                type="password"
                value={passwords.newPassword}
                onChange={e => setPasswords(p => ({ ...p, newPassword: e.target.value }))}
                placeholder="Новый пароль"
                className="input-field"
                required
              />
              <input
                type="password"
                value={passwords.confirmPassword}
                onChange={e => setPasswords(p => ({ ...p, confirmPassword: e.target.value }))}
                placeholder="Подтвердить"
                className="input-field"
                required
              />
            </div>
            {passwordError && <p className="text-sm text-red-600">{passwordError}</p>}
            {passwordMessage && <p className="text-sm text-medical-700">{passwordMessage}</p>}
            <button type="submit" disabled={savingPassword} className="btn-primary py-2 px-4 text-sm disabled:opacity-50">
              {savingPassword ? 'Сохранение...' : 'Сменить пароль'}
            </button>
          </form>
        </section>

        {doctor.role === 'admin' && (
          <section className="glass-card rounded-2xl p-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <h2 className="section-header mb-0">
                <Shield className="w-5 h-5" />
                Управление врачами
              </h2>
              <button onClick={() => setShowCreate(v => !v)} className="btn-secondary flex items-center justify-center gap-2 py-2 px-3 text-sm">
                <Plus className="w-4 h-4" />
                Добавить врача
              </button>
            </div>

            {showCreate && (
              <form onSubmit={createDoctor} className="mb-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                <input value={newDoctor.name} onChange={e => setNewDoctor(d => ({ ...d, name: e.target.value }))} placeholder="ФИО" className="input-field" required />
                <input type="email" value={newDoctor.email} onChange={e => setNewDoctor(d => ({ ...d, email: e.target.value }))} placeholder="Email" className="input-field" required />
                <input type="password" value={newDoctor.password} onChange={e => setNewDoctor(d => ({ ...d, password: e.target.value }))} placeholder="Пароль" className="input-field" required />
                <input value={newDoctor.specialty} onChange={e => setNewDoctor(d => ({ ...d, specialty: e.target.value }))} placeholder="Специальность" className="input-field" />
                <div className="flex gap-2">
                  <select value={newDoctor.role} onChange={e => setNewDoctor(d => ({ ...d, role: e.target.value as DoctorInfo['role'] }))} className="input-field">
                    <option value="doctor">Врач</option>
                    <option value="admin">Администратор</option>
                  </select>
                  <button type="submit" disabled={creatingDoctor} className="btn-primary py-2 px-3 text-sm disabled:opacity-50">
                    {creatingDoctor ? '...' : 'OK'}
                  </button>
                </div>
              </form>
            )}

            {adminError && <p className="text-sm text-red-600 mb-3">{adminError}</p>}
            {loadingDoctors ? (
              <div className="py-8 text-center text-text-muted">Загрузка...</div>
            ) : (
              <div className="space-y-2">
                {doctors.map(item => {
                  const isSelf = item.id === doctor.id;
                  return (
                    <div key={item.id} className="bg-white rounded-xl border border-slate-200 p-3 flex flex-col lg:flex-row lg:items-center gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-medical-900 truncate">{item.name}</span>
                          {isSelf && <span className="text-xs text-medical-700 bg-medical-50 border border-medical-200 rounded-full px-2 py-0.5">вы</span>}
                        </div>
                        <p className="text-sm text-text-secondary truncate">{item.email}</p>
                        <p className="text-xs text-text-muted truncate">{item.specialty || 'Специальность не указана'}</p>
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                        <select
                          value={item.role}
                          onChange={e => void updateDoctorRole(item, e.target.value as DoctorInfo['role'])}
                          disabled={isSelf}
                          className="input-field py-2 text-sm sm:w-44 disabled:opacity-60"
                        >
                          <option value="doctor">{ROLE_LABEL.doctor}</option>
                          <option value="admin">{ROLE_LABEL.admin}</option>
                        </select>
                        <span className={`text-xs font-medium rounded-full px-3 py-2 text-center ${item.isActive ? 'bg-medical-50 text-medical-700' : 'bg-slate-100 text-slate-500'}`}>
                          {item.isActive ? 'Активен' : 'Деактивирован'}
                        </span>
                        <button
                          type="button"
                          onClick={() => void toggleDoctorStatus(item)}
                          disabled={isSelf}
                          className="btn-secondary flex items-center justify-center gap-2 py-2 px-3 text-sm disabled:opacity-50"
                        >
                          {item.isActive ? <Ban className="w-4 h-4" /> : <RotateCcw className="w-4 h-4" />}
                          {item.isActive ? 'Деактивировать' : 'Восстановить'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}

        <button
          type="button"
          onClick={onLogout}
          className="w-full flex items-center justify-center gap-2 rounded-xl bg-red-600 hover:bg-red-700 text-white font-medium px-6 py-3 active:scale-[0.98] transition-all"
        >
          <LogOut className="w-5 h-5" />
          Выйти из системы
        </button>
      </main>
    </div>
  );
}
