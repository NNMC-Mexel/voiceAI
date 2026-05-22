import { useEffect, useState } from 'react';
import { Lock, UserPlus } from 'lucide-react';
import { apiClient } from '../api/client';
import type { DoctorInfo } from '../api/client';

interface LoginScreenProps {
  onLogin: (doctor: DoctorInfo) => void;
}

export function LoginScreen({ onLogin }: LoginScreenProps) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [name, setName]         = useState('');
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [specialty, setSpecialty] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);
  const [setupRequired, setSetupRequired] = useState(false);

  useEffect(() => {
    let cancelled = false;
    apiClient.getSetupStatus()
      .then((status) => {
        if (cancelled) return;
        setSetupRequired(status.setupRequired);
        if (!status.setupRequired) setMode('login');
      })
      .catch(() => {
        if (!cancelled) setSetupRequired(false);
      });
    return () => { cancelled = true; };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) return;
    if (mode === 'register' && !name.trim()) return;
    if (mode === 'register' && !setupRequired) return;

    setLoading(true);
    setError('');

    try {
      if (mode === 'login') {
        const result = await apiClient.loginDoctor(email.trim(), password);
        if (result.success && result.doctor) {
          onLogin(result.doctor);
        } else {
          setError('Неверный email или пароль');
        }
      } else {
        const result = await apiClient.registerDoctor(name.trim(), email.trim(), password, specialty.trim());
        if (result.success && result.doctor) {
          onLogin(result.doctor);
        } else {
          setError('Ошибка регистрации');
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Ошибка соединения';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-6 sm:p-6">
      <div className="w-full max-w-sm mx-auto box-border">
        <div className="text-center mb-8 slide-up">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-medical-100 rounded-full mb-4">
            <Lock className="w-8 h-8 text-medical-600" />
          </div>
          <h1 className="text-2xl font-display font-bold text-medical-900 mb-2">МедДок</h1>
          <p className="text-text-secondary">
            {mode === 'login' ? 'Войдите в свой кабинет' : 'Создайте кабинет врача'}
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="glass-card rounded-2xl p-5 sm:p-6 slide-up box-border w-full space-y-3"
          style={{ animationDelay: '0.1s' }}
        >
          {mode === 'register' && (
            <>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="ФИО врача"
                autoFocus
                required
                className="block w-full box-border px-4 py-3 rounded-xl border border-slate-200 bg-white text-medical-900 placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-medical-400 focus:border-transparent"
              />
              <input
                type="text"
                value={specialty}
                onChange={(e) => setSpecialty(e.target.value)}
                placeholder="Специальность (необязательно)"
                className="block w-full box-border px-4 py-3 rounded-xl border border-slate-200 bg-white text-medical-900 placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-medical-400 focus:border-transparent"
              />
            </>
          )}

          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            autoFocus={mode === 'login'}
            required
            className="block w-full box-border px-4 py-3 rounded-xl border border-slate-200 bg-white text-medical-900 placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-medical-400 focus:border-transparent"
          />

          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={mode === 'register' ? 'Пароль (минимум 8 символов)' : 'Пароль'}
            required
            className="block w-full box-border px-4 py-3 rounded-xl border border-slate-200 bg-white text-medical-900 placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-medical-400 focus:border-transparent"
          />

          {error && <p className="text-red-600 text-sm">{error}</p>}

          <button
            type="submit"
            disabled={loading || !email.trim() || !password.trim() || (mode === 'register' && !name.trim())}
            className="btn-primary w-full py-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (mode === 'login' ? 'Вход...' : 'Регистрация...') : (mode === 'login' ? 'Войти' : 'Создать аккаунт')}
          </button>

          <button
            type="button"
            disabled={mode === 'login' && !setupRequired}
            onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(''); }}
            className="w-full text-center text-sm text-medical-600 hover:text-medical-700 py-1 disabled:cursor-default disabled:text-text-muted"
          >
            {mode === 'login'
              ? setupRequired
                ? <><UserPlus className="inline w-4 h-4 mr-1" />Первый вход? Создать аккаунт</>
                : 'Вход только для добавленных врачей'
              : 'Уже есть аккаунт? Войти'}
          </button>
        </form>
      </div>
    </div>
  );
}
