import { useState } from 'react';
import { Lock } from 'lucide-react';
import { apiClient } from '../api/client';

interface LoginScreenProps {
  onLogin: () => void;
}

export function LoginScreen({ onLogin }: LoginScreenProps) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password.trim()) return;

    setLoading(true);
    setError('');

    try {
      const ok = await apiClient.login(password);
      if (ok) {
        onLogin();
      } else {
        setError('Неверный пароль');
      }
    } catch {
      setError('Ошибка соединения с сервером');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8 slide-up">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-medical-100 rounded-full mb-4">
            <Lock className="w-8 h-8 text-medical-600" />
          </div>
          <h1 className="text-2xl font-display font-bold text-medical-900 mb-2">МедДок</h1>
          <p className="text-text-secondary">Введите пароль для входа</p>
        </div>

        <form onSubmit={handleSubmit} className="glass-card rounded-2xl p-6 slide-up" style={{ animationDelay: '0.1s' }}>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Пароль"
            autoFocus
            className="w-full px-4 py-3 rounded-xl border border-slate-200 bg-white text-medical-900 placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-medical-400 focus:border-transparent mb-4"
          />

          {error && (
            <p className="text-red-600 text-sm mb-4">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading || !password.trim()}
            className="btn-primary w-full py-3 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Вход...' : 'Войти'}
          </button>
        </form>
      </div>
    </div>
  );
}
