import { useState, useEffect, useCallback, useRef } from 'react';
import apiClient from '../api/client';
import type { HealthStatus } from '../api/client';

export interface ServiceHealth {
  /** null — статус ещё не известен (первый запрос в полёте) */
  health: HealthStatus | null;
  /** true — бэкенд вообще не отвечает (сеть/сервер лежит) */
  backendDown: boolean;
  /** Принудительная перепроверка (например, по кнопке «Проверить снова») */
  refresh: () => void;
}

const POLL_INTERVAL_MS = 30_000;

/**
 * Pre-flight мониторинг ИИ-сервисов (ASR + LLM).
 *
 * Зачем: без него врач узнаёт о недоступности LLM только ПОСЛЕ того, как
 * надиктовал приём и получил 500 на структурировании — потерянная запись
 * и потерянное время. Хук позволяет показать предупреждение ДО записи.
 *
 * Поведение: опрос /api/health каждые 30с + немедленная перепроверка при
 * возврате фокуса окна (типовой кейс: сервер перезапустили, врач вернулся
 * к вкладке — статус актуализируется сразу, а не через полминуты).
 */
export function useServiceHealth(enabled: boolean = true): ServiceHealth {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  // Защита от гонки: устаревший медленный ответ не должен затирать свежий
  const requestSeq = useRef(0);

  const check = useCallback(async () => {
    const seq = ++requestSeq.current;
    try {
      const result = await apiClient.healthCheck();
      if (seq !== requestSeq.current) return;
      setHealth(result);
      setBackendDown(false);
    } catch {
      if (seq !== requestSeq.current) return;
      setHealth(null);
      setBackendDown(true);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    void check();
    const interval = window.setInterval(() => void check(), POLL_INTERVAL_MS);

    const onFocus = () => void check();
    window.addEventListener('focus', onFocus);

    return () => {
      window.clearInterval(interval);
      window.removeEventListener('focus', onFocus);
    };
  }, [enabled, check]);

  return { health, backendDown, refresh: check };
}
