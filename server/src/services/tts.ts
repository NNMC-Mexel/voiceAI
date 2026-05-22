import type { TtsConfig } from '../types.js';

export class TtsService {
  constructor(private config: TtsConfig) {}

  get isEnabled(): boolean {
    return this.config.enabled && !!this.config.serverUrl;
  }

  async healthCheck(): Promise<boolean> {
    if (!this.isEnabled) return false;
    try {
      const response = await fetch(`${this.config.serverUrl}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async synthesize(text: string, language = 'ru'): Promise<string> {
    if (!this.isEnabled) {
      throw new Error('TTS is not enabled or TTS_SERVER_URL is not set');
    }

    const response = await fetch(`${this.config.serverUrl}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language }),
      signal: AbortSignal.timeout(60000),
    });

    if (!response.ok) {
      let msg = `TTS server error: ${response.status}`;
      try {
        const text = await response.text();
        if (text) {
          try {
            const err = JSON.parse(text) as { error?: unknown; message?: unknown };
            const detail = typeof err.error === 'string' && err.error.trim()
              ? err.error.trim()
              : typeof err.message === 'string' && err.message.trim()
                ? err.message.trim()
                : '';
            if (detail) msg = `TTS error: ${detail}`;
          } catch {
            msg = text;
          }
        }
      } catch { /* ignore */ }
      throw new Error(msg);
    }

    const data = await response.json() as { audio_base64?: string };
    if (!data.audio_base64) {
      throw new Error('TTS response does not contain audio_base64');
    }
    return data.audio_base64;
  }
}
