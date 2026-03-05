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
        const err = await response.json() as { error?: string };
        if (err?.error) msg = `TTS error: ${err.error}`;
      } catch { /* ignore */ }
      throw new Error(msg);
    }

    const data = await response.json() as { audio_base64: string };
    return data.audio_base64;
  }
}
