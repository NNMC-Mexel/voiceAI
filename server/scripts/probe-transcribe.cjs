#!/usr/bin/env node
/**
 * Прямая проверка /api/transcribe на уже загруженном webm/mp3 в uploads/.
 * Цель: воспроизвести сценарий пользователя и узнать реальные duration/chars/chunks
 * без участия фронтенда.
 */
const SERVER = process.env.SERVER_URL || 'http://localhost:3001';
const AUTH_PASS = process.env.AUTH_PASS || 'meddok2026';
const FILENAME = process.argv[2] || '1776653548591_recording.webm';

async function main() {
  const loginResp = await fetch(`${SERVER}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password: AUTH_PASS }),
  });
  const { token } = await loginResp.json();

  console.log(`[probe] POST /api/transcribe filename=${FILENAME}`);
  const start = Date.now();
  const resp = await fetch(`${SERVER}/api/transcribe`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ filename: FILENAME }),
  });
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);

  if (!resp.ok) {
    console.error(`FAIL ${resp.status}: ${await resp.text()}`);
    process.exit(1);
  }
  const data = await resp.json();
  console.log(`\n[probe] elapsed total:  ${elapsed}s`);
  console.log(`[probe] whisper duration field: ${data.duration}`);
  console.log(`[probe] language:        ${data.language}`);
  console.log(`[probe] text chars:      ${data.text?.length || 0}`);
  console.log(`\n--- FIRST 300 CHARS ---\n${(data.text || '').slice(0, 300)}`);
  console.log(`\n--- LAST 300 CHARS ---\n${(data.text || '').slice(-300)}`);
}

main().catch((e) => { console.error('Fatal:', e); process.exit(1); });
