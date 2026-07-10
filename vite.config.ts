import { defineConfig } from 'vite'
import { readFileSync } from 'node:fs'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'

// Порт бэкенда берём из server/.env, чтобы прокси dev-сервера всегда совпадал
// с реальным портом API (иначе /api уходит в никуда → пустой ответ на фронте).
function backendPort(): string {
  try {
    const env = readFileSync('./server/.env', 'utf8')
    const m = env.match(/^\s*PORT\s*=\s*(\d+)/m)
    if (m) return m[1]
  } catch { /* нет server/.env — используем дефолт */ }
  return '1337'
}

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || `http://localhost:${backendPort()}`

export default defineConfig({
  plugins: [react(), tailwindcss(), basicSsl()],
  server: {
    host: '0.0.0.0',
    https: {},
    port: 5173,
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
