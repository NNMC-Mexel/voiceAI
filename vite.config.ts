import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:3001'

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
