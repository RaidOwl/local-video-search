import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In Docker dev the API lives at http://local-vss-api:8000; on a bare host it's
// typically http://localhost:8000. The nginx production config strips the /api
// prefix before proxying, so the dev proxy does the same to stay consistent.
const apiTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // FastAPI's interactive docs live at the API root. Proxy them in dev so the
      // "API" nav link (/docs) works just like it does behind nginx in production.
      '^/(docs|redoc|openapi.json)': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})