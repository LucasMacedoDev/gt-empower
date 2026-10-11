import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dentro do Docker, "localhost" aponta para o contêiner do frontend.
// O serviço FastAPI é acessível pelo nome da rede Compose: "backend".
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://backend:8000',
        changeOrigin: true,
      },
    },
  },
});
