# ─────────────────────────────────────────────
# Stage 1: Build React frontend (Vite)
# ─────────────────────────────────────────────
FROM node:22-slim AS frontend-builder

WORKDIR /frontend

COPY package*.json ./
# NODE_ENV=development чтобы npm ci установил devDependencies (vite, typescript и т.д.)
RUN NODE_ENV=development npm ci

COPY . .

# VITE_API_URL пустой — фронт и бэк на одном домене (same-origin)
RUN npm run build
# Результат: /frontend/dist/


# ─────────────────────────────────────────────
# Stage 2: Build Node.js backend (TypeScript)
# ─────────────────────────────────────────────
FROM node:22-slim AS backend-builder

WORKDIR /backend

COPY server/package*.json ./
# NODE_ENV=development чтобы npm ci установил devDependencies (typescript/tsc и т.д.)
RUN NODE_ENV=development npm ci

COPY server/ ./
RUN npm run build
# Результат: /backend/dist/


# ─────────────────────────────────────────────
# Stage 3: Production runtime
# ─────────────────────────────────────────────
FROM node:22-slim

WORKDIR /app

# Backend
COPY --from=backend-builder /backend/dist         ./dist
COPY --from=backend-builder /backend/node_modules ./node_modules
COPY --from=backend-builder /backend/package.json ./

# Frontend статика — раздаётся backend-ом через fastifyStatic
COPY --from=frontend-builder /frontend/dist       ./public

RUN mkdir -p uploads temp

ENV NODE_ENV=production
ENV PORT=3001
ENV HOST=0.0.0.0
# Путь к frontend-сборке — читается в server/src/index.ts
ENV STATIC_DIR=/app/public

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD node -e "fetch('http://localhost:3001/api/health').then(r=>r.ok?process.exit(0):process.exit(1)).catch(()=>process.exit(1))"

CMD ["node", "dist/index.js"]
