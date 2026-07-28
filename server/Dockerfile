# Build stage
FROM node:22-slim AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

# Build TypeScript
RUN npm run build

# Production stage
FROM node:22-slim

WORKDIR /app

# The production API is an orchestrator. ASR runs only in the pinned GigaAM
# service; no Python/Whisper fallback is installed in this image.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy built files
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

# Create directories
RUN mkdir -p uploads temp models

# Environment
ENV NODE_ENV=production
ENV PORT=1337

EXPOSE 1337

CMD ["node", "dist/index.js"]
