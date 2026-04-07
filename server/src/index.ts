import 'dotenv/config';
import Fastify from 'fastify';
import cors from '@fastify/cors';
import multipart from '@fastify/multipart';
import fastifyStatic from '@fastify/static';
import path from 'path';
import { fileURLToPath } from 'url';

import { loadConfig } from './config.js';
import { registerRoutes } from './routes.js';
import { WhisperService } from './services/whisper.js';
import { LLMService } from './services/llm.js';
import { TtsService } from './services/tts.js';
import { loadUserCorrections } from './services/medical-dictionary.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function isDevLocalOrigin(origin: string): boolean {
  let url: URL;
  try {
    url = new URL(origin);
  } catch {
    return false;
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;

  const host = url.hostname;
  if (host === 'localhost' || host === '127.0.0.1') return true;

  if (/^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host)) return true;
  if (/^192\.168\.\d{1,3}\.\d{1,3}$/.test(host)) return true;

  const m = host.match(/^172\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/);
  if (m) {
    const secondOctet = Number.parseInt(m[1], 10);
    return secondOctet >= 16 && secondOctet <= 31;
  }

  return false;
}

async function main() {
  const config = loadConfig();
  const isProduction = process.env.NODE_ENV === 'production';

  const fastify = Fastify({
    // За Coolify/Nginx/CloudFlare берём реальный IP клиента из X-Forwarded-For,
    // иначе rate-limit/brute-force защита считают всех одним IP прокси.
    trustProxy: true,
    logger: {
      level: 'info',
      transport: {
        target: 'pino-pretty',
        options: {
          translateTime: 'HH:MM:ss Z',
          ignore: 'pid,hostname',
        },
      },
    },
  });

  await fastify.register(cors, {
    origin: (origin, cb) => {
      if (!origin) return cb(null, true);
      if (config.security.corsOrigins.includes(origin)) return cb(null, true);
      if (!isProduction && isDevLocalOrigin(origin)) return cb(null, true);
      return cb(new Error('CORS blocked'), false);
    },
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  });

  await fastify.register(multipart, {
    limits: {
      fileSize: 100 * 1024 * 1024,
    },
  });

  // STATIC_DIR задаётся через env в продакшене (Coolify),
  // по умолчанию ищем ../../dist (локальная разработка)
  const staticPath = process.env.STATIC_DIR || path.join(__dirname, '../../dist');
  try {
    await fastify.register(fastifyStatic, {
      root: staticPath,
      prefix: '/',
    });
    fastify.log.info(`Serving frontend from: ${staticPath}`);
  } catch {
    fastify.log.info('No static files to serve (frontend not built yet)');
  }

  // Загружаем пользовательские замены медицинского словаря
  await loadUserCorrections();

  const whisperService = new WhisperService(config.whisper);
  await whisperService.init();

  const llmService = new LLMService(config.llm);
  const ttsService = new TtsService(config.tts);

  await registerRoutes(fastify, config, whisperService, llmService, ttsService);

  try {
    await fastify.listen({
      port: config.port,
      host: config.host,
    });

    fastify.log.info(`Server started on http://${config.host}:${config.port}`);
    fastify.log.info('Health endpoint: GET /api/health');
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

main();
