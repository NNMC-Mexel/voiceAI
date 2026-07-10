// HTTP-маршруты движка лучевой диагностики (structured reporting).
// Stateless: клиент присылает шаблон + список голосовых команд сессии,
// сервер прогоняет их через свежий движок и возвращает собранный протокол.
// Такой replay-контракт совпадает с потоком чанков whisper: накопили команды —
// переслали — получили актуальный протокол; идемпотентно и переживает reconnect.

import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { RadiologyEngine, getTemplate, templates } from './radiology/index.js';
import { DocEngine } from './radiology/doc-engine.js';
import { docTemplates, getDocTemplate } from './radiology/doc-registry.js';
import { buildHints } from './radiology/doc-hints.js';

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

// Разбить строку диктовки на отдельные команды («жидкости нет. газа нет.»)
function splitCommands(line: string): string[] {
  return line.split(/[.\n;]+/).map((s) => s.trim()).filter(Boolean);
}

export function registerRadiologyRoutes(fastify: FastifyInstance): void {
  // ─── Fill-in движок (рабочие шаблоны Михайлова) ───────────────────────────
  fastify.get('/api/radiology/doc-templates', async () => ({
    templates: docTemplates.map((t) => ({ id: t.id, name: t.name, modality: t.modality, title: t.title })),
  }));

  // Подсказки «что можно диктовать» по шаблону (примеры команд из конфига).
  fastify.get('/api/radiology/doc-templates/:id/hints', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const tpl = getDocTemplate(id);
    if (!tpl) return reply.status(404).send({ error: 'Шаблон не найден' });
    return { hints: buildHints(tpl) };
  });

  fastify.post('/api/radiology/doc-build', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Некорректное тело запроса' });
    const templateId = typeof body.templateId === 'string' ? body.templateId : '';
    const commands = Array.isArray(body.commands)
      ? body.commands.filter((c): c is string => typeof c === 'string') : [];

    const tpl = getDocTemplate(templateId);
    if (!tpl) return reply.status(404).send({ error: `Неизвестный шаблон: ${templateId}` });

    const engine = new DocEngine(tpl);
    const applied: { command: string; ok: boolean; action: string; blockId?: string }[] = [];
    for (const line of commands) {
      for (const cmd of splitCommands(line)) {
        const r = engine.apply(cmd);
        applied.push({ command: cmd, ok: r.ok, action: r.action, blockId: r.blockId });
      }
    }
    return { report: engine.build(), applied };
  });

  // Список доступных шаблонов лучевой диагностики (для экрана выбора).
  fastify.get('/api/radiology/templates', async () => ({
    templates: templates.map((t) => ({
      id: t.id, name: t.name, modality: t.modality, aliases: t.aliases,
    })),
  }));

  // Структура секций шаблона (для превью/каркаса фронта).
  fastify.get('/api/radiology/templates/:id', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const tpl = getTemplate(id);
    if (!tpl) return reply.status(404).send({ error: 'Шаблон не найден' });
    return {
      id: tpl.id, name: tpl.name, modality: tpl.modality,
      sections: tpl.sections.map((s) => ({ id: s.id, organ: s.organ })),
    };
  });

  // Прогнать команды сессии и собрать протокол.
  fastify.post('/api/radiology/build', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) return reply.status(400).send({ error: 'Некорректное тело запроса' });

    const templateId = typeof body.templateId === 'string' ? body.templateId : '';
    const commands = Array.isArray(body.commands)
      ? body.commands.filter((c): c is string => typeof c === 'string')
      : [];

    const tpl = getTemplate(templateId);
    if (!tpl) return reply.status(404).send({ error: `Неизвестный шаблон: ${templateId}` });

    const engine = new RadiologyEngine(tpl);
    // Разбиваем каждую строку на отдельные фразы-команды (по точкам/переносам),
    // т.к. врач диктует потоком: «жидкости нет. газа нет.»
    const applied: { command: string; ok: boolean; handled: string; warnings?: string[] }[] = [];
    for (const line of commands) {
      for (const cmd of line.split(/[.\n;]+/).map((s) => s.trim()).filter(Boolean)) {
        const r = engine.apply(cmd);
        applied.push({ command: cmd, ok: r.ok, handled: r.handled, warnings: r.warnings });
      }
    }

    const report = engine.build();
    return { report, applied };
  });
}
