// HTTP-контракт маршрутов лучевой диагностики (через fastify.inject, без БД/авторизации).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import Fastify from 'fastify';
import { registerRadiologyRoutes } from './routes-radiology.js';

function app() {
  const f = Fastify();
  registerRadiologyRoutes(f);
  return f;
}

test('GET /api/radiology/templates возвращает КТ ОБП', async () => {
  const f = app();
  const res = await f.inject({ method: 'GET', url: '/api/radiology/templates' });
  assert.equal(res.statusCode, 200);
  const body = res.json();
  assert.ok(body.templates.some((t: { id: string }) => t.id === 'CT_ABDOMEN'));
  await f.close();
});

test('GET /api/radiology/templates/:id отдаёт секции', async () => {
  const f = app();
  const res = await f.inject({ method: 'GET', url: '/api/radiology/templates/CT_ABDOMEN' });
  assert.equal(res.statusCode, 200);
  const body = res.json();
  assert.ok(body.sections.some((s: { id: string }) => s.id === 'liver'));
  await f.close();
});

test('POST /api/radiology/build собирает протокол из сессии команд', async () => {
  const f = app();
  const res = await f.inject({
    method: 'POST', url: '/api/radiology/build',
    payload: {
      templateId: 'CT_ABDOMEN',
      commands: [
        'ОБП контраст. Печень плотность 56 норма.',
        'Селезёнка размеры 14 7 5.',
        'Правая почка Босняк 1 18 мм.',
        'Жидкости нет. Газа нет.',
        'остальное норма',
      ],
    },
  });
  assert.equal(res.statusCode, 200);
  const { report, applied } = res.json();
  assert.match(report.description, /Средняя плотность паренхимы печени.*56 HU/);
  assert.match(report.conclusion, /спленомегалии/);
  assert.match(report.conclusion, /Простая киста правой почки Bosniak I/);
  // «ОБП контраст» распознано как техника, «печень…» как секция
  assert.ok(applied.some((a: { command: string; handled: string }) => a.command.includes('контраст') && a.handled === 'technique'));
  await f.close();
});

test('POST /api/radiology/build с неизвестным шаблоном → 404', async () => {
  const f = app();
  const res = await f.inject({
    method: 'POST', url: '/api/radiology/build',
    payload: { templateId: 'NOPE', commands: [] },
  });
  assert.equal(res.statusCode, 404);
  await f.close();
});
