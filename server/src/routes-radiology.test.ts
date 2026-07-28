// HTTP-контракт маршрутов лучевой диагностики (через fastify.inject, без БД/авторизации).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import Fastify from 'fastify';
import { registerRadiologyRoutes } from './routes-radiology.js';

function app(enableLegacyPipelines?: boolean) {
  const f = Fastify();
  registerRadiologyRoutes(f, {
    ...(enableLegacyPipelines === undefined ? {} : { enableLegacyPipelines }),
  });
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

test('production gate disables all three legacy radiology write pipelines', async () => {
  const f = app(false);
  for (const url of [
    '/api/radiology/build',
    '/api/radiology/doc-build',
    '/api/radiology/structure',
  ]) {
    const response = await f.inject({
      method: 'POST',
      url,
      payload: {},
    });
    assert.equal(response.statusCode, 410, url);
    assert.equal(response.json().error, 'legacy_radiology_pipeline_disabled');
    assert.equal(response.headers['cache-control'], 'no-store');
  }
  await f.close();
});

test('GET template preview exposes defaults without enabling legacy writes', async () => {
  const f = app(false);
  const res = await f.inject({
    method: 'GET',
    url: '/api/radiology/doc-templates/CT_ABDOMEN_MIKHAILOV/preview',
  });
  assert.equal(res.statusCode, 200);
  const { preview } = res.json();
  assert.equal(preview.templateId, 'CT_ABDOMEN_MIKHAILOV');
  assert.ok(preview.title);
  assert.ok(preview.blocks.length > 0);
  assert.ok(
    preview.blocks.every(
      (block: { origin: string }) => block.origin === 'template_default',
    ),
  );
  assert.ok(preview.blocks.some((block: { id: string }) => block.id === 'liver'));
  assert.match(preview.text, /Печень/u);
  await f.close();
});

test('GET template preview returns 404 for an unknown template', async () => {
  const f = app();
  const res = await f.inject({
    method: 'GET',
    url: '/api/radiology/doc-templates/NOPE/preview',
  });
  assert.equal(res.statusCode, 404);
  await f.close();
});
