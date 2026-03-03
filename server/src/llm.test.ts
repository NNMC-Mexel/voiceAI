import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LLMService } from './services/llm.ts';
import type { LLMConfig, MedicalDocument } from './types.js';

const config: LLMConfig = {
  serverUrl: 'http://127.0.0.1:65535',
  model: 'test-model',
  maxTokens: 128,
  temperature: 0,
  parallelSlots: 1,
  requestTimeoutMs: 100,
  allowMockOnFailure: false,
};

const baseDocument: MedicalDocument = {
  patient: {
    fullName: '',
    age: '',
    gender: '',
    complaintDate: '',
  },
  complaints: 'Боль в пояснице.',
  anamnesis: '',
  objectiveStatus: '',
  diagnosis: '',
  clinicalCourse: '',
  conclusion: '',
  recommendations: '',
  doctorNotes: '',
};

test('applyInstruction maps "жалобы ..." command to complaints section', async () => {
  const service = new LLMService(config);
  const result = await service.applyInstruction(baseDocument, 'жалобы температура 38,8, озноб');

  assert.match(result.complaints, /температура 38,8, озноб/i);
  assert.equal(result.doctorNotes, '');
});

test('applyInstruction maps "заметки врача ..." command to doctorNotes section', async () => {
  const service = new LLMService(config);
  const result = await service.applyInstruction(baseDocument, 'заметки врача пациент отказался от госпитализации');

  assert.equal(result.complaints, baseDocument.complaints);
  assert.match(result.doctorNotes, /отказался от госпитализации/i);
});

test('addendum fallback routes "жалобы ..." text to complaints instead of doctorNotes', () => {
  const service = new LLMService(config) as any;
  const emptyComplaints: MedicalDocument = { ...baseDocument, complaints: '' };
  const result = service.mergeDocumentWithPatch(emptyComplaints, {}, 'Жалобы боли в пояснице, температура 38,8.');

  assert.match(result.complaints, /боли в пояснице, температура 38,8/i);
  assert.equal(result.doctorNotes, '');
});
