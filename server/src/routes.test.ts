import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveUploadPath, toSafeUploadFilename, isValidMedicalDocument } from './routes.ts';

test('toSafeUploadFilename removes unsafe chars', () => {
  assert.equal(toSafeUploadFilename('../../bad name?.webm'), 'bad_name_.webm');
});

test('resolveUploadPath blocks traversal outside upload root', () => {
  assert.throws(() => resolveUploadPath('./uploads', '../secret.txt'));
});

test('resolveUploadPath allows valid file path', () => {
  const resolved = resolveUploadPath('./uploads', '1700000_audio.webm');
  assert.match(resolved, /uploads/);
});

test('isValidMedicalDocument validates minimal shape', () => {
  const valid = {
    patient: {
      fullName: '',
      age: '',
      gender: '',
      complaintDate: '',
    },
    complaints: '',
    anamnesis: '',
    objectiveStatus: '',
    diagnosis: '',
    clinicalCourse: '',
    conclusion: '',
    recommendations: '',
    doctorNotes: '',
  };

  assert.equal(isValidMedicalDocument(valid), true);
  assert.equal(isValidMedicalDocument({}), false);
});
