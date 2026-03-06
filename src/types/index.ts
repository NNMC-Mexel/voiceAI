export interface PatientInfo {
  fullName: string;
  age: string;
  gender: string;
  complaintDate: string;
}

export interface MedicalDocument {
  patient: PatientInfo;
  complaints: string;
  anamnesis: string;
  clinicalCourse: string;       // Перенесённые заболевания
  allergyHistory: string;       // Аллергологический анамнез
  objectiveStatus: string;      // Объективный статус
  neurologicalStatus: string;   // Неврологический статус
  diagnosis: string;            // Диагноз
  conclusion: string;           // Амбулаторная терапия
  recommendations: string;      // Рекомендации / План лечения
  doctorNotes: string;          // План обследования
}

export type AppStep = 'recording' | 'processing' | 'editing' | 'preview';

export interface RecordingState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  audioBlob: Blob | null;
}

export const emptyDocument: MedicalDocument = {
  patient: {
    fullName: '',
    age: '',
    gender: '',
    complaintDate: new Date().toISOString().split('T')[0],
  },
  complaints: '',
  anamnesis: '',
  clinicalCourse: '',
  allergyHistory: '',
  objectiveStatus: '',
  neurologicalStatus: '',
  diagnosis: '',
  conclusion: '',
  recommendations: '',
  doctorNotes: '',
};

export const fieldLabels: Record<keyof Omit<MedicalDocument, 'patient'>, string> = {
  complaints: 'Жалобы',
  anamnesis: 'Анамнез заболевания',
  clinicalCourse: 'Перенесённые заболевания',
  allergyHistory: 'Аллергологический анамнез',
  objectiveStatus: 'Объективный статус',
  neurologicalStatus: 'Неврологический статус',
  diagnosis: 'Диагноз',
  conclusion: 'Амбулаторная терапия',
  recommendations: 'Рекомендации / План лечения',
  doctorNotes: 'План обследования',
};

export const patientFieldLabels: Record<keyof PatientInfo, string> = {
  fullName: 'ФИО пациента',
  age: 'Возраст',
  gender: 'Пол',
  complaintDate: 'Дата обращения',
};
