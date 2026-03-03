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
  objectiveStatus: string;
  diagnosis: string;
  clinicalCourse: string;
  conclusion: string;
  recommendations: string;
  doctorNotes: string;
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
  objectiveStatus: '',
  diagnosis: '',
  clinicalCourse: '',
  conclusion: '',
  recommendations: '',
  doctorNotes: '',
};

export const fieldLabels: Record<keyof Omit<MedicalDocument, 'patient'>, string> = {
  complaints: 'Жалобы',
  anamnesis: 'Анамнез заболевания',
  objectiveStatus: 'Объективный статус',
  diagnosis: 'Диагноз',
  clinicalCourse: 'Клиническое течение',
  conclusion: 'Заключение',
  recommendations: 'Рекомендации',
  doctorNotes: 'Заметки врача',
};

export const patientFieldLabels: Record<keyof PatientInfo, string> = {
  fullName: 'ФИО пациента',
  age: 'Возраст',
  gender: 'Пол',
  complaintDate: 'Дата обращения',
};
