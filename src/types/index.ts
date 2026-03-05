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
  clinicalCourse: string;       // Анамнез жизни
  allergyHistory: string;       // Аллергологический анамнез
  objectiveStatus: string;      // Объективные данные
  neurologicalStatus: string;   // Неврологический статус
  diagnosis: string;            // Предварительный диагноз (основной)
  conclusion: string;           // Сопутствующий диагноз
  recommendations: string;      // План лечения
  doctorNotes: string;          // Прочее
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
  complaints: 'Жалобы при поступлении',
  anamnesis: 'Анамнез заболевания',
  clinicalCourse: 'Анамнез жизни',
  allergyHistory: 'Аллергологический анамнез',
  objectiveStatus: 'Объективные данные',
  neurologicalStatus: 'Неврологический статус',
  diagnosis: 'Предварительный диагноз (основной)',
  conclusion: 'Сопутствующий диагноз',
  recommendations: 'План лечения',
  doctorNotes: 'Прочее',
};

export const patientFieldLabels: Record<keyof PatientInfo, string> = {
  fullName: 'ФИО пациента',
  age: 'Возраст',
  gender: 'Пол',
  complaintDate: 'Дата обращения',
};
