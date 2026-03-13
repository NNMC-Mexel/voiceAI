export interface PatientInfo {
  fullName: string;
  age: string;
  gender: string;
  complaintDate: string;
}

export interface RiskAssessment {
  fallInLast3Months: string;   // "да" | "нет"
  dizzinessOrWeakness: string; // "да" | "нет"
  needsEscort: string;         // "да" | "нет"
  painScore: string;           // "0б", "1б", ... "10б"
}

export interface MedicalDocument {
  patient: PatientInfo;
  riskAssessment: RiskAssessment;
  complaints: string;
  anamnesis: string;
  outpatientExams: string;      // Амбулаторные обследования
  clinicalCourse: string;       // Перенесённые заболевания
  allergyHistory: string;       // Аллергологический анамнез
  objectiveStatus: string;      // Объективный статус
  neurologicalStatus: string;   // Неврологический статус
  diagnosis: string;            // Предварительный диагноз
  finalDiagnosis: string;       // Заключительный диагноз
  conclusion: string;           // Амбулаторная терапия
  doctorNotes: string;          // План обследования
  recommendations: string;      // Рекомендации / План лечения
  diet: string;                 // Диета
}

export type AppStep = 'recording' | 'processing' | 'editing' | 'preview';

export interface RecordingState {
  isRecording: boolean;
  isPaused: boolean;
  duration: number;
  audioBlob: Blob | null;
}

export const emptyRiskAssessment: RiskAssessment = {
  fallInLast3Months: 'нет',
  dizzinessOrWeakness: 'нет',
  needsEscort: 'нет',
  painScore: '0',
};

export const emptyDocument: MedicalDocument = {
  patient: {
    fullName: '',
    age: '',
    gender: '',
    complaintDate: new Date().toISOString().split('T')[0],
  },
  riskAssessment: { ...emptyRiskAssessment },
  complaints: '',
  anamnesis: '',
  outpatientExams: '',
  clinicalCourse: '',
  allergyHistory: '',
  objectiveStatus: '',
  neurologicalStatus: '',
  diagnosis: '',
  finalDiagnosis: '',
  conclusion: '',
  doctorNotes: '',
  recommendations: '',
  diet: '',
};

export const fieldLabels: Record<keyof Omit<MedicalDocument, 'patient' | 'riskAssessment'>, string> = {
  complaints: 'Жалобы',
  anamnesis: 'Анамнез заболевания',
  outpatientExams: 'Амбулаторные обследования',
  clinicalCourse: 'Анамнез жизни',
  allergyHistory: 'Аллергологический анамнез',
  objectiveStatus: 'Объективный статус',
  neurologicalStatus: 'Неврологический статус',
  diagnosis: 'Предварительный диагноз',
  finalDiagnosis: 'Заключительный диагноз',
  doctorNotes: 'План обследования',
  recommendations: 'Рекомендации / План лечения',
  conclusion: 'Амбулаторная терапия',
  diet: 'Диета',
};

export const patientFieldLabels: Record<keyof PatientInfo, string> = {
  fullName: 'ФИО пациента',
  age: 'Возраст',
  gender: 'Пол',
  complaintDate: 'Дата обращения',
};

export const riskAssessmentLabels: Record<keyof RiskAssessment, string> = {
  fallInLast3Months: 'Падал ли в последние 3 месяца',
  dizzinessOrWeakness: 'Головокружение или слабость',
  needsEscort: 'Нужно сопровождение',
  painScore: 'Оценка боли',
};
