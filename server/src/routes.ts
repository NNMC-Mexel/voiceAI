import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { pipeline } from 'stream/promises';
import { createWriteStream } from 'fs';
import { mkdir, unlink, appendFile } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { WhisperService } from './services/whisper.js';
import { LLMService } from './services/llm.js';
import { TtsService } from './services/tts.js';
import { DocumentExtractorService } from './services/document-extractor.js';
import type { AppDb } from './db/index.js';
import { doctors } from './db/schema.js';
import { registerDoctorRoutes } from './routes-doctors.js';
import { registerRadiologyRoutes } from './routes-radiology.js';
import { eq } from 'drizzle-orm';
import {
  getUserCorrections,
  addUserCorrection,
  deleteUserCorrection,
} from './services/medical-dictionary.js';
import type { UserCorrectionScope } from './services/medical-dictionary.js';
import type { ServerConfig, MedicalDocument, QualityWarning } from './types.js';

interface RateState {
  count: number;
  windowStartedAt: number;
}

const rewriteableFields = [
  'complaints',
  'anamnesis',
  'clinicalCourse',
  'allergyHistory',
  'objectiveStatus',
  'neurologicalStatus',
  'diagnosis',
  'finalDiagnosis',
  'conclusion',
  'recommendations',
  'doctorNotes',
  'outpatientExams',
] as const;

type RewriteableField = (typeof rewriteableFields)[number];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null;
}

function normalizeCorrectionScope(scope: string | undefined): UserCorrectionScope {
  return scope === 'medications' || scope === 'exams' || scope === 'neurological'
    ? scope
    : 'global';
}

export function toSafeUploadFilename(originalName: string): string {
  const base = path.basename(originalName);
  const sanitized = base.replace(/[^A-Za-z0-9._-]/g, '_');
  return sanitized || 'audio.webm';
}

export function resolveUploadPath(uploadDir: string, filename: string): string {
  const uploadRoot = path.resolve(uploadDir);
  const candidate = path.resolve(uploadRoot, filename);
  const normalizedRoot = uploadRoot.endsWith(path.sep) ? uploadRoot : `${uploadRoot}${path.sep}`;

  if (!candidate.startsWith(normalizedRoot)) {
    throw new Error('Invalid file path');
  }

  return candidate;
}

function toCleanOcrLines(rawText: string): string[] {
  return rawText
    .replace(/\r/g, '\n')
    .split('\n')
    .map((line) => line.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
}

function parseDmyDate(value: string): Date | null {
  const match = value.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (!match) return null;

  const [, dd, mm, yyyy] = match;
  const date = new Date(Number(yyyy), Number(mm) - 1, Number(dd));
  if (
    date.getFullYear() !== Number(yyyy) ||
    date.getMonth() !== Number(mm) - 1 ||
    date.getDate() !== Number(dd)
  ) {
    return null;
  }
  return date;
}

function dmyToIso(value: string): string {
  if (!parseDmyDate(value)) return '';
  const [, dd, mm, yyyy] = value.match(/^(\d{2})\.(\d{2})\.(\d{4})$/) || [];
  return yyyy && mm && dd ? `${yyyy}-${mm}-${dd}` : '';
}

function calculateAgeYears(birthDate: Date, atDate: Date): string {
  let age = atDate.getFullYear() - birthDate.getFullYear();
  const beforeBirthday =
    atDate.getMonth() < birthDate.getMonth() ||
    (atDate.getMonth() === birthDate.getMonth() && atDate.getDate() < birthDate.getDate());
  if (beforeBirthday) age -= 1;
  return age >= 0 && age < 130 ? `${age} лет` : '';
}

function extractDocumentDateIso(rawText: string): string {
  const patterns = [
    /Дата\s+взятия:\s*(\d{2}\.\d{2}\.\d{4})/iu,
    /Заказ\s*№?\S*\s+от\s*(\d{2}\.\d{2}\.\d{4})/iu,
    /Выполнено:\s*(\d{2}\.\d{2}\.\d{4})/iu,
  ];

  for (const pattern of patterns) {
    const match = rawText.match(pattern);
    if (match?.[1]) {
      const iso = dmyToIso(match[1]);
      if (iso) return iso;
    }
  }
  return '';
}

function looksLikePatientNameLine(line: string): boolean {
  const clean = line.replace(/\([^)]*\)/g, '').trim();
  if (!/^[А-ЯЁA-Z][А-ЯЁA-Z'’` -]+$/u.test(clean)) return false;

  const words = clean.split(/\s+/).filter(Boolean);
  if (words.length < 2 || words.length > 4) return false;
  return !/(?:МИНИСТЕРСТВО|MINISTRY|ORGANIZATION|ОРГАНИЗАЦ|НАИМЕНОВАНИЕ|ТОО|АО|HEALTH|CENTER|MEDICAL|ВРАЧ|КАЗАХСТАН|РЕСПУБЛИК)/iu.test(clean);
}

function extractPatientFromExactText(rawText: string): MedicalDocument['patient'] {
  const lines = toCleanOcrLines(rawText);
  const birthMatch = rawText.match(/Дата\s+рождения:\s*(\d{2}\.\d{2}\.\d{4})/iu);
  const birthDate = birthMatch?.[1] ? parseDmyDate(birthMatch[1]) : null;
  const documentDateIso = extractDocumentDateIso(rawText);
  const documentDate = documentDateIso ? new Date(`${documentDateIso}T00:00:00`) : new Date();

  let fullName = '';
  let gender = '';

  const birthLineIndex = lines.findIndex((line) => /Дата\s+рождения:/iu.test(line));
  const candidates = birthLineIndex >= 0
    ? lines.slice(Math.max(0, birthLineIndex - 6), birthLineIndex).reverse()
    : lines;

  for (const line of candidates) {
    if (!looksLikePatientNameLine(line)) continue;

    const genderMatch = line.match(/\((муж|жен|м|ж|male|female)\)/iu);
    if (genderMatch) {
      const value = genderMatch[1].toLowerCase();
      gender = value.startsWith('м') || value === 'male' ? 'мужской' : 'женский';
    }

    fullName = line.replace(/\([^)]*\)/g, '').replace(/\s+/g, ' ').trim();
    break;
  }

  const explicitGender = rawText.match(/(?:Пол|Gender):\s*(мужской|женский|муж|жен|male|female)/iu);
  if (!gender && explicitGender?.[1]) {
    const value = explicitGender[1].toLowerCase();
    gender = value.startsWith('м') || value === 'male' ? 'мужской' : 'женский';
  }

  return {
    fullName,
    age: birthDate ? calculateAgeYears(birthDate, documentDate) : '',
    gender,
    complaintDate: documentDateIso || new Date().toISOString().slice(0, 10),
    birthDate: birthMatch?.[1] ? dmyToIso(birthMatch[1]) : '',
  };
}

function isOcrServiceLine(line: string): boolean {
  return /\[неразборчиво\]/iu.test(line) ||
    /(?:Интерпретацию\s+полученных|Исполнители:|Напечатано|Адрес:|Тел\.?:|laboratory@|стр\.\s*\d+|Қазақстан Республикасы|Денсаулық сақтау|Министерство здравоохранения|Ministry of Health|Уйымның атауы|Наименование организации|Organization Name|Форма\s*№|Утверждена|Приказ|Зарегистрирован|Министр|Әділет|медициналық|EFQM|Recognised for excellence|^\d+\s*star$|^Организация:$|^Врач:$|^ТОО\b|^АО\b|National Scientific Medical Center|Национальный научный медицинский центр)/iu.test(line);
}

function isExactLabLine(line: string): boolean {
  if (isOcrServiceLine(line)) return false;

  return /^(?:№\s*(?:карты|направления)|(?:ИИН|ИНН)\s*:|Заказ\s*№|Кровь\b|Дата\s+взятия:|Выполнено:|IDs?:|Показатель\b)/iu.test(line) ||
    /(?:анализ\s+(?:крови|мочи)|Общий\s+анализ|Биохимическ|Гематолог)/iu.test(line) ||
    /^[А-ЯЁA-Z][^:\n]{1,90}?\s+(?:↑|↓)?\s*\d+(?:[.,]\d+)?\s+(?:10E\d+\/л|10\^\d+\/л|г\/л|мм\/ч|%|фл|пг|мкмоль\/л|ммоль\/л|Ед\/л|\/100WBC|\([^)]+\))/u.test(line);
}

function cleanExactOutpatientExams(rawText: string, patient: MedicalDocument['patient']): string {
  const lines = toCleanOcrLines(rawText);
  const result: string[] = [];
  const seen = new Set<string>();

  const add = (line: string) => {
    const clean = line.trim();
    if (!clean || seen.has(clean)) return;
    seen.add(clean);
    result.push(clean);
  };

  if (patient.fullName) {
    add(`Пациент: ${patient.fullName}${patient.gender ? `, ${patient.gender}` : ''}${patient.age ? `, ${patient.age}` : ''}`);
  }

  for (const line of lines) {
    if (looksLikePatientNameLine(line) || /Дата\s+рождения:/iu.test(line)) continue;
    if (isExactLabLine(line)) add(line);
  }

  if (result.length > (patient.fullName ? 1 : 0)) {
    return result.join('\n');
  }

  return lines
    .filter((line) => !isOcrServiceLine(line))
    .filter((line) => !/\[неразборчиво\]/iu.test(line))
    .join('\n')
    .trim();
}

const CONSULTATION_SECTION_FIELDS: Array<[RegExp, RewriteableField]> = [
  [/^Жалобы$/iu, 'complaints'],
  [/^Анамнез\s+заболевания$/iu, 'anamnesis'],
  [/^Амбулаторные\s+обследования$/iu, 'outpatientExams'],
  [/^Анамнез\s+жизни$/iu, 'clinicalCourse'],
  [/^Аллергологический\s+анамнез$/iu, 'allergyHistory'],
  [/^Объективный\s+статус$/iu, 'objectiveStatus'],
  [/^Неврологический\s+статус$/iu, 'neurologicalStatus'],
  [/^Предварительный\s+диагноз$/iu, 'diagnosis'],
  [/^Заключение$/iu, 'finalDiagnosis'],
  [/^План\s+обследования$/iu, 'doctorNotes'],
  [/^Рекомендации\s*\/\s*План\s+лечения$/iu, 'recommendations'],
  [/^План\s+лечения$/iu, 'recommendations'],
  [/^Рекомендации$/iu, 'recommendations'],
  [/^Амбулаторная\s+терапия$/iu, 'conclusion'],
];

function stripPlaceholderValue(value: string): string {
  const clean = value.replace(/\s+/g, ' ').trim();
  return clean === '-' || clean === '—' ? '' : clean;
}

function normalizeDmyOrIsoDate(value: string): string {
  const clean = stripPlaceholderValue(value);
  const dmy = clean.match(/\b(\d{2}\.\d{2}\.\d{4})\b/u)?.[1];
  if (dmy) return dmyToIso(dmy);

  const iso = clean.match(/\b(\d{4}-\d{2}-\d{2})\b/u)?.[1];
  return iso || clean;
}

function consultationFieldFromHeading(line: string): RewriteableField | null {
  const heading = line.replace(/\s+/g, ' ').replace(/[:.]+$/u, '').trim();
  for (const [pattern, field] of CONSULTATION_SECTION_FIELDS) {
    if (pattern.test(heading)) return field;
  }
  return null;
}

function consultationInlineSection(line: string): { field: RewriteableField; content: string } | null {
  const match = line.match(/^(.{3,80}?):\s*(.+)$/u);
  if (!match) return null;

  const field = consultationFieldFromHeading(match[1]);
  if (!field) return null;

  return { field, content: match[2].trim() };
}

function isConsultationServiceLine(line: string): boolean {
  return /^--\s*\d+\s+of\s+\d+\s*--$/iu.test(line) ||
    /^Подпись\s+врача$/iu.test(line) ||
    /^Документ\s+сформирован\s+автоматически(?:\s|$)/iu.test(line);
}

function appendProtocolLine(
  sections: Record<RewriteableField, string[]>,
  field: RewriteableField | null,
  line: string,
) {
  if (!field) return;
  const clean = line
    .trim()
    .replace(/(?<![А-ЯЁа-яёA-Za-z])Н(?:ь|Ь|b|B|в|В)(?=\s*[-–—]\s*\d)/gu, 'Hb');
  if (!clean) return;
  sections[field].push(clean);
}

export function documentFromConsultationProtocolText(rawText: string): MedicalDocument | null {
  const lines = toCleanOcrLines(rawText).filter((line) => !isConsultationServiceLine(line));
  const hasProtocolMarker = lines.some((line) => /^КОНСУЛЬТАЦИЯ$/iu.test(line)) ||
    lines.some((line) => /^Дата\s+составления:/iu.test(line));
  const sectionHeadingCount = lines.filter((line) => consultationFieldFromHeading(line)).length;
  const hasOutpatientSection = lines.some((line) =>
    consultationFieldFromHeading(line) === 'outpatientExams' ||
    consultationInlineSection(line)?.field === 'outpatientExams',
  );

  if (!hasProtocolMarker && sectionHeadingCount < 2 && !hasOutpatientSection) {
    return null;
  }

  const today = new Date().toISOString().slice(0, 10);
  const patient: MedicalDocument['patient'] = {
    fullName: '',
    age: '',
    gender: '',
    complaintDate: '',
    birthDate: '',
  };
  const riskAssessment: MedicalDocument['riskAssessment'] = {
    fallInLast3Months: '',
    dizzinessOrWeakness: '',
    needsEscort: '',
    painScore: '',
  };
  const sections = Object.fromEntries(
    rewriteableFields.map((field) => [field, [] as string[]]),
  ) as Record<RewriteableField, string[]>;

  let documentDate = '';
  let currentField: RewriteableField | null = null;
  let matchedProtocolField = false;

  for (const line of lines) {
    const inlineSection = consultationInlineSection(line);
    if (inlineSection) {
      currentField = inlineSection.field;
      appendProtocolLine(sections, currentField, inlineSection.content);
      matchedProtocolField = true;
      continue;
    }

    const headingField = consultationFieldFromHeading(line);
    if (headingField) {
      currentField = headingField;
      matchedProtocolField = true;
      continue;
    }

    if (/^(?:КОНСУЛЬТАЦИЯ|Оценка\s+риска\s+\(шкала\s+Морзе\))$/iu.test(line)) {
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    if (currentField) {
      appendProtocolLine(sections, currentField, line);
      continue;
    }

    let match = line.match(/^Дата\s+составления:\s*(.+)$/iu);
    if (match) {
      documentDate = normalizeDmyOrIsoDate(match[1]);
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    match = line.match(/^ФИО\s+пациента:\s*(.+)$/iu);
    if (match) {
      patient.fullName = stripPlaceholderValue(match[1]);
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    match = line.match(/^Возраст:\s*(.+)$/iu);
    if (match) {
      patient.age = stripPlaceholderValue(match[1]);
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    match = line.match(/^Пол:\s*(.+)$/iu);
    if (match) {
      patient.gender = stripPlaceholderValue(match[1]);
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    match = line.match(/^Дата\s+обращения:\s*(.+)$/iu);
    if (match) {
      patient.complaintDate = normalizeDmyOrIsoDate(match[1]);
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    match = line.match(/^Падал\s*\(3\s*мес\.?\):\s*(.+)$/iu);
    if (match) {
      riskAssessment.fallInLast3Months = stripPlaceholderValue(match[1]);
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    match = line.match(/^Головокружение:\s*(.+)$/iu);
    if (match) {
      riskAssessment.dizzinessOrWeakness = stripPlaceholderValue(match[1]);
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    match = line.match(/^Сопровождение:\s*(.+)$/iu);
    if (match) {
      riskAssessment.needsEscort = stripPlaceholderValue(match[1]);
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    match = line.match(/^Оценка\s+боли:\s*(.+)$/iu);
    if (match) {
      riskAssessment.painScore = stripPlaceholderValue(match[1]).replace(/\s*б(?:алл[а-я]*)?\.?$/iu, '');
      currentField = null;
      matchedProtocolField = true;
      continue;
    }

    appendProtocolLine(sections, currentField, line);
  }

  patient.complaintDate = patient.complaintDate || documentDate || today;

  const document: MedicalDocument = {
    patient,
    riskAssessment,
    complaints: sections.complaints.join('\n').trim(),
    anamnesis: sections.anamnesis.join('\n').trim(),
    outpatientExams: sections.outpatientExams.join('\n').trim(),
    clinicalCourse: sections.clinicalCourse.join('\n').trim(),
    allergyHistory: sections.allergyHistory.join('\n').trim(),
    objectiveStatus: sections.objectiveStatus.join('\n').trim(),
    neurologicalStatus: sections.neurologicalStatus.join('\n').trim(),
    diagnosis: sections.diagnosis.join('\n').trim(),
    finalDiagnosis: sections.finalDiagnosis.join('\n').trim(),
    conclusion: sections.conclusion.join('\n').trim(),
    doctorNotes: sections.doctorNotes.join('\n').trim(),
    recommendations: sections.recommendations.join('\n').trim(),
  };

  const hasSectionContent = rewriteableFields.some((field) => document[field].trim().length > 0);
  return matchedProtocolField && (hasProtocolMarker || hasSectionContent) ? document : null;
}

export function documentFromExactSourceText(rawText: string, base?: MedicalDocument): MedicalDocument {
  const patient = extractPatientFromExactText(rawText);
  const text = cleanExactOutpatientExams(rawText, patient);
  const today = new Date().toISOString().slice(0, 10);

  return {
    patient: {
      fullName: patient.fullName || base?.patient?.fullName || '',
      age: patient.age || base?.patient?.age || '',
      gender: patient.gender || base?.patient?.gender || '',
      complaintDate: patient.complaintDate || base?.patient?.complaintDate || today,
      birthDate: patient.birthDate || base?.patient?.birthDate || '',
    },
    riskAssessment: {
      fallInLast3Months: '',
      dizzinessOrWeakness: '',
      needsEscort: '',
      painScore: '',
    },
    complaints: '',
    anamnesis: '',
    outpatientExams: text,
    clinicalCourse: '',
    allergyHistory: '',
    objectiveStatus: '',
    neurologicalStatus: '',
    diagnosis: '',
    finalDiagnosis: '',
    conclusion: '',
    doctorNotes: '',
    recommendations: '',
    manualCheck: text
      ? 'Документ получен из фото/OCR. Оставлены только распознанные фактические данные анализа; служебный текст бланка скрыт.'
      : base?.manualCheck,
  };
}

// Заголовки разделов, которые LLM/Whisper иногда оставляют как «контент» поля
// (например `complaints = "Жалобы"`). Такие фрагменты должны считаться пустыми
// при оценке полезности документа: иначе фронт получает success:true с
// псевдо-заполненным протоколом.
const SECTION_HEADER_RE =
  /^(?:жалобы|анамнез(?:\s+(?:заболевани[а-яёА-ЯЁ\w]*|жизни))?|объективн[а-яёА-ЯЁ\w]*\s+статус|объективно|перенесённые\s+заболевани[а-яёА-ЯЁ\w]*|аллерголог[а-яёА-ЯЁ\w]*\s+анамнез|неврологическ[а-яёА-ЯЁ\w]*\s+статус|диагноз(?:\s+(?:предварительн[а-яёА-ЯЁ\w]*|заключительн[а-яёА-ЯЁ\w]*|основн[а-яёА-ЯЁ\w]*))?|план(?:\s+(?:обследовани[а-яёА-ЯЁ\w]*|лечени[а-яёА-ЯЁ\w]*))?|рекомендации|рекомендация|заключение|сопутствующ[а-яёА-ЯЁ\w]*\s+диагноз|амбулаторн[а-яёА-ЯЁ\w]*\s+терапи[а-яёА-ЯЁ\w]*|анамнез|данные|статус)\.?$/iu;

function stripSectionHeaders(s: string): string {
  if (!s || !s.trim()) return '';
  return s
    .split(/[.!?\n]+/u)
    .map((f) => f.trim())
    .filter((f) => f && !SECTION_HEADER_RE.test(f))
    .join('. ')
    .trim();
}

// Поле считается «содержательным», если после удаления одиночных заголовков
// остаётся хотя бы 10 символов И ≥2 слов (≥3 букв каждое). Это режет случаи
// «Диагноз: АГ» (длина после strip 4) и «Жалобы» (после strip 0).
export function isFieldMeaningful(s: string): boolean {
  const stripped = stripSectionHeaders(s);
  if (stripped.length < 10) return false;
  const words = stripped.match(/[а-яёА-ЯЁa-zA-Z]{3,}/gu) || [];
  return words.length >= 2;
}

export type DocumentUsefulness = {
  status: 'ok' | 'labs_only' | 'empty';
  emptyFields: string[];
  placeholderFields: string[];
  meaningfulFields: string[];
  reason?: string;
};

const NON_EXAMS_CLINICAL_FIELDS = [
  'complaints',
  'anamnesis',
  'clinicalCourse',
  'allergyHistory',
  'objectiveStatus',
  'neurologicalStatus',
  'diagnosis',
  'finalDiagnosis',
  'conclusion',
  'doctorNotes',
  'recommendations',
] as const;

const SUSPICIOUS_UNIT_GARBAGE_RE =
  /(?<![А-ЯЁа-яёA-Za-z])(?:СЛЧЭЛЬ|СЛЦАЛЬ|ЛЩЕ|ММС\s+ЛЩЕ|ГЭСЛЧЭЛЬ|ГЕСЛШЕЛЬ|КОЭСЛЧ|МОЛЬСЛЧ|МГСЛЧЭЛЬ)(?![А-ЯЁа-яёA-Za-z])/iu;

function documentClinicalText(doc: MedicalDocument): string {
  return [
    doc.patient.fullName,
    doc.patient.age,
    doc.patient.gender,
    doc.patient.complaintDate,
    doc.complaints,
    doc.anamnesis,
    doc.outpatientExams,
    doc.clinicalCourse,
    doc.allergyHistory,
    doc.objectiveStatus,
    doc.neurologicalStatus,
    doc.diagnosis,
    doc.finalDiagnosis,
    doc.conclusion,
    doc.doctorNotes,
    doc.recommendations,
    doc.manualCheck || '',
  ].join('\n');
}

export function collectDocumentQualityWarnings(
  rawText: string,
  doc: MedicalDocument,
  usefulness: DocumentUsefulness,
): string[] {
  return [...new Set(collectDocumentQualityWarningDetails(rawText, doc, usefulness)
    .map((warning) => warning.code === 'important_number_missing'
      ? `important_number_missing:${warning.evidence || 'value'}`
      : warning.code))];
}

export function collectDocumentQualityWarningDetails(
  rawText: string,
  doc: MedicalDocument,
  usefulness: DocumentUsefulness,
): QualityWarning[] {
  const warnings: QualityWarning[] = [];
  const seen = new Set<string>();
  const add = (warning: QualityWarning) => {
    const key = `${warning.code}:${warning.field || ''}:${warning.evidence || ''}`;
    if (seen.has(key)) return;
    seen.add(key);
    warnings.push(warning);
  };
  const docText = documentClinicalText(doc);

  if (isWeakPatientName(doc.patient?.fullName)) {
    add({
      code: 'patient_identity_missing',
      severity: 'info',
      field: 'patient',
      message: 'ФИО пациента не найдено в диктовке. Для live-диктанта врач должен продиктовать пациента или заполнить поле вручную.',
    });
  }

  if (usefulness.status === 'labs_only') {
    add({
      code: 'document_labs_only',
      severity: 'warning',
      field: 'outpatientExams',
      message: 'В документе распознаны в основном обследования, клинические разделы почти пустые.',
    });
  }

  if (usefulness.status === 'ok') {
    const meaningfulNonExams = usefulness.meaningfulFields.filter((f) => f !== 'outpatientExams');
    if (meaningfulNonExams.length < 3) {
      add({
        code: 'suspiciously_few_clinical_fields',
        severity: 'warning',
        field: 'document',
        message: 'Заполнено мало клинических разделов; проверьте, не потерялись ли жалобы, анамнез или диагноз.',
      });
    }
  }

  if (SUSPICIOUS_UNIT_GARBAGE_RE.test(docText)) {
    add({
      code: 'suspicious_unit_garbage_in_document',
      severity: 'warning',
      field: 'document',
      message: 'В документе остались подозрительные единицы измерения после распознавания.',
    });
  }

  const rawBpValues = rawText.match(/\b\d{2,3}\s*\/\s*\d{2,3}\b/gu) || [];
  for (const value of rawBpValues) {
    const normalized = value.replace(/\s+/g, '');
    if (!docText.replace(/\s+/g, '').includes(normalized)) {
      add({
        code: 'important_number_missing',
        severity: 'critical',
        field: 'document',
        message: `В raw есть значение АД ${normalized}, но оно не найдено в итоговых полях.`,
        evidence: `bp_${normalized}`,
      });
    }
  }

  collectAdvancedQualityWarnings(rawText, doc, add);

  return warnings;
}

function normalizeForCoverage(value: string): string {
  return value
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^a-zа-я0-9\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function appendManualCheck(doc: MedicalDocument, message: string): void {
  const current = (doc.manualCheck || '').trim();
  if (current.includes(message)) return;
  doc.manualCheck = current ? `${current}\n${message}` : message;
}

function tokenCoverage(reference: string, actual: string): number | null {
  const refTokens = new Set(
    normalizeForCoverage(reference)
      .split(/\s+/)
      .filter((token) => token.length >= 4),
  );
  if (refTokens.size === 0) return null;

  const actualTokens = new Set(
    normalizeForCoverage(actual)
      .split(/\s+/)
      .filter((token) => token.length >= 4),
  );

  let found = 0;
  for (const token of refTokens) {
    if (actualTokens.has(token)) found++;
  }
  return Math.round((found / refTokens.size) * 100);
}

function sentenceCoverage(reference: string, actual: string): { total: number; found: number; percent: number | null; missing: string[] } {
  const actualNorm = normalizeForCoverage(actual);
  const sentences = reference
    .split(/(?<=[.!?])\s+|\n+/u)
    .map((sentence) => sentence.trim())
    .filter((sentence) => normalizeForCoverage(sentence).length >= 25);

  let found = 0;
  const missing: string[] = [];

  for (const sentence of sentences) {
    const normalized = normalizeForCoverage(sentence);
    const probes = [
      normalized.slice(0, Math.min(35, normalized.length)),
      normalized.slice(Math.max(0, Math.floor(normalized.length / 2) - 18), Math.floor(normalized.length / 2) + 18),
      normalized.slice(Math.max(0, normalized.length - 35)),
    ].filter((probe) => probe.length >= 18);

    if (probes.some((probe) => actualNorm.includes(probe))) {
      found++;
    } else {
      missing.push(sentence);
    }
  }

  return {
    total: sentences.length,
    found,
    percent: sentences.length ? Math.round((found / sentences.length) * 100) : null,
    missing: missing.slice(0, 5),
  };
}

function isWeakPatientName(value: string | undefined): boolean {
  const clean = String(value || '').replace(/\s+/g, ' ').trim();
  if (!clean) return true;
  if (/^(?:пациент|пациентка|фио(?:\s+пациента)?|не\s+указано)$/iu.test(clean)) return true;
  if (/^(?:и\.?\s*о\.?|ф\.?\s*и\.?\s*о\.?)[\s!.,:;-]+/iu.test(clean)) return true;
  if (clean.length > 90 && /(?:диагноз|диагнос|жалоб|анамнез|лечени)/iu.test(clean)) return true;
  return clean.split(/\s+/).filter(Boolean).length < 2;
}

export function extractPatientHintsFromFilename(originalName: string): Partial<MedicalDocument['patient']> {
  const base = path.parse(path.basename(originalName)).name
    .replace(/__\d+_audio$/iu, '')
    .replace(/__audio$/iu, '')
    .replace(/[_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const firstPart = base.split(',')[0]?.trim() || '';
  const nameMatch = firstPart.match(/^([А-ЯЁӘІҢҒҮҰҚӨҺA-Z][А-ЯЁӘІҢҒҮҰҚӨҺа-яёәіңғүұқөһA-Za-z'-]+(?:\s+[А-ЯЁӘІҢҒҮҰҚӨҺA-Z][А-ЯЁӘІҢҒҮҰҚӨҺа-яёәіңғүұқөһA-Za-z'-]+){1,3})$/u);
  const fullName = nameMatch?.[1] || '';
  const birthDate = base.match(/\b(\d{2})\.(\d{2})\.(\d{4})\s*г?\.?\s*р?\.?/iu);
  const yearOnly = !birthDate ? base.match(/\b((?:19|20)\d{2})\s*г?\.?\s*р?\.?/iu) : null;
  const gender = /\bжен(?:ский)?\b/iu.test(base)
    ? 'женский'
    : /\bмуж(?:ской)?\b/iu.test(base)
      ? 'мужской'
      : '';

  const hints: Partial<MedicalDocument['patient']> = {};
  if (fullName && !/^(?:документ|шаблон|консультаци|назначения|выписки|протоколы)$/iu.test(fullName)) {
    hints.fullName = fullName;
  }
  if (birthDate) {
    hints.birthDate = `${birthDate[3]}-${birthDate[2]}-${birthDate[1]}`;
  } else if (yearOnly) {
    hints.birthDate = yearOnly[1];
  }
  if (gender) hints.gender = gender;
  return hints;
}

export function enrichDocumentFromSourceName(
  doc: MedicalDocument,
  originalName: string,
): { document: MedicalDocument; warnings: QualityWarning[] } {
  const hints = extractPatientHintsFromFilename(originalName);
  const warnings: QualityWarning[] = [];
  if (!hints.fullName && !hints.birthDate && !hints.gender) {
    return { document: doc, warnings };
  }

  const enriched: MedicalDocument = {
    ...doc,
    patient: { ...doc.patient },
    riskAssessment: { ...doc.riskAssessment },
  };

  if (hints.fullName && isWeakPatientName(enriched.patient.fullName)) {
    enriched.patient.fullName = hints.fullName;
    appendManualCheck(enriched, `ФИО пациента подставлено из имени файла: ${hints.fullName}. Проверьте вручную.`);
    warnings.push({
      code: 'patientNameFromFilename',
      severity: 'info',
      field: 'patient',
      message: 'ФИО пациента не было надежно извлечено из диктовки и подставлено из имени файла.',
      evidence: hints.fullName,
    });
  }
  if (hints.birthDate && !enriched.patient.birthDate) {
    enriched.patient.birthDate = hints.birthDate;
  }
  if (hints.gender && !enriched.patient.gender) {
    enriched.patient.gender = hints.gender;
  }

  return { document: enriched, warnings };
}

function warningCodes(warnings: QualityWarning[]): string[] {
  return [...new Set(warnings.map((warning) => warning.code === 'important_number_missing'
    ? `important_number_missing:${warning.evidence || 'value'}`
    : warning.code))];
}

function isRetryableLlmError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /timeout|таймаут|aborted|ECONNRESET|ECONNREFUSED|fetch failed|socket|503|502|500|json|parse|expected|unterminated|text\.trim is not a function/i.test(message);
}

async function structureTextWithRetry(
  llmService: LLMService,
  text: string,
  attempts = 2,
): Promise<Awaited<ReturnType<LLMService['structureText']>>> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await llmService.structureText(text);
    } catch (error) {
      lastError = error;
      if (attempt >= attempts || !isRetryableLlmError(error)) break;
      const delayMs = 1200 * attempt;
      console.warn(`[llm] structureText failed on attempt ${attempt}/${attempts}; retrying in ${delayMs}ms`, error);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw lastError;
}

export async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, fallback: T): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((resolve) => {
        timeout = setTimeout(() => resolve(fallback), timeoutMs);
      }),
    ]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

function collectAdvancedQualityWarnings(
  rawText: string,
  doc: MedicalDocument,
  add: (warning: QualityWarning) => void,
): void {
  const rawNorm = normalizeForCoverage(rawText);
  const docText = documentClinicalText(doc);

  const tokenPct = tokenCoverage(rawText, docText);
  const sentencePct = sentenceCoverage(rawText, docText);
  if (tokenPct !== null && tokenPct < 60) {
    add({
      code: 'lowDocumentCoverage',
      severity: tokenPct < 40 ? 'critical' : 'warning',
      field: 'document',
      message: `Итоговый документ покрывает только ${tokenPct}% значимых слов raw-текста. Возможна потеря клинических данных.`,
      evidence: `token_${tokenPct}`,
    });
  }
  if (sentencePct.percent !== null && sentencePct.percent < 50) {
    add({
      code: 'lowDocumentCoverage',
      severity: sentencePct.percent < 25 ? 'critical' : 'warning',
      field: 'document',
      message: `Итоговый документ покрывает только ${sentencePct.percent}% предложений raw-текста. Проверьте пропущенные фрагменты.`,
      evidence: `sentence_${sentencePct.percent}`,
    });
  }

  const criticalFieldChecks: Array<{ field: keyof Pick<MedicalDocument, 'complaints' | 'diagnosis' | 'recommendations'>; raw: RegExp; label: string }> = [
    { field: 'complaints', raw: /(?:жалоб[а-яё]*|беспоко[а-яё]*|предъявля[а-яё]*|отмеча[а-яё]*)/iu, label: 'жалобы' },
    { field: 'diagnosis', raw: /(?:диагноз|код\s+мкб|мкб|основн[а-яё]*\s+заболеван[а-яё]*)/iu, label: 'диагноз' },
    { field: 'recommendations', raw: /(?:рекоменд[а-яё]*|назнач[а-яё]*|план\s+лечени[а-яё]*|принимать|контроль)/iu, label: 'рекомендации' },
  ];
  for (const check of criticalFieldChecks) {
    if (check.raw.test(rawText) && !doc[check.field]?.trim()) {
      add({
        code: 'criticalFieldMissing',
        severity: 'warning',
        field: check.field,
        message: `В raw есть маркеры раздела "${check.label}", но итоговое поле пустое.`,
        evidence: check.field,
      });
    }
  }

  const rawLabish = /(?:оак|оам|биохим|анализ|гемоглобин|креатинин|глюкоз|холестерин|лпнп|лпвп|триглицерид|лейкоцит|эритроцит|тромбоцит|соэ|hba1c|гликирован)/iu.test(rawText);
  if (rawLabish) {
    const valueMatches = rawText.matchAll(/\d+(?:[,.]\d+)?/gu);
    for (const match of valueMatches) {
      const value = match[0];
      const index = match.index ?? 0;
      const after = rawText.slice(index + value.length, index + value.length + 8);
      const compact = value.replace(',', '.');
      if (/^\d{1,2}$/.test(compact)) continue;
      if (/^(?:19|20)\d{2}$/.test(compact)) continue;
      if (/^\d{1,2}\.\d{1,2}$/.test(value) && /^\.\d{2,4}\b/.test(after)) continue;
      const docHas = docText.includes(value) || docText.includes(value.replace(',', '.')) || docText.includes(value.replace('.', ','));
      if (!docHas) {
        add({
          code: 'possibleLostLabValue',
          severity: 'warning',
          field: 'outpatientExams',
          message: `В raw есть лабораторное число ${value}, но оно не найдено в итоговых обследованиях.`,
          evidence: value,
        });
      }
    }
  }

  const examLines = (doc.outpatientExams || '').split(/\n+/).map((line) => line.trim()).filter(Boolean);
  for (const line of examLines) {
    const label = line.match(/(?:^|\s)(КТ|МРТ|МСКТ|ЭКГ|ЭхоКГ|УЗИ|УЗДГ|Холтер|СМАД)(?:\s|$)/iu)?.[1];
    if (!label) continue;
    const labelRe = new RegExp(`(^|[^а-яёa-z])${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}($|[^а-яёa-z])`, 'iu');
    if (!labelRe.test(rawText)) {
      add({
        code: 'suspiciousExamRescue',
        severity: 'warning',
        field: 'outpatientExams',
        message: `В итоговых обследованиях есть ${label}, но в raw нет такого отдельного маркера.`,
        evidence: line.slice(0, 160),
      });
    }
  }

  // Whisper пропустил значение максимального АД: фраза есть, числа нет
  const hasMaxBpPhrase = /максимальн\S+\s+цифр\S+\s*(?:артериального\s+давления\s*)?/iu.test(rawText);
  const hasMaxBpValue = /максимальн\S+\s+цифр[^\n.!?]{0,120}\d{2,3}\/\d{2,3}/iu.test(rawText);
  if (hasMaxBpPhrase && !hasMaxBpValue) {
    add({
      code: 'max_bp_value_missing',
      severity: 'warning',
      field: 'anamnesis',
      message: 'Есть фраза "максимальные цифры АД" но числовое значение X/Y не найдено — возможно Whisper пропустил число.',
    });
  }

  const lifeHistoryInRecommendations = (doc.recommendations || '')
    .split(/\n+/)
    .find((line) => /(?:алкогол[а-яё]*\s+употреб|наследственност|туберкул|операци|травм|профессиональн[а-яё]*\s+вредност|физическ[а-яё]*\s+активность\s+низк|курит\s+\d)/iu.test(line));
  if (lifeHistoryInRecommendations) {
    add({
      code: 'sectionRoutingIssue',
      severity: 'warning',
      field: 'recommendations',
      message: 'В рекомендациях обнаружена фраза, похожая на анамнез жизни.',
      evidence: lifeHistoryInRecommendations.slice(0, 160),
    });
  }

  const rawHasRecommendationIntent = /(?:рекоменд|назнач|план\s+лечени|лечение|принимать|контроль|повторн\S+\s+(?:осмотр|прием|приём)|консультаци|диет|стол\s*№|режим|огранич)/iu.test(rawText);
  const unsupportedRecommendation = (doc.recommendations || '')
    .split(/\n+/)
    .map((line) => line.replace(/^\s*\d+[\.)]\s*/, '').trim())
    .find((line) => {
      if (!line || rawHasRecommendationIntent) return false;
      return /(?:диет|питани|ограничить|исключить|алкогол|курени|физическ\S+\s+активн|контроль\s+(?:ад|веса|массы)|здоров\S+\s+образ)/iu.test(line);
    });
  if (unsupportedRecommendation) {
    add({
      code: 'unsupportedRecommendation',
      severity: 'warning',
      field: 'recommendations',
      message: 'В рекомендациях есть типовой пункт, но в raw-диктовке нет явного назначения или рекомендации.',
      evidence: unsupportedRecommendation.slice(0, 160),
    });
  }

  const conclusionLifeTail = (doc.conclusion || '')
    .split(/\n+/)
    .find((line) => /(?:туберкул|вирусн[а-яё]*\s+гепатит|вич|операци|травм|наследственност|алкогол|курение|профессиональн[а-яё]*\s+вредност)/iu.test(line));
  if (conclusionLifeTail) {
    add({
      code: 'sectionRoutingIssue',
      severity: 'warning',
      field: 'conclusion',
      message: 'В амбулаторной терапии обнаружена фраза, похожая на анамнез жизни.',
      evidence: conclusionLifeTail.slice(0, 160),
    });
  }

  for (const field of ['conclusion', 'recommendations'] as const) {
    const line = (doc[field] || '')
      .split(/\n+/)
      .find((item) => (item.match(/\d+(?:[,.]\d+)?\s*(?:мг|мкг|г|мл)/giu) || []).length >= 2);
    if (line) {
      add({
        code: 'drugListMayBeMerged',
        severity: 'warning',
        field,
        message: 'Пункт содержит несколько дозировок; возможно, несколько препаратов склеились в один пункт.',
        evidence: line.slice(0, 180),
      });
    }
  }

  const docSentences = docText
    .split(/(?<=[.!?])\s+|\n+/u)
    .map((s) => s.trim())
    .filter((s) => s.length >= 35);
  for (const sentence of docSentences) {
    if (!/(?:кт|мрт|мскт|тредмил|коронар|ацетилсалицил|нитроглицерин|эмпаглифлозин|розувастатин|диагноз|стенокард|инфаркт)/iu.test(sentence)) {
      continue;
    }
    const ns = normalizeForCoverage(sentence);
    if (!ns || rawNorm.includes(ns.slice(0, Math.min(35, ns.length)))) continue;
    const words = ns.split(/\s+/).filter((w) => w.length >= 5);
    const overlap = words.filter((w) => rawNorm.includes(w)).length;
    if (words.length >= 4 && overlap / words.length < 0.45) {
      add({
        code: 'possibleAddedFact',
        severity: 'warning',
        field: 'document',
        message: 'Итоговый документ содержит медицинский факт, который плохо подтверждается raw-текстом.',
        evidence: sentence.slice(0, 180),
      });
      break;
    }
  }
}

/**
 * Оценивает «полезность» структурированного документа. Не учитывает patient —
 * один заполненный fullName не должен спасать документ без клинического
 * контента.
 *
 *   ok          — есть ≥1 содержательное клиническое поле (не лаборатория).
 *   labs_only   — содержательно только outpatientExams (валидный частичный
 *                 документ, но фронт должен показать баннер «только анализы»).
 *   empty       — ни одного содержательного поля; placeholder-заголовки и/или
 *                 одиночное ФИО не считаются.
 */
export function assessDocumentUsefulness(doc: MedicalDocument): DocumentUsefulness {
  const empty: string[] = [];
  const placeholder: string[] = [];
  const meaningful: string[] = [];

  for (const f of [...NON_EXAMS_CLINICAL_FIELDS, 'outpatientExams'] as const) {
    const v = doc[f];
    if (!v || !v.trim()) {
      empty.push(f);
    } else if (!isFieldMeaningful(v)) {
      placeholder.push(f);
    } else {
      meaningful.push(f);
    }
  }

  const meaningfulNonExams = meaningful.filter((f) => f !== 'outpatientExams');
  const examsMeaningful = meaningful.includes('outpatientExams');

  if (meaningfulNonExams.length === 0 && !examsMeaningful) {
    return {
      status: 'empty',
      emptyFields: empty,
      placeholderFields: placeholder,
      meaningfulFields: [],
      reason: 'document_appears_empty',
    };
  }
  if (meaningfulNonExams.length === 0 && examsMeaningful) {
    return {
      status: 'labs_only',
      emptyFields: empty,
      placeholderFields: placeholder,
      meaningfulFields: meaningful,
    };
  }
  return {
    status: 'ok',
    emptyFields: empty,
    placeholderFields: placeholder,
    meaningfulFields: meaningful,
  };
}

export function isValidMedicalDocument(doc: unknown): doc is MedicalDocument {
  if (!isRecord(doc)) return false;
  if (!isRecord(doc.patient)) return false;

  const patient = doc.patient as Record<string, unknown>;
  const patientKeys = ['fullName', 'age', 'gender', 'complaintDate'];
  for (const key of patientKeys) {
    if (typeof patient[key] !== 'string') return false;
  }

  const textKeys = [
    'complaints',
    'anamnesis',
    'outpatientExams',
    'clinicalCourse',
    'allergyHistory',
    'objectiveStatus',
    'neurologicalStatus',
    'diagnosis',
    'finalDiagnosis',
    'conclusion',
    'doctorNotes',
    'recommendations',
  ];

  for (const key of textKeys) {
    if (typeof doc[key] !== 'string') return false;
  }

  return true;
}

// ─── Streaming session store ──────────────────────────────────────────────────

interface ChunkJob {
  index: number;
  textPromise: Promise<string>;
}

interface StreamSession {
  id: string;
  jobs: ChunkJob[];
  createdAt: number;
}

type AudioJobStatus = 'queued' | 'transcribing' | 'structuring' | 'done' | 'failed';

interface AudioProcessSuccess {
  success: true;
  transcription: {
    text: string;
    duration: number;
    language: string;
  };
  document: MedicalDocument;
  processingTime: number;
  warnings: string[];
  qualityWarnings: QualityWarning[];
  timingsMs: {
    saveFile: number;
    whisper: number;
    llm: number;
    total: number;
  };
}

interface AudioProcessFailurePayload {
  success: false;
  error: string;
  message?: string;
  reason?: string;
  emptyFields?: string[];
  placeholderFields?: string[];
  transcription?: AudioProcessSuccess['transcription'];
}

interface AudioProcessHttpError extends Error {
  statusCode: number;
  payload: AudioProcessFailurePayload;
}

interface AudioProcessJob {
  id: string;
  status: AudioJobStatus;
  filename: string;
  sourceName: string;
  createdAt: string;
  updatedAt: string;
  startedAt?: string;
  finishedAt?: string;
  result?: AudioProcessSuccess;
  error?: string;
  message?: string;
  statusCode?: number;
  transcription?: AudioProcessSuccess['transcription'];
}

const streamSessions = new Map<string, StreamSession>();
const STREAM_SESSION_TTL_MS = 30 * 60 * 1000;

// ─────────────────────────────────────────────────────────────────────────────

export async function registerRoutes(
  fastify: FastifyInstance,
  config: ServerConfig,
  whisperService: WhisperService,
  llmService: LLMService,
  ttsService: TtsService,
  db?: AppDb,
): Promise<void> {
  if (!existsSync(config.uploadDir)) {
    await mkdir(config.uploadDir, { recursive: true });
  }

  const visionProvider = process.env.DOCUMENT_VISION_PROVIDER?.trim().toLowerCase() === 'anthropic'
    ? 'anthropic'
    : 'ollama';
  const documentExtractor = new DocumentExtractorService({
    visionProvider,
    anthropicApiKey: process.env.ANTHROPIC_API_KEY?.trim() || config.llm.anthropic?.apiKey,
    ollama: {
      serverUrl: process.env.DOCUMENT_VISION_SERVER_URL?.trim() || config.llm.serverUrl,
      model: process.env.DOCUMENT_VISION_MODEL?.trim() || config.llm.model,
      timeoutMs: Number.parseInt(process.env.DOCUMENT_VISION_TIMEOUT_MS || '90000', 10),
    },
  });

  // Регистрируем маршруты врачей/пациентов/синхронизации если БД доступна
  if (db) {
    await registerDoctorRoutes(fastify, db, documentExtractor, llmService);
  }

  // Маршруты движка лучевой диагностики (structured reporting, без БД/LLM)
  registerRadiologyRoutes(fastify);

  const rateMap = new Map<string, RateState>();
  const audioJobs = new Map<string, AudioProcessJob>();
  const AUDIO_JOB_TTL_MS = 24 * 60 * 60 * 1000;

  // Сессионные токены с TTL. Без TTL — утечка памяти от мёртвых сессий.
  const AUTH_TOKEN_TTL_MS = 24 * 60 * 60 * 1000; // 24 часа
  const authTokens = new Map<string, number>(); // token → expiresAt
  const isTokenValid = (token: string): boolean => {
    const exp = authTokens.get(token);
    if (!exp) return false;
    if (Date.now() > exp) {
      authTokens.delete(token);
      return false;
    }
    return true;
  };
  // Периодическая очистка истёкших записей (раз в час).
  // Без неё authTokens/loginAttempts/rateMap утекают в памяти при долгом аптайме.
  const cleanupInterval = setInterval(() => {
    const now = Date.now();
    for (const [token, exp] of authTokens) {
      if (now > exp) authTokens.delete(token);
    }
    for (const [ip, st] of loginAttempts) {
      // Удаляем записи, у которых lockout истёк И счётчик не растёт > 1ч
      if (st.lockedUntil > 0 && now > st.lockedUntil) loginAttempts.delete(ip);
    }
    for (const [ip, st] of rateMap) {
      if (now - st.windowStartedAt > config.security.rateLimitWindowMs * 2) {
        rateMap.delete(ip);
      }
    }
    for (const [sid, sess] of streamSessions) {
      if (now - sess.createdAt > STREAM_SESSION_TTL_MS) streamSessions.delete(sid);
    }
    for (const [jobId, job] of audioJobs) {
      const updatedAt = Date.parse(job.updatedAt);
      if (Number.isFinite(updatedAt) && now - updatedAt > AUDIO_JOB_TTL_MS) {
        audioJobs.delete(jobId);
      }
    }
  }, 60 * 60 * 1000);
  cleanupInterval.unref?.();

  // Brute-force защита на /api/auth/login: lockout по IP после N неудачных попыток.
  const LOGIN_MAX_ATTEMPTS = 5;
  const LOGIN_LOCKOUT_MS = 15 * 60 * 1000; // 15 минут
  const LOGIN_FAIL_DELAY_MS = 500; // задержка ответа на неверный пароль
  const loginAttempts = new Map<string, { count: number; lockedUntil: number }>();

  fastify.addHook('onRequest', async (request, reply) => {
    const url = request.url;
    // Only protect API routes; let static files (frontend) pass through
    if (!url.startsWith('/api/')) return;
    if (url.startsWith('/api/health')) return;
    if (url.startsWith('/api/auth/')) return;

    if (db) {
      try {
        await request.jwtVerify();
      } catch {
        return reply.status(401).send({ error: 'Unauthorized' });
      }

      const doctor = db.select({ isActive: doctors.isActive })
        .from(doctors)
        .where(eq(doctors.id, request.user.doctorId))
        .get();
      if (!doctor?.isActive) {
        return reply.status(403).send({ error: 'Аккаунт деактивирован' });
      }
    } else {
      // Auth by password (session token)
      if (config.security.authPassword) {
        const auth = request.headers.authorization;
        const token = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';

        if (!token || !isTokenValid(token)) {
          return reply.status(401).send({ error: 'Unauthorized' });
        }
      }

      // Legacy API key auth
      if (config.security.apiKey) {
        const rawApiKey = request.headers['x-api-key'];
        const headerApiKey = typeof rawApiKey === 'string' ? rawApiKey : '';
        const auth = request.headers.authorization;
        const bearerApiKey = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
        const provided = headerApiKey || bearerApiKey;

        if (!provided || provided !== config.security.apiKey) {
          return reply.status(401).send({ error: 'Unauthorized' });
        }
      }
    }

    const now = Date.now();
    const key = request.ip || 'unknown';
    const state = rateMap.get(key);

    if (!state || now - state.windowStartedAt >= config.security.rateLimitWindowMs) {
      rateMap.set(key, { count: 1, windowStartedAt: now });
      return;
    }

    state.count += 1;
    if (state.count > config.security.rateLimitMaxRequests) {
      return reply.status(429).send({ error: 'Too many requests' });
    }
  });

  if (!db) {
    // --- Legacy auth endpoint ---
    fastify.post('/api/auth/login', async (request, reply) => {
      if (!config.security.authPassword) {
        return { success: true, token: 'no-auth' };
      }

      const ip = request.ip || 'unknown';
      const now = Date.now();
      const attempt = loginAttempts.get(ip);

      // Lockout по IP
      if (attempt && attempt.lockedUntil > now) {
        const minutesLeft = Math.ceil((attempt.lockedUntil - now) / 60000);
        return reply.status(429).send({
          error: `Слишком много попыток. Попробуйте через ${minutesLeft} мин.`,
        });
      }

      const body = request.body as Record<string, unknown> | null;
      const password = typeof body?.password === 'string' ? body.password : '';

      if (password !== config.security.authPassword) {
        // Задержка ответа — замедляем перебор
        await new Promise((resolve) => setTimeout(resolve, LOGIN_FAIL_DELAY_MS));

        const next = attempt && attempt.lockedUntil <= now
          ? { count: attempt.count + 1, lockedUntil: 0 }
          : { count: (attempt?.count ?? 0) + 1, lockedUntil: 0 };

        if (next.count >= LOGIN_MAX_ATTEMPTS) {
          next.lockedUntil = now + LOGIN_LOCKOUT_MS;
          next.count = 0;
          loginAttempts.set(ip, next);
          request.log.warn({ ip }, 'Login lockout triggered');
          return reply.status(429).send({
            error: 'Слишком много неверных попыток. Заблокировано на 15 минут.',
          });
        }

        loginAttempts.set(ip, next);
        return reply.status(401).send({ error: 'Неверный пароль' });
      }

      // Успешный логин — сбрасываем счётчик
      loginAttempts.delete(ip);

      const token = randomUUID();
      authTokens.set(token, now + AUTH_TOKEN_TTL_MS);
      return { success: true, token };
    });

    fastify.post('/api/auth/logout', async (request) => {
      const auth = request.headers.authorization;
      const token = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
      if (token) authTokens.delete(token);
      return { success: true };
    });

    fastify.get('/api/auth/check', async (request, reply) => {
      if (!config.security.authPassword) {
        return { authenticated: true };
      }
      const auth = request.headers.authorization;
      const token = typeof auth === 'string' && auth.startsWith('Bearer ') ? auth.slice(7).trim() : '';
      if (!token || !isTokenValid(token)) {
        return reply.status(401).send({ authenticated: false });
      }
      return { authenticated: true };
    });
  }

  // ─── Streaming session endpoints ─────────────────────────────────────────────

  fastify.post('/api/session/start', async (_request, reply) => {
    if (!config.whisper.serverUrl) {
      return reply.status(503).send({ error: 'Streaming unavailable: Whisper HTTP server not configured' });
    }
    if (!(await whisperService.healthCheck())) {
      return reply.status(503).send({ error: 'Streaming unavailable: Whisper HTTP server is not reachable' });
    }
    const sessionId = randomUUID();
    streamSessions.set(sessionId, { id: sessionId, jobs: [], createdAt: Date.now() });
    return { sessionId };
  });

  fastify.post('/api/session/:id/chunk', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const session = streamSessions.get(id);
    if (!session) {
      return reply.status(404).send({ error: 'Session not found or expired' });
    }

    const body = request.body;
    if (!isRecord(body) || typeof body.audio_base64 !== 'string' || !body.audio_base64) {
      return reply.status(400).send({ error: 'audio_base64 required' });
    }

    const chunkIndex = typeof body.chunk_index === 'number' ? body.chunk_index : session.jobs.length;
    const audioBase64 = body.audio_base64 as string;

    const textPromise = whisperService.transcribeBase64(audioBase64).catch((err) => {
      fastify.log.warn({ sessionId: id, chunkIndex }, `chunk transcription failed: ${err}`);
      return '';
    });

    session.jobs.push({ index: chunkIndex, textPromise });
    return { ok: true, chunkIndex };
  });

  fastify.post('/api/session/:id/finish', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const session = streamSessions.get(id);
    if (!session) {
      return reply.status(404).send({ error: 'Session not found or expired' });
    }

    streamSessions.delete(id);

    if (session.jobs.length === 0) {
      return reply.status(422).send({ error: 'No chunks received in session' });
    }

    try {
      const t0 = Date.now();

      const results = await Promise.all(
        session.jobs
          .sort((a, b) => a.index - b.index)
          .map((job) => job.textPromise)
      );

      const fullText = results.filter((t) => t.trim()).join(' ');

      if (!fullText.trim()) {
        return reply.status(422).send({ error: 'All chunks failed transcription' });
      }

      const t1 = Date.now();
      fastify.log.info(
        { sessionId: id, chunks: results.length, chars: fullText.length, whisperWaitMs: t1 - t0 },
        'session chunks merged'
      );

      const structured = await structureTextWithRetry(llmService, fullText);
      const t2 = Date.now();

      const usefulness = assessDocumentUsefulness(structured.document);
      if (usefulness.status === 'empty') {
        return reply.status(422).send({
          success: false,
          error: 'document_appears_empty',
          reason: usefulness.reason,
          emptyFields: usefulness.emptyFields,
          placeholderFields: usefulness.placeholderFields,
          transcription: { text: fullText, language: 'ru' },
        });
      }

      const qualityWarnings = collectDocumentQualityWarningDetails(fullText, structured.document, usefulness);

      fastify.log.info(
        { timings_ms: { whisper_wait: t1 - t0, llm: t2 - t1, total: t2 - t0 } },
        'session/finish timings'
      );

      return {
        success: true,
        transcription: { text: fullText, language: 'ru' },
        document: structured.document,
        processingTime: t2 - t0,
        warnings: warningCodes(qualityWarnings),
        qualityWarnings,
      };
    } catch (error) {
      fastify.log.error({ sessionId: id }, `session/finish error: ${error}`);
      const message = error instanceof Error ? error.message : 'Unknown error';
      const isTimeout = /timeout|aborted/i.test(message);
      return reply.status(isTimeout ? 408 : 500).send({ error: 'Session finish failed', message });
    }
  });

  // ─────────────────────────────────────────────────────────────────────────────

  fastify.get('/api/health', async () => {
    const healthTimeoutMs = Number.parseInt(process.env.HEALTH_CHECK_TIMEOUT_MS || '3000', 10);
    const [llmReady, whisperReady, ttsReady] = await Promise.all([
      withTimeout(llmService.healthCheck(), healthTimeoutMs, false),
      withTimeout(whisperService.healthCheck(), healthTimeoutMs, false),
      ttsService.isEnabled ? withTimeout(ttsService.healthCheck(), healthTimeoutMs, false) : Promise.resolve(false),
    ]);

    return {
      status: llmReady && whisperReady ? 'ok' : 'degraded',
      timestamp: new Date().toISOString(),
      services: {
        whisper: whisperReady ? 'ready' : 'unavailable',
        llm: llmReady ? 'ready' : 'unavailable',
        tts: ttsService.isEnabled ? (ttsReady ? 'ready' : 'unavailable') : 'disabled',
      },
    };
  });

  fastify.post('/api/upload', async (request: FastifyRequest, reply: FastifyReply) => {
    const data = await request.file();

    if (!data) {
      return reply.status(400).send({ error: 'No file uploaded' });
    }

    const originalName = data.filename || 'audio';
    const sourceName = toSafeUploadFilename(originalName);
    const filename = `${Date.now()}_${sourceName}`;
    const filepath = resolveUploadPath(config.uploadDir, filename);

    await pipeline(data.file, createWriteStream(filepath));

    return {
      success: true,
      filename,
      mimetype: data.mimetype,
    };
  });

  fastify.post('/api/transcribe', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    const filename = isRecord(body) && typeof body.filename === 'string' ? body.filename : undefined;

    if (!filename) {
      return reply.status(400).send({ error: 'Filename is required' });
    }

    if (filename !== path.basename(filename)) {
      return reply.status(400).send({ error: 'Invalid filename' });
    }

    const filepath = resolveUploadPath(config.uploadDir, filename);

    if (!existsSync(filepath)) {
      return reply.status(404).send({ error: 'File not found' });
    }

    try {
      const result = await whisperService.transcribeFile(filepath, filename);

      try {
        await unlink(filepath);
      } catch {
        // Ignore cleanup errors
      }

      return {
        success: true,
        ...result,
      };
    } catch (error) {
      console.error('Transcription error:', error);
      return reply.status(503).send({
        error: 'Сервис распознавания речи недоступен',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/structure', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    const text = isRecord(body) && typeof body.text === 'string' ? body.text.trim() : '';

    if (!text) {
      return reply.status(400).send({ error: 'Text is required' });
    }

    try {
      const result = await structureTextWithRetry(llmService, text);
      // PHI-дамп (raw + JSON документа) пишется только при STRUCTURE_LOG_DUMP=true.
      // По умолчанию выключен: содержимое жалоб/анамнеза/диагноза — медданные.
      if (process.env.STRUCTURE_LOG_DUMP === 'true') {
        try {
          const ts = new Date().toISOString().replace(/[:.]/g, '-');
          const logDir = path.join(config.uploadDir, '..', 'temp', 'structure-logs');
          await mkdir(logDir, { recursive: true });
          const logPath = path.join(logDir, `${ts}.log`);
          const dump = [
            `=== ${ts} ===`,
            `--- RAW WHISPER TEXT (${text.length} chars) ---`,
            text,
            '',
            `--- LLM STRUCTURED RESULT ---`,
            JSON.stringify(result, null, 2),
            '',
          ].join('\n');
          await appendFile(logPath, dump, 'utf-8');
          console.log(`[structure-log] wrote ${logPath}`);
        } catch (logErr) {
          console.warn('[structure-log] failed:', logErr);
        }
      }

      // Жёсткая проверка «полезности» — не отдаём success:true на пустой документ.
      const usefulness = assessDocumentUsefulness(result.document);
      if (usefulness.status === 'empty') {
        request.log.warn(
          { emptyFields: usefulness.emptyFields, placeholderFields: usefulness.placeholderFields },
          'structure: document_appears_empty',
        );
        return reply.status(422).send({
          success: false,
          error: 'document_appears_empty',
          reason: usefulness.reason,
          emptyFields: usefulness.emptyFields,
          placeholderFields: usefulness.placeholderFields,
        });
      }

      const qualityWarnings = collectDocumentQualityWarningDetails(text, result.document, usefulness);

      return {
        success: true,
        ...result,
        warnings: warningCodes(qualityWarnings),
        qualityWarnings,
      };
    } catch (error) {
      console.error('Structuring error:', error);
      const message = error instanceof Error ? error.message : 'Unknown error';
      const isTimeout = /timeout|таймаут|aborted|ECONNABORTED/i.test(message);
      return reply.status(isTimeout ? 408 : 500).send({
        error: isTimeout ? 'LLM request timeout' : 'Text structuring failed',
        message,
      });
    }
  });

  fastify.post('/api/rewrite-field', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const fieldRaw = typeof body.field === 'string' ? body.field : '';
    const text = typeof body.text === 'string' ? body.text.trim() : '';
    const field = rewriteableFields.find((x) => x === fieldRaw) as RewriteableField | undefined;

    if (!field) {
      return reply.status(400).send({ error: 'Valid field is required' });
    }

    if (!text) {
      return reply.status(400).send({ error: 'Text is required' });
    }

    try {
      const rewrittenText = await llmService.rewriteDocumentField(field, text);
      return {
        success: true,
        field,
        text: rewrittenText || text,
      };
    } catch (error) {
      console.error('Rewrite field error:', error);
      return reply.status(500).send({
        error: 'Field rewrite failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/recommendations', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    const docCandidate = isRecord(body) ? body.document : undefined;

    if (!isValidMedicalDocument(docCandidate)) {
      return reply.status(400).send({ error: 'Valid document is required' });
    }

    try {
      const recommendations = await llmService.generateRecommendations(docCandidate);
      return { success: true, recommendations };
    } catch (error) {
      console.error('Recommendations error:', error);
      return reply.status(500).send({
        error: 'Recommendations failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/chat', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const question = typeof body.question === 'string' ? body.question.trim() : '';
    const historyRaw = Array.isArray(body.history) ? body.history : [];
    const history = historyRaw
      .filter((m) => isRecord(m) && (m.role === 'user' || m.role === 'assistant') && typeof m.text === 'string')
      .map((m) => ({ role: m.role as 'user' | 'assistant', text: String(m.text) }))
      .slice(-12);

    const document = isValidMedicalDocument(body.document) ? body.document : undefined;

    if (!question) {
      return reply.status(400).send({ error: 'Question is required' });
    }

    try {
      const answer = await llmService.chat(question, history, document);
      return { success: true, answer };
    } catch (error) {
      console.error('Chat error:', error);
      return reply.status(500).send({
        error: 'Chat failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/augment', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const text = typeof body.text === 'string' ? body.text.trim() : '';
    const document = isValidMedicalDocument(body.document) ? body.document : undefined;

    if (!document) {
      return reply.status(400).send({ error: 'Valid document is required' });
    }

    if (!text) {
      return reply.status(400).send({ error: 'Text is required' });
    }

    try {
      const updated = await llmService.applyAddendum(document, text);

      return {
        success: true,
        document: updated,
      };
    } catch (error) {
      console.error('Augment error:', error);
      return reply.status(500).send({
        error: 'Document augmentation failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.post('/api/instruct', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const instruction = typeof body.instruction === 'string' ? body.instruction.trim() : '';
    const document = isValidMedicalDocument(body.document) ? body.document : undefined;

    if (!document) {
      return reply.status(400).send({ error: 'Valid document is required' });
    }

    if (!instruction) {
      return reply.status(400).send({ error: 'Instruction is required' });
    }

    try {
      const updated = await llmService.applyInstruction(document, instruction);
      const changedFields = rewriteableFields.filter((key) => updated[key] !== document[key]);
      const patientChanged =
        updated.patient.fullName !== document.patient.fullName ||
        updated.patient.age !== document.patient.age ||
        updated.patient.gender !== document.patient.gender ||
        updated.patient.complaintDate !== document.patient.complaintDate;

      return {
        success: true,
        document: updated,
        changedFields,
        patientChanged,
      };
    } catch (error) {
      console.error('Instruction apply error:', error);
      return reply.status(500).send({
        error: 'Instruction apply failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  const audioProcessError = (
    statusCode: number,
    payload: AudioProcessFailurePayload,
  ): AudioProcessHttpError => {
    const error = new Error(payload.message || payload.error) as AudioProcessHttpError;
    error.statusCode = statusCode;
    error.payload = payload;
    return error;
  };

  const processSavedAudioFile = async (
    filepath: string,
    filename: string,
    originalName: string,
    sourceName: string,
    t0: number,
    t1: number,
    onStatus?: (status: AudioJobStatus) => void,
  ): Promise<AudioProcessSuccess> => {
    try {
      onStatus?.('transcribing');
      const transcription = await whisperService.transcribeFile(filepath, filename);
      const t2 = Date.now();

      onStatus?.('structuring');
      let structured: Awaited<ReturnType<LLMService['structureText']>>;
      try {
        structured = await structureTextWithRetry(llmService, transcription.text);
      } catch (structureError) {
        const message = structureError instanceof Error ? structureError.message : 'Unknown error';
        fastify.log.error(
          { sourceName, transcriptionChars: transcription.text.length },
          `process structure failed after successful transcription: ${message}`,
        );
        throw audioProcessError(isRetryableLlmError(structureError) ? 503 : 422, {
          success: false,
          error: 'structure_failed_after_transcription',
          message,
          transcription: {
            text: transcription.text,
            duration: transcription.duration,
            language: transcription.language,
          },
        });
      }

      const sourceEnrichment = enrichDocumentFromSourceName(structured.document, originalName);
      const document = sourceEnrichment.document;
      const t3 = Date.now();

      if (process.env.STRUCTURE_LOG_DUMP === 'true') {
        try {
          const ts = new Date().toISOString().replace(/[:.]/g, '-');
          const logDir = path.join(config.uploadDir, '..', 'temp', 'structure-logs');
          await mkdir(logDir, { recursive: true });
          const logPath = path.join(logDir, `${ts}_${sourceName}.log`);
          const dump = [
            `=== ${ts} | source=${sourceName} ===`,
            `--- RAW WHISPER TEXT (${transcription.text.length} chars) ---`,
            transcription.text,
            '',
            `--- LLM STRUCTURED RESULT ---`,
            JSON.stringify(document, null, 2),
            '',
          ].join('\n');
          await appendFile(logPath, dump, 'utf-8');
          console.log(`[structure-log] wrote ${logPath}`);
        } catch (logErr) {
          console.warn('[structure-log] failed:', logErr);
        }
      }

      const timingsMs = {
        saveFile: t1 - t0,
        whisper: t2 - t1,
        llm: t3 - t2,
        total: t3 - t0,
      };
      fastify.log.info({ timings_ms: timingsMs }, 'process timings');

      const usefulness = assessDocumentUsefulness(document);
      if (usefulness.status === 'empty') {
        fastify.log.warn(
          {
            emptyFields: usefulness.emptyFields,
            placeholderFields: usefulness.placeholderFields,
            transcriptionChars: transcription.text.length,
          },
          'process: document_appears_empty',
        );
        throw audioProcessError(422, {
          success: false,
          error: 'document_appears_empty',
          reason: usefulness.reason,
          emptyFields: usefulness.emptyFields,
          placeholderFields: usefulness.placeholderFields,
          transcription: {
            text: transcription.text,
            duration: transcription.duration,
            language: transcription.language,
          },
        });
      }

      const qualityWarnings = [
        ...sourceEnrichment.warnings,
        ...collectDocumentQualityWarningDetails(transcription.text, document, usefulness),
      ];

      return {
        success: true,
        transcription: {
          text: transcription.text,
          duration: transcription.duration,
          language: transcription.language,
        },
        document,
        processingTime: transcription.duration + structured.processingTime,
        warnings: warningCodes(qualityWarnings),
        qualityWarnings,
        timingsMs,
      };
    } finally {
      try {
        await unlink(filepath);
      } catch {
        // Ignore cleanup errors
      }
    }
  };

  const markAudioJob = (job: AudioProcessJob, status: AudioJobStatus): void => {
    job.status = status;
    job.updatedAt = new Date().toISOString();
    if (status === 'transcribing' && !job.startedAt) {
      job.startedAt = job.updatedAt;
    }
  };

  const runAudioJob = async (
    jobId: string,
    filepath: string,
    filename: string,
    originalName: string,
    sourceName: string,
    t0: number,
    t1: number,
  ): Promise<void> => {
    const job = audioJobs.get(jobId);
    if (!job) return;

    try {
      const result = await processSavedAudioFile(filepath, filename, originalName, sourceName, t0, t1, (status) => {
        markAudioJob(job, status);
      });
      job.status = 'done';
      job.result = result;
      job.transcription = result.transcription;
      job.finishedAt = new Date().toISOString();
      job.updatedAt = job.finishedAt;
    } catch (error) {
      const httpError = error as Partial<AudioProcessHttpError>;
      const payload = httpError.payload;
      job.status = 'failed';
      job.statusCode = typeof httpError.statusCode === 'number' ? httpError.statusCode : 500;
      job.error = payload?.error || 'Processing failed';
      job.message = payload?.message || (error instanceof Error ? error.message : 'Unknown error');
      job.transcription = payload?.transcription;
      job.finishedAt = new Date().toISOString();
      job.updatedAt = job.finishedAt;
      fastify.log.error({ jobId, sourceName, statusCode: job.statusCode }, `audio job failed: ${job.message}`);
    }
  };

  fastify.post('/api/jobs/audio', async (request: FastifyRequest, reply: FastifyReply) => {
    const data = await request.file();

    if (!data) {
      return reply.status(400).send({ error: 'No file uploaded' });
    }

    const originalName = data.filename || 'audio';
    const sourceName = toSafeUploadFilename(originalName);
    const filename = `${Date.now()}_${sourceName}`;
    const filepath = resolveUploadPath(config.uploadDir, filename);
    const t0 = Date.now();

    try {
      await pipeline(data.file, createWriteStream(filepath));
      const t1 = Date.now();
      const nowIso = new Date().toISOString();
      const jobId = randomUUID();
      const job: AudioProcessJob = {
        id: jobId,
        status: 'queued',
        filename,
        sourceName,
        createdAt: nowIso,
        updatedAt: nowIso,
      };
      audioJobs.set(jobId, job);
      void runAudioJob(jobId, filepath, filename, originalName, sourceName, t0, t1);

      return reply.status(202).send({
        success: true,
        jobId,
        status: job.status,
        statusUrl: `/api/jobs/${jobId}`,
      });
    } catch (error) {
      try {
        await unlink(filepath);
      } catch {
        // Ignore cleanup errors
      }
      fastify.log.error({ sourceName }, `audio job upload failed: ${error}`);
      return reply.status(500).send({
        error: 'Audio job upload failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.get('/api/jobs/:id', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const job = audioJobs.get(id);
    if (!job) {
      return reply.status(404).send({ error: 'Job not found or expired' });
    }
    return job;
  });

  fastify.delete('/api/jobs/:id', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };
    const deleted = audioJobs.delete(id);
    return reply.status(deleted ? 200 : 404).send(deleted ? { success: true } : { error: 'Job not found or expired' });
  });

  fastify.post('/api/process', async (request: FastifyRequest, reply: FastifyReply) => {
    const data = await request.file();

    if (!data) {
      return reply.status(400).send({ error: 'No file uploaded' });
    }

    const originalName = data.filename || 'audio';
    const sourceName = toSafeUploadFilename(originalName);
    const filename = `${Date.now()}_${sourceName}`;
    const filepath = resolveUploadPath(config.uploadDir, filename);

    try {
      const t0 = Date.now();
      await pipeline(data.file, createWriteStream(filepath));
      const t1 = Date.now();
      return await processSavedAudioFile(filepath, filename, originalName, sourceName, t0, t1);
    } catch (error) {
      console.error('Processing error:', error);

      const httpError = error as Partial<AudioProcessHttpError>;
      if (httpError.payload && typeof httpError.statusCode === 'number') {
        return reply.status(httpError.statusCode).send(httpError.payload);
      }

      return reply.status(500).send({
        error: 'Processing failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  // ─── Document upload: PDF / Word / Image → LLM ───────────────────────────────

  fastify.post('/api/process-document', async (request: FastifyRequest, reply: FastifyReply) => {
    const data = await request.file();

    if (!data) {
      return reply.status(400).send({ error: 'Файл не загружен' });
    }

    const MAX_SIZE = 20 * 1024 * 1024; // 20 MB
    const chunks: Buffer[] = [];
    let totalSize = 0;

    for await (const chunk of data.file) {
      totalSize += chunk.length;
      if (totalSize > MAX_SIZE) {
        return reply.status(413).send({ error: 'Файл слишком большой. Максимум 20 МБ.' });
      }
      chunks.push(chunk);
    }

    const buffer = Buffer.concat(chunks);

    if (buffer.length === 0) {
      return reply.status(400).send({ error: 'Загружен пустой файл' });
    }

    const safeFilename = toSafeUploadFilename(data.filename || 'document');

    try {
      const t0 = Date.now();

      const extraction = await documentExtractor.extract(buffer, data.mimetype, safeFilename);

      const t1 = Date.now();
      fastify.log.info(
        { method: extraction.extractionMethod, chars: extraction.text.length, ms: t1 - t0 },
        'document extracted',
      );

      const protocolDocument = documentFromConsultationProtocolText(extraction.text);
      const document = protocolDocument ||
        (extraction.extractionMethod === 'vision'
          ? documentFromExactSourceText(extraction.text)
          : (await structureTextWithRetry(llmService, extraction.text)).document);
      const t2 = Date.now();

      const usefulness = assessDocumentUsefulness(document);
      if (usefulness.status === 'empty') {
        request.log.warn(
          { emptyFields: usefulness.emptyFields, method: extraction.extractionMethod },
          'process-document: document_appears_empty',
        );
        return reply.status(422).send({
          success: false,
          error: 'document_appears_empty',
          reason: usefulness.reason,
          emptyFields: usefulness.emptyFields,
          placeholderFields: usefulness.placeholderFields,
          source: { text: extraction.text, extractionMethod: extraction.extractionMethod },
        });
      }

      const qualityWarnings = collectDocumentQualityWarningDetails(
        extraction.text,
        document,
        usefulness,
      );

      fastify.log.info(
        {
          parser: protocolDocument ? 'consultation-protocol' : extraction.extractionMethod,
          timings_ms: { extract: t1 - t0, structure: t2 - t1, total: t2 - t0 },
        },
        'process-document timings',
      );

      return {
        success: true,
        transcription: {
          text: extraction.text,
          language: 'ru',
          extractionMethod: extraction.extractionMethod,
          pageCount: extraction.pageCount,
          warning: extraction.warning,
        },
        document,
        processingTime: t2 - t0,
        warnings: warningCodes(qualityWarnings),
        qualityWarnings,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      fastify.log.error({ filename: safeFilename }, `process-document error: ${message}`);
      const isUserError = /формат|слишком|содержит|отсканирован|требует|Anthropic|баланс|средств|Vision|распознаван|ключ|ограничил/i.test(message);
      return reply.status(isUserError ? 422 : 500).send({
        error: isUserError ? message : 'Ошибка обработки документа',
        message,
      });
    }
  });

  // ─────────────────────────────────────────────────────────────────────────────

  fastify.post('/api/documents', async (request: FastifyRequest, reply: FastifyReply) => {
    const document = request.body;

    if (!isValidMedicalDocument(document)) {
      return reply.status(400).send({ error: 'Valid document is required' });
    }

    return {
      success: true,
      document,
      id: `doc_${Date.now()}`,
      savedAt: new Date().toISOString(),
    };
  });

  fastify.get('/api/document-capabilities', async () => {
    return {
      pdf: true,
      word: true,
      image: documentExtractor.canProcessImages,
    };
  });

  fastify.get('/api/config', async () => {
    return {
      maxRecordingDuration: 30 * 60,
      supportedAudioFormats: ['audio/webm', 'audio/wav', 'audio/mp3', 'audio/ogg'],
      language: config.whisper.language,
      llmModel: config.llm.model,
      ttsEnabled: ttsService.isEnabled,
    };
  });

  fastify.post('/api/tts', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    const text = isRecord(body) && typeof body.text === 'string' ? body.text.trim() : '';

    if (!text) {
      return reply.status(400).send({ error: 'text is required' });
    }

    if (!ttsService.isEnabled) {
      return reply.status(503).send({ error: 'TTS is not enabled on this server' });
    }

    try {
      const audioBase64 = await ttsService.synthesize(text);
      return { success: true, audio_base64: audioBase64, format: 'wav' };
    } catch (error) {
      console.error('TTS error:', error);
      return reply.status(500).send({
        error: 'TTS synthesis failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  // ─── Corrections API (пользовательские замены медицинского словаря) ────────

  fastify.get('/api/corrections', async () => {
    const corrections = getUserCorrections();
    return { corrections, total: corrections.length };
  });

  fastify.post('/api/corrections', async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body;
    if (!isRecord(body)) {
      return reply.status(400).send({ error: 'Invalid request body' });
    }

    const wrong = typeof body.wrong === 'string' ? body.wrong.trim() : '';
    const correct = typeof body.correct === 'string' ? body.correct.trim() : '';
    const scope = normalizeCorrectionScope(typeof body.scope === 'string' ? body.scope : undefined);
    const requireDose = body.requireDose === true;

    if (!wrong || !correct) {
      return reply.status(400).send({ error: "Поля 'wrong' и 'correct' обязательны" });
    }

    if (wrong === correct) {
      return reply.status(400).send({ error: 'Значения должны отличаться' });
    }

    try {
      const correction = await addUserCorrection(wrong, correct, { scope, requireDose });
      const all = getUserCorrections();
      return { success: true, id: correction.id, totalCorrections: all.length };
    } catch (error) {
      console.error('Add correction error:', error);
      return reply.status(500).send({
        error: 'Failed to add correction',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  });

  fastify.delete('/api/corrections/:id', async (request: FastifyRequest, reply: FastifyReply) => {
    const { id } = request.params as { id: string };

    if (!id) {
      return reply.status(400).send({ error: 'ID is required' });
    }

    const deleted = await deleteUserCorrection(id);
    if (!deleted) {
      return reply.status(404).send({ error: 'Замена не найдена' });
    }

    return { success: true };
  });
}

