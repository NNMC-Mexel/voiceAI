import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Copy,
  Download,
  FileText,
  Lightbulb,
  LogOut,
  Mic,
  RotateCcw,
  Search,
  Settings,
  Shield,
  Stethoscope,
} from 'lucide-react';
import { apiClient, ApiRequestError } from '../api/client';
import type {
  DoctorInfo,
  RadiologyApplied,
  RadiologyApprovedReport,
  RadiologyBlockHint,
  RadiologyDictationReport,
  RadiologyReport,
  RadiologyRecomposeRevision,
  RadiologySpanCorrection,
  RadiologyTemplateReviewDraft,
  RadiologyTemplateReviewSegment,
  RadiologyTemplatePreview,
  RadiologyTemplateSummary,
  RadiologyTranscriptionArtifact,
  RadiologyWordTiming,
} from '../api/client';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';

const BROWSER_ASR_FALLBACK_ENABLED =
  import.meta.env.VITE_ENABLE_BROWSER_ASR_FALLBACK === 'true';
const PENDING_RADIOLOGY_SESSION_KEY = 'voicemed:radiology:pending-session:v1';
const LAST_RADIOLOGY_ARTIFACT_KEY = 'voicemed:radiology:last-artifact:v1';

interface StoredRadiologyPointer {
  sessionId: string;
  templateId: string;
}

function isStoredRadiologyPointer(value: unknown): value is StoredRadiologyPointer {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<StoredRadiologyPointer>;
  return (
    typeof candidate.sessionId === 'string'
    && candidate.sessionId.trim() === candidate.sessionId
    && candidate.sessionId.length >= 8
    && candidate.sessionId.length <= 200
    && typeof candidate.templateId === 'string'
    && candidate.templateId.trim() === candidate.templateId
    && candidate.templateId.length >= 1
    && candidate.templateId.length <= 200
  );
}

function clearStoredRadiologyPointer(key: string): void {
  try {
    window.sessionStorage.removeItem(key);
  } catch {
    // Storage can be unavailable in hardened/private browser contexts.
  }
}

function readStoredRadiologyPointer(key: string): StoredRadiologyPointer | null {
  try {
    const raw = window.sessionStorage.getItem(key);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (isStoredRadiologyPointer(parsed)) return parsed;
  } catch {
    // Invalid or unavailable storage is handled as an absent pointer.
  }
  clearStoredRadiologyPointer(key);
  return null;
}

function writeStoredRadiologyPointer(
  key: string,
  pointer: StoredRadiologyPointer,
): void {
  try {
    window.sessionStorage.setItem(key, JSON.stringify(pointer));
  } catch {
    // The workflow remains usable without crash recovery.
  }
}

function audioFileExtension(mimeType: string): string {
  const normalized = mimeType.toLocaleLowerCase('en').split(';', 1)[0].trim();
  const knownExtensions: Record<string, string> = {
    'audio/aac': 'aac',
    'audio/mp4': 'm4a',
    'audio/mpeg': 'mp3',
    'audio/mp3': 'mp3',
    'audio/ogg': 'ogg',
    'audio/wav': 'wav',
    'audio/x-m4a': 'm4a',
    'audio/x-wav': 'wav',
    'audio/webm': 'webm',
  };
  return knownExtensions[normalized] ?? 'webm';
}

function safeProtocolFilename(value: string): string {
  const normalized = value
    .normalize('NFKC')
    .replace(/[^\p{L}\p{N}._-]+/gu, '_')
    .replace(/^[_.]+|[_.]+$/gu, '')
    .slice(0, 80);
  return normalized || 'radiology-protocol';
}

function downloadUtf8Text(text: string, filename: string): void {
  const blob = new Blob(['\uFEFF', text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  try {
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
  } finally {
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  }
}

function localDateStamp(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function formatLocalDateTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? 'дата не указана'
    : date.toLocaleString('ru-RU');
}

async function validateApprovedReport(
  approvedReport: RadiologyApprovedReport,
  artifact: RadiologyTranscriptionArtifact,
): Promise<void> {
  const revision = approvedReport.recomposeRevision;
  const effectiveApprovedReport = revision?.report ?? artifact.report;
  const approvedDraft = effectiveApprovedReport?.reviewDraft;
  const acceptedIds = new Set(approvedReport.acceptedTemplateSegmentIds);
  const reviewedResidualIds = new Set(approvedReport.reviewedResidualAtomIds);
  if (
    approvedReport.sessionId !== artifact.sessionId
    || approvedReport.templateId !== artifact.templateId
    || !/^[a-f0-9]{64}$/u.test(approvedReport.sourceArtifactSha256)
    || !approvedReport.verbatimTranscript.trim()
    || !approvedReport.finalReport.trim()
    || !/^[a-f0-9]{64}$/u.test(approvedReport.finalReportSha256)
    || acceptedIds.size !== approvedReport.acceptedTemplateSegmentIds.length
    || reviewedResidualIds.size !== approvedReport.reviewedResidualAtomIds.length
    || (
      revision !== null
      && (
        revision.sessionId !== artifact.sessionId
        || revision.templateId !== artifact.templateId
        || revision.sourceArtifactSha256 !== approvedReport.sourceArtifactSha256
        || revision.verbatimTranscript.text !== approvedReport.verbatimTranscript
      )
    )
    || (approvedReport.baseDraftSha256 ?? null) !== (approvedDraft?.sha256 ?? null)
    || [...acceptedIds].some((id) => !approvedDraft?.segments.some((segment) => (
      segment.id === id
      && segment.confirmationRequired
      && segment.defaultKind !== 'placeholder'
    )))
    || [...reviewedResidualIds].some((id) => !approvedDraft?.residualAtomIds.includes(id))
  ) {
    throw new Error('Сервер вернул подтверждённый протокол для другого исследования');
  }
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return;
  const digestText = async (text: string): Promise<string> => {
    const digest = await subtle.digest('SHA-256', new TextEncoder().encode(text));
    return [...new Uint8Array(digest)]
      .map((value) => value.toString(16).padStart(2, '0'))
      .join('');
  };
  const finalReportSha256 = await digestText(approvedReport.finalReport);
  const verbatimSha256 = await digestText(approvedReport.verbatimTranscript);
  const expectedVerbatimSha256 = revision?.verbatimTranscript.sha256
    ?? artifact.rawTranscript.sha256;
  if (
    finalReportSha256 !== approvedReport.finalReportSha256
    || verbatimSha256 !== expectedVerbatimSha256
  ) {
    throw new Error('Контрольная сумма подтверждённого протокола не совпала');
  }
}

function createIdempotencyKey(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  const randomPart = Math.random().toString(36).slice(2);
  return `feedback-${Date.now().toString(36)}-${randomPart}-${randomPart}`;
}

interface RadiologyWorkspaceScreenProps {
  doctor: DoctorInfo;
  onOpenSettings?: () => void;
  onOpenAdmin?: () => void;
  onOpenTherapy?: () => void;
  onLogout?: () => void;
}

// Минимальный тип браузерного распознавания речи (нет в стандартных DOM-типах).
interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onstart: (() => void) | null;
  onresult: ((e: { resultIndex: number; results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }> }) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start(): void;
  stop(): void;
}

function getSpeechRecognition(): SpeechRecognitionLike | null {
  const w = window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognitionLike; SpeechRecognition?: new () => SpeechRecognitionLike };
  const Ctor = w.SpeechRecognition || w.webkitSpeechRecognition;
  return Ctor ? new Ctor() : null;
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = typeof reader.result === 'string' ? reader.result : '';
      const encoded = result.split(',')[1];
      if (encoded) resolve(encoded);
      else reject(new Error('Не удалось подготовить аудиочанк'));
    };
    reader.onerror = () => reject(reader.error ?? new Error('Не удалось прочитать аудиочанк'));
    reader.readAsDataURL(blob);
  });
}

function correctionFromTranscriptEdit(
  original: string,
  corrected: string,
  modality: string,
  author: string,
  words: RadiologyWordTiming[] = [],
): RadiologySpanCorrection[] {
  if (original === corrected) return [];

  const lowerOriginal = original.toLocaleLowerCase('ru');
  const wordSpans: Array<{ start: number; end: number; confidence: number }> = [];
  let wordCursor = 0;
  for (const word of words) {
    if (word.confidence === null) continue;
    const needle = word.text.toLocaleLowerCase('ru');
    const start = lowerOriginal.indexOf(needle, wordCursor);
    if (start < 0) continue;
    wordSpans.push({ start, end: start + needle.length, confidence: word.confidence });
    wordCursor = start + needle.length;
  }

  const entityType = (before: string, after: string): string => {
    const value = `${before} ${after}`.toLocaleLowerCase('ru');
    if (/\d|мм|см|миллиметр|сантиметр|hu|хаунсфилд/u.test(value)) return 'number_unit';
    if (/справа|слева|прав(?:ый|ая|ое)|лев(?:ый|ая|ое)/u.test(value)) return 'laterality';
    if (/\b(?:нет|без|не\s+выяв|есть|имеется)\b/u.test(value)) return 'negation';
    if (/контраст/u.test(value)) return 'contrast';
    return 'transcript-span';
  };

  const result: RadiologySpanCorrection[] = [];
  const appendCorrection = (
    originalStart: number,
    originalEnd: number,
    correctedStart: number,
    correctedEnd: number,
  ) => {
    const originalLength = originalEnd - originalStart;
    const correctedLength = correctedEnd - correctedStart;
    const parts = Math.max(
      1,
      Math.ceil(originalLength / 9_000),
      Math.ceil(correctedLength / 9_000),
    );
    for (let part = 0; part < parts; part++) {
      const from = originalStart + Math.floor((originalLength * part) / parts);
      const to = originalStart + Math.floor((originalLength * (part + 1)) / parts);
      const replacementFrom = correctedStart + Math.floor((correctedLength * part) / parts);
      const replacementTo = correctedStart + Math.floor((correctedLength * (part + 1)) / parts);
      const originalText = original.slice(from, to);
      const correctedText = corrected.slice(replacementFrom, replacementTo);
      const overlappingConfidence = wordSpans
        .filter((word) => word.start < to && word.end > from)
        .map((word) => word.confidence);
      result.push({
        start: from,
        end: to,
        originalText,
        correctedText,
        entityType: entityType(originalText, correctedText),
        ...(overlappingConfidence.length
          ? { confidence: Math.min(...overlappingConfidence) }
          : {}),
        modality,
        author,
      });
    }
  };

  const anchorLength = 12;
  const searchWindow = 2_000;
  let originalCursor = 0;
  let correctedCursor = 0;
  while (originalCursor < original.length || correctedCursor < corrected.length) {
    while (
      originalCursor < original.length
      && correctedCursor < corrected.length
      && original[originalCursor] === corrected[correctedCursor]
    ) {
      originalCursor++;
      correctedCursor++;
    }
    if (originalCursor === original.length && correctedCursor === corrected.length) break;

    let synchronization: { original: number; corrected: number } | null = null;
    let bestCost = Number.POSITIVE_INFINITY;
    const originalSearchEnd = Math.min(
      original.length - anchorLength,
      originalCursor + searchWindow,
    );
    for (let candidate = originalCursor; candidate <= originalSearchEnd; candidate++) {
      const anchor = original.slice(candidate, candidate + anchorLength);
      const correctedCandidate = corrected.indexOf(anchor, correctedCursor);
      if (
        correctedCandidate < 0
        || correctedCandidate > correctedCursor + searchWindow
      ) continue;
      const cost = (candidate - originalCursor) + (correctedCandidate - correctedCursor);
      if (cost < bestCost) {
        synchronization = { original: candidate, corrected: correctedCandidate };
        bestCost = cost;
        if (cost === 0) break;
      }
    }

    if (!synchronization) {
      appendCorrection(
        originalCursor,
        original.length,
        correctedCursor,
        corrected.length,
      );
      break;
    }
    appendCorrection(
      originalCursor,
      synchronization.original,
      correctedCursor,
      synchronization.corrected,
    );
    originalCursor = synchronization.original;
    correctedCursor = synchronization.corrected;
  }
  return result.filter(
    (correction) => correction.originalText !== correction.correctedText,
  );
}

type RecognitionMode = 'gigaam' | 'browser';

function toDisplayReport(report: RadiologyDictationReport): RadiologyReport {
  return {
    title: report.title,
    blocks: report.blocks
      .filter((block) => block.origin !== 'template_default')
      .map((block) => ({
        id: block.id,
        label: block.label,
        text: block.text,
      })),
    text: report.fullText,
    conclusion: report.blocks.find((block) => block.source === 'conclusion')?.text ?? '',
  };
}

function reviewSegmentTitle(segment: RadiologyTemplateReviewSegment): string {
  const evidence = segment.evidence.map((item) => item.text).filter(Boolean).join(' · ');
  switch (segment.kind) {
    case 'transcript_value':
      return evidence ? `Подставлено из диктовки: «${evidence}»` : 'Подставлено из диктовки';
    case 'template_choice':
      return evidence
        ? `Вариант выбран по диктовке: «${evidence}»`
        : 'Вариант шаблона';
    case 'derived':
      return evidence
        ? `Детерминированно вычислено из: «${evidence}»`
        : 'Детерминированно вычисленное значение';
    case 'verbatim':
      return evidence ? `Дословный остаточный фрагмент: «${evidence}»` : 'Дословный фрагмент';
    case 'template_default':
      return segment.defaultKind === 'placeholder'
        ? 'Незаполненное поле шаблона'
        : 'Норма или значение по умолчанию из шаблона';
    default:
      return segment.confirmationRequired
        ? 'Текст шаблона, требующий подтверждения'
        : 'Служебный текст шаблона';
  }
}

function ReviewDraftSegment({
  segment,
  accepted,
}: {
  segment: RadiologyTemplateReviewSegment;
  accepted: boolean;
}) {
  const evidenceBackedChoice = segment.kind === 'template_choice' && segment.evidence.length > 0;
  const className = [
    'rounded-sm',
    segment.kind === 'transcript_value' || evidenceBackedChoice
      ? 'bg-emerald-100 text-emerald-950 font-semibold px-0.5'
      : '',
    segment.kind === 'template_default'
      ? segment.defaultKind === 'placeholder'
        ? 'bg-amber-50 text-amber-900 px-0.5'
        : 'bg-slate-100 text-slate-700 px-0.5'
      : '',
    segment.kind === 'template_literal' ? 'text-slate-600' : '',
    segment.kind === 'template_choice' && !evidenceBackedChoice
      ? 'bg-slate-100 text-slate-700 px-0.5'
      : '',
    segment.kind === 'derived' ? 'bg-indigo-50 text-indigo-900 px-0.5' : '',
    segment.kind === 'verbatim' ? 'bg-amber-100 text-amber-950 px-0.5' : '',
    segment.confirmationRequired && !accepted ? 'opacity-50 line-through' : '',
  ].filter(Boolean).join(' ');

  return (
    <span className={className} title={reviewSegmentTitle(segment)}>
      {segment.text}
      {segment.kind === 'derived' && (
        <sup className="ml-0.5 text-[9px] font-bold text-indigo-600">fx</sup>
      )}
    </span>
  );
}

function segmentsForSection(
  draft: RadiologyTemplateReviewDraft,
  segmentIds: string[],
): RadiologyTemplateReviewSegment[] {
  const byId = new Map(draft.segments.map((segment) => [segment.id, segment]));
  return segmentIds
    .map((segmentId) => byId.get(segmentId))
    .filter((segment): segment is RadiologyTemplateReviewSegment => Boolean(segment));
}

function composeAcceptedReviewDraftText(
  draft: RadiologyTemplateReviewDraft,
  acceptedSegmentIds: Set<string>,
  sourceAtoms: RadiologyTranscriptionArtifact['routing']['atoms'] = [],
): string {
  const confirmationSegments = draft.segments.filter(
    (segment) => segment.confirmationRequired,
  );
  if (
    confirmationSegments.every((segment) => acceptedSegmentIds.has(segment.id))
  ) {
    return draft.fullText;
  }

  let text = draft.title;
  const atomById = new Map(sourceAtoms.map((sourceAtom) => [sourceAtom.id, sourceAtom]));
  for (const section of draft.sections) {
    const sectionSegments = segmentsForSection(draft, section.segmentIds);
    const confirmable = sectionSegments.filter(
      (segment) => segment.confirmationRequired,
    );
    const acceptedConfirmable = confirmable.filter(
      (segment) => acceptedSegmentIds.has(segment.id),
    );
    let body = sectionSegments
      .filter(
        (segment) => (
          !segment.confirmationRequired || acceptedSegmentIds.has(segment.id)
        ),
      )
      .map((segment) => segment.text)
      .join('')
      .trim();
    const hasUnacceptedPlaceholder = sectionSegments.some(
      (segment) => (
        segment.confirmationRequired
        && segment.defaultKind === 'placeholder'
        && !acceptedSegmentIds.has(segment.id)
      ),
    );

    // If the doctor disables all template text for a section that contains
    // dictated evidence, keep the exact evidence instead of leaving isolated
    // slot values such as "150 60" without their acoustic context.
    if (
      hasUnacceptedPlaceholder
      || (confirmable.length > 0 && acceptedConfirmable.length === 0)
    ) {
      const evidenceSpans = sectionSegments
        .flatMap((segment) => segment.evidence)
        .sort((left, right) => left.start - right.start || left.end - right.end)
        .filter((item, index, all) => (
          index === 0
          || item.atomId !== all[index - 1].atomId
          || item.start !== all[index - 1].start
          || item.end !== all[index - 1].end
        ));
      const evidenceAtoms = [...new Set(evidenceSpans.map((item) => item.atomId))]
        .map((atomId) => atomById.get(atomId))
        .filter(
          (sourceAtom): sourceAtom is RadiologyTranscriptionArtifact['routing']['atoms'][number] => (
            Boolean(sourceAtom)
          ),
        )
        .sort((left, right) => left.start - right.start || left.end - right.end);
      body = (
        evidenceAtoms.length > 0
          ? evidenceAtoms.map((sourceAtom) => sourceAtom.text)
          : evidenceSpans.map((item) => item.text)
      ).join(' ').trim();
    }
    if (!body) continue;
    text += `\n${section.label}: ${body}`;
  }
  return text;
}

function initiallyAcceptedTemplateSegments(
  draft: RadiologyTemplateReviewDraft | undefined,
): Set<string> {
  const residualSectionIds = new Set(
    draft?.segments
      .filter((segment) => segment.origin === 'transcript_append')
      .map((segment) => segment.sectionId) ?? [],
  );
  return new Set(
    draft?.segments
      .filter((segment) => segment.confirmationRequired)
      .filter((segment) => segment.defaultKind !== 'placeholder')
      // A section with out-of-schema clinical text must be reviewed as a
      // whole; its normal template literals are never pre-accepted.
      .filter((segment) => !residualSectionIds.has(segment.sectionId))
      .map((segment) => segment.id) ?? [],
  );
}

export function RadiologyWorkspaceScreen({ doctor, onOpenSettings, onOpenAdmin, onOpenTherapy, onLogout }: RadiologyWorkspaceScreenProps) {
  const [templates, setTemplates] = useState<RadiologyTemplateSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [commands, setCommands] = useState<string[]>([]);
  const [report, setReport] = useState<RadiologyReport | null>(null);
  const [applied, setApplied] = useState<RadiologyApplied[]>([]);
  const [input, setInput] = useState('');
  const [recognitionMode, setRecognitionMode] = useState<RecognitionMode>('gigaam');
  const [retainAudioConsent, setRetainAudioConsent] = useState(false);
  const [retentionNotice, setRetentionNotice] = useState('');
  const [browserPhiConsent, setBrowserPhiConsent] = useState(false);
  const [browserListening, setBrowserListening] = useState(false);
  const [startingRecording, setStartingRecording] = useState(false);
  const [processingAudio, setProcessingAudio] = useState(false);
  const [recoveringArtifact, setRecoveringArtifact] = useState(false);
  const [interim, setInterim] = useState('');
  const [copied, setCopied] = useState(false);
  const [hints, setHints] = useState<RadiologyBlockHint[]>([]);
  const [showHints, setShowHints] = useState(true);
  const [templatePreview, setTemplatePreview] = useState<RadiologyTemplatePreview | null>(null);
  const [templatePreviewLoading, setTemplatePreviewLoading] = useState(false);
  const [artifact, setArtifact] = useState<RadiologyTranscriptionArtifact | null>(null);
  const [approvedReport, setApprovedReport] = useState<RadiologyApprovedReport | null>(null);
  const [reviewRevision, setReviewRevision] = useState<RadiologyRecomposeRevision | null>(null);
  const [recomposingReview, setRecomposingReview] = useState(false);
  const [acceptedTemplateSegmentIds, setAcceptedTemplateSegmentIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [reviewedResidualAtomIds, setReviewedResidualAtomIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [verbatimTranscript, setVerbatimTranscript] = useState('');
  const [finalReportText, setFinalReportText] = useState('');
  const [finalReportManuallyEdited, setFinalReportManuallyEdited] = useState(false);
  const [finalReportEditing, setFinalReportEditing] = useState(false);
  const [compositionReviewOpen, setCompositionReviewOpen] = useState(false);
  const [sourceReviewOpen, setSourceReviewOpen] = useState(false);
  const [reviewError, setReviewError] = useState('');
  const [feedbackState, setFeedbackState] = useState<
    'idle' | 'saving' | 'saved' | 'saved_unverified'
  >('idle');
  const recogRef = useRef<SpeechRecognitionLike | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const attemptedAudioBlobRef = useRef<Blob | null>(null);
  const browserGenerationRef = useRef(0);
  const operationInFlightRef = useRef(false);
  const feedbackSavingRef = useRef(false);
  const finalReportEditBaselineRef = useRef('');
  const finalReportEditFeedbackBaselineRef = useRef<'idle' | 'saved'>('idle');
  const finalReportEditManualBaselineRef = useRef(false);
  const recomposeGenerationRef = useRef(0);
  const copyResetTimerRef = useRef<number | null>(null);
  const feedbackSubmissionRef = useRef<{
    payloadSignature: string;
    idempotencyKey: string;
  } | null>(null);
  const selectedTemplateIdRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const recoveryAttemptedRef = useRef(false);
  const pendingClientTranscriptRef = useRef<{
    sessionId?: string;
    transcript: string;
    source: 'browser' | 'manual';
    templateId: string;
  } | null>(null);
  const [audioRetryToken, setAudioRetryToken] = useState(0);

  const selected = templates.find((t) => t.id === selectedId) || null;
  const effectiveReport = reviewRevision?.report ?? artifact?.report ?? null;
  const effectiveNormalization = reviewRevision?.normalization ?? artifact?.normalization ?? null;
  const effectiveRouting = reviewRevision?.routing ?? artifact?.routing ?? null;
  const effectiveSafety = reviewRevision?.safety ?? artifact?.safety ?? null;
  const reviewDraft = effectiveReport?.reviewDraft ?? null;
  const displayedDocument = reviewDraft ? null : report ?? templatePreview;
  const showingTemplateDefaults = report === null && templatePreview !== null;
  const reviewConfirmationSegments = reviewDraft?.segments.filter(
    (segment) => segment.confirmationRequired,
  ) ?? [];
  const confirmationSegments = reviewConfirmationSegments.filter(
    (segment) => segment.defaultKind !== 'placeholder',
  ) ?? [];
  const unacceptedTemplateSegmentIds = confirmationSegments
    .filter((segment) => !acceptedTemplateSegmentIds.has(segment.id))
    .map((segment) => segment.id);
  const unreviewedResidualAtomIds = (reviewDraft?.residualAtomIds ?? [])
    .filter((atomId) => !reviewedResidualAtomIds.has(atomId));
  const incompleteRequiredFieldIds = (reviewDraft?.fieldAssignments ?? [])
    .filter((assignment) => assignment.status === 'incomplete' && assignment.evidence.length === 0)
    .map((assignment) => assignment.fieldId);
  const unresolvedCriticalDraftIssues = (reviewDraft?.issues ?? []).filter((issue) => (
    issue.severity === 'critical'
    && (!issue.atomId || !reviewedResidualAtomIds.has(issue.atomId))
  ));
  selectedTemplateIdRef.current = selectedId;
  const reviewSpanCorrections = useMemo(
    () => (
      artifact && selected
        ? correctionFromTranscriptEdit(
            artifact.rawTranscript.text,
            verbatimTranscript,
            selected.modality,
            doctor.name,
            artifact.rawTranscript.words,
          )
        : []
    ),
    [artifact, doctor.name, selected, verbatimTranscript],
  );
  const residualEvidenceByAtomId = useMemo(() => {
    const result = new Map<string, string>();
    if (!reviewDraft) return result;
    for (const segment of reviewDraft.segments) {
      if (segment.kind !== 'verbatim') continue;
      for (const evidence of segment.evidence) {
        if (!result.has(evidence.atomId)) result.set(evidence.atomId, evidence.text);
      }
    }
    for (const issue of reviewDraft.issues) {
      if (!issue.atomId || result.has(issue.atomId)) continue;
      const text = issue.evidence?.map((item) => item.text).filter(Boolean).join(' ');
      if (text) result.set(issue.atomId, text);
    }
    return result;
  }, [reviewDraft]);
  const unresolvedCriticalNormalization = effectiveNormalization
    ? effectiveNormalization.issues.some((issue) => {
        if (issue.severity !== 'critical') return false;
        if (reviewRevision) return true;
        if (!issue.source) return true;
        return !reviewSpanCorrections.some(
          (correction) =>
            correction.start <= issue.source!.start
            && correction.end >= issue.source!.end
            && correction.correctedText.trim().length > 0,
        );
      })
    : false;
  const reviewRequiresRecompose = (
    reviewSpanCorrections.length > 0
    && reviewRevision?.verbatimTranscript.text !== verbatimTranscript
  );
  const reviewMutationBusy = recomposingReview || feedbackState === 'saving';
  const feedbackLocked = feedbackState === 'saved' || feedbackState === 'saved_unverified';
  const approvedReady = Boolean(
    feedbackState === 'saved'
    && approvedReport
    && approvedReport.finalReport === finalReportText,
  );
  const approvalBlockReasons = useMemo(() => {
    if (!artifact) return [];
    const reasons: string[] = [];
    if (artifact.legacySchemaVersion === 1) {
      reasons.push('Artifact v1 нужно повторно прогнать через pipeline v2.');
    }
    if (!finalReportText.trim()) {
      reasons.push('Финальный протокол пуст. Добавьте текст перед подтверждением.');
    }
    if (reviewRequiresRecompose) {
      reasons.push('После исправления дословной расшифровки пересоберите шаблонный черновик.');
    }
    if (effectiveReport === null) {
      reasons.push('Безопасный черновик не построен; запись нужно обработать повторно.');
    }
    if (unresolvedCriticalNormalization) {
      reasons.push('Есть неоднозначность в распознанном фрагменте. Исправьте дословную расшифровку и пересоберите протокол.');
    }
    if (
      artifact.longform.degraded
      || artifact.longform.seamConflicts.some((seam) => seam.critical)
    ) {
      reasons.push('Длинная запись обработана в резервном режиме или содержит критический конфликт на стыке фрагментов.');
    }
    if ((effectiveRouting?.unmatchedAtomIds.length ?? 0) > 0) {
      reasons.push('Есть клинические фрагменты без секции; сначала требуется повторная маршрутизация.');
    }
    if (incompleteRequiredFieldIds.length > 0) {
      reasons.push(`Заполните обязательные поля шаблона: ${incompleteRequiredFieldIds.join(', ')}.`);
    }
    if (reviewDraft?.status === 'failed') {
      reasons.push('Шаблонный черновик не построен безопасно. Проверьте исходный текст и повторите обработку.');
    }
    if (unresolvedCriticalDraftIssues.length > 0) {
      reasons.push('В шаблонном черновике остались критические ошибки разбора.');
    }
    if (unreviewedResidualAtomIds.length > 0) {
      reasons.push('Проверьте все остаточные дословные фрагменты перед подтверждением.');
    }
    if (effectiveSafety?.approvalBlocked) {
      reasons.push('Проверки безопасности не пройдены. Подробности доступны в разделе расшифровки и ошибок.');
    }
    return [...new Set(reasons)];
  }, [
    artifact,
    effectiveReport,
    effectiveRouting,
    effectiveSafety,
    finalReportText,
    incompleteRequiredFieldIds,
    reviewDraft,
    reviewRequiresRecompose,
    unresolvedCriticalDraftIssues,
    unresolvedCriticalNormalization,
    unreviewedResidualAtomIds,
  ]);
  const hardApprovalBlockReason = approvalBlockReasons[0] ?? null;

  const confirmDiscardCurrentReview = useCallback((ignoreManualInput = false): boolean => {
    const hasUnsavedManualInput = !ignoreManualInput && input.trim().length > 0;
    if (!artifact) {
      if (!hasUnsavedManualInput) return true;
      return window.confirm(
        'В поле ручной расшифровки есть несохранённый текст. Уйти и потерять его?',
      );
    }
    if (approvedReady && !hasUnsavedManualInput) return true;
    if (approvedReady) {
      return window.confirm(
        'В поле ручной расшифровки есть несохранённый текст. Уйти и потерять его?',
      );
    }
    if (feedbackState === 'saved_unverified') {
      return window.confirm(
        `Подтверждение сохранено, но контрольная копия ещё не сверена.${
          hasUnsavedManualInput ? ' В поле ручной расшифровки также есть несохранённый текст.' : ''
        } Уйти до повторной сверки с сервером?`,
      );
    }
    return window.confirm(
      `Текущий протокол ещё не подтверждён.${
        hasUnsavedManualInput ? ' В поле ручной расшифровки также есть несохранённый текст.' : ''
      } Уйти и потерять несохранённые правки?`,
    );
  }, [approvedReady, artifact, feedbackState, input]);

  const {
    isRecording,
    isStopping,
    audioBlob,
    formattedDuration,
    startRecording,
    stopRecording,
    resetRecording,
  } = useVoiceRecorder();

  const listening = recognitionMode === 'gigaam' ? isRecording : browserListening;
  const hasPendingServerAudio = recognitionMode === 'gigaam'
    && Boolean(audioBlob);
  const hasPendingClientTranscript = pendingClientTranscriptRef.current !== null;
  const workflowBusy =
    listening
    || startingRecording
    || isStopping
    || processingAudio
    || recoveringArtifact
    || recomposingReview
    || hasPendingServerAudio
    || hasPendingClientTranscript
    || feedbackState === 'saving';

  const applyArtifact = useCallback((nextArtifact: RadiologyTranscriptionArtifact) => {
    if (nextArtifact.templateId !== selectedTemplateIdRef.current) {
      throw new Error('Получен результат для уже сменённого шаблона; он не был применён');
    }
    const transcript = nextArtifact.normalization.text.trim();
    const displayReport = nextArtifact.report ? toDisplayReport(nextArtifact.report) : null;
    const nextReviewDraft = nextArtifact.report?.reviewDraft;
    const appliedFieldCount = nextReviewDraft?.fieldAssignments.filter(
      (assignment) => assignment.status === 'applied',
    ).length ?? 0;
    const residualCount = nextReviewDraft?.residualAtomIds.length ?? 0;
    setArtifact(nextArtifact);
    setApprovedReport(null);
    setReviewRevision(null);
    setRecomposingReview(false);
    const initiallyAcceptedTemplateSegmentIds = initiallyAcceptedTemplateSegments(
      nextReviewDraft,
    );
    setAcceptedTemplateSegmentIds(initiallyAcceptedTemplateSegmentIds);
    setReviewedResidualAtomIds(new Set());
    setVerbatimTranscript(nextArtifact.rawTranscript.text);
    setFinalReportText(
      nextReviewDraft
        ? composeAcceptedReviewDraftText(
            nextReviewDraft,
            initiallyAcceptedTemplateSegmentIds,
            nextArtifact.routing.atoms,
          )
        : nextArtifact.report?.fullText ?? '',
    );
    setFinalReportManuallyEdited(false);
    setFinalReportEditing(false);
    setCompositionReviewOpen(Boolean(
      nextReviewDraft
      && (
        nextReviewDraft.status === 'failed'
        || nextReviewDraft.residualAtomIds.length > 0
        || nextReviewDraft.fieldAssignments.some((assignment) => assignment.status !== 'applied')
      )
    ));
    setSourceReviewOpen(
      nextArtifact.normalization.issues.some((issue) => issue.severity === 'critical'),
    );
    setCopied(false);
    setReviewError('');
    feedbackSubmissionRef.current = null;
    setCommands(transcript ? [transcript] : []);
    setReport(displayReport);
    setApplied([{
      command: transcript,
      ok: !nextArtifact.safety.approvalBlocked,
      action: 'server-artifact',
      detail: nextReviewDraft
        ? [
            `Подставлено полей: ${appliedFieldCount}`,
            residualCount > 0
              ? `Остаточных фрагментов для проверки: ${residualCount}`
              : '',
          ].filter(Boolean).join('; ')
        : nextArtifact.safety.status === 'passed'
          ? 'проверки безопасности пройдены'
          : 'нужна проверка врача',
    }]);
    writeStoredRadiologyPointer(LAST_RADIOLOGY_ARTIFACT_KEY, {
      sessionId: nextArtifact.sessionId,
      templateId: nextArtifact.templateId,
    });
    clearStoredRadiologyPointer(PENDING_RADIOLOGY_SESSION_KEY);
  }, []);

  const applyApprovedReport = useCallback((
    nextApprovedReport: RadiologyApprovedReport,
    sourceArtifact: RadiologyTranscriptionArtifact,
  ) => {
    const revision = nextApprovedReport.recomposeRevision;
    const nextReport = revision?.report ?? sourceArtifact.report;
    const nextRouting = revision?.routing ?? sourceArtifact.routing;
    const nextDraft = nextReport?.reviewDraft;
    const acceptedIds = new Set(nextApprovedReport.acceptedTemplateSegmentIds);
    const reviewedIds = new Set(nextApprovedReport.reviewedResidualAtomIds);
    const composedText = nextDraft
      ? composeAcceptedReviewDraftText(nextDraft, acceptedIds, nextRouting.atoms)
      : nextReport?.fullText ?? '';
    setApprovedReport(nextApprovedReport);
    setReviewRevision(revision);
    setVerbatimTranscript(nextApprovedReport.verbatimTranscript);
    setAcceptedTemplateSegmentIds(acceptedIds);
    setReviewedResidualAtomIds(reviewedIds);
    setFinalReportText(nextApprovedReport.finalReport);
    setFinalReportManuallyEdited(nextApprovedReport.finalReport !== composedText);
    setFinalReportEditing(false);
    setCompositionReviewOpen(false);
    setSourceReviewOpen(false);
    setCopied(false);
    setReviewError('');
    setReport(nextReport ? toDisplayReport(nextReport) : null);
    feedbackSubmissionRef.current = null;
    setFeedbackState('saved');
  }, []);

  useEffect(() => {
    let cancelled = false;
    apiClient.getRadiologyTemplates()
      .then((list) => { if (!cancelled) setTemplates(list); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : 'Не удалось загрузить шаблоны'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (loading || templates.length === 0 || recoveryAttemptedRef.current) return;
    recoveryAttemptedRef.current = true;

    const candidates = [
      {
        key: PENDING_RADIOLOGY_SESSION_KEY,
        pointer: readStoredRadiologyPointer(PENDING_RADIOLOGY_SESSION_KEY),
      },
      {
        key: LAST_RADIOLOGY_ARTIFACT_KEY,
        pointer: readStoredRadiologyPointer(LAST_RADIOLOGY_ARTIFACT_KEY),
      },
    ].filter(
      (candidate): candidate is { key: string; pointer: StoredRadiologyPointer } =>
        candidate.pointer !== null,
    );
    if (candidates.length === 0) return;

    let cancelled = false;
    setRecoveringArtifact(true);
    setError('');
    void (async () => {
      try {
        for (const candidate of candidates) {
          const { key, pointer } = candidate;
          if (!templates.some((template) => template.id === pointer.templateId)) {
            clearStoredRadiologyPointer(key);
            continue;
          }

          let persisted: { artifact: RadiologyTranscriptionArtifact };
          try {
            persisted = await apiClient.getRadiologyArtifact(pointer.sessionId);
          } catch (requestError) {
            if (cancelled) return;
            const terminal =
              requestError instanceof ApiRequestError
              && [400, 403, 404, 409, 410].includes(requestError.status);
            if (terminal) {
              clearStoredRadiologyPointer(key);
              continue;
            }
            setError(
              requestError instanceof Error
                ? requestError.message
                : 'Не удалось восстановить последний протокол',
            );
            return;
          }

          if (cancelled) return;
          const recoveredArtifact = persisted.artifact;
          if (
            !recoveredArtifact
            || recoveredArtifact.sessionId !== pointer.sessionId
            || recoveredArtifact.templateId !== pointer.templateId
          ) {
            clearStoredRadiologyPointer(key);
            continue;
          }

          let recoveredApprovedReport: RadiologyApprovedReport | null = null;
          let approvedReportLoadError = '';
          try {
            const response = await apiClient.getRadiologyApprovedReport(pointer.sessionId);
            await validateApprovedReport(response.approvedReport, recoveredArtifact);
            recoveredApprovedReport = response.approvedReport;
          } catch (approvedError) {
            const notFound = approvedError instanceof ApiRequestError
              && approvedError.status === 404;
            if (!notFound) {
              approvedReportLoadError = approvedError instanceof Error
                ? approvedError.message
                : 'Не удалось восстановить подтверждённую версию протокола';
            }
          }

          selectedTemplateIdRef.current = pointer.templateId;
          setSelectedId(pointer.templateId);
          applyArtifact(recoveredArtifact);
          if (recoveredApprovedReport) {
            applyApprovedReport(recoveredApprovedReport, recoveredArtifact);
          } else if (approvedReportLoadError) {
            setFeedbackState('saved_unverified');
            setReviewError(
              `Черновик восстановлен, но подтверждённую версию загрузить не удалось: ${approvedReportLoadError}`,
            );
          }
          setError('');
          return;
        }
      } finally {
        if (!cancelled) setRecoveringArtifact(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [applyApprovedReport, applyArtifact, loading, templates]);

  // Подсказки «что можно диктовать» для выбранного шаблона.
  useEffect(() => {
    if (!selectedId) {
      setHints([]);
      setTemplatePreview(null);
      setTemplatePreviewLoading(false);
      return;
    }
    let cancelled = false;
    setTemplatePreview(null);
    setTemplatePreviewLoading(true);
    apiClient.getRadiologyHints(selectedId)
      .then((h) => { if (!cancelled) setHints(h); })
      .catch(() => { if (!cancelled) setHints([]); });
    apiClient.getRadiologyTemplatePreview(selectedId)
      .then((preview) => {
        if (!cancelled) setTemplatePreview(preview);
      })
      .catch((previewError) => {
        if (!cancelled) {
          setTemplatePreview(null);
          setError(
            previewError instanceof Error
              ? previewError.message
              : 'Не удалось загрузить предпросмотр шаблона',
          );
        }
      })
      .finally(() => {
        if (!cancelled) setTemplatePreviewLoading(false);
      });
    return () => { cancelled = true; };
  }, [selectedId]);

  const fillExample = useCallback((ex: string) => {
    // Пример с многоточием («… добавь …») — просто в поле для правки; иначе тоже в поле.
    setInput(ex.replace(/\s*…\s*$/, ' '));
    inputRef.current?.focus();
  }, []);

  const clearClinicalResult = useCallback(() => {
    setArtifact(null);
    setApprovedReport(null);
    setReviewRevision(null);
    setRecomposingReview(false);
    setAcceptedTemplateSegmentIds(new Set());
    setReviewedResidualAtomIds(new Set());
    setReport(null);
    setCommands([]);
    setApplied([]);
    setInput('');
    setVerbatimTranscript('');
    setFinalReportText('');
    setFinalReportManuallyEdited(false);
    setFinalReportEditing(false);
    setCompositionReviewOpen(false);
    setSourceReviewOpen(false);
    setReviewError('');
    setCopied(false);
    setFeedbackState('idle');
  }, []);

  const stopBrowserListening = useCallback(() => {
    const recognition = recogRef.current;
    if (!recognition) return;
    try {
      recognition.stop();
    } catch {
      // The recognizer may already be stopping; onend owns finalization.
    }
  }, []);

  const finishClientTranscriptArtifact = useCallback(async (
    transcript: string,
    source: 'browser' | 'manual' = 'browser',
  ): Promise<void> => {
    const clean = transcript.trim();
    if (!selectedId || !clean || operationInFlightRef.current) return;
    if (!confirmDiscardCurrentReview(source === 'manual')) return;
    clearClinicalResult();
    operationInFlightRef.current = true;
    let pending = pendingClientTranscriptRef.current;
    if (
      !pending
      || pending.transcript !== clean
      || pending.source !== source
      || pending.templateId !== selectedId
    ) {
      pending = {
        transcript: clean,
        source,
        templateId: selectedId,
      };
      pendingClientTranscriptRef.current = pending;
    }
    setProcessingAudio(true);
    setError('');
    setRetentionNotice('');
    try {
      let sessionId = pending.sessionId;
      if (!sessionId) {
        const session = await apiClient.startRadiologySession(selectedId, source, false);
        sessionId = session.sessionId;
        pendingClientTranscriptRef.current = {
          ...pending,
          sessionId: session.sessionId,
        };
        writeStoredRadiologyPointer(PENDING_RADIOLOGY_SESSION_KEY, {
          sessionId,
          templateId: selectedId,
        });
      }
      const result = await apiClient.finishRadiologySession(sessionId, clean);
      if (mountedRef.current) {
        applyArtifact(result.artifact);
        pendingClientTranscriptRef.current = null;
        if (source === 'browser') setBrowserPhiConsent(false);
      }
    } catch (e) {
      if (mountedRef.current) {
        setError(e instanceof Error ? e.message : 'Не удалось обработать клиентскую расшифровку');
      }
    } finally {
      operationInFlightRef.current = false;
      if (mountedRef.current) setProcessingAudio(false);
    }
  }, [applyArtifact, clearClinicalResult, confirmDiscardCurrentReview, selectedId]);

  const startBrowserListening = useCallback(() => {
    if (!BROWSER_ASR_FALLBACK_ENABLED || !browserPhiConsent) {
      setError('Browser fallback отключён политикой или не подтверждён врачом.');
      return;
    }
    if (
      recogRef.current
      || operationInFlightRef.current
      || pendingClientTranscriptRef.current
      || recoveringArtifact
    ) return;
    if (!confirmDiscardCurrentReview()) return;
    const recog = getSpeechRecognition();
    if (!recog) {
      setBrowserPhiConsent(false);
      setError('Голосовой ввод недоступен в этом браузере — используйте поле ввода команды ниже.');
      return;
    }
    const generation = ++browserGenerationRef.current;
    const transcriptParts: string[] = [];
    let recognitionFailed = false;
    recog.lang = 'ru-RU';
    recog.continuous = true;
    recog.interimResults = true;
    recog.onstart = () => {
      if (generation !== browserGenerationRef.current || recogRef.current !== recog) return;
      clearClinicalResult();
      setError('');
    };
    recog.onresult = (e) => {
      if (generation !== browserGenerationRef.current || recogRef.current !== recog) return;
      let finalText = '';
      let interimText = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) finalText += r[0].transcript;
        else interimText += r[0].transcript;
      }
      if (finalText) {
        const clean = finalText.trim();
        if (clean) {
          transcriptParts.push(clean);
        }
      }
      setInterim(interimText);
    };
    recog.onend = () => {
      if (generation !== browserGenerationRef.current || recogRef.current !== recog) return;
      recogRef.current = null;
      setBrowserListening(false);
      setInterim('');
      if (!recognitionFailed) {
        const transcript = transcriptParts.join(' ').trim();
        if (transcript) {
          void finishClientTranscriptArtifact(transcript, 'browser');
        } else {
          setBrowserPhiConsent(false);
          setError('Browser SpeechRecognition не вернул финальный текст.');
        }
      }
    };
    recog.onerror = () => {
      if (generation !== browserGenerationRef.current || recogRef.current !== recog) return;
      recognitionFailed = true;
      recogRef.current = null;
      setBrowserListening(false);
      setInterim('');
      setBrowserPhiConsent(false);
      setError('Browser SpeechRecognition завершился с ошибкой; запись не была сохранена.');
    };
    recogRef.current = recog;
    try {
      recog.start();
      setBrowserListening(true);
    } catch (error) {
      recogRef.current = null;
      recognitionFailed = true;
      setBrowserPhiConsent(false);
      setError(error instanceof Error ? error.message : 'Не удалось запустить browser fallback');
    }
  }, [
    browserPhiConsent,
    clearClinicalResult,
    confirmDiscardCurrentReview,
    finishClientTranscriptArtifact,
    recoveringArtifact,
  ]);

  const startGigaamListening = useCallback(async () => {
    if (
      !selectedId
      || operationInFlightRef.current
      || pendingClientTranscriptRef.current
      || audioBlob
      || processingAudio
      || recoveringArtifact
      || isRecording
      || isStopping
      || startingRecording
    ) return;
    if (!confirmDiscardCurrentReview()) return;
    setStartingRecording(true);
    setError('');
    setRetentionNotice('');
    attemptedAudioBlobRef.current = null;

    try {
      const started = await startRecording();
      if (started) clearClinicalResult();
    } catch (e) {
      sessionIdRef.current = null;
      resetRecording();
      setError(e instanceof Error ? e.message : 'Не удалось начать серверное распознавание');
    } finally {
      setStartingRecording(false);
    }
  }, [
    audioBlob,
    clearClinicalResult,
    confirmDiscardCurrentReview,
    isRecording,
    isStopping,
    processingAudio,
    recoveringArtifact,
    resetRecording,
    selectedId,
    startRecording,
    startingRecording,
  ]);

  const stopListening = useCallback(() => {
    if (recognitionMode === 'gigaam') {
      stopRecording();
      return;
    }
    stopBrowserListening();
  }, [recognitionMode, stopBrowserListening, stopRecording]);

  const startListening = useCallback(() => {
    if (recognitionMode === 'gigaam') {
      void startGigaamListening();
      return;
    }
    startBrowserListening();
  }, [recognitionMode, startBrowserListening, startGigaamListening]);

  useEffect(() => {
    if (!audioBlob || !selectedId || isRecording || isStopping || processingAudio) return;
    if (attemptedAudioBlobRef.current === audioBlob) return;
    if (operationInFlightRef.current) return;
    attemptedAudioBlobRef.current = audioBlob;
    operationInFlightRef.current = true;
    setProcessingAudio(true);

    void (async () => {
      try {
        if (audioBlob.size > 32 * 1024 * 1024) {
          throw new Error(
            'Запись превышает безопасный лимит 32 МиБ. Скачайте исходное аудио и начните более короткую запись.',
          );
        }
        let sessionId = sessionIdRef.current;
        if (sessionId) {
          // A previous finish may have committed the immutable artifact while
          // its HTTP response was lost. Recover it before re-uploading audio.
          try {
            const persisted = await apiClient.getRadiologyArtifact(sessionId);
            if (mountedRef.current) {
              applyArtifact(persisted.artifact);
              sessionIdRef.current = null;
              attemptedAudioBlobRef.current = null;
              setRetainAudioConsent(false);
              resetRecording();
            }
            return;
          } catch {
            // The active session may not be finished yet.
          }
          try {
            const finished = await apiClient.finishRadiologySession(sessionId);
            if (mountedRef.current) {
              applyArtifact(finished.artifact);
              sessionIdRef.current = null;
              attemptedAudioBlobRef.current = null;
              setRetainAudioConsent(false);
              resetRecording();
            }
            return;
          } catch {
            // No accepted chunk yet, or its ASR failed: retry the same bytes.
          }
        }
        if (!sessionId) {
          const session = await apiClient.startRadiologySession(
            selectedId,
            'gigaam',
            retainAudioConsent,
          );
          sessionId = session.sessionId;
          sessionIdRef.current = sessionId;
          writeStoredRadiologyPointer(PENDING_RADIOLOGY_SESSION_KEY, {
            sessionId,
            templateId: selectedId,
          });
          if (retainAudioConsent && !session.retainAudio && mountedRef.current) {
            setRetentionNotice(
              'Серверная политика хранения аудио выключена: запись будет обработана, но не сохранена для датасета.',
            );
          }
        }
        const audioBase64 = await blobToBase64(audioBlob);
        await apiClient.sendRadiologyChunk(
          sessionId,
          audioBase64,
          0,
          audioBlob.type || 'audio/webm',
        );
        const result = await apiClient.finishRadiologySession(sessionId);
        if (mountedRef.current) {
          applyArtifact(result.artifact);
          sessionIdRef.current = null;
          attemptedAudioBlobRef.current = null;
          setRetainAudioConsent(false);
          resetRecording();
        }
      } catch (e) {
        if (mountedRef.current) {
          setError(e instanceof Error ? e.message : 'Ошибка обработки серверной записи');
        }
      } finally {
        operationInFlightRef.current = false;
        if (mountedRef.current) setProcessingAudio(false);
      }
    })();
  }, [
    applyArtifact,
    audioBlob,
    audioRetryToken,
    isRecording,
    isStopping,
    processingAudio,
    resetRecording,
    retainAudioConsent,
    selectedId,
  ]);

  const retryAudioProcessing = useCallback(() => {
    attemptedAudioBlobRef.current = null;
    setError('');
    setAudioRetryToken((value) => value + 1);
  }, []);

  const retryClientTranscript = useCallback(() => {
    const pending = pendingClientTranscriptRef.current;
    if (!pending) return;
    setError('');
    void finishClientTranscriptArtifact(pending.transcript, pending.source);
  }, [finishClientTranscriptArtifact]);

  const discardClientTranscript = useCallback(() => {
    const pending = pendingClientTranscriptRef.current;
    if (!pending) return;
    if (pending.source === 'manual') setInput(pending.transcript);
    if (pending.source === 'browser') setBrowserPhiConsent(false);
    pendingClientTranscriptRef.current = null;
    clearStoredRadiologyPointer(PENDING_RADIOLOGY_SESSION_KEY);
    setError('');
  }, []);

  const downloadPendingAudio = useCallback(() => {
    if (!audioBlob) return;
    const url = URL.createObjectURL(audioBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `radiology-dictation-${new Date().toISOString().replace(/[:.]/gu, '-')}.${audioFileExtension(audioBlob.type)}`;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  }, [audioBlob]);

  const discardPendingAudio = useCallback(() => {
    if (!audioBlob) return;
    sessionIdRef.current = null;
    attemptedAudioBlobRef.current = null;
    clearStoredRadiologyPointer(PENDING_RADIOLOGY_SESSION_KEY);
    setError('');
    setRetentionNotice('');
    setRetainAudioConsent(false);
    resetRecording();
  }, [audioBlob, resetRecording]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      browserGenerationRef.current += 1;
      const recognition = recogRef.current;
      recogRef.current = null;
      if (recognition) {
        recognition.onstart = null;
        recognition.onresult = null;
        recognition.onend = null;
        recognition.onerror = null;
        try {
          recognition.stop();
        } catch {
          // Already inactive.
        }
      }
      if (copyResetTimerRef.current !== null) {
        window.clearTimeout(copyResetTimerRef.current);
        copyResetTimerRef.current = null;
      }
      sessionIdRef.current = null;
      pendingClientTranscriptRef.current = null;
    };
  }, []);

  useEffect(() => {
    const hasUnpersistedWork = Boolean(
      input.trim()
      || (artifact && !approvedReady)
      || listening
      || startingRecording
      || isStopping
      || processingAudio
      || hasPendingServerAudio
      || hasPendingClientTranscript,
    );
    if (!hasUnpersistedWork) return;
    const warnBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', warnBeforeUnload);
    return () => window.removeEventListener('beforeunload', warnBeforeUnload);
  }, [
    approvedReady,
    artifact,
    hasPendingClientTranscript,
    hasPendingServerAudio,
    input,
    isStopping,
    listening,
    processingAudio,
    startingRecording,
  ]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return templates;
    return templates.filter((t) => (t.name + ' ' + t.title).toLowerCase().includes(q));
  }, [query, templates]);

  const handleCopy = useCallback(async () => {
    if (artifact && (!approvedReady || finalReportEditing)) return;
    const text = artifact
      ? approvedReport?.finalReport ?? ''
      : report?.text ?? templatePreview?.text ?? '';
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      setReviewError('');
      setCopied(true);
      if (copyResetTimerRef.current !== null) {
        window.clearTimeout(copyResetTimerRef.current);
      }
      copyResetTimerRef.current = window.setTimeout(() => {
        copyResetTimerRef.current = null;
        setCopied(false);
      }, 1500);
    } catch {
      setReviewError('Не удалось скопировать протокол. Разрешите доступ к буферу обмена или выделите текст вручную.');
    }
  }, [approvedReady, approvedReport, artifact, finalReportEditing, report, templatePreview]);

  const handleDownloadReport = useCallback(() => {
    if (
      !artifact
      || !selected
      || !approvedReady
      || !approvedReport
      || finalReportEditing
      || !approvedReport.finalReport.trim()
    ) return;
    const approvedDate = new Date(approvedReport.approvedAt);
    const date = localDateStamp(
      Number.isNaN(approvedDate.getTime()) ? new Date() : approvedDate,
    );
    downloadUtf8Text(
      approvedReport.finalReport,
      `${safeProtocolFilename(selected.name)}-${date}-final.txt`,
    );
  }, [approvedReady, approvedReport, artifact, finalReportEditing, selected]);

  const composeReviewText = useCallback((acceptedIds: Set<string>): string => {
    if (reviewDraft) {
      return composeAcceptedReviewDraftText(
        reviewDraft,
        acceptedIds,
        effectiveRouting?.atoms,
      );
    }
    return effectiveReport?.fullText ?? '';
  }, [effectiveReport, effectiveRouting, reviewDraft]);

  const resetManualFinalReport = useCallback(() => {
    setFinalReportText(composeReviewText(acceptedTemplateSegmentIds));
    setFinalReportManuallyEdited(false);
    setFinalReportEditing(false);
    setReviewError('');
    setCopied(false);
    setFeedbackState('idle');
  }, [acceptedTemplateSegmentIds, composeReviewText]);

  const beginFinalReportEdit = useCallback(() => {
    if (reviewMutationBusy || feedbackState === 'saved_unverified') return;
    finalReportEditBaselineRef.current = finalReportText;
    finalReportEditFeedbackBaselineRef.current = feedbackState === 'saved' ? 'saved' : 'idle';
    finalReportEditManualBaselineRef.current = finalReportManuallyEdited;
    setReviewError('');
    setCopied(false);
    setFeedbackState('idle');
    setFinalReportEditing(true);
  }, [feedbackState, finalReportManuallyEdited, finalReportText, reviewMutationBusy]);

  const cancelFinalReportEdit = useCallback(() => {
    setFinalReportText(finalReportEditBaselineRef.current);
    setFinalReportManuallyEdited(finalReportEditManualBaselineRef.current);
    setFeedbackState(finalReportEditFeedbackBaselineRef.current);
    setCopied(false);
    setReviewError('');
    setFinalReportEditing(false);
  }, []);

  const finishFinalReportEdit = useCallback(() => {
    if (finalReportText === finalReportEditBaselineRef.current) {
      setFinalReportManuallyEdited(finalReportEditManualBaselineRef.current);
      setFeedbackState(finalReportEditFeedbackBaselineRef.current);
    }
    setFinalReportEditing(false);
  }, [finalReportText]);

  const recomposeReviewDraft = useCallback(async () => {
    if (
      !artifact
      || !selected
      || reviewSpanCorrections.length === 0
      || reviewMutationBusy
      || feedbackLocked
    ) {
      return;
    }
    if (
      finalReportManuallyEdited
      && !window.confirm(
        'Пересборка заменит ручные изменения финального протокола. Продолжить?',
      )
    ) {
      return;
    }
    const requestGeneration = ++recomposeGenerationRef.current;
    const requestedVerbatimTranscript = verbatimTranscript;
    const previouslyAcceptedIds = new Set(acceptedTemplateSegmentIds);
    setRecomposingReview(true);
    setReviewError('');
    try {
      const { revision } = await apiClient.recomposeRadiologyReview(
        artifact.sessionId,
        {
          verbatimTranscript: requestedVerbatimTranscript,
          spanCorrections: reviewSpanCorrections,
        },
      );
      if (
        !mountedRef.current
        || requestGeneration !== recomposeGenerationRef.current
      ) {
        return;
      }
      if (
        revision.sessionId !== artifact.sessionId
        || revision.templateId !== artifact.templateId
        || !/^[a-f0-9]{64}$/u.test(revision.sourceArtifactSha256)
        || revision.verbatimTranscript.text !== requestedVerbatimTranscript
      ) {
        throw new Error('Сервер вернул пересборку для другого artifact');
      }
      const nextDraft = revision.report?.reviewDraft;
      const accepted = reviewDraft
        ? new Set(
            nextDraft?.segments
              .filter((segment) => (
                segment.confirmationRequired
                && segment.defaultKind !== 'placeholder'
                && previouslyAcceptedIds.has(segment.id)
              ))
              .map((segment) => segment.id) ?? [],
          )
        : initiallyAcceptedTemplateSegments(nextDraft);
      setReviewRevision(revision);
      setAcceptedTemplateSegmentIds(accepted);
      setReviewedResidualAtomIds(new Set());
      setFinalReportText(
        nextDraft
          ? composeAcceptedReviewDraftText(
              nextDraft,
              accepted,
              revision.routing.atoms,
            )
          : revision.report?.fullText ?? '',
      );
      setFinalReportManuallyEdited(false);
      setFinalReportEditing(false);
      setCompositionReviewOpen(Boolean(
        nextDraft
        && (
          nextDraft.status === 'failed'
          || nextDraft.residualAtomIds.length > 0
          || nextDraft.fieldAssignments.some((assignment) => assignment.status !== 'applied')
        )
      ));
      setSourceReviewOpen(
        revision.normalization.issues.some((issue) => issue.severity === 'critical'),
      );
      setCopied(false);
      setReport(revision.report ? toDisplayReport(revision.report) : null);
      feedbackSubmissionRef.current = null;
      setFeedbackState('idle');
    } catch (recomposeError) {
      setReviewError(
        recomposeError instanceof Error
          ? recomposeError.message
          : 'Не удалось пересобрать черновик после исправления расшифровки',
      );
    } finally {
      if (mountedRef.current && requestGeneration === recomposeGenerationRef.current) {
        setRecomposingReview(false);
      }
    }
  }, [
    acceptedTemplateSegmentIds,
    artifact,
    feedbackLocked,
    finalReportManuallyEdited,
    reviewMutationBusy,
    reviewDraft,
    reviewSpanCorrections,
    selected,
    verbatimTranscript,
  ]);

  const submitFeedback = useCallback(async () => {
    if (
      !artifact
      || !selected
      || feedbackSavingRef.current
      || feedbackLocked
      || finalReportEditing
      || hardApprovalBlockReason !== null
      || reviewRequiresRecompose
    ) return;
    const spanCorrections = reviewSpanCorrections;
    const currentReviewDraft = effectiveReport?.reviewDraft;
    const acceptedTemplateSegments = currentReviewDraft
      ? [...acceptedTemplateSegmentIds].sort()
      : undefined;
    const reviewedResidualAtoms = currentReviewDraft
      ? [...reviewedResidualAtomIds].sort()
      : undefined;
    const normalizationResolutions = artifact.normalization.issues.flatMap((issue) => {
      if (!issue.source) return [];
      const correction = spanCorrections.find(
        (candidate) =>
          candidate.start <= issue.source!.start
          && candidate.end >= issue.source!.end,
      );
      if (!correction || !correction.correctedText.trim()) return [];
      const replacementText = correction.correctedText.trim();
      const explicitRange = /(?:\bот\b[\s\S]*\bдо\b|\d\s*[-–—]\s*\d)/iu.test(replacementText);
      const numericValues = replacementText.match(/[+-]?\d+(?:[.,]\d+)?/gu) ?? [];
      return [{
        issueId: issue.id,
        replacementText,
        resolution: explicitRange
          ? 'confirmed_range' as const
          : numericValues.length === 1
            ? 'confirmed_single' as const
            : 'confirmed_verbatim' as const,
      }];
    });
    const approved = true;
    const payloadSignature = JSON.stringify({
      verbatimTranscript,
      finalReport: finalReportText,
      spanCorrections,
      normalizationResolutions,
      baseDraftSha256: currentReviewDraft?.sha256,
      acceptedTemplateSegmentIds: acceptedTemplateSegments,
      reviewedResidualAtomIds: reviewedResidualAtoms,
      approved,
    });
    let submission = feedbackSubmissionRef.current;
    if (!submission || submission.payloadSignature !== payloadSignature) {
      submission = {
        payloadSignature,
        idempotencyKey: createIdempotencyKey(),
      };
      feedbackSubmissionRef.current = submission;
    }
    feedbackSavingRef.current = true;
    setFeedbackState('saving');
    setReviewError('');
    try {
      const savedFeedback = await apiClient.submitRadiologyFeedback(artifact.sessionId, {
        idempotencyKey: submission.idempotencyKey,
        verbatimTranscript,
        finalReport: finalReportText,
        spanCorrections,
        normalizationResolutions,
        baseDraftSha256: currentReviewDraft?.sha256,
        acceptedTemplateSegmentIds: acceptedTemplateSegments,
        reviewedResidualAtomIds: reviewedResidualAtoms,
        approved,
        author: doctor.name,
      });
      feedbackSubmissionRef.current = null;
      setApprovedReport(null);
      try {
        const response = await apiClient.getRadiologyApprovedReport(artifact.sessionId);
        await validateApprovedReport(response.approvedReport, artifact);
        if (
          response.approvedReport.feedbackId !== savedFeedback.feedbackId
          || response.approvedReport.revision !== savedFeedback.revision
          || response.approvedReport.finalReport !== finalReportText
        ) {
          throw new Error('Сервер вернул не ту подтверждённую ревизию протокола');
        }
        applyApprovedReport(response.approvedReport, artifact);
      } catch (approvedReportError) {
        setFinalReportEditing(false);
        setFeedbackState('saved_unverified');
        setReviewError(
          `Подтверждение сохранено, но повторная загрузка финальной версии не удалась: ${
            approvedReportError instanceof Error
              ? approvedReportError.message
              : 'неизвестная ошибка'
          }`,
        );
      }
    } catch (e) {
      setFeedbackState('idle');
      setReviewError(e instanceof Error ? e.message : 'Не удалось сохранить подтверждение');
    } finally {
      feedbackSavingRef.current = false;
    }
  }, [
    applyApprovedReport,
    artifact,
    acceptedTemplateSegmentIds,
    doctor.name,
    effectiveReport,
    feedbackLocked,
    finalReportEditing,
    finalReportText,
    hardApprovalBlockReason,
    reviewSpanCorrections,
    reviewRequiresRecompose,
    reviewedResidualAtomIds,
    selected,
    verbatimTranscript,
  ]);

  const refreshApprovedReport = useCallback(async () => {
    if (!artifact || feedbackSavingRef.current) return;
    feedbackSavingRef.current = true;
    setFeedbackState('saving');
    setReviewError('');
    try {
      const response = await apiClient.getRadiologyApprovedReport(artifact.sessionId);
      await validateApprovedReport(response.approvedReport, artifact);
      applyApprovedReport(response.approvedReport, artifact);
    } catch (refreshError) {
      setFeedbackState('saved_unverified');
      setReviewError(
        refreshError instanceof Error
          ? refreshError.message
          : 'Не удалось загрузить подтверждённую версию протокола',
      );
    } finally {
      feedbackSavingRef.current = false;
    }
  }, [applyApprovedReport, artifact]);

  const resetToSelection = useCallback(() => {
    if (!confirmDiscardCurrentReview()) return;
    stopListening();
    resetRecording();
    setSelectedId(null);
    setCommands([]);
    setReport(null);
    setApplied([]);
    setInput('');
    setArtifact(null);
    setApprovedReport(null);
    setReviewRevision(null);
    setRecomposingReview(false);
    setAcceptedTemplateSegmentIds(new Set());
    setReviewedResidualAtomIds(new Set());
    setTemplatePreview(null);
    setTemplatePreviewLoading(false);
    setVerbatimTranscript('');
    setFinalReportText('');
    setFinalReportManuallyEdited(false);
    setFinalReportEditing(false);
    setCompositionReviewOpen(false);
    setSourceReviewOpen(false);
    setReviewError('');
    setFeedbackState('idle');
    setRetainAudioConsent(false);
    setRetentionNotice('');
    setBrowserPhiConsent(false);
    browserGenerationRef.current += 1;
    feedbackSavingRef.current = false;
    feedbackSubmissionRef.current = null;
    pendingClientTranscriptRef.current = null;
    sessionIdRef.current = null;
    attemptedAudioBlobRef.current = null;
    clearStoredRadiologyPointer(PENDING_RADIOLOGY_SESSION_KEY);
  }, [confirmDiscardCurrentReview, resetRecording, stopListening]);

  // ─── Верхняя панель ────────────────────────────────────────────────────────
  const TopBar = (
    <div className="flex items-center justify-between px-6 py-3 border-b border-slate-200 bg-white">
      <div className="flex items-center gap-2 text-medical-800 font-semibold">
        <Stethoscope size={18} /> Лучевая диагностика
      </div>
      <div className="flex flex-wrap items-center justify-end gap-3 text-sm text-text-muted">
        <span className="hidden sm:inline">{doctor.name}</span>
        {onOpenTherapy && <button disabled={workflowBusy} onClick={() => { if (confirmDiscardCurrentReview()) onOpenTherapy(); }} className="hover:text-medical-700 disabled:text-slate-300">Терапия</button>}
        {onOpenAdmin && <button disabled={workflowBusy} onClick={() => { if (confirmDiscardCurrentReview()) onOpenAdmin(); }} className="flex items-center gap-1 hover:text-medical-700 disabled:text-slate-300"><Shield size={16} /> Админка</button>}
        {onOpenSettings && <button disabled={workflowBusy} onClick={() => { if (confirmDiscardCurrentReview()) onOpenSettings(); }} className="flex items-center gap-1 hover:text-medical-700 disabled:text-slate-300"><Settings size={16} /> Настройки</button>}
        {onLogout && <button disabled={workflowBusy} onClick={() => { if (confirmDiscardCurrentReview()) onLogout(); }} className="flex items-center gap-1 hover:text-red-600 disabled:text-slate-300"><LogOut size={16} /> Выйти</button>}
      </div>
    </div>
  );

  // ─── Экран выбора шаблона ────────────────────────────────────────────────────
  if (!selected) {
    return (
      <div className="min-h-screen bg-medical-50">
        {TopBar}
        <div className="max-w-3xl mx-auto px-6 py-10">
          <h1 className="text-2xl font-bold text-medical-900 mb-1">Выберите шаблон исследования</h1>
          <p className="text-text-muted mb-6">После выбора можно начать голосовую диктовку по шаблону.</p>

          <div className="relative mb-6">
            <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Поиск шаблона…"
              className="w-full box-border pl-10 pr-4 py-3 rounded-xl border border-slate-200 bg-white focus:outline-none focus:ring-2 focus:ring-medical-400"
            />
          </div>

          {loading && <p className="text-text-muted">Загрузка шаблонов…</p>}
          {recoveringArtifact && (
            <p className="text-text-muted mb-4">Восстанавливаем последний протокол…</p>
          )}
          {error && <p className="text-red-600 mb-4">{error}</p>}

          <div className="grid gap-3 sm:grid-cols-2">
            {filtered.map((t) => (
              <button
                key={t.id}
                disabled={recoveringArtifact}
                onClick={() => { setSelectedId(t.id); setError(''); }}
                className="text-left p-4 rounded-xl border border-slate-200 bg-white hover:border-medical-400 hover:shadow-sm transition disabled:opacity-60 disabled:hover:border-slate-200 disabled:hover:shadow-none"
              >
                <div className="text-xs font-semibold text-medical-600 mb-1">{t.modality}</div>
                <div className="font-semibold text-medical-900">{t.name}</div>
                <div className="text-sm text-text-muted mt-1 line-clamp-2">{t.title}</div>
              </button>
            ))}
          </div>
          {!loading && filtered.length === 0 && <p className="text-text-muted">Шаблоны не найдены.</p>}
        </div>
      </div>
    );
  }

  // ─── Рабочий экран: запись + живой документ ─────────────────────────────────
  return (
    <div className="min-h-screen bg-medical-50">
      {TopBar}
      <div className="max-w-5xl mx-auto px-6 py-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        {/* Документ */}
        <div className={artifact ? 'order-1 lg:order-1' : 'order-2 lg:order-1'}>
          <div className="flex items-center justify-between mb-3">
            <button
              onClick={resetToSelection}
              disabled={workflowBusy}
              className="text-sm text-medical-700 hover:underline disabled:text-slate-300 disabled:no-underline"
            >
              ← Сменить шаблон
            </button>
            {!artifact && (
              <button
                onClick={() => void handleCopy()}
                disabled={!reviewDraft && !displayedDocument}
                className="flex items-center gap-1 text-sm text-medical-700 hover:underline disabled:text-slate-300 disabled:no-underline"
              >
                {copied
                  ? <><Check size={15} /> Скопировано</>
                  : <><Copy size={15} /> {showingTemplateDefaults ? 'Копировать шаблон' : 'Копировать'}</>}
              </button>
            )}
          </div>

          {artifact && (
            <section
              aria-labelledby="radiology-final-report-heading"
              className="mb-4 rounded-xl border-2 border-medical-200 bg-white p-5 shadow-sm"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <FileText size={20} className="text-medical-700" />
                    <h2 id="radiology-final-report-heading" className="text-lg font-bold text-medical-950">
                      Готовый протокол
                    </h2>
                  </div>
                  <p id="radiology-final-report-help" className="mt-1 text-sm text-text-muted">
                    Это итоговый текст. Именно он будет подтверждён, скопирован и скачан.
                    Для изменения нажмите «Редактировать».
                  </p>
                </div>
                <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                  finalReportEditing
                    ? 'bg-amber-100 text-amber-900'
                    : approvedReady
                    ? 'bg-emerald-100 text-emerald-800'
                    : feedbackState === 'saved_unverified'
                      ? 'bg-amber-100 text-amber-900'
                    : reviewRequiresRecompose || finalReportManuallyEdited
                      ? 'bg-amber-100 text-amber-900'
                      : hardApprovalBlockReason
                        ? 'bg-red-100 text-red-800'
                        : 'bg-sky-100 text-sky-800'
                }`}>
                  {finalReportEditing
                    ? 'Режим редактирования'
                    : approvedReady
                    ? 'Подтверждён врачом'
                    : feedbackState === 'saved_unverified'
                      ? 'Сохранён — нужна сверка'
                    : feedbackState === 'saving'
                      ? 'Проверяем на сервере…'
                      : reviewRequiresRecompose
                      ? 'Нужна пересборка'
                      : finalReportManuallyEdited
                        ? 'Изменён — не проверен'
                        : hardApprovalBlockReason
                          ? 'Подтверждение заблокировано'
                          : 'Черновик'}
                </span>
              </div>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
                <label htmlFor="radiology-final-report" className="block text-sm font-semibold text-medical-900">
                  Текст протокола
                </label>
                {finalReportEditing ? (
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={cancelFinalReportEdit}
                      disabled={reviewMutationBusy}
                      className="rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                    >
                      Отменить правки
                    </button>
                    <button
                      type="button"
                      onClick={finishFinalReportEdit}
                      disabled={reviewMutationBusy}
                      className="rounded-md bg-medical-700 px-2.5 py-1.5 text-xs font-semibold text-white hover:bg-medical-800 disabled:opacity-50"
                    >
                      Готово
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={beginFinalReportEdit}
                    disabled={reviewMutationBusy || feedbackState === 'saved_unverified'}
                    className="rounded-md border border-medical-300 bg-medical-50 px-2.5 py-1.5 text-xs font-semibold text-medical-800 hover:bg-medical-100 disabled:opacity-50"
                  >
                    {approvedReady ? 'Создать новую версию' : 'Редактировать'}
                  </button>
                )}
              </div>
              <textarea
                id="radiology-final-report"
                aria-describedby="radiology-final-report-help radiology-final-report-status"
                value={finalReportText}
                readOnly={!finalReportEditing}
                disabled={reviewMutationBusy}
                onChange={(event) => {
                  const nextText = event.target.value;
                  const changedFromBaseline = nextText !== finalReportEditBaselineRef.current;
                  setFinalReportText(nextText);
                  setFinalReportManuallyEdited(
                    changedFromBaseline
                      ? true
                      : finalReportEditManualBaselineRef.current,
                  );
                  setCopied(false);
                  setReviewError('');
                  setFeedbackState(
                    changedFromBaseline
                      ? 'idle'
                      : finalReportEditFeedbackBaselineRef.current,
                  );
                }}
                rows={16}
                className={`mt-1 w-full box-border rounded-lg border px-3 py-3 text-[15px] leading-6 text-medical-950 focus:border-medical-500 focus:outline-none focus:ring-2 focus:ring-medical-300 disabled:bg-slate-50 ${
                  finalReportEditing
                    ? 'border-medical-400 bg-white'
                    : 'cursor-default border-slate-200 bg-slate-50'
                }`}
              />

              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => void handleCopy()}
                  disabled={
                    !approvedReady
                    || reviewMutationBusy
                    || finalReportEditing
                    || !finalReportText.trim()
                  }
                  className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-medical-800 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {copied ? <Check size={16} /> : <Copy size={16} />}
                  {copied ? 'Скопировано' : 'Копировать'}
                </button>
                <button
                  type="button"
                  onClick={handleDownloadReport}
                  disabled={
                    !approvedReady
                    || reviewMutationBusy
                    || finalReportEditing
                    || !finalReportText.trim()
                  }
                  className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-medical-800 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Download size={16} />
                  Скачать TXT
                </button>
              </div>

              {!approvedReady && feedbackState !== 'saved_unverified' && (
                <p className="mt-2 text-xs text-amber-800">
                  Копирование и скачивание откроются после серверной проверки и подтверждения врачом.
                </p>
              )}
              {feedbackState === 'saved_unverified' && (
                <p className="mt-2 text-xs text-amber-800">
                  Подтверждение уже сохранено на сервере. Скачивание откроется после контрольной сверки сохранённой версии.
                </p>
              )}
              {approvedReady && approvedReport && (
                <p className="mt-2 text-xs text-emerald-800">
                  Подтверждено {formatLocalDateTime(approvedReport.approvedAt)}
                  {' · '}врач: {approvedReport.author}
                  {' · '}ревизия {approvedReport.revision}
                </p>
              )}

              {reviewError && (
                <div
                  role="alert"
                  aria-live="assertive"
                  className="mt-4 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-900"
                >
                  {reviewError}
                </div>
              )}

              {feedbackState === 'saved_unverified' ? (
                <div id="radiology-final-report-status" className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
                  <div className="font-semibold">Сервер сохранил подтверждение, но браузер ещё не сверил контрольную копию.</div>
                  <p className="mt-1">
                    Правки временно заблокированы, чтобы не создать дублирующую ревизию. Повторите безопасную загрузку подтверждённого текста.
                  </p>
                  <button
                    type="button"
                    onClick={() => void refreshApprovedReport()}
                    disabled={reviewMutationBusy}
                    className="mt-3 rounded-md border border-amber-400 bg-white px-3 py-2 text-xs font-semibold hover:bg-amber-100 disabled:opacity-50"
                  >
                    Повторить сверку с сервером
                  </button>
                </div>
              ) : finalReportEditing ? (
                <p id="radiology-final-report-status" className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                  Завершите редактирование кнопкой «Готово». После этого протокол можно проверить и подтвердить.
                </p>
              ) : hardApprovalBlockReason ? (
                <div id="radiology-final-report-status" className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-900">
                  <div className="flex items-start gap-2">
                    <AlertTriangle size={18} className="mt-0.5 shrink-0" />
                    <div>
                      <div className="font-semibold">Что нужно исправить перед подтверждением</div>
                      <ul className="mt-1 space-y-1">
                        {approvalBlockReasons.map((reason) => (
                          <li key={reason}>• {reason}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {reviewDraft && (
                      <button
                        type="button"
                        onClick={() => setCompositionReviewOpen(true)}
                        className="rounded-md border border-red-300 bg-white px-2.5 py-1.5 text-xs font-semibold hover:bg-red-100"
                      >
                        Проверить подстановки
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => setSourceReviewOpen(true)}
                      className="rounded-md border border-red-300 bg-white px-2.5 py-1.5 text-xs font-semibold hover:bg-red-100"
                    >
                      Открыть расшифровку и ошибки
                    </button>
                  </div>
                </div>
              ) : (
                <p id="radiology-final-report-status" className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
                  Текст готов к серверной проверке. Нажмите кнопку ниже, чтобы сохранить точную подтверждённую версию.
                </p>
              )}

              <button
                type="button"
                onClick={() => void submitFeedback()}
                disabled={
                  feedbackLocked
                  || feedbackState === 'saving'
                  || finalReportEditing
                  || hardApprovalBlockReason !== null
                }
                className="mt-4 w-full rounded-lg bg-emerald-600 py-3 font-semibold text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                {approvedReady
                  ? 'Протокол подтверждён врачом'
                  : feedbackState === 'saved_unverified'
                    ? 'Подтверждение сохранено — требуется сверка'
                  : feedbackState === 'saving'
                    ? 'Проверяем и сохраняем…'
                    : 'Проверить и подтвердить протокол'}
              </button>
            </section>
          )}

          <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
            {artifact && (
              <button
                type="button"
                aria-expanded={compositionReviewOpen}
                aria-controls="radiology-composition-review"
                onClick={() => setCompositionReviewOpen((current) => !current)}
                className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-medical-300"
              >
                <span>
                  <span className="block font-semibold text-medical-900">Как шаблон собрал протокол</span>
                  <span className="mt-0.5 block text-xs text-text-muted">
                    Подстановки из речи, нормы шаблона и вычисленные значения
                  </span>
                </span>
                <span className="flex shrink-0 items-center gap-2 text-xs text-text-muted">
                  {reviewDraft?.fieldAssignments.filter((assignment) => assignment.status === 'applied').length ?? 0} полей
                  <ChevronDown
                    size={18}
                    className={`transition-transform ${compositionReviewOpen ? 'rotate-180' : ''}`}
                  />
                </span>
              </button>
            )}
            <div id="radiology-composition-review" className={artifact
              ? compositionReviewOpen ? 'border-t border-slate-200 p-5' : 'hidden'
              : 'p-5'}>
            {showingTemplateDefaults && (
              <div className="mb-4 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-sm text-sky-900">
                <span className="font-semibold">Предпросмотр шаблона.</span>{' '}
                Это нормы по умолчанию, а не результат распознавания и не подтверждённые данные исследования.
              </div>
            )}
            {reviewDraft && (
              <div className="mb-4 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-950">
                <div className="font-semibold">Черновик собран по выбранному шаблону.</div>
                <p className="mt-1 text-xs">
                  Зелёным выделены значения из диктовки, серым — нормы шаблона,
                  <span className="mx-1 rounded-sm bg-indigo-50 px-0.5 text-indigo-900">fx</span>
                  — детерминированно вычисленные значения.
                </p>
                {finalReportManuallyEdited && (
                  <div className="mt-3 rounded-md border border-amber-300 bg-amber-50 p-2 text-xs text-amber-950">
                    <div className="font-semibold">Финальный текст изменён вручную.</div>
                    <p className="mt-1">
                      {feedbackLocked
                        ? 'Подтверждённая версия защищена. Для изменения сначала нажмите «Создать новую версию» в готовом протоколе.'
                        : 'Переключатели норм временно отключены, чтобы не потерять ваши правки.'}
                    </p>
                    {!feedbackLocked && (
                      <button
                        type="button"
                        onClick={resetManualFinalReport}
                        disabled={reviewMutationBusy || finalReportEditing}
                        className="mt-2 inline-flex items-center gap-1 font-semibold text-amber-900 hover:underline disabled:opacity-50"
                      >
                        <RotateCcw size={13} /> Вернуть собранный вариант
                      </button>
                    )}
                  </div>
                )}
                {confirmationSegments.length > 0 && (
                  <label className="mt-2 flex items-start gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={unacceptedTemplateSegmentIds.length === 0}
                      disabled={
                        reviewMutationBusy
                        || feedbackLocked
                        || finalReportEditing
                        || finalReportManuallyEdited
                      }
                      onChange={(event) => {
                        const checked = event.target.checked;
                        const next = new Set(acceptedTemplateSegmentIds);
                        for (const segment of confirmationSegments) {
                          if (checked) next.add(segment.id);
                          else next.delete(segment.id);
                        }
                        setAcceptedTemplateSegmentIds(next);
                        setFinalReportText(composeReviewText(next));
                        setFinalReportManuallyEdited(false);
                        setCopied(false);
                        setReviewError('');
                        setFeedbackState('idle');
                      }}
                      className="mt-0.5"
                    />
                    <span>
                      Использовать и подтвердить нормы всего выбранного шаблона
                      ({confirmationSegments.length})
                    </span>
                  </label>
                )}
              </div>
            )}
            <div className="text-center font-semibold text-medical-900 mb-4">
              {reviewDraft?.title ?? displayedDocument?.title ?? selected.title}
            </div>
            {templatePreviewLoading && !report && (
              <div className="py-8 text-center text-sm text-text-muted">
                Загружаем структуру шаблона…
              </div>
            )}
            <div className="space-y-2 text-[15px] leading-relaxed text-medical-900">
              {reviewDraft
                ? reviewDraft.sections.map((section) => {
                    const sectionSegments = segmentsForSection(reviewDraft, section.segmentIds);
                    const confirmationIds = sectionSegments
                      .filter(
                        (segment) => (
                          segment.confirmationRequired
                          && segment.defaultKind !== 'placeholder'
                        ),
                      )
                      .map((segment) => segment.id);
                    const sectionAccepted = confirmationIds.length > 0
                      && confirmationIds.every((segmentId) => (
                        acceptedTemplateSegmentIds.has(segmentId)
                      ));
                    const isConclusion = section.mode === 'conclusion';
                    return (
                      <div
                        key={section.id}
                        className={[
                          isConclusion ? 'pt-2 mt-2 border-t border-slate-200' : '',
                          section.mode === 'verbatim_fallback'
                            ? 'rounded-md border border-amber-200 bg-amber-50 p-2'
                            : '',
                        ].filter(Boolean).join(' ')}
                      >
                        <p className="whitespace-pre-wrap">
                          <span className="font-semibold">{section.label}:</span>{' '}
                          {sectionSegments.length > 0
                            ? sectionSegments.map((segment) => (
                                <ReviewDraftSegment
                                  key={segment.id}
                                  segment={segment}
                                  accepted={
                                    !segment.confirmationRequired
                                    || acceptedTemplateSegmentIds.has(segment.id)
                                  }
                                />
                              ))
                            : section.text}
                        </p>
                        {confirmationIds.length > 0 && (
                          <label className="mt-1 flex items-center gap-1.5 text-[11px] text-slate-500">
                            <input
                              type="checkbox"
                              checked={sectionAccepted}
                              disabled={
                                reviewMutationBusy
                                || feedbackLocked
                                || finalReportEditing
                                || finalReportManuallyEdited
                              }
                              onChange={(event) => {
                                const checked = event.target.checked;
                                const next = new Set(acceptedTemplateSegmentIds);
                                for (const segmentId of confirmationIds) {
                                  if (checked) next.add(segmentId);
                                  else next.delete(segmentId);
                                }
                                setAcceptedTemplateSegmentIds(next);
                                setFinalReportText(composeReviewText(next));
                                setFinalReportManuallyEdited(false);
                                setCopied(false);
                                setReviewError('');
                                setFeedbackState('idle');
                              }}
                            />
                            Подтвердить нормы раздела ({confirmationIds.length})
                          </label>
                        )}
                        {section.issues.length > 0 && (
                          <ul className="mt-1 space-y-0.5 text-xs text-amber-800">
                            {section.issues.map((issue, index) => (
                              <li key={`${issue.code}-${issue.fieldId ?? issue.atomId ?? index}`}>
                                • {issue.message}
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    );
                  })
                : displayedDocument?.blocks.map((b) => {
                    const sep = b.text.indexOf(': ');
                    const label = sep > 0 ? b.text.slice(0, sep) : b.label;
                    const body = sep > 0 ? b.text.slice(sep + 2) : b.text;
                    const isConclusion = b.id === 'conclusion';
                    return (
                      <p
                        key={b.id}
                        className={[
                          isConclusion ? 'pt-2 mt-2 border-t border-slate-200' : '',
                          showingTemplateDefaults ? 'text-slate-600' : '',
                        ].filter(Boolean).join(' ')}
                      >
                        <span className="font-semibold">{label}:</span> {body}
                      </p>
                    );
                  })}
            </div>
            {reviewDraft && reviewDraft.residualAtomIds.length > 0 && (
              <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-3">
                <div className="text-sm font-semibold text-amber-950">
                  Остаточные дословные фрагменты требуют проверки
                </div>
                <div className="mt-2 space-y-2">
                  {reviewDraft.residualAtomIds.map((atomId) => (
                    <label key={atomId} className="flex items-start gap-2 text-sm text-amber-950">
                      <input
                        type="checkbox"
                        checked={reviewedResidualAtomIds.has(atomId)}
                        disabled={
                          reviewMutationBusy
                          || feedbackLocked
                          || finalReportEditing
                        }
                        onChange={(event) => {
                          const checked = event.target.checked;
                          setReviewedResidualAtomIds((current) => {
                            const next = new Set(current);
                            if (checked) next.add(atomId);
                            else next.delete(atomId);
                            return next;
                          });
                          setCopied(false);
                          setReviewError('');
                          setFeedbackState('idle');
                        }}
                        className="mt-0.5"
                      />
                      <span>
                        {residualEvidenceByAtomId.get(atomId) ?? `Фрагмент ${atomId}`}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}
            {reviewDraft && reviewDraft.issues.some((issue) => !issue.sectionId) && (
              <ul className="mt-4 space-y-1 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-950">
                {reviewDraft.issues
                  .filter((issue) => !issue.sectionId)
                  .map((issue, index) => (
                    <li key={`${issue.code}-${issue.atomId ?? issue.fieldId ?? index}`}>
                      • {issue.message}
                    </li>
                  ))}
              </ul>
            )}
            {!reviewDraft && effectiveReport?.templateDefaults.length ? (
              <details className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                <summary className="cursor-pointer text-sm font-semibold text-slate-700">
                  Неподтверждённые нормы шаблона ({effectiveReport.templateDefaults.length})
                </summary>
                <p className="mt-2 text-xs text-slate-500">
                  Эти разделы не были произнесены и не входят в доказательный протокол.
                </p>
                <div className="mt-3 space-y-2 text-sm text-slate-600">
                  {effectiveReport.templateDefaults.map((item) => (
                    <p key={item.id}>
                      <span className="font-semibold">{item.label}:</span> {item.text}
                    </p>
                  ))}
                </div>
              </details>
            ) : null}
          </div>
          </div>

          {artifact && (
            <section className="mt-4 overflow-hidden rounded-xl border border-slate-200 bg-white">
              <button
                type="button"
                aria-expanded={sourceReviewOpen}
                aria-controls="radiology-source-review"
                onClick={() => setSourceReviewOpen((current) => !current)}
                className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-medical-300"
              >
                <span>
                  <span className="block font-semibold text-medical-900">
                    Исходная расшифровка и техническая проверка
                  </span>
                  <span className="mt-0.5 block text-xs text-text-muted">
                    Открывайте этот раздел, если ASR распознал фразу неверно или подтверждение заблокировано
                  </span>
                </span>
                <span className="flex shrink-0 items-center gap-2">
                  <span className={`rounded-full px-2 py-1 text-xs font-semibold ${
                    effectiveSafety?.status === 'passed'
                      ? 'bg-emerald-100 text-emerald-800'
                      : effectiveSafety?.status === 'failed'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-amber-100 text-amber-800'
                  }`}>
                    {effectiveSafety?.status === 'passed' ? 'Текст проверен' : 'Есть замечания'}
                  </span>
                  <ChevronDown
                    size={18}
                    className={`text-text-muted transition-transform ${sourceReviewOpen ? 'rotate-180' : ''}`}
                  />
                </span>
              </button>

              {sourceReviewOpen && (
                <div id="radiology-source-review" className="border-t border-slate-200 p-5">
                  <div className="mb-4 rounded-lg bg-slate-50 px-3 py-2 text-xs text-text-muted">
                    <div>
                      ASR: {artifact.model.asr.name} · decoder: {artifact.model.decoder.name} · SHA аудио: {
                        artifact.audio.sha256 ? `${artifact.audio.sha256.slice(0, 12)}…` : 'нет аудио'
                      }
                    </div>
                    <div className="mt-1">
                      Dataset: {artifact.training.eligible
                        ? 'может быть включён после врачебного ревью'
                        : `исключён (${artifact.training.exclusionReasons.join(', ') || 'нет основания'})`}
                    </div>
                  </div>

                  <label htmlFor="radiology-verbatim-transcript" className="block text-sm font-semibold text-medical-800 mb-1">
                    Дословная расшифровка — исправление ошибок ASR
                  </label>
                  <textarea
                    id="radiology-verbatim-transcript"
                    value={verbatimTranscript}
                    disabled={
                      reviewMutationBusy
                      || feedbackLocked
                      || finalReportEditing
                    }
                    onChange={(event) => {
                      recomposeGenerationRef.current += 1;
                      setVerbatimTranscript(event.target.value);
                      setReviewRevision(null);
                      setReviewedResidualAtomIds(new Set());
                      setCopied(false);
                      setReviewError('');
                      setFeedbackState('idle');
                    }}
                    rows={5}
                    className="w-full box-border rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-medical-400"
                  />
                  <p className="mb-4 mt-1 text-xs text-text-muted">
                    Исправляйте только то, что реально произнесено. Протокольные нормы сюда не добавляются.
                  </p>
                  {reviewSpanCorrections.length > 0 && (
                    <button
                      type="button"
                      onClick={() => void recomposeReviewDraft()}
                      disabled={
                        reviewMutationBusy
                        || feedbackLocked
                        || finalReportEditing
                      }
                      className="mb-4 w-full rounded-lg border border-medical-300 bg-medical-50 px-3 py-2 text-sm font-semibold text-medical-800 hover:bg-medical-100 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {recomposingReview
                        ? 'Пересобираем протокол…'
                        : reviewRevision?.verbatimTranscript.text === verbatimTranscript
                          ? 'Протокол пересобран по исправленному тексту'
                          : 'Применить исправления и пересобрать протокол'}
                    </button>
                  )}

                  <div className="mb-1 text-sm font-semibold text-medical-800">
                    Нормализованный текст — только для контроля
                  </div>
                  <div className="mb-4 whitespace-pre-wrap rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-medical-900">
                    {effectiveNormalization?.text || 'Нормализация не выполнена'}
                  </div>
                  {(effectiveNormalization?.issues.length ?? 0) > 0 && (
                    <ul className="mb-4 space-y-1 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
                      {effectiveNormalization?.issues.map((issue) => (
                        <li key={issue.id}>
                          • {issue.message}
                          {issue.source?.text ? ` Исходный фрагмент: «${issue.source.text}».` : ''}
                        </li>
                      ))}
                    </ul>
                  )}

                  {(effectiveReport?.unmatched || artifact.unmatchedText) && (
                    <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                      <span className="font-semibold">Не удалось разнести по секциям:</span>{' '}
                      {effectiveReport?.unmatched || artifact.unmatchedText}
                    </div>
                  )}
                  {(effectiveSafety?.issues.length ?? 0) > 0 && (
                    <ul className="mt-3 space-y-1 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-900">
                      {effectiveSafety?.issues.map((issue, index) => (
                        <li key={`${issue.code}-${index}`}>• {issue.message}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </section>
          )}
        </div>

        {/* Панель управления диктовкой */}
        <div className={artifact ? 'order-2 lg:order-2' : 'order-1 lg:order-2'}>
          <div className="rounded-xl border border-slate-200 bg-white p-5 lg:sticky lg:top-6">
            <div className="text-sm font-semibold text-medical-800 mb-1">{selected.name}</div>
            <p className="text-xs text-text-muted mb-3">Органы и параметры можно диктовать в естественном порядке — норма подставится только для неупомянутых секций.</p>

            <div className={`grid ${BROWSER_ASR_FALLBACK_ENABLED ? 'grid-cols-2' : 'grid-cols-1'} gap-1 rounded-lg bg-slate-100 p-1 mb-3`}>
              <button
                type="button"
                disabled={workflowBusy}
                onClick={() => { stopBrowserListening(); setRecognitionMode('gigaam'); setError(''); }}
                className={`rounded-md px-2 py-1.5 text-xs font-semibold ${
                  recognitionMode === 'gigaam' ? 'bg-white text-medical-800 shadow-sm' : 'text-text-muted'
                }`}
              >
                Серверный GigaAM
              </button>
              {BROWSER_ASR_FALLBACK_ENABLED && (
                <button
                  type="button"
                  disabled={workflowBusy}
                  onClick={() => {
                    resetRecording();
                    setBrowserPhiConsent(false);
                    setRecognitionMode('browser');
                    setError('');
                  }}
                  className={`rounded-md px-2 py-1.5 text-xs font-semibold ${
                    recognitionMode === 'browser' ? 'bg-white text-medical-800 shadow-sm' : 'text-text-muted'
                  }`}
                >
                  Browser fallback
                </button>
              )}
            </div>
            {recognitionMode === 'browser' && (
              <label className="mb-3 flex items-start gap-2 text-xs rounded-md bg-red-50 border border-red-200 p-2 text-red-800">
                <input
                  type="checkbox"
                  checked={browserPhiConsent}
                  disabled={listening || processingAudio}
                  onChange={(event) => setBrowserPhiConsent(event.target.checked)}
                  className="mt-0.5"
                />
                <span>
                  Браузер может передавать медицинскую речь внешнему сервису. Использовать только после утверждения политики и согласия; запись получит source=browser и не попадёт в обучение или benchmark.
                </span>
              </label>
            )}
            {recognitionMode === 'gigaam' && (
              <>
                <label className="mb-3 flex items-start gap-2 rounded-md bg-slate-50 border border-slate-200 p-2 text-xs text-medical-800">
                  <input
                    type="checkbox"
                    checked={retainAudioConsent}
                    disabled={workflowBusy}
                    onChange={(e) => setRetainAudioConsent(e.target.checked)}
                    className="mt-0.5"
                  />
                  <span>
                    Сохранить аудиозапись on-premise для проверенного датасета. Включать только при наличии согласия и политики хранения.
                  </span>
                </label>
                {retentionNotice && (
                  <p className="mb-3 rounded-md border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800">
                    {retentionNotice}
                  </p>
                )}
              </>
            )}

            <button
              onClick={listening ? stopListening : startListening}
              disabled={
                processingAudio
                || startingRecording
                || isStopping
                || recoveringArtifact
                || (recognitionMode === 'browser' && !browserPhiConsent)
                || (!listening && workflowBusy)
              }
              className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold transition ${
                processingAudio
                || startingRecording
                || isStopping
                || recoveringArtifact
                || (!listening && workflowBusy)
                  ? 'bg-slate-300 text-slate-600'
                  : listening
                    ? 'bg-red-500 text-white animate-pulse'
                    : 'bg-medical-600 text-white hover:bg-medical-700'
              }`}
            >
              <Mic size={18} /> {
                processingAudio
                  ? `${recognitionMode === 'gigaam' ? 'GigaAM' : 'Сервер'} обрабатывает запись…`
                  : startingRecording
                    ? 'Подготавливаем микрофон…'
                  : isStopping
                    ? 'Завершаем запись…'
                  : listening
                    ? `Остановить запись${recognitionMode === 'gigaam' ? ` · ${formattedDuration}` : ''}`
                    : 'Начать запись'
              }
            </button>
            {interim && <p className="mt-2 text-sm text-text-muted italic">«{interim}»</p>}
            {!processingAudio && !startingRecording && audioBlob && error && (
              <div className="mt-2 grid gap-2">
                <button
                  type="button"
                  onClick={retryAudioProcessing}
                  className="w-full rounded-lg border border-medical-300 px-3 py-2 text-sm font-semibold text-medical-800 hover:bg-medical-50"
                >
                  Повторить отправку этой записи
                </button>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={downloadPendingAudio}
                    className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-semibold text-medical-800 hover:bg-slate-50"
                  >
                    Скачать исходное аудио
                  </button>
                  <button
                    type="button"
                    onClick={discardPendingAudio}
                    className="rounded-lg border border-red-200 px-3 py-2 text-xs font-semibold text-red-700 hover:bg-red-50"
                  >
                    Удалить из формы
                  </button>
                </div>
              </div>
            )}
            {!processingAudio && !audioBlob && hasPendingClientTranscript && error && (
              <div className="mt-2 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={retryClientTranscript}
                  className="rounded-lg border border-medical-300 px-3 py-2 text-xs font-semibold text-medical-800 hover:bg-medical-50"
                >
                  Повторить формирование
                </button>
                <button
                  type="button"
                  onClick={discardClientTranscript}
                  className="rounded-lg border border-red-200 px-3 py-2 text-xs font-semibold text-red-700 hover:bg-red-50"
                >
                  Отменить
                </button>
              </div>
            )}

            <div className="mt-4">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (workflowBusy || operationInFlightRef.current) return;
                  const manualTranscript = input.trim();
                  if (!manualTranscript) return;
                  void finishClientTranscriptArtifact(manualTranscript, 'manual');
                }}
                className="flex gap-2"
              >
                <input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  disabled={workflowBusy}
                  placeholder="Ручная расшифровка (не попадёт в ASR dataset)…"
                  className="flex-1 min-w-0 box-border px-3 py-2 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-medical-400 text-sm"
                />
                <button
                  type="submit"
                  disabled={workflowBusy}
                  className="px-3 py-2 rounded-lg bg-medical-100 text-medical-800 font-medium text-sm hover:bg-medical-200 disabled:bg-slate-100 disabled:text-slate-300"
                >
                  Добавить
                </button>
              </form>
            </div>

            {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

            <div className="mt-4 text-sm text-text-muted">Фрагментов в canonical artifact: {commands.length}</div>

            {commands.length > 0 && (
              <ul className="mt-3 space-y-1 max-h-48 overflow-auto text-sm">
                {commands.map((c, i) => {
                  const a = applied[i];
                  const unknown = a && !a.ok;
                  return (
                    <li key={i} className={`px-2 py-1 rounded ${unknown ? 'bg-amber-50 text-amber-700' : 'bg-slate-50 text-medical-800'}`}>
                      {c}
                      {unknown && (
                        <span className="block text-xs mt-0.5 opacity-90">{a.detail || 'не распознано'}</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          {/* Подсказки: что можно диктовать */}
          <div className="bg-white rounded-xl border border-slate-200 mt-4">
            <button
              onClick={() => setShowHints((v) => !v)}
              className="w-full flex items-center gap-2 px-4 py-3 text-sm font-semibold text-medical-800"
            >
              <Lightbulb size={16} className="text-amber-500" /> Что можно диктовать
              <span className="ml-auto text-text-muted">{showHints ? '▾' : '▸'}</span>
            </button>
            {showHints && (
              <div className="px-4 pb-4 max-h-105 overflow-auto">
                <p className="text-xs text-text-muted mb-3">Назовите орган, параметр и значение в удобном порядке. Можно вернуться к уже названному органу. Нажмите пример, чтобы подставить его в поле.</p>
                <div className="space-y-3">
                  {hints.map((h) => (
                    <div key={h.blockId}>
                      <div className="text-xs font-semibold text-medical-700 mb-1">{h.label}</div>
                      <div className="flex flex-wrap gap-1.5">
                        {h.examples.map((ex, i) => (
                          <button
                            key={i}
                            onClick={() => fillExample(ex)}
                            className="px-2 py-1 rounded-md bg-medical-50 border border-slate-200 text-xs text-medical-800 hover:border-medical-400"
                          >
                            {ex}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
