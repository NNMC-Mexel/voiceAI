import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Mic, Search, Copy, Check, LogOut, Settings, Shield, Stethoscope, Lightbulb } from 'lucide-react';
import { apiClient, ApiRequestError } from '../api/client';
import type {
  DoctorInfo,
  RadiologyApplied,
  RadiologyBlockHint,
  RadiologyDictationReport,
  RadiologyReport,
  RadiologySpanCorrection,
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
  const [verbatimTranscript, setVerbatimTranscript] = useState('');
  const [finalReportText, setFinalReportText] = useState('');
  const [feedbackState, setFeedbackState] = useState<'idle' | 'saving' | 'saved'>('idle');
  const recogRef = useRef<SpeechRecognitionLike | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const attemptedAudioBlobRef = useRef<Blob | null>(null);
  const browserGenerationRef = useRef(0);
  const operationInFlightRef = useRef(false);
  const feedbackSavingRef = useRef(false);
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
  const displayedDocument = report ?? templatePreview;
  const showingTemplateDefaults = report === null && templatePreview !== null;
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
  const unresolvedCriticalNormalization = artifact
    ? artifact.normalization.issues.some((issue) => {
        if (issue.severity !== 'critical') return false;
        if (!issue.source) return true;
        return !reviewSpanCorrections.some(
          (correction) =>
            correction.start <= issue.source!.start
            && correction.end >= issue.source!.end
            && correction.correctedText.trim().length > 0,
        );
      })
    : false;
  const hardApprovalBlockReason = artifact
    ? artifact.legacySchemaVersion === 1
      ? 'Artifact v1 нужно повторно прогнать через pipeline v2.'
      : artifact.report === null
        ? 'Безопасный черновик не построен; запись нужно обработать повторно.'
          : unresolvedCriticalNormalization
            ? 'Есть неоднозначность нормализации; сначала нужна явная resolution врача.'
        : artifact.longform.degraded
          || artifact.longform.seamConflicts.some((seam) => seam.critical)
          ? 'Long-form декодирование degraded или содержит критический overlap-конфликт.'
          : artifact.routing.unmatchedAtomIds.length > 0
            ? 'Есть клинические фрагменты без секции; сначала требуется повторная маршрутизация.'
            : null
    : null;

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
    || hasPendingServerAudio
    || hasPendingClientTranscript
    || feedbackState === 'saving';

  const applyArtifact = useCallback((nextArtifact: RadiologyTranscriptionArtifact) => {
    if (nextArtifact.templateId !== selectedTemplateIdRef.current) {
      throw new Error('Получен результат для уже сменённого шаблона; он не был применён');
    }
    const transcript = nextArtifact.normalization.text.trim();
    const displayReport = nextArtifact.report ? toDisplayReport(nextArtifact.report) : null;
    setArtifact(nextArtifact);
    setVerbatimTranscript(nextArtifact.rawTranscript.text);
    setFinalReportText(nextArtifact.report?.fullText ?? '');
    feedbackSubmissionRef.current = null;
    setCommands(transcript ? [transcript] : []);
    setReport(displayReport);
    setApplied([{
      command: transcript,
      ok: !nextArtifact.safety.approvalBlocked,
      action: 'server-artifact',
      detail: nextArtifact.safety.status === 'passed'
        ? 'проверки безопасности пройдены'
        : 'нужна проверка врача',
    }]);
    writeStoredRadiologyPointer(LAST_RADIOLOGY_ARTIFACT_KEY, {
      sessionId: nextArtifact.sessionId,
      templateId: nextArtifact.templateId,
    });
    clearStoredRadiologyPointer(PENDING_RADIOLOGY_SESSION_KEY);
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
              && [400, 403, 404, 410].includes(requestError.status);
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

          selectedTemplateIdRef.current = pointer.templateId;
          setSelectedId(pointer.templateId);
          applyArtifact(recoveredArtifact);
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
  }, [applyArtifact, loading, templates]);

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
    setReport(null);
    setCommands([]);
    setApplied([]);
    setVerbatimTranscript('');
    setFinalReportText('');
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
  ) => {
    const clean = transcript.trim();
    if (!selectedId || !clean || operationInFlightRef.current) return;
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
  }, [applyArtifact, selectedId]);

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
      sessionIdRef.current = null;
      pendingClientTranscriptRef.current = null;
    };
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return templates;
    return templates.filter((t) => (t.name + ' ' + t.title).toLowerCase().includes(q));
  }, [query, templates]);

  const handleCopy = useCallback(() => {
    const text = artifact && report
      ? finalReportText
      : report?.text ?? templatePreview?.text ?? '';
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  }, [artifact, finalReportText, report, templatePreview]);

  const submitFeedback = useCallback(async () => {
    if (!artifact || !selected || feedbackSavingRef.current || feedbackState === 'saved') return;
    const spanCorrections = reviewSpanCorrections;
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
    setError('');
    try {
      await apiClient.submitRadiologyFeedback(artifact.sessionId, {
        idempotencyKey: submission.idempotencyKey,
        verbatimTranscript,
        finalReport: finalReportText,
        spanCorrections,
        normalizationResolutions,
        approved,
        author: doctor.name,
      });
      feedbackSubmissionRef.current = null;
      setFeedbackState('saved');
    } catch (e) {
      setFeedbackState('idle');
      setError(e instanceof Error ? e.message : 'Не удалось сохранить подтверждение');
    } finally {
      feedbackSavingRef.current = false;
    }
  }, [
    artifact,
    doctor.name,
    feedbackState,
    finalReportText,
    reviewSpanCorrections,
    selected,
    verbatimTranscript,
  ]);

  const resetToSelection = useCallback(() => {
    stopListening();
    resetRecording();
    setSelectedId(null);
    setCommands([]);
    setReport(null);
    setApplied([]);
    setInput('');
    setArtifact(null);
    setTemplatePreview(null);
    setTemplatePreviewLoading(false);
    setVerbatimTranscript('');
    setFinalReportText('');
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
  }, [resetRecording, stopListening]);

  // ─── Верхняя панель ────────────────────────────────────────────────────────
  const TopBar = (
    <div className="flex items-center justify-between px-6 py-3 border-b border-slate-200 bg-white">
      <div className="flex items-center gap-2 text-medical-800 font-semibold">
        <Stethoscope size={18} /> Лучевая диагностика
      </div>
      <div className="flex items-center gap-3 text-sm text-text-muted">
        <span className="hidden sm:inline">{doctor.name}</span>
        {onOpenTherapy && <button disabled={workflowBusy} onClick={onOpenTherapy} className="hover:text-medical-700 disabled:text-slate-300">Терапия</button>}
        {onOpenAdmin && <button disabled={workflowBusy} onClick={onOpenAdmin} className="flex items-center gap-1 hover:text-medical-700 disabled:text-slate-300"><Shield size={16} /> Админка</button>}
        {onOpenSettings && <button disabled={workflowBusy} onClick={onOpenSettings} className="flex items-center gap-1 hover:text-medical-700 disabled:text-slate-300"><Settings size={16} /> Настройки</button>}
        {onLogout && <button disabled={workflowBusy} onClick={onLogout} className="flex items-center gap-1 hover:text-red-600 disabled:text-slate-300"><LogOut size={16} /> Выйти</button>}
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
        <div className="order-2 lg:order-1">
          <div className="flex items-center justify-between mb-3">
            <button
              onClick={resetToSelection}
              disabled={workflowBusy}
              className="text-sm text-medical-700 hover:underline disabled:text-slate-300 disabled:no-underline"
            >
              ← Сменить шаблон
            </button>
            <button
              onClick={handleCopy}
              disabled={!displayedDocument}
              className="flex items-center gap-1 text-sm text-medical-700 hover:underline disabled:text-slate-300 disabled:no-underline"
            >
              {copied
                ? <><Check size={15} /> Скопировано</>
                : <><Copy size={15} /> {showingTemplateDefaults ? 'Копировать шаблон' : 'Копировать'}</>}
            </button>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            {showingTemplateDefaults && (
              <div className="mb-4 rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-sm text-sky-900">
                <span className="font-semibold">Предпросмотр шаблона.</span>{' '}
                Это нормы по умолчанию, а не результат распознавания и не подтверждённые данные исследования.
              </div>
            )}
            <div className="text-center font-semibold text-medical-900 mb-4">
              {displayedDocument?.title ?? selected.title}
            </div>
            {templatePreviewLoading && !report && (
              <div className="py-8 text-center text-sm text-text-muted">
                Загружаем структуру шаблона…
              </div>
            )}
            <div className="space-y-2 text-[15px] leading-relaxed text-medical-900">
              {displayedDocument?.blocks.map((b) => {
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
            {artifact?.report?.templateDefaults.length ? (
              <details className="mt-4 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                <summary className="cursor-pointer text-sm font-semibold text-slate-700">
                  Неподтверждённые нормы шаблона ({artifact.report.templateDefaults.length})
                </summary>
                <p className="mt-2 text-xs text-slate-500">
                  Эти разделы не были произнесены и не входят в доказательный протокол.
                </p>
                <div className="mt-3 space-y-2 text-sm text-slate-600">
                  {artifact.report.templateDefaults.map((item) => (
                    <p key={item.id}>
                      <span className="font-semibold">{item.label}:</span> {item.text}
                    </p>
                  ))}
                </div>
              </details>
            ) : null}
          </div>

          {artifact && (
            <div className="bg-white rounded-xl border border-slate-200 p-5 mt-4">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                <div>
                  <div className="font-semibold text-medical-900">Проверка перед подтверждением</div>
                  <div className="text-xs text-text-muted">
                    ASR: {artifact.model.asr.name} · decoder: {artifact.model.decoder.name} · SHA аудио: {
                      artifact.audio.sha256 ? `${artifact.audio.sha256.slice(0, 12)}…` : 'нет аудио'
                    }
                  </div>
                  <div className="text-xs text-text-muted">
                    Dataset: {artifact.training.eligible
                      ? 'может быть включён после врачебного ревью'
                      : `исключён (${artifact.training.exclusionReasons.join(', ') || 'нет основания'})`}
                  </div>
                </div>
                <span className={`text-xs font-semibold px-2 py-1 rounded-full ${
                  artifact.safety.status === 'passed'
                    ? 'bg-emerald-100 text-emerald-800'
                    : artifact.safety.status === 'failed'
                      ? 'bg-red-100 text-red-800'
                      : 'bg-amber-100 text-amber-800'
                }`}>
                  {artifact.safety.status === 'passed' ? 'Проверки пройдены' : 'Требуется проверка'}
                </span>
              </div>

              <label className="block text-sm font-semibold text-medical-800 mb-1">
                Дословная расшифровка
              </label>
              <textarea
                value={verbatimTranscript}
                disabled={feedbackState === 'saving'}
                onChange={(e) => {
                  setVerbatimTranscript(e.target.value);
                  setFeedbackState('idle');
                }}
                rows={5}
                className="w-full box-border px-3 py-2 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-medical-400 text-sm"
              />
              <p className="text-xs text-text-muted mt-1 mb-4">
                Исправляйте только то, что реально произнесено. Протокольные нормы сюда не добавляются.
              </p>

              <label className="block text-sm font-semibold text-medical-800 mb-1">
                Нормализованный текст
              </label>
              <div className="mb-4 whitespace-pre-wrap rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-medical-900">
                {artifact.normalization.text || 'Нормализация не выполнена'}
              </div>
              {artifact.normalization.issues.length > 0 && (
                <ul className="mb-4 space-y-1 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
                  {artifact.normalization.issues.map((issue) => (
                    <li key={issue.id}>
                      • {issue.message}
                      {issue.source?.text ? ` Исходный фрагмент: «${issue.source.text}».` : ''}
                    </li>
                  ))}
                </ul>
              )}

              <label className="block text-sm font-semibold text-medical-800 mb-1">
                Финальный протокол
              </label>
              <textarea
                value={finalReportText}
                disabled={feedbackState === 'saving'}
                onChange={(e) => {
                  setFinalReportText(e.target.value);
                  setFeedbackState('idle');
                }}
                rows={10}
                className="w-full box-border px-3 py-2 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-medical-400 text-sm"
              />

              {artifact.unmatchedText && (
                <div className="mt-3 rounded-lg bg-amber-50 border border-amber-200 p-3 text-sm text-amber-900">
                  <span className="font-semibold">Не удалось разнести по секциям:</span> {artifact.unmatchedText}
                </div>
              )}
              {artifact.safety.issues.length > 0 && (
                <ul className="mt-3 space-y-1 rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-900">
                  {artifact.safety.issues.map((issue, index) => (
                    <li key={`${issue.code}-${index}`}>• {issue.message}</li>
                  ))}
                </ul>
              )}

              <button
                type="button"
                onClick={() => void submitFeedback()}
                disabled={
                  feedbackState === 'saving'
                  || feedbackState === 'saved'
                  || hardApprovalBlockReason !== null
                }
                className="mt-4 w-full py-2.5 rounded-lg bg-emerald-600 text-white font-semibold hover:bg-emerald-700 disabled:bg-slate-300"
              >
                {feedbackState === 'saved'
                  ? 'Подтверждение сохранено'
                  : feedbackState === 'saving'
                    ? 'Сохраняем…'
                    : 'Подтвердить протокол врачом'}
              </button>
              <p className="text-xs text-text-muted mt-2">
                {hardApprovalBlockReason
                  ?? 'Правки сохраняются как отдельное versioned-событие и не становятся глобальным regex автоматически.'}
              </p>
            </div>
          )}
        </div>

        {/* Панель управления диктовкой */}
        <div className="order-1 lg:order-2">
          <div className="bg-white rounded-xl border border-slate-200 p-5 sticky top-6">
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
                  setInput('');
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
