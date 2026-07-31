import { createHash, randomUUID } from 'crypto';
import { link, mkdir, readFile, readdir, rm, stat, unlink, writeFile } from 'fs/promises';
import path from 'path';
import type { DictationReport } from './radiology/dictation.js';
import {
  ARRANGER_PROMPT_VERSION,
  buildArrangerPrompt,
} from './radiology/arranger.js';
import { getDocTemplate } from './radiology/doc-registry.js';
import type { NumberCheck } from './radiology/number-check.js';
import {
  denormalizeDetailed,
  type GigaAMNormalizationResult,
  type NormalizationAlignmentSpan,
} from './services/gigaam-denormalize.js';
import {
  verifyRawToNormalizedSafety,
  verifyRadiologySafety,
  type RadiologySafetyReport,
  type SafetyEntityCheck,
  type SafetyIssue,
} from './radiology/safety.js';
import {
  composeTemplateReviewDraft,
  TEMPLATE_COMPOSER_VERSION,
  templateSha256,
  type TemplateSectionAtom,
} from './radiology/template-composer.js';
import { extractNumbers } from './radiology/numbers.js';

export const RADIOLOGY_ARTIFACT_SCHEMA_VERSION = 2 as const;
export const RADIOLOGY_FEEDBACK_SCHEMA_VERSION = 2 as const;
const LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION = 1 as const;
const LEGACY_RADIOLOGY_FEEDBACK_SCHEMA_VERSION = 1 as const;

export type RadiologyTranscriptionSource =
  | 'gigaam'
  | 'whisper'
  | 'browser'
  | 'manual'
  | 'unknown';

export interface RadiologyWordTiming {
  text: string;
  startMs: number;
  endMs: number;
  confidence: number | null;
  avgLogprob?: number | null;
  scoreType?: string | null;
  chunkIndex?: number;
}

export interface RadiologyASRContractVerification {
  metadataAvailable: boolean;
  metadataSchema: boolean;
  transcriptionSchema: boolean;
  runtimeIdentity: boolean;
  checkpoint: boolean;
  decoder: boolean;
  hashes: boolean;
  wordEvidence: boolean;
  productionReady: boolean;
}

export interface RadiologyComponentVersion {
  name: string;
  version: string;
  checksum?: string;
  configSha256?: string;
}

export interface RadiologyModelMetadata {
  asr: RadiologyComponentVersion;
  vad: RadiologyComponentVersion | null;
  decoder: RadiologyComponentVersion;
  languageModel: RadiologyComponentVersion | null;
  contextVocabulary: RadiologyComponentVersion | null;
  dictionary: RadiologyComponentVersion;
  normalizer: RadiologyComponentVersion;
  template: RadiologyComponentVersion;
  router: RadiologyComponentVersion;
  prompt: RadiologyComponentVersion;
  structurer: RadiologyComponentVersion;
  /** Added additively so persisted v2 artifacts without a composer remain readable. */
  composer?: RadiologyComponentVersion;
  llm: RadiologyComponentVersion | null;
  safety: RadiologyComponentVersion;
}

export interface RadiologyASRContextBiasMetadata {
  scope: string | null;
  active: boolean;
  terms: number;
}

export interface RadiologyASRHashMetadata {
  audioSha256?: string;
  normalizedAudioSha256?: string;
  rawTextSha256: string;
  outputTextSha256?: string;
  finalTextSha256?: string;
  normalizedTextSha256: string;
}

export interface RadiologyASRRuntimeProvenance {
  schemaVersion: string;
  runtimeId: string;
  acousticDecoder: string | null;
  ctcDecoder: unknown | null;
  contextBias: RadiologyASRContextBiasMetadata;
  hashes: RadiologyASRHashMetadata;
  verification: {
    schema: boolean;
    runtime: boolean;
    checkpoint: boolean;
    hashes: boolean;
    metadata: boolean;
    decoder: boolean;
    wordEvidence: boolean;
    productionContract: boolean;
  };
}

export interface RadiologyArtifactTranscriptionChunk {
  index: number;
  rawText: string;
  rawTextSha256: string;
  normalizedText: string;
  normalizedTextSha256: string;
  rawAvailable: boolean;
  language: string;
  source: RadiologyTranscriptionSource;
  words: RadiologyWordTiming[];
  longform?: RadiologyChunkTranscription['longform'];
  provenance: RadiologyASRRuntimeProvenance;
}

export interface RadiologyEvidenceSpan {
  transcript: 'raw' | 'normalized';
  start: number;
  end: number;
  text: string;
}

export interface RadiologyArtifactSection {
  id: string;
  label: string;
  text: string;
  source: 'dictated' | 'normal' | 'conclusion' | 'unmatched';
  evidence: RadiologyEvidenceSpan[];
  origin?:
    | 'verbatim'
    | 'explicit-normal-template'
    | 'missing-template-default'
    | 'dictated-conclusion'
    | 'extractive-conclusion'
    | 'unmatched';
  assignmentMethod?: 'anchor' | 'rule' | 'llm' | 'unmatched' | 'template' | 'mixed';
}

export interface RadiologyNormalizationTransformation {
  kind: string;
  source: {
    start: number;
    end: number;
    text: string;
  };
  normalized: {
    start: number;
    end: number;
    text: string;
  };
}

export interface RadiologyNormalizationIssue {
  id: string;
  code: string;
  severity: 'critical' | 'warning';
  message: string;
  source?: {
    start: number;
    end: number;
    text: string;
  };
  normalized?: {
    start: number;
    end: number;
    text: string;
  };
  values?: number[];
}

export interface RadiologyTranscriptAtom {
  id: string;
  start: number;
  end: number;
  text: string;
  candidateSectionIds: string[];
  anchorRuleIds: string[];
}

export interface RadiologySpanAssignment {
  atomId: string;
  sectionId: string | null;
  method: 'anchor' | 'rule' | 'llm' | 'unmatched';
}

export interface RadiologyTemplateDefault {
  id: string;
  label: string;
  text: string;
}

export interface RadiologyEvidenceBackedConclusion {
  text: string;
  mode: 'dictated' | 'extractive';
  evidence: RadiologyEvidenceSpan[];
}

export interface RadiologySafetyCheck<T = unknown> {
  status: 'passed' | 'failed' | 'not_run';
  details?: T;
}

export interface RadiologySafetyStageResult {
  stage: 'raw_to_normalized' | 'normalized_to_report' | 'verbatim_to_final_report';
  status: 'passed' | 'failed' | 'incomplete';
  sourceSha256: string | null;
  outputSha256: string | null;
  numbers: RadiologySafetyCheck<NumberCheck>;
  units: RadiologySafetyCheck<SafetyEntityCheck>;
  negations: RadiologySafetyCheck<SafetyEntityCheck>;
  laterality: RadiologySafetyCheck<SafetyEntityCheck>;
  contrast: RadiologySafetyCheck<SafetyEntityCheck>;
  criticalFacts: RadiologySafetyCheck<SafetyEntityCheck>;
  issues: SafetyIssue[];
}

export interface RadiologySafetyResult {
  status: 'passed' | 'failed' | 'incomplete';
  stages: RadiologySafetyStageResult[];
  numbers: RadiologySafetyCheck<NumberCheck>;
  units: RadiologySafetyCheck<SafetyEntityCheck>;
  negations: RadiologySafetyCheck<SafetyEntityCheck>;
  laterality: RadiologySafetyCheck<SafetyEntityCheck>;
  contrast: RadiologySafetyCheck<SafetyEntityCheck>;
  criticalFacts: RadiologySafetyCheck<SafetyEntityCheck>;
  requiresReview: boolean;
  approvalBlocked: boolean;
  issues: SafetyIssue[];
}

export type RadiologyArtifactReport = Omit<DictationReport, 'templateDefaults'> & {
  sections: RadiologyArtifactSection[];
  conclusion: RadiologyEvidenceBackedConclusion | null;
  evidenceBackedText: string;
  evidenceSha256: string;
  templateDefaults: RadiologyTemplateDefault[];
};

export interface RadiologyTranscriptionArtifact {
  schemaVersion: typeof RADIOLOGY_ARTIFACT_SCHEMA_VERSION;
  legacySchemaVersion?: typeof LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION;
  kind: 'radiology-transcription';
  sessionId: string;
  /**
   * Object-level PHI owner. `null` is reserved for the explicitly enabled
   * no-auth/test compatibility mode; authenticated sessions always persist a
   * normalized string id (JWT ids are numeric in the current application).
   */
  ownerDoctorId: string | null;
  templateId: string;
  createdAt: string;
  completedAt: string;
  source: {
    type: RadiologyTranscriptionSource;
    audioSha256: string | null;
  };
  audio: {
    sha256: string | null;
    hashKind: 'sha256-bytes' | 'sha256-index-length-prefixed-chunks-v1' | 'none';
    bytes: number;
    mimeType?: string;
    stored: boolean;
    chunks: Array<{
      index: number;
      sha256: string;
      bytes: number;
      stored: boolean;
    }>;
  };
  rawTranscript: {
    text: string;
    sha256: string;
    language: string;
    rawAvailable: boolean;
    words: RadiologyWordTiming[];
  };
  normalizedTranscript: {
    text: string;
    sha256: string;
  };
  normalization: {
    text: string;
    sha256: string;
    version: string;
    transformations: RadiologyNormalizationTransformation[];
    issues: RadiologyNormalizationIssue[];
  };
  /**
   * Exact per-request ASR outputs. Unlike the aggregate transcript, these
   * strings are not trimmed or joined, so hashes supplied by the ASR runtime
   * remain independently verifiable.
   */
  asrChunks: RadiologyArtifactTranscriptionChunk[];
  longform: {
    degraded: boolean;
    seamConflicts: Array<{
      chunkIndex: number;
      startMs: number;
      endMs: number;
      critical: boolean;
      leftText?: string;
      rightText?: string;
    }>;
  };
  sections: RadiologyArtifactSection[];
  routing: {
    atoms: RadiologyTranscriptAtom[];
    assignments: RadiologySpanAssignment[];
    unmatchedAtomIds: string[];
  };
  unmatchedText: string;
  report: RadiologyArtifactReport | null;
  reportSha256: string | null;
  safety: RadiologySafetyResult;
  components: RadiologyModelMetadata;
  /** @deprecated Use components. Kept for one compatibility release. */
  model: RadiologyModelMetadata;
  training: {
    eligible: boolean;
    exclusionReasons: string[];
  };
}

export interface RadiologyChunkTranscription {
  rawText?: string;
  normalizedText: string;
  normalization?: {
    version: string;
    transformations: RadiologyNormalizationTransformation[];
    issues: RadiologyNormalizationIssue[];
  };
  rawAvailable?: boolean;
  language?: string;
  words?: RadiologyWordTiming[];
  source?: RadiologyTranscriptionSource;
  model?: Partial<RadiologyModelMetadata>;
  schemaVersion?: string;
  runtimeId?: string;
  contextBias?: RadiologyASRContextBiasMetadata;
  hashes?: Partial<RadiologyASRHashMetadata>;
  verification?: RadiologyASRContractVerification;
  provenance?: {
    schemaVersion: string;
    runtimeId: string;
    acousticDecoder: string | null;
    ctcDecoder: unknown | null;
    contextBias: RadiologyASRContextBiasMetadata;
    hashes: Partial<RadiologyASRHashMetadata>;
    checkpointVerified?: boolean;
    verification?: RadiologyASRContractVerification;
  };
  checkpointVerified?: boolean;
  longform?: {
    mode: 'vad' | 'emission_stitch' | 'text_fallback' | 'single';
    degraded: boolean;
    vad?: RadiologyComponentVersion | null;
    seams?: Array<{
      startMs: number;
      endMs: number;
      conflict: boolean;
      critical: boolean;
      leftText?: string;
      rightText?: string;
    }>;
  };
}

export type RadiologyChunkTranscriber = (
  audioBase64: string,
  context: { sessionId: string; templateId: string; chunkIndex: number },
) => Promise<string | RadiologyChunkTranscription>;

export type RadiologyTranscriptStructurer = (
  templateId: string,
  normalizedTranscript: string,
  context?: {
    allowLLM: boolean;
    normalizationAmbiguous: boolean;
    rawTranscript: string;
    normalizationAlignment: NormalizationAlignmentSpan[];
  },
) => Promise<DictationReport>;

export interface CreateRadiologySessionInput {
  templateId: string;
  source?: RadiologyTranscriptionSource;
  mimeType?: string;
  retainAudio?: boolean;
}

export interface RadiologySessionActor {
  authenticated: true;
  doctorId?: string;
  role?: string;
}

export interface CreateRadiologySessionResult {
  sessionId: string;
  mode: 'radiology';
  templateId: string;
  source: RadiologyTranscriptionSource;
  retainAudio: boolean;
  createdAt: string;
  chunkUrl: string;
  finishUrl: string;
}

export interface SpanCorrectionInput {
  start: number;
  end: number;
  originalText: string;
  correctedText: string;
  entityType: string;
  confidence?: number | null;
  modality: string;
  author?: string;
}

export interface RadiologyFeedbackInput {
  idempotencyKey: string;
  verbatimTranscript: string;
  finalReport: string;
  spanCorrections: SpanCorrectionInput[];
  normalizationResolutions?: NormalizationResolutionInput[];
  baseDraftSha256?: string;
  acceptedTemplateSegmentIds?: string[];
  reviewedResidualAtomIds?: string[];
  approved: boolean;
  author?: string;
}

export interface RadiologyRecomposeInput {
  verbatimTranscript: string;
  spanCorrections: SpanCorrectionInput[];
}

/**
 * A deterministic, non-persisted projection built from an immutable source
 * artifact plus physician span corrections. The original ASR artifact is
 * never modified; feedback can bind to this revision's review-draft SHA.
 */
export interface RadiologyRecomposeRevision {
  schemaVersion: 1;
  kind: 'radiology-recompose-revision';
  sessionId: string;
  templateId: string;
  sourceArtifactSha256: string;
  verbatimTranscript: {
    text: string;
    sha256: string;
  };
  normalization: RadiologyTranscriptionArtifact['normalization'];
  routing: RadiologyTranscriptionArtifact['routing'];
  report: RadiologyArtifactReport | null;
  safety: RadiologySafetyResult;
  components: RadiologyModelMetadata;
}

export interface NormalizationResolutionInput {
  issueId: string;
  replacementText: string;
  resolution: 'confirmed_single' | 'confirmed_range' | 'confirmed_verbatim';
}

export interface StoredSpanCorrection extends SpanCorrectionInput {
  confidence: number | null;
  author: string;
}

export interface RadiologyFeedbackEvent {
  schemaVersion: typeof RADIOLOGY_FEEDBACK_SCHEMA_VERSION;
  kind: 'radiology-feedback';
  datasetVersion: 'radiology-feedback/v2';
  feedbackId: string;
  revision: number;
  idempotencyKey: string;
  contentSha256: string;
  sessionId: string;
  templateId: string;
  createdAt: string;
  source: RadiologyTranscriptionSource;
  author: string;
  verbatimTranscript: string;
  verbatimTranscriptSha256: string;
  finalReport: string;
  finalReportSha256: string;
  spanCorrections: StoredSpanCorrection[];
  normalizationResolutions: NormalizationResolutionInput[];
  baseDraftSha256: string | null;
  acceptedTemplateSegmentIds: string[];
  reviewedResidualAtomIds: string[];
  recomposeRevision: RadiologyRecomposeRevision | null;
  approved: boolean;
  safety: RadiologySafetyReport;
  normalizationSafetyStage: RadiologySafetyStageResult;
  safetyStage: RadiologySafetyStageResult;
  training: {
    eligible: boolean;
    exclusionReasons: string[];
  };
}

export class RadiologySessionError extends Error {
  constructor(
    public readonly statusCode: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = 'RadiologySessionError';
  }
}

const DEFAULT_FEEDBACK_REVISION_LIMIT = 50;
const DEFAULT_STORAGE_RETENTION_MS = 30 * 24 * 60 * 60 * 1000;
const DEFAULT_ORPHAN_GRACE_MS = 24 * 60 * 60 * 1000;
const DEFAULT_STORAGE_CLEANUP_INTERVAL_MS = 10 * 60 * 1000;

function sha256Text(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function canonicalValue(value: unknown): unknown {
  if (typeof value === 'function') {
    return { $function: Function.prototype.toString.call(value) };
  }
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, canonicalValue(entry)]),
    );
  }
  return value;
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

function immutableArtifactFingerprint(artifact: RadiologyTranscriptionArtifact): string {
  const { completedAt: _completedAt, ...immutable } = artifact;
  return sha256Text(canonicalJson(immutable));
}

function builtInTemplateComponent(templateId: string): RadiologyComponentVersion {
  const template = getDocTemplate(templateId);
  if (!template) return { name: templateId, version: 'unknown' };
  return {
    name: templateId,
    version: 'content-v1',
    checksum: sha256Text(canonicalJson(template)),
  };
}

function sha256ChunkManifest(chunks: Array<{ index: number; audio: Buffer }>): string | null {
  if (chunks.length === 0) return null;
  const hash = createHash('sha256');
  hash.update('voicemed-radiology-chunks-v1\0', 'utf8');
  for (const chunk of chunks) {
    const prefix = Buffer.alloc(12);
    prefix.writeUInt32BE(chunk.index, 0);
    prefix.writeBigUInt64BE(BigInt(chunk.audio.length), 4);
    hash.update(prefix);
    hash.update(chunk.audio);
  }
  return hash.digest('hex');
}

function audioArtifactHash(chunks: Array<{ index: number; audio: Buffer }>): {
  sha256: string | null;
  hashKind: RadiologyTranscriptionArtifact['audio']['hashKind'];
} {
  if (chunks.length === 0) return { sha256: null, hashKind: 'none' };
  if (chunks.length === 1) {
    return {
      sha256: createHash('sha256').update(chunks[0].audio).digest('hex'),
      hashKind: 'sha256-bytes',
    };
  }
  return {
    sha256: sha256ChunkManifest(chunks),
    hashKind: 'sha256-index-length-prefixed-chunks-v1',
  };
}

function safeSessionId(sessionId: string): string {
  if (!/^[A-Za-z0-9_-]{8,128}$/u.test(sessionId)) {
    throw new RadiologySessionError(400, 'invalid_session_id', 'Invalid radiology session id');
  }
  return sessionId;
}

function asErrorCode(error: unknown): string | undefined {
  return typeof error === 'object' && error !== null && 'code' in error
    ? String((error as { code?: unknown }).code)
    : undefined;
}

async function atomicCreateFile(filename: string, content: string | Buffer): Promise<void> {
  await mkdir(path.dirname(filename), { recursive: true, mode: 0o700 });
  const temp = `${filename}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temp, content, { flag: 'wx', mode: 0o600 });
    // A hard link publishes only a completely written file and fails if the
    // immutable destination already exists.
    await link(temp, filename);
  } finally {
    await unlink(temp).catch(() => undefined);
  }
}

async function atomicCreateOrVerify(filename: string, content: Buffer): Promise<void> {
  try {
    await atomicCreateFile(filename, content);
  } catch (error) {
    if (asErrorCode(error) !== 'EEXIST') throw error;
    const existing = await readFile(filename);
    if (!existing.equals(content)) {
      throw new RadiologySessionError(
        409,
        'immutable_file_conflict',
        'A different immutable file already exists',
      );
    }
  }
}

export interface RadiologyArtifactStoreOptions {
  maxFeedbackRevisionsPerSession?: number;
  storageRetentionMs?: number;
  orphanGraceMs?: number;
  cleanupIntervalMs?: number;
  now?: () => Date;
}

export interface SaveRadiologyFeedbackResult {
  feedback: RadiologyFeedbackEvent;
  idempotentReplay: boolean;
}

function notRunSafetyStage(
  stage: RadiologySafetyStageResult['stage'],
  sourceSha256: string | null = null,
  outputSha256: string | null = null,
): RadiologySafetyStageResult {
  return {
    stage,
    status: 'incomplete',
    sourceSha256,
    outputSha256,
    numbers: { status: 'not_run' },
    units: { status: 'not_run' },
    negations: { status: 'not_run' },
    laterality: { status: 'not_run' },
    contrast: { status: 'not_run' },
    criticalFacts: { status: 'not_run' },
    issues: [],
  };
}

function adaptLegacyArtifact(value: unknown): RadiologyTranscriptionArtifact | null {
  if (!value || typeof value !== 'object') return null;
  const legacy = value as Record<string, any>;
  if (legacy.schemaVersion !== LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION) return null;
  if (typeof legacy.sessionId !== 'string' || typeof legacy.templateId !== 'string') return null;

  const legacyReport = (legacy.report ?? null) as DictationReport | null;
  const sections = Array.isArray(legacy.sections)
    ? legacy.sections as RadiologyArtifactSection[]
    : reportSections(legacyReport);
  const evidenceBackedText = legacyReport
    ? [
        ...legacyReport.blocks
          .filter((block) => block.source === 'dictated' || block.source === 'conclusion')
          .map((block) => block.text),
        legacyReport.unmatched,
      ].filter(Boolean).join(' ')
    : '';
  const report = legacyReport
    ? {
        ...legacyReport,
        sections,
        conclusion: null,
        evidenceBackedText,
        evidenceSha256: sha256Text(evidenceBackedText),
        templateDefaults: [],
      } satisfies RadiologyArtifactReport
    : null;
  const legacyModel = modelMetadata(
    legacy.templateId,
    (legacy.model ?? {}) as Partial<RadiologyModelMetadata>,
    [],
  );
  const rawSha256 = legacy.rawTranscript?.sha256
    ?? sha256Text(String(legacy.rawTranscript?.text ?? ''));
  const normalizedText = String(
    legacy.normalizedTranscript?.text
    ?? legacy.rawTranscript?.text
    ?? '',
  );
  const normalizedSha256 = legacy.normalizedTranscript?.sha256
    ?? sha256Text(normalizedText);
  const legacySafety = (legacy.safety ?? {}) as Partial<RadiologySafetyResult>;
  const normalizedNotRun = notRunSafetyStage(
    'normalized_to_report',
    normalizedSha256,
    report?.evidenceSha256 ?? null,
  );
  const fallbackCheck = <T>(): RadiologySafetyCheck<T> => ({ status: 'not_run' });

  return {
    ...legacy,
    schemaVersion: RADIOLOGY_ARTIFACT_SCHEMA_VERSION,
    legacySchemaVersion: LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION,
    source: {
      type: typeof legacy.source === 'string' ? legacy.source : 'unknown',
      audioSha256: legacy.audio?.sha256 ?? null,
    },
    normalizedTranscript: {
      text: normalizedText,
      sha256: normalizedSha256,
    },
    normalization: {
      text: normalizedText,
      sha256: normalizedSha256,
      version: 'legacy-unverified',
      transformations: [],
      issues: [{
        id: 'legacy-normalization-unverified',
        code: 'legacy_normalization_unverified',
        severity: 'critical',
        message: 'Artifact v1 has no reproducible raw-to-normalized safety stage.',
      }],
    },
    asrChunks: Array.isArray(legacy.asrChunks) ? legacy.asrChunks : [],
    longform: {
      degraded: true,
      seamConflicts: [],
    },
    sections,
    routing: {
      atoms: [],
      assignments: [],
      unmatchedAtomIds: [],
    },
    unmatchedText: String(legacy.unmatchedText ?? legacyReport?.unmatched ?? ''),
    report,
    reportSha256: report ? sha256Text(report.fullText) : null,
    safety: {
      status: 'incomplete',
      stages: [
        notRunSafetyStage('raw_to_normalized', rawSha256, normalizedSha256),
        normalizedNotRun,
      ],
      numbers: legacySafety.numbers ?? fallbackCheck<NumberCheck>(),
      units: legacySafety.units ?? fallbackCheck<SafetyEntityCheck>(),
      negations: legacySafety.negations ?? fallbackCheck<SafetyEntityCheck>(),
      laterality: legacySafety.laterality ?? fallbackCheck<SafetyEntityCheck>(),
      contrast: legacySafety.contrast ?? fallbackCheck<SafetyEntityCheck>(),
      criticalFacts: legacySafety.criticalFacts ?? fallbackCheck<SafetyEntityCheck>(),
      requiresReview: true,
      approvalBlocked: true,
      issues: legacySafety.issues ?? [],
    },
    components: legacyModel,
    model: legacyModel,
    training: {
      eligible: false,
      exclusionReasons: uniqueReasons([
        ...(legacy.training?.exclusionReasons ?? []),
        'legacy_artifact_schema',
        'raw_to_normalized_safety_not_run',
      ]),
    },
  } as unknown as RadiologyTranscriptionArtifact;
}

export class RadiologyArtifactStore {
  private writeQueue: Promise<void> = Promise.resolve();
  private readonly maxFeedbackRevisionsPerSession: number;
  private readonly storageRetentionMs: number;
  private readonly orphanGraceMs: number;
  private readonly cleanupIntervalMs: number;
  private readonly now: () => Date;
  private lastCleanupAtMs = Number.NEGATIVE_INFINITY;

  constructor(
    private readonly rootDir: string,
    options: RadiologyArtifactStoreOptions = {},
  ) {
    this.maxFeedbackRevisionsPerSession =
      options.maxFeedbackRevisionsPerSession ?? DEFAULT_FEEDBACK_REVISION_LIMIT;
    this.storageRetentionMs = options.storageRetentionMs ?? DEFAULT_STORAGE_RETENTION_MS;
    this.orphanGraceMs = options.orphanGraceMs ?? DEFAULT_ORPHAN_GRACE_MS;
    this.cleanupIntervalMs =
      options.cleanupIntervalMs ?? DEFAULT_STORAGE_CLEANUP_INTERVAL_MS;
    this.now = options.now ?? (() => new Date());
  }

  private artifactPathForVersion(
    sessionId: string,
    version: typeof RADIOLOGY_ARTIFACT_SCHEMA_VERSION | typeof LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION,
  ): string {
    return path.join(
      this.rootDir,
      `schema-v${version}`,
      'artifacts',
      `${safeSessionId(sessionId)}.json`,
    );
  }

  private artifactPath(sessionId: string): string {
    return this.artifactPathForVersion(sessionId, RADIOLOGY_ARTIFACT_SCHEMA_VERSION);
  }

  private legacyArtifactPath(sessionId: string): string {
    return this.artifactPathForVersion(sessionId, LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION);
  }

  private audioChunkPath(sessionId: string, chunkIndex: number): string {
    return path.join(
      this.rootDir,
      `schema-v${RADIOLOGY_ARTIFACT_SCHEMA_VERSION}`,
      'audio',
      safeSessionId(sessionId),
      `${String(chunkIndex).padStart(8, '0')}.chunk`,
    );
  }

  private feedbackDirForVersion(
    sessionId: string,
    version: typeof RADIOLOGY_FEEDBACK_SCHEMA_VERSION | typeof LEGACY_RADIOLOGY_FEEDBACK_SCHEMA_VERSION,
  ): string {
    return path.join(
      this.rootDir,
      `schema-v${version}`,
      'feedback',
      safeSessionId(sessionId),
    );
  }

  private feedbackDir(sessionId: string): string {
    return this.feedbackDirForVersion(sessionId, RADIOLOGY_FEEDBACK_SCHEMA_VERSION);
  }

  private legacyFeedbackDir(sessionId: string): string {
    return this.feedbackDirForVersion(sessionId, LEGACY_RADIOLOGY_FEEDBACK_SCHEMA_VERSION);
  }

  private artifactDir(): string {
    return path.join(
      this.rootDir,
      `schema-v${RADIOLOGY_ARTIFACT_SCHEMA_VERSION}`,
      'artifacts',
    );
  }

  private audioRootDir(): string {
    return path.join(
      this.rootDir,
      `schema-v${RADIOLOGY_ARTIFACT_SCHEMA_VERSION}`,
      'audio',
    );
  }

  private legacyAudioRootDir(): string {
    return path.join(
      this.rootDir,
      `schema-v${LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION}`,
      'audio',
    );
  }

  private feedbackRootDir(): string {
    return path.join(
      this.rootDir,
      `schema-v${RADIOLOGY_FEEDBACK_SCHEMA_VERSION}`,
      'feedback',
    );
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.writeQueue.then(operation, operation);
    this.writeQueue = result.then(() => undefined, () => undefined);
    return result;
  }

  async saveArtifact(
    artifact: RadiologyTranscriptionArtifact,
    audioChunks: Array<{ index: number; audio: Buffer }> | null,
  ): Promise<RadiologyTranscriptionArtifact> {
    return this.enqueue(async () => {
      await this.maybeCleanupExpiredData();
      const existing = await this.getArtifact(artifact.sessionId);
      if (existing) {
        if (
          existing.rawTranscript.sha256 === artifact.rawTranscript.sha256
          && existing.normalizedTranscript.sha256 === artifact.normalizedTranscript.sha256
          && existing.audio.sha256 === artifact.audio.sha256
          && (existing.ownerDoctorId ?? null) === artifact.ownerDoctorId
          && immutableArtifactFingerprint(existing) === immutableArtifactFingerprint(artifact)
        ) {
          return existing;
        }
        throw new RadiologySessionError(
          409,
          'artifact_already_exists',
          'A different immutable artifact already exists for this session',
        );
      }

      if (audioChunks) {
        const actualAudio = audioArtifactHash(audioChunks);
        if (
          actualAudio.sha256 !== artifact.audio.sha256
          || actualAudio.hashKind !== artifact.audio.hashKind
        ) {
          throw new RadiologySessionError(500, 'audio_hash_mismatch', 'Audio hash validation failed');
        }
        for (const chunk of audioChunks) {
          const metadata = artifact.audio.chunks.find((item) => item.index === chunk.index);
          const actualChunkHash = createHash('sha256').update(chunk.audio).digest('hex');
          if (!metadata || metadata.sha256 !== actualChunkHash || metadata.bytes !== chunk.audio.length) {
            throw new RadiologySessionError(
              500,
              'audio_chunk_hash_mismatch',
              `Audio chunk ${chunk.index} hash validation failed`,
            );
          }
          await atomicCreateOrVerify(this.audioChunkPath(artifact.sessionId, chunk.index), chunk.audio);
        }
      }

      try {
        await atomicCreateFile(
          this.artifactPath(artifact.sessionId),
          `${JSON.stringify(artifact, null, 2)}\n`,
        );
      } catch (error) {
        if (asErrorCode(error) !== 'EEXIST') throw error;
        const raced = await this.getArtifact(artifact.sessionId);
        if (
          raced
          && raced.rawTranscript.sha256 === artifact.rawTranscript.sha256
          && raced.normalizedTranscript.sha256 === artifact.normalizedTranscript.sha256
          && raced.audio.sha256 === artifact.audio.sha256
          && (raced.ownerDoctorId ?? null) === artifact.ownerDoctorId
          && immutableArtifactFingerprint(raced) === immutableArtifactFingerprint(artifact)
        ) {
          return raced;
        }
        if (raced) {
          throw new RadiologySessionError(
            409,
            'artifact_already_exists',
            'A different immutable artifact already exists for this session',
          );
        }
        throw error;
      }
      return artifact;
    });
  }

  async getArtifact(sessionId: string): Promise<RadiologyTranscriptionArtifact | null> {
    let artifact: RadiologyTranscriptionArtifact | null = null;
    try {
      const raw = await readFile(this.artifactPath(sessionId), 'utf8');
      const parsed = JSON.parse(raw) as RadiologyTranscriptionArtifact;
      artifact = parsed.schemaVersion === RADIOLOGY_ARTIFACT_SCHEMA_VERSION ? parsed : null;
    } catch (error) {
      if (asErrorCode(error) !== 'ENOENT') throw error;
    }
    if (!artifact) {
      try {
        const raw = await readFile(this.legacyArtifactPath(sessionId), 'utf8');
        artifact = adaptLegacyArtifact(JSON.parse(raw));
      } catch (error) {
        if (asErrorCode(error) !== 'ENOENT') throw error;
      }
    }
    if (!artifact) return null;
    const completedAtMs = Date.parse(artifact.completedAt || artifact.createdAt);
    if (
      Number.isFinite(completedAtMs)
      && this.now().getTime() - completedAtMs > this.storageRetentionMs
    ) {
      await this.removeSessionData(sessionId);
      return null;
    }
    return artifact;
  }

  async saveFeedback(
    event: Omit<RadiologyFeedbackEvent, 'revision'>,
  ): Promise<SaveRadiologyFeedbackResult> {
    return this.enqueue(async () => {
      await this.maybeCleanupExpiredData();
      const dir = this.feedbackDir(event.sessionId);
      await mkdir(dir, { recursive: true, mode: 0o700 });
      const entries = (await readdir(dir)).filter((name) => name.endsWith('.json')).sort();
      for (const name of entries) {
        const raw = await readFile(path.join(dir, name), 'utf8');
        const existing = JSON.parse(raw) as Partial<RadiologyFeedbackEvent>;
        if (existing.idempotencyKey !== event.idempotencyKey) continue;
        if (
          existing.contentSha256 === event.contentSha256
          && existing.schemaVersion === RADIOLOGY_FEEDBACK_SCHEMA_VERSION
        ) {
          return {
            feedback: existing as RadiologyFeedbackEvent,
            idempotentReplay: true,
          };
        }
        throw new RadiologySessionError(
          409,
          'feedback_idempotency_conflict',
          'idempotencyKey is already bound to a different feedback payload',
        );
      }
      if (entries.length >= this.maxFeedbackRevisionsPerSession) {
        throw new RadiologySessionError(
          409,
          'feedback_revision_limit',
          'This radiology session reached the configured feedback revision limit',
        );
      }
      const revision = entries.length + 1;
      const stored: RadiologyFeedbackEvent = { ...event, revision };
      const filename = path.join(
        dir,
        `${String(revision).padStart(6, '0')}-${event.feedbackId}.json`,
      );
      await atomicCreateFile(filename, `${JSON.stringify(stored, null, 2)}\n`);
      return { feedback: stored, idempotentReplay: false };
    });
  }

  async listFeedback(sessionId: string): Promise<RadiologyFeedbackEvent[]> {
    const currentDir = this.feedbackDir(sessionId);
    let entries: string[] = [];
    try {
      entries = (await readdir(currentDir)).filter((name) => name.endsWith('.json')).sort();
    } catch (error) {
      if (asErrorCode(error) !== 'ENOENT') throw error;
    }
    const current = await Promise.all(entries.map(async (name) => {
      const raw = await readFile(path.join(currentDir, name), 'utf8');
      return JSON.parse(raw) as RadiologyFeedbackEvent;
    }));
    if (current.length) return current;

    const legacyDir = this.legacyFeedbackDir(sessionId);
    let legacyEntries: string[] = [];
    try {
      legacyEntries = (await readdir(legacyDir)).filter((name) => name.endsWith('.json')).sort();
    } catch (error) {
      if (asErrorCode(error) !== 'ENOENT') throw error;
    }
    return Promise.all(legacyEntries.map(async (name) => {
      const raw = await readFile(path.join(legacyDir, name), 'utf8');
      const legacy = JSON.parse(raw) as Record<string, any>;
      return {
        ...legacy,
        schemaVersion: RADIOLOGY_FEEDBACK_SCHEMA_VERSION,
        datasetVersion: 'radiology-feedback/v2',
        normalizationResolutions: [],
        baseDraftSha256: null,
        acceptedTemplateSegmentIds: [],
        reviewedResidualAtomIds: [],
        recomposeRevision: null,
        normalizationSafetyStage: notRunSafetyStage('raw_to_normalized'),
        safetyStage: notRunSafetyStage('verbatim_to_final_report'),
        training: {
          eligible: false,
          exclusionReasons: uniqueReasons([
            ...(legacy.training?.exclusionReasons ?? []),
            'legacy_feedback_schema',
          ]),
        },
      } as unknown as RadiologyFeedbackEvent;
    }));
  }

  async getFeedbackByIdempotencyKey(
    sessionId: string,
    idempotencyKey: string,
  ): Promise<RadiologyFeedbackEvent | null> {
    const feedback = await this.listFeedback(sessionId);
    return feedback.find((event) => event.idempotencyKey === idempotencyKey) ?? null;
  }

  private async maybeCleanupExpiredData(): Promise<void> {
    const nowMs = this.now().getTime();
    if (nowMs - this.lastCleanupAtMs < this.cleanupIntervalMs) return;
    this.lastCleanupAtMs = nowMs;
    await this.cleanupExpiredData(nowMs);
  }

  private async cleanupExpiredData(nowMs: number): Promise<void> {
    let artifactEntries: string[] = [];
    try {
      artifactEntries = (await readdir(this.artifactDir()))
        .filter((name) => name.endsWith('.json'));
    } catch (error) {
      if (asErrorCode(error) !== 'ENOENT') throw error;
    }

    const liveSessionIds = new Set<string>();
    for (const name of artifactEntries) {
      const sessionId = name.slice(0, -'.json'.length);
      if (!/^[A-Za-z0-9_-]{8,128}$/u.test(sessionId)) continue;
      try {
        const raw = await readFile(path.join(this.artifactDir(), name), 'utf8');
        const artifact = JSON.parse(raw) as Partial<RadiologyTranscriptionArtifact>;
        const completedAtMs = Date.parse(artifact.completedAt || artifact.createdAt || '');
        if (
          Number.isFinite(completedAtMs)
          && nowMs - completedAtMs > this.storageRetentionMs
        ) {
          await this.removeSessionData(sessionId);
        } else {
          liveSessionIds.add(sessionId);
        }
      } catch (error) {
        // A corrupt artifact must not make the whole store unusable. Leave it
        // for operator review; orphan cleanup below only removes directories
        // whose age exceeds the explicit grace period.
        if (asErrorCode(error) === 'ENOENT') continue;
      }
    }

    await Promise.all([
      this.cleanupOrphanDirectories(this.audioRootDir(), liveSessionIds, nowMs),
      this.cleanupOrphanDirectories(this.feedbackRootDir(), liveSessionIds, nowMs),
    ]);
  }

  private async cleanupOrphanDirectories(
    root: string,
    liveSessionIds: Set<string>,
    nowMs: number,
  ): Promise<void> {
    let entries: string[] = [];
    try {
      entries = await readdir(root);
    } catch (error) {
      if (asErrorCode(error) === 'ENOENT') return;
      throw error;
    }
    for (const sessionId of entries) {
      if (
        liveSessionIds.has(sessionId)
        || !/^[A-Za-z0-9_-]{8,128}$/u.test(sessionId)
      ) {
        continue;
      }
      const target = path.join(root, sessionId);
      try {
        const metadata = await stat(target);
        if (nowMs - metadata.mtimeMs > this.orphanGraceMs) {
          await rm(target, { recursive: true, force: true });
        }
      } catch (error) {
        if (asErrorCode(error) !== 'ENOENT') throw error;
      }
    }
  }

  private async removeSessionData(sessionId: string): Promise<void> {
    const safeId = safeSessionId(sessionId);
    await Promise.all([
      rm(this.artifactPath(safeId), { force: true }),
      rm(this.legacyArtifactPath(safeId), { force: true }),
      rm(path.join(this.audioRootDir(), safeId), { recursive: true, force: true }),
      rm(path.join(this.legacyAudioRootDir(), safeId), { recursive: true, force: true }),
      rm(this.feedbackDir(safeId), { recursive: true, force: true }),
      rm(this.legacyFeedbackDir(safeId), { recursive: true, force: true }),
    ]);
  }
}

interface SessionChunk {
  index: number;
  audio: Buffer;
  audioSha256: string;
  transcription: Promise<RadiologyChunkTranscription> | null;
  transcriptionState: 'pending' | 'fulfilled' | 'rejected';
}

interface ActiveRadiologySession {
  id: string;
  ownerDoctorId: string | null;
  templateId: string;
  source: RadiologyTranscriptionSource;
  mimeType?: string;
  retainAudio: boolean;
  createdAt: string;
  lastActivityAtMs: number;
  totalAudioBytes: number;
  chunks: Map<number, SessionChunk>;
  finishPromise?: Promise<RadiologyTranscriptionArtifact>;
}

export interface RadiologySessionServiceOptions {
  store: RadiologyArtifactStore;
  transcribeChunk?: RadiologyChunkTranscriber;
  structureTranscript?: RadiologyTranscriptStructurer;
  model?: Partial<RadiologyModelMetadata>;
  sessionTtlMs?: number;
  maxChunkBytes?: number;
  maxChunksPerSession?: number;
  maxTotalAudioBytes?: number;
  maxActiveSessions?: number;
  maxActiveAudioBytes?: number;
  maxOwnerActiveSessions?: number;
  maxOwnerActiveAudioBytes?: number;
  maxPendingTranscriptions?: number;
  /**
   * Compatibility escape hatch for isolated tests/development without an
   * authenticated user object. Keep disabled for PHI-bearing deployments.
   */
  allowUnownedSessions?: boolean;
  allowAudioPersistence?: boolean;
  now?: () => Date;
  idFactory?: () => string;
}

function component(
  value: Partial<RadiologyComponentVersion> | undefined,
  fallbackName: string,
): RadiologyComponentVersion {
  const name = value?.name?.trim() || fallbackName;
  const version = value?.version?.trim() || 'unknown';
  const checksum = value?.checksum?.trim();
  return {
    name,
    version,
    ...(checksum ? { checksum } : {}),
    configSha256: value?.configSha256
      ?? sha256Text(canonicalJson({ name, version, checksum: checksum ?? null })),
  };
}

function modelMetadata(
  templateId: string,
  defaults: Partial<RadiologyModelMetadata> | undefined,
  chunks: RadiologyChunkTranscription[],
): RadiologyModelMetadata {
  const chunkModel = chunks.find((chunk) => chunk.model)?.model;
  const selected = {
    ...defaults,
    ...chunkModel,
  };
  const builtInTemplate = builtInTemplateComponent(templateId);
  // The template is executable server-side schema, not ASR runtime metadata.
  // Never let a remote transcription response override its version or hash.
  const template = builtInTemplate.checksum
    ? component(builtInTemplate, templateId)
    : component(selected.template, templateId);
  const defaultRouter = component(
    selected.router ?? selected.structurer ?? { version: '2' },
    'radiology-span-router',
  );
  const templateDefinition = getDocTemplate(templateId);
  const promptText = templateDefinition
    ? buildArrangerPrompt(templateDefinition, []).system
    : ARRANGER_PROMPT_VERSION;
  const prompt = selected.prompt
    ? component(selected.prompt, 'radiology-span-router-prompt')
    : component({
        version: ARRANGER_PROMPT_VERSION,
        configSha256: sha256Text(promptText),
      }, 'radiology-span-router-prompt');
  return {
    asr: component(selected.asr, 'unknown'),
    vad: selected.vad === null
      ? null
      : selected.vad
        ? component(selected.vad, 'voice-activity-detector')
        : null,
    decoder: component(selected.decoder, 'unknown'),
    languageModel: selected.languageModel === null
      ? null
      : selected.languageModel
        ? component(selected.languageModel, 'unknown')
        : null,
    contextVocabulary: selected.contextVocabulary === null
      ? null
      : selected.contextVocabulary
        ? component(selected.contextVocabulary, 'radiology-context-vocabulary')
        : null,
    dictionary: component(selected.dictionary, 'medical-dictionary'),
    normalizer: selected.normalizer
      ? component(selected.normalizer, 'gigaam-radiology-normalizer')
      : component({ version: '2.1' }, 'gigaam-radiology-normalizer'),
    template,
    router: defaultRouter,
    prompt,
    structurer: selected.structurer
      ? component(selected.structurer, 'radiology-structurer-guardrails')
      : defaultRouter,
    ...(templateId === 'CT_ABDOMEN_MIKHAILOV'
      ? {
          composer: selected.composer
            ? component(selected.composer, 'radiology-template-composer')
            : component({
                version: TEMPLATE_COMPOSER_VERSION,
                configSha256: sha256Text(canonicalJson({
                  version: TEMPLATE_COMPOSER_VERSION,
                  templateChecksum: template.checksum ?? null,
                })),
              }, 'radiology-template-composer'),
        }
      : {}),
    llm: selected.llm === null
      ? null
      : selected.llm
        ? component(selected.llm, 'radiology-span-classifier')
        : null,
    safety: selected.safety
      ? component(selected.safety, 'radiology-safety-verifier')
      : component({ version: '2' }, 'radiology-safety-verifier'),
  };
}

function normalizeChunkResult(
  result: string | RadiologyChunkTranscription,
): RadiologyChunkTranscription {
  if (typeof result === 'string') {
    // Legacy WhisperService returns text after dictionary/denormalization.
    // Keep the snapshot but explicitly prevent it from being treated as raw
    // ASR training truth.
    return {
      rawText: result,
      normalizedText: result,
      rawAvailable: false,
      language: 'ru',
    };
  }
  const normalizedText = result.normalizedText?.trim() || result.rawText?.trim() || '';
  return {
    ...result,
    rawText: result.rawText ?? normalizedText,
    normalizedText,
    rawAvailable:
      result.rawAvailable === true
      && typeof result.rawText === 'string'
      && result.rawText.trim().length > 0,
    language: result.language || 'ru',
    words: result.words ?? [],
  };
}

function artifactNormalization(
  result: GigaAMNormalizationResult,
  sourceText: string,
): RadiologyTranscriptionArtifact['normalization'] {
  const transformations: RadiologyNormalizationTransformation[] = [];
  let sourceCursor = 0;
  for (const span of result.alignment) {
    if (span.sourceStart > sourceCursor) {
      transformations.push({
        kind: sourceText.slice(sourceCursor, span.sourceStart).trim().length === 0
          ? 'whitespace'
          : 'aligned_normalization',
        source: {
          start: sourceCursor,
          end: span.sourceStart,
          text: sourceText.slice(sourceCursor, span.sourceStart),
        },
        normalized: {
          start: span.normalizedStart,
          end: span.normalizedStart,
          text: '',
        },
      });
    }
    const source = sourceText.slice(span.sourceStart, span.sourceEnd);
    const normalized = result.text.slice(span.normalizedStart, span.normalizedEnd);
    if (source !== normalized) {
      transformations.push({
        kind: span.kind ?? 'aligned_normalization',
        source: {
          start: span.sourceStart,
          end: span.sourceEnd,
          text: source,
        },
        normalized: {
          start: span.normalizedStart,
          end: span.normalizedEnd,
          text: normalized,
        },
      });
    }
    sourceCursor = Math.max(sourceCursor, span.sourceEnd);
  }
  if (sourceCursor < sourceText.length) {
    transformations.push({
      kind: sourceText.slice(sourceCursor).trim().length === 0
        ? 'whitespace'
        : 'aligned_normalization',
      source: {
        start: sourceCursor,
        end: sourceText.length,
        text: sourceText.slice(sourceCursor),
      },
      normalized: {
        start: result.text.length,
        end: result.text.length,
        text: '',
      },
    });
  }
  return {
    text: result.text,
    sha256: sha256Text(result.text),
    version: result.version,
    transformations,
    issues: result.issues.map((issue) => {
      const aligned = result.alignment.filter((span) => (
        span.sourceStart < issue.end && span.sourceEnd > issue.start
      ));
      const normalizedStart = aligned[0]?.normalizedStart ?? issue.start;
      const normalizedEnd = aligned[aligned.length - 1]?.normalizedEnd
        ?? normalizedStart + issue.normalizedText.length;
      return {
      id: sha256Text(canonicalJson({
        code: issue.code,
        start: issue.start,
        end: issue.end,
        sourceText: issue.sourceText,
        values: issue.values,
      })).slice(0, 24),
      code: issue.code,
      severity: issue.severity,
      message: issue.message,
      source: {
        start: issue.start,
        end: issue.end,
        text: issue.sourceText,
      },
      normalized: {
        start: normalizedStart,
        end: normalizedEnd,
        text: result.text.slice(normalizedStart, normalizedEnd),
      },
      values: issue.values,
      };
    }),
  };
}

function contextBiasForChunk(
  chunk: RadiologyChunkTranscription,
): RadiologyASRContextBiasMetadata {
  return chunk.contextBias
    ?? chunk.provenance?.contextBias
    ?? { scope: null, active: false, terms: 0 };
}

function runtimeValue(
  direct: string | undefined,
  nested: string | undefined,
  field: 'schemaVersion' | 'runtimeId',
): string {
  const directValue = direct?.trim();
  const nestedValue = nested?.trim();
  if (directValue && nestedValue && directValue !== nestedValue) {
    throw new RadiologySessionError(
      422,
      'inconsistent_asr_provenance',
      `ASR ${field} differs between top-level and nested provenance`,
    );
  }
  return directValue || nestedValue || 'unknown';
}

function artifactTranscriptionChunk(
  chunk: RadiologyChunkTranscription,
  index: number,
  audioSha256?: string,
): RadiologyArtifactTranscriptionChunk {
  const rawText = chunk.rawText ?? chunk.normalizedText;
  const normalizedText = chunk.normalizedText;
  const rawTextSha256 = sha256Text(rawText);
  const normalizedTextSha256 = sha256Text(normalizedText);
  const nestedHashes = chunk.provenance?.hashes ?? {};
  const directHashes = chunk.hashes ?? {};
  for (const key of Object.keys(directHashes) as Array<keyof RadiologyASRHashMetadata>) {
    if (
      directHashes[key]
      && nestedHashes[key]
      && directHashes[key] !== nestedHashes[key]
    ) {
      throw new RadiologySessionError(
        422,
        'inconsistent_asr_provenance',
        `ASR hash ${key} differs between top-level and nested provenance`,
      );
    }
  }
  const suppliedHashes = {
    ...nestedHashes,
    ...directHashes,
  };
  for (const [name, hash] of Object.entries(suppliedHashes)) {
    if (hash && !/^[a-f0-9]{64}$/u.test(hash)) {
      throw new RadiologySessionError(
        422,
        'invalid_asr_provenance_hash',
        `ASR hash ${name} is not a lowercase SHA-256 digest`,
      );
    }
  }
  const hashChecks: Array<[string, string | undefined, string]> = [
    ['audio', suppliedHashes.audioSha256, audioSha256 ?? ''],
    ['raw transcript', suppliedHashes.rawTextSha256, rawTextSha256],
    ['normalized transcript', suppliedHashes.normalizedTextSha256, normalizedTextSha256],
  ];
  for (const [label, supplied, actual] of hashChecks) {
    if (supplied && supplied !== actual) {
      throw new RadiologySessionError(
        422,
        'asr_provenance_hash_mismatch',
        `ASR ${label} SHA-256 does not match the canonical session payload`,
      );
    }
  }

  const directContext = chunk.contextBias;
  const nestedContext = chunk.provenance?.contextBias;
  if (
    directContext
    && nestedContext
    && canonicalJson(directContext) !== canonicalJson(nestedContext)
  ) {
    throw new RadiologySessionError(
      422,
      'inconsistent_asr_provenance',
      'ASR context bias differs between top-level and nested provenance',
    );
  }
  const contextBias = contextBiasForChunk(chunk);
  const schemaVersion = runtimeValue(
    chunk.schemaVersion,
    chunk.provenance?.schemaVersion,
    'schemaVersion',
  );
  const runtimeId = runtimeValue(chunk.runtimeId, chunk.provenance?.runtimeId, 'runtimeId');
  const directCheckpointVerified = chunk.checkpointVerified;
  const nestedCheckpointVerified = chunk.provenance?.checkpointVerified;
  if (
    directCheckpointVerified !== undefined
    && nestedCheckpointVerified !== undefined
    && directCheckpointVerified !== nestedCheckpointVerified
  ) {
    throw new RadiologySessionError(
      422,
      'inconsistent_asr_provenance',
      'ASR checkpoint verification differs between top-level and nested provenance',
    );
  }
  const directContractVerification = chunk.verification;
  const nestedContractVerification = chunk.provenance?.verification;
  if (
    directContractVerification
    && nestedContractVerification
    && canonicalJson(directContractVerification)
      !== canonicalJson(nestedContractVerification)
  ) {
    throw new RadiologySessionError(
      422,
      'inconsistent_asr_provenance',
      'ASR production-contract verification differs between top-level and nested provenance',
    );
  }
  const contractVerification =
    directContractVerification ?? nestedContractVerification;
  const source = chunk.source ?? 'unknown';
  const expectedSchema = source === 'gigaam'
    ? 'gigaam.transcription.v2'
    : source === 'whisper'
      ? 'whisper.transcription.v1'
      : null;
  const requiredHashesWereSupplied = Boolean(
    audioSha256
    && suppliedHashes.audioSha256 === audioSha256
    && suppliedHashes.rawTextSha256 === rawTextSha256
    && suppliedHashes.normalizedTextSha256 === normalizedTextSha256,
  );
  const wordEvidenceIsAcoustic = Boolean(
    (chunk.words?.length ?? 0) > 0
    && chunk.words?.every((word) => {
      const scoreType = word.scoreType?.trim().toLowerCase() ?? '';
      return (
        typeof word.confidence === 'number'
        && Number.isFinite(word.confidence)
        && word.confidence >= 0
        && word.confidence <= 1
        && typeof word.avgLogprob === 'number'
        && Number.isFinite(word.avgLogprob)
        && !scoreType.includes('fused')
        && !scoreType.includes('language_model')
        && (
          scoreType.includes('acoustic')
          || scoreType.includes('emission')
          || scoreType.includes('ctc_')
        )
      );
    })
  );
  const verification = {
    schema: Boolean(
      expectedSchema !== null
      && schemaVersion === expectedSchema
      && (
        source !== 'gigaam'
        || contractVerification?.transcriptionSchema === true
      )
    ),
    runtime: Boolean(
      /^[a-f0-9]{64}$/u.test(runtimeId)
      && (
        source !== 'gigaam'
        || contractVerification?.runtimeIdentity === true
      )
    ),
    checkpoint: Boolean(
      (directCheckpointVerified ?? nestedCheckpointVerified) === true
      && (
        source !== 'gigaam'
        || contractVerification?.checkpoint === true
      )
    ),
    hashes: Boolean(
      requiredHashesWereSupplied
      && (
        source !== 'gigaam'
        || contractVerification?.hashes === true
      )
    ),
    metadata: Boolean(
      source !== 'gigaam'
      || (
        contractVerification?.metadataAvailable === true
        && contractVerification.metadataSchema === true
      )
    ),
    decoder: Boolean(
      source !== 'gigaam'
      || contractVerification?.decoder === true
    ),
    wordEvidence: Boolean(
      source !== 'gigaam'
      || (
        contractVerification?.wordEvidence === true
        && wordEvidenceIsAcoustic
      )
    ),
    productionContract: Boolean(
      source !== 'gigaam'
      || contractVerification?.productionReady === true
    ),
  };
  return {
    index,
    rawText,
    rawTextSha256,
    normalizedText,
    normalizedTextSha256,
    rawAvailable: chunk.rawAvailable === true,
    language: chunk.language || 'ru',
    source,
    words: (chunk.words ?? []).map((word) => ({
      ...word,
      chunkIndex: word.chunkIndex ?? index,
    })),
    ...(chunk.longform ? { longform: chunk.longform } : {}),
    provenance: {
      schemaVersion,
      runtimeId,
      acousticDecoder: chunk.provenance?.acousticDecoder ?? null,
      ctcDecoder: chunk.provenance?.ctcDecoder ?? null,
      contextBias,
      hashes: {
        ...suppliedHashes,
        ...(audioSha256 ? { audioSha256 } : {}),
        rawTextSha256,
        normalizedTextSha256,
      },
      verification,
    },
  };
}

function assertUniformASRProvenance(
  templateId: string,
  defaults: Partial<RadiologyModelMetadata> | undefined,
  chunks: RadiologyChunkTranscription[],
  artifactChunks: RadiologyArtifactTranscriptionChunk[],
): void {
  if (chunks.length < 2) return;
  const identities = chunks.map((chunk, index) => canonicalJson({
    source: artifactChunks[index].source,
    language: artifactChunks[index].language,
    schemaVersion: artifactChunks[index].provenance.schemaVersion,
    runtimeId: artifactChunks[index].provenance.runtimeId,
    acousticDecoder: artifactChunks[index].provenance.acousticDecoder,
    ctcDecoder: artifactChunks[index].provenance.ctcDecoder,
    contextBias: artifactChunks[index].provenance.contextBias,
    model: modelMetadata(templateId, defaults, [chunk]),
  }));
  if (new Set(identities).size > 1) {
    throw new RadiologySessionError(
      409,
      'mixed_asr_provenance',
      'Chunks were transcribed by different ASR/model/runtime configurations',
    );
  }
}

function uniqueReasons(reasons: string[]): string[] {
  return [...new Set(reasons)];
}

function trainingExclusionReasons(input: {
  source: RadiologyTranscriptionSource;
  rawAvailable: boolean;
  audioStored: boolean;
  asrChunks: RadiologyArtifactTranscriptionChunk[];
  model: RadiologyModelMetadata;
}): string[] {
  const reasons: string[] = [];
  if (input.source === 'browser') reasons.push('browser_asr_source');
  if (input.source === 'whisper') reasons.push('whisper_challenger_source');
  if (input.source === 'manual') reasons.push('manual_transcript_source');
  if (input.source === 'unknown') reasons.push('asr_source_unknown');
  if (!input.rawAvailable) reasons.push('raw_asr_unavailable');
  if (!input.audioStored) reasons.push('audio_not_retained');
  if (input.asrChunks.length === 0) reasons.push('asr_chunks_missing');
  if (
    input.asrChunks.length === 0
    || input.asrChunks.some((chunk) => chunk.provenance?.verification?.schema !== true)
  ) {
    reasons.push('asr_schema_unverified');
  }
  if (
    input.asrChunks.length === 0
    || input.asrChunks.some((chunk) => chunk.provenance?.verification?.runtime !== true)
  ) {
    reasons.push('asr_runtime_unverified');
  }
  if (
    input.asrChunks.length === 0
    || input.asrChunks.some((chunk) => chunk.provenance?.verification?.checkpoint !== true)
  ) {
    reasons.push('asr_checkpoint_unverified');
  }
  if (
    input.asrChunks.length === 0
    || input.asrChunks.some((chunk) => chunk.provenance?.verification?.hashes !== true)
  ) {
    reasons.push('asr_hashes_unverified');
  }
  if (
    input.asrChunks.length === 0
    || input.asrChunks.some((chunk) => chunk.provenance?.verification?.metadata !== true)
  ) {
    reasons.push('asr_metadata_unverified');
  }
  if (
    input.asrChunks.length === 0
    || input.asrChunks.some((chunk) => chunk.provenance?.verification?.decoder !== true)
  ) {
    reasons.push('asr_decoder_unverified');
  }
  if (
    input.asrChunks.length === 0
    || input.asrChunks.some((chunk) => (
      chunk.provenance?.verification?.wordEvidence !== true
    ))
  ) {
    reasons.push('asr_word_evidence_unverified');
  }
  if (
    input.asrChunks.length === 0
    || input.asrChunks.some((chunk) => (
      chunk.provenance?.verification?.productionContract !== true
    ))
  ) {
    reasons.push('asr_contract_unverified');
  }
  if (!/^[a-f0-9]{64}$/u.test(input.model.asr.checksum ?? '')) {
    reasons.push('asr_checkpoint_checksum_unverified');
  }
  if (
    !input.model.asr.version.trim()
    || input.model.asr.version.trim().toLowerCase() === 'unknown'
  ) {
    reasons.push('asr_model_version_unknown');
  }
  return uniqueReasons(reasons);
}

function decodeBase64Audio(value: string): Buffer {
  const compact = value.replace(/\s+/gu, '');
  if (!compact || compact.length % 4 === 1 || !/^[A-Za-z0-9+/]*={0,2}$/u.test(compact)) {
    throw new RadiologySessionError(400, 'invalid_audio_base64', 'audio_base64 is invalid');
  }
  const audio = Buffer.from(compact, 'base64');
  if (audio.length === 0) {
    throw new RadiologySessionError(400, 'empty_audio_chunk', 'Audio chunk is empty');
  }
  return audio;
}

function reportSections(report: DictationReport | null): RadiologyArtifactSection[] {
  if (!report) return [];
  const sections: RadiologyArtifactSection[] = report.blocks
    .filter((block) => block.origin !== 'template_default')
    .map((block) => ({
    id: block.id,
    label: block.label,
    text: block.text,
    source: block.source,
    evidence: block.evidence.map((span) => ({
      transcript: 'normalized',
      start: span.start,
      end: span.end,
      text: span.text,
    })),
    origin: block.origin === 'template_default'
      ? 'missing-template-default'
      : block.origin === 'generated_extract'
        ? 'extractive-conclusion'
        : block.source === 'conclusion'
          ? 'dictated-conclusion'
          : block.normalReason === 'explicit'
            ? 'explicit-normal-template'
            : 'verbatim',
    ...(block.assignmentMethod ? { assignmentMethod: block.assignmentMethod } : {}),
    }));
  if (report.unmatched.trim()) {
    sections.push({
      id: 'unmatched',
      label: 'Unmatched',
      text: report.unmatched,
      source: 'unmatched',
      origin: 'unmatched',
      assignmentMethod: 'unmatched',
      evidence: report.unmatchedSpans.map((span) => ({
        transcript: 'normalized',
        start: span.start,
        end: span.end,
        text: span.text,
      })),
    });
  }
  return sections;
}

type ExtendedDictationReport = DictationReport & {
  routing?: {
    atoms?: RadiologyTranscriptAtom[];
    assignments?: RadiologySpanAssignment[];
    unmatchedAtomIds?: string[];
  };
  conclusion?: {
    text: string;
    mode: 'dictated' | 'extractive';
    evidence: Array<{ start: number; end: number; text: string }>;
  } | null;
  evidenceBackedText?: string;
};

function evidenceBackedTextForReport(report: DictationReport | null): string {
  if (!report) return '';
  return [
    ...report.blocks
      .filter((block) => block.origin === 'transcript' && block.evidence.length > 0)
      .map((block) => block.text),
    report.unmatched,
  ].filter(Boolean).join(' ');
}

function routingForReport(
  report: DictationReport | null,
): RadiologyTranscriptionArtifact['routing'] {
  const routing = (report as ExtendedDictationReport | null)?.routing;
  return {
    atoms: routing?.atoms ?? [],
    assignments: routing?.assignments ?? [],
    unmatchedAtomIds: routing?.unmatchedAtomIds ?? [],
  };
}

function artifactReport(report: DictationReport | null): RadiologyArtifactReport | null {
  if (!report) return null;
  const extended = report as ExtendedDictationReport;
  const sections = reportSections(report);
  const evidenceBackedText = evidenceBackedTextForReport(report);
  const conclusionBlock = report.blocks.find(
    (block) => block.source === 'conclusion' && block.text.trim().length > 0,
  );
  const reportConclusion = extended.conclusion ?? (
    conclusionBlock
      ? {
          text: conclusionBlock.text,
          mode: conclusionBlock.origin === 'transcript'
            ? 'dictated' as const
            : 'extractive' as const,
          evidence: conclusionBlock.evidence,
        }
      : null
  );
  const conclusion = reportConclusion
    ? {
        text: reportConclusion.text,
        mode: reportConclusion.mode,
        evidence: reportConclusion.evidence.map((span) => ({
          transcript: 'normalized' as const,
          start: span.start,
          end: span.end,
          text: span.text,
        })),
      }
    : null;
  return {
    ...report,
    sections,
    conclusion,
    evidenceBackedText,
    evidenceSha256: sha256Text(evidenceBackedText),
    templateDefaults: report.templateDefaults.map((item) => ({
      id: item.sectionId,
      label: item.label,
      text: item.text,
    })),
  };
}

function safetyStageFromReport(
  stage: RadiologySafetyStageResult['stage'],
  sourceText: string,
  outputText: string,
  safety: RadiologySafetyReport,
): RadiologySafetyStageResult {
  return {
    stage,
    status: safety.ok ? 'passed' : 'failed',
    sourceSha256: sha256Text(sourceText),
    outputSha256: sha256Text(outputText),
    numbers: {
      status: safety.numbers.ok ? 'passed' : 'failed',
      details: safety.numbers,
    },
    units: {
      status: safety.numberUnits.ok ? 'passed' : 'failed',
      details: safety.numberUnits,
    },
    negations: {
      status: safety.negations.ok ? 'passed' : 'failed',
      details: safety.negations,
    },
    laterality: {
      status: safety.lateralities.ok ? 'passed' : 'failed',
      details: safety.lateralities,
    },
    contrast: {
      status: safety.contrast.ok ? 'passed' : 'failed',
      details: safety.contrast,
    },
    criticalFacts: {
      status: safety.criticalFacts.ok ? 'passed' : 'failed',
      details: safety.criticalFacts,
    },
    issues: safety.issues,
  };
}

function maskNormalizationLedger(
  rawText: string,
  normalization: RadiologyTranscriptionArtifact['normalization'],
  kinds: ReadonlySet<string>,
): { rawText: string; normalizedText: string } {
  let maskedRaw = rawText;
  let maskedNormalized = normalization.text;
  const maskOnce = (text: string, fragment: string): string => {
    if (!fragment) return text;
    const index = text.indexOf(fragment);
    if (index < 0) return text;
    return `${text.slice(0, index)}${' '.repeat(fragment.length)}${text.slice(index + fragment.length)}`;
  };

  // Only exact source/output pairs from the immutable ledger are masked. The
  // caller chooses which stages may be ignored by a particular independent
  // verifier; unrecorded edits remain visible and fail closed.
  for (const transformation of normalization.transformations) {
    if (!kinds.has(transformation.kind)) continue;
    maskedRaw = maskOnce(maskedRaw, transformation.source.text);
    maskedNormalized = maskOnce(maskedNormalized, transformation.normalized.text);
  }
  return { rawText: maskedRaw, normalizedText: maskedNormalized };
}

function rawToNormalizedStage(
  rawText: string,
  normalization: RadiologyTranscriptionArtifact['normalization'],
  rawAvailable: boolean,
): {
  stage: RadiologySafetyStageResult;
  destructive: boolean;
  ambiguous: boolean;
} {
  if (!rawAvailable) {
    return {
      stage: notRunSafetyStage(
        'raw_to_normalized',
        sha256Text(rawText),
        normalization.sha256,
      ),
      destructive: false,
      ambiguous: false,
    };
  }

  // Dates/decimals intentionally change the cardinal surface and are verified
  // by their strict grammar ledger. Cardinal transforms remain visible to the
  // independent number multiset check, which is what catches 50+53→103.
  const numberMasked = maskNormalizationLedger(
    rawText,
    normalization,
    new Set(['date', 'decimal']),
  );
  const detailedSafety = verifyRawToNormalizedSafety(numberMasked.rawText, {
    text: numberMasked.normalizedText,
    issues: normalization.issues.map((issue) => ({
      code: issue.code,
      severity: issue.severity,
      message: issue.message,
      start: issue.source?.start,
      end: issue.source?.end,
      sourceText: issue.source?.text,
      normalizedText: issue.normalized?.text,
    })),
  });
  // The generic entity checker cannot pair a spoken number with the digit form
  // produced by normalization. Mask every explicitly ledgered deterministic
  // transform for this second pass, while still comparing all untouched
  // negations, lateralities, contrast markers and clinical facts.
  const entityMasked = maskNormalizationLedger(
    rawText,
    normalization,
    new Set(['date', 'decimal', 'cardinal', 'unit', 'numeric_format', 'whitespace']),
  );
  const entitySafety = verifyRadiologySafety(entityMasked.rawText, entityMasked.normalizedText);
  const stage = safetyStageFromReport(
    'raw_to_normalized',
    rawText,
    normalization.text,
    entitySafety,
  );
  const issues = [...detailedSafety.issues, ...entitySafety.issues].filter(
    (issue, index, all) =>
      all.findIndex((candidate) =>
        candidate.code === issue.code
        && candidate.message === issue.message
      ) === index,
  );
  const ambiguous = detailedSafety.status === 'incomplete';
  const destructive = detailedSafety.status === 'failed' || !entitySafety.ok;
  return {
    stage: {
      ...stage,
      status: destructive ? 'failed' : ambiguous ? 'incomplete' : 'passed',
      issues,
    },
    destructive,
    ambiguous,
  };
}

function safetyResult(
  report: DictationReport | null,
  stages: RadiologySafetyStageResult[] = [],
): RadiologySafetyResult {
  const notRun = <T>(): RadiologySafetyCheck<T> => ({ status: 'not_run' });
  if (!report) {
    const status = stages.some((stage) => stage.status === 'failed')
      ? 'failed'
      : 'incomplete';
    return {
      status,
      stages,
      numbers: { status: 'not_run' },
      units: notRun<SafetyEntityCheck>(),
      negations: notRun<SafetyEntityCheck>(),
      laterality: notRun<SafetyEntityCheck>(),
      contrast: notRun<SafetyEntityCheck>(),
      criticalFacts: notRun<SafetyEntityCheck>(),
      requiresReview: true,
      approvalBlocked: true,
      issues: stages.flatMap((stage) => stage.issues),
    };
  }
  const safety: RadiologySafetyReport = report.safety;
  const stageIssues = stages.flatMap((stage) => stage.issues);
  const issues = [...stageIssues, ...safety.issues].filter(
    (issue, index, all) =>
      all.findIndex((candidate) =>
        candidate.code === issue.code
        && candidate.message === issue.message
      ) === index,
  );
  const status: RadiologySafetyResult['status'] = stages.some((stage) => stage.status === 'failed')
    || !safety.ok
    ? 'failed'
    : stages.some((stage) => stage.status === 'incomplete') || stages.length < 2
      ? 'incomplete'
      : 'passed';
  return {
    status,
    stages,
    numbers: {
      status: safety.numbers.ok ? 'passed' : 'failed',
      details: safety.numbers,
    },
    units: { status: safety.numberUnits.ok ? 'passed' : 'failed', details: safety.numberUnits },
    negations: { status: safety.negations.ok ? 'passed' : 'failed', details: safety.negations },
    laterality: { status: safety.lateralities.ok ? 'passed' : 'failed', details: safety.lateralities },
    contrast: { status: safety.contrast.ok ? 'passed' : 'failed', details: safety.contrast },
    criticalFacts: {
      status: safety.criticalFacts.ok ? 'passed' : 'failed',
      details: safety.criticalFacts,
    },
    requiresReview: status !== 'passed' || safety.requiresReview,
    approvalBlocked: status !== 'passed',
    issues,
  };
}

function applySpanCorrections(
  rawTranscript: string,
  corrections: StoredSpanCorrection[],
): string {
  const ordered = [...corrections].sort((left, right) => left.start - right.start || left.end - right.end);
  let cursor = 0;
  let result = '';
  for (const correction of ordered) {
    if (correction.start < cursor) {
      throw new RadiologySessionError(
        409,
        'correction_spans_overlap',
        'spanCorrections must not overlap',
      );
    }
    result += rawTranscript.slice(cursor, correction.start);
    result += correction.correctedText;
    cursor = correction.end;
  }
  return result + rawTranscript.slice(cursor);
}

function validateArtifactCorrections(
  artifact: RadiologyTranscriptionArtifact,
  verbatimTranscript: string,
  submitted: StoredSpanCorrection[],
): StoredSpanCorrection[] {
  const templateModality = getDocTemplate(artifact.templateId)?.modality;
  const corrections = submitted.map((correction, index) => {
    if (
      !Number.isSafeInteger(correction.start)
      || !Number.isSafeInteger(correction.end)
      || correction.start < 0
      || correction.end < correction.start
      || correction.end > artifact.rawTranscript.text.length
    ) {
      throw new RadiologySessionError(
        400,
        'correction_span_out_of_bounds',
        `spanCorrections[${index}] is outside the raw transcript`,
      );
    }
    const originalAtSpan = artifact.rawTranscript.text.slice(correction.start, correction.end);
    if (originalAtSpan !== correction.originalText) {
      throw new RadiologySessionError(
        409,
        'correction_span_mismatch',
        `spanCorrections[${index}].originalText does not match the immutable raw transcript`,
      );
    }
    if (
      templateModality
      && correction.modality.trim().toUpperCase() !== templateModality.toUpperCase()
    ) {
      throw new RadiologySessionError(
        409,
        'correction_modality_mismatch',
        `spanCorrections[${index}].modality does not match template modality ${templateModality}`,
      );
    }
    return correction;
  });
  const reconstructedVerbatim = applySpanCorrections(artifact.rawTranscript.text, corrections);
  if (reconstructedVerbatim !== verbatimTranscript) {
    throw new RadiologySessionError(
      409,
      'verbatim_corrections_mismatch',
      'verbatimTranscript must equal the immutable raw transcript with spanCorrections applied',
    );
  }
  return corrections;
}

function validateNormalizationResolutions(
  artifact: RadiologyTranscriptionArtifact,
  input: RadiologyFeedbackInput,
  sourceArtifact?: RadiologyTranscriptionArtifact,
): NormalizationResolutionInput[] {
  const resolutions = input.normalizationResolutions ?? [];
  const issueById = new Map(
    [
      ...(sourceArtifact?.normalization.issues ?? []),
      ...artifact.normalization.issues,
    ].map((issue) => [issue.id, issue]),
  );
  const seen = new Set<string>();
  for (const resolution of resolutions) {
    if (seen.has(resolution.issueId)) {
      throw new RadiologySessionError(
        400,
        'duplicate_normalization_resolution',
        `Normalization issue ${resolution.issueId} was resolved more than once`,
      );
    }
    seen.add(resolution.issueId);
    if (!issueById.has(resolution.issueId)) {
      throw new RadiologySessionError(
        409,
        'normalization_issue_mismatch',
        `Normalization issue ${resolution.issueId} does not belong to this artifact`,
      );
    }
    if (!input.verbatimTranscript.includes(resolution.replacementText)) {
      throw new RadiologySessionError(
        409,
        'normalization_resolution_mismatch',
        `Resolution text for ${resolution.issueId} is absent from verbatimTranscript`,
      );
    }
    if (
      resolution.resolution === 'confirmed_range'
      && !/(?:\bот\b[\s\S]*\bдо\b|[-–—])/iu.test(resolution.replacementText)
    ) {
      throw new RadiologySessionError(
        422,
        'normalization_resolution_invalid',
        `Resolution ${resolution.issueId} does not contain an explicit range marker`,
      );
    }
  }
  if (input.approved) {
    const unresolved = artifact.normalization.issues.filter(
      (issue) => issue.severity === 'critical' && !seen.has(issue.id),
    );
    if (unresolved.length) {
      throw new RadiologySessionError(
        422,
        'normalization_resolution_required',
        `Critical normalization issue ${unresolved[0].id} requires an explicit physician resolution`,
      );
    }
  }
  return resolutions;
}

function removeOnce(text: string, fragment: string): string {
  const index = text.indexOf(fragment);
  if (index < 0) return text;
  return `${text.slice(0, index)}${' '.repeat(fragment.length)}${text.slice(index + fragment.length)}`;
}

function reviewSectionBodies(
  finalReport: string,
  draft: NonNullable<RadiologyArtifactReport['reviewDraft']>,
): Map<string, string> {
  const located = draft.sections
    .map((section) => {
      const marker = `${section.label}:`;
      const markerStart = finalReport.indexOf(marker);
      return {
        section,
        markerStart,
        bodyStart: markerStart < 0 ? -1 : markerStart + marker.length,
      };
    })
    .filter((item) => item.markerStart >= 0)
    .sort((left, right) => left.markerStart - right.markerStart);
  const result = new Map<string, string>();
  for (let index = 0; index < located.length; index++) {
    const current = located[index];
    const end = located[index + 1]?.markerStart ?? finalReport.length;
    result.set(
      current.section.id,
      finalReport.slice(current.bodyStart, end),
    );
  }
  return result;
}

function validateFinalFieldBindings(
  report: RadiologyArtifactReport,
  sectionBodies: Map<string, string>,
  acceptedTemplateSegmentIds: ReadonlySet<string>,
): void {
  const draft = report.reviewDraft;
  if (!draft) return;
  const template = getDocTemplate(draft.templateId);
  if (!template) {
    throw new RadiologySessionError(
      409,
      'review_template_unavailable',
      `Template ${draft.templateId} is unavailable for final field validation`,
    );
  }
  if (templateSha256(template) !== draft.templateSha256) {
    throw new RadiologySessionError(
      409,
      'review_template_version_changed',
      'The template schema changed after this draft was created; reprocessing is required',
    );
  }

  const sectionsWithAcceptedTemplateContext = new Set(
    draft.segments
      .filter((segment) => (
        segment.confirmationRequired
        && acceptedTemplateSegmentIds.has(segment.id)
      ))
      .map((segment) => segment.sectionId),
  );
  const expectedAssignments = draft.fieldAssignments.filter((assignment) => (
    assignment.status === 'applied'
    && assignment.kind !== 'explicit_normal'
    && !sectionsWithAcceptedTemplateContext.has(assignment.sectionId)
  ));
  const sectionsRequiringParsing = new Set(
    expectedAssignments.map((assignment) => assignment.sectionId),
  );

  let transcript = '';
  const atoms: TemplateSectionAtom[] = [];
  for (const section of draft.sections) {
    const body = (sectionBodies.get(section.id) ?? '').trim();
    if (!body) continue;
    if (body.length > 20_000 || extractNumbers(body).length > 64) {
      throw new RadiologySessionError(
        422,
        'final_section_too_large',
        `Section ${section.id} exceeds deterministic review limits`,
      );
    }
    if (!sectionsRequiringParsing.has(section.id)) continue;
    if (transcript) transcript += '\n';
    const start = transcript.length;
    transcript += body;
    atoms.push({
      atomId: `final:${section.id}`,
      sectionId: section.id,
      start,
      end: start + body.length,
      text: body,
    });
  }
  const recomposed = composeTemplateReviewDraft(template, transcript, atoms);
  const finalAssignments = new Map(
    recomposed.fieldAssignments
      .filter((assignment) => assignment.status === 'applied')
      .map((assignment) => [assignment.fieldId, assignment]),
  );

  for (const expected of expectedAssignments) {
    const actual = finalAssignments.get(expected.fieldId);
    if (
      !actual
      || canonicalJson(actual.value) !== canonicalJson(expected.value)
      || actual.canonicalUnit !== expected.canonicalUnit
    ) {
      throw new RadiologySessionError(
        422,
        'final_field_binding_changed',
        `Final report changed the value, unit, or section binding of ${expected.fieldId}`,
      );
    }
  }

  for (const placeholder of draft.segments.filter(
    (segment) => segment.defaultKind === 'placeholder' && segment.fieldId,
  )) {
    const body = (sectionBodies.get(placeholder.sectionId) ?? '').trim();
    const addressedSection = draft.fieldAssignments.some(
      (assignment) => assignment.sectionId === placeholder.sectionId,
    ) || draft.residualAtomIds.some((atomId) => (
      draft.segments.some((segment) => (
        segment.sectionId === placeholder.sectionId
        && segment.evidence.some((evidence) => evidence.atomId === atomId)
      ))
    ));
    if (body && addressedSection && !finalAssignments.has(placeholder.fieldId!)) {
      throw new RadiologySessionError(
        422,
        'incomplete_final_template_field',
        `Section ${placeholder.sectionId} still contains unresolved field ${placeholder.fieldId}`,
      );
    }
  }
}

function segmentIndex(text: string, fragment: string): number {
  const trimmed = fragment.trim();
  if (/^[+-]?\d+(?:[.,]\d+)?$/u.test(trimmed)) {
    const escaped = trimmed.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
    const match = new RegExp(
      `(?<![\\d.,])${escaped}(?![\\d.,])`,
      'u',
    ).exec(text);
    return match?.index ?? -1;
  }
  return text.indexOf(fragment);
}

function removeSegmentOnce(text: string, fragment: string): string {
  const index = segmentIndex(text, fragment);
  if (index < 0) return text;
  return `${text.slice(0, index)}${' '.repeat(fragment.length)}${text.slice(index + fragment.length)}`;
}

function maskSegmentAt(text: string, index: number, fragment: string): string {
  if (index < 0) return text;
  return `${text.slice(0, index)}${' '.repeat(fragment.length)}${text.slice(index + fragment.length)}`;
}

interface ValidatedReviewDraftFeedback {
  baseDraftSha256: string | null;
  acceptedTemplateSegmentIds: string[];
  reviewedResidualAtomIds: string[];
  acceptedSegments: NonNullable<RadiologyArtifactReport['reviewDraft']>['segments'];
  requireCompleteDraftEvidence: boolean;
}

function validateReviewDraftFeedback(
  artifact: RadiologyTranscriptionArtifact,
  input: RadiologyFeedbackInput,
): ValidatedReviewDraftFeedback {
  const report = artifact.report;
  const draft = report?.reviewDraft;
  const acceptedIds = input.acceptedTemplateSegmentIds ?? [];
  const reviewedResidualIds = input.reviewedResidualAtomIds ?? [];
  const submittedReviewState =
    input.baseDraftSha256 !== undefined
    || input.acceptedTemplateSegmentIds !== undefined
    || input.reviewedResidualAtomIds !== undefined;

  if (!draft) {
    if (submittedReviewState) {
      throw new RadiologySessionError(
        409,
        'review_draft_unavailable',
        'This v2 artifact has no immutable template review draft',
      );
    }
    return {
      baseDraftSha256: null,
      acceptedTemplateSegmentIds: [],
      reviewedResidualAtomIds: [],
      acceptedSegments: [],
      requireCompleteDraftEvidence: false,
    };
  }

  if (draft.sha256 !== sha256Text(draft.fullText)) {
    throw new RadiologySessionError(
      409,
      'review_draft_integrity_mismatch',
      'The immutable review draft does not match its stored SHA-256',
    );
  }
  if (
    input.baseDraftSha256 !== undefined
    && input.baseDraftSha256 !== draft.sha256
  ) {
    throw new RadiologySessionError(
      409,
      'review_draft_sha_mismatch',
      'baseDraftSha256 does not match the immutable artifact review draft',
    );
  }
  if (
    (input.approved || acceptedIds.length > 0 || reviewedResidualIds.length > 0)
    && input.baseDraftSha256 === undefined
  ) {
    throw new RadiologySessionError(
      422,
      'review_draft_sha_required',
      'Template review decisions require baseDraftSha256',
    );
  }
  if (input.approved && draft.status === 'failed') {
    throw new RadiologySessionError(
      422,
      'review_draft_approval_blocked',
      'The template review draft has critical composition issues',
    );
  }
  const segmentById = new Map(draft.segments.map((segment) => [segment.id, segment]));
  const acceptedSegments = acceptedIds.map((id) => {
    const segment = segmentById.get(id);
    if (!segment) {
      throw new RadiologySessionError(
        422,
        'template_segment_mismatch',
        `Template segment ${id} does not belong to this artifact`,
      );
    }
    if (!segment.confirmationRequired) {
      throw new RadiologySessionError(
        422,
        'template_segment_not_confirmable',
        `Template segment ${id} does not require physician confirmation`,
      );
    }
    if (segment.defaultKind === 'placeholder') {
      throw new RadiologySessionError(
        422,
        'template_placeholder_not_approvable',
        `Template placeholder ${id} must be filled or removed, not accepted as clinical text`,
      );
    }
    if (
      draft.fullText.slice(segment.start, segment.end) !== segment.text
    ) {
      throw new RadiologySessionError(
        409,
        'template_segment_integrity_mismatch',
        `Template segment ${id} does not match its immutable draft span`,
      );
    }
    return segment;
  });
  const residualIds = new Set(draft.residualAtomIds);
  for (const atomId of reviewedResidualIds) {
    if (!residualIds.has(atomId)) {
      throw new RadiologySessionError(
        422,
        'residual_atom_mismatch',
        `Residual atom ${atomId} does not belong to this artifact`,
      );
    }
  }
  if (input.approved) {
    const reviewed = new Set(reviewedResidualIds);
    const missing = draft.residualAtomIds.filter((atomId) => !reviewed.has(atomId));
    if (missing.length > 0) {
      throw new RadiologySessionError(
        422,
        'residual_atom_review_required',
        `Residual atom ${missing[0]} requires explicit physician review`,
      );
    }
    const unresolvedAssignment = draft.fieldAssignments.find(
      (assignment) => assignment.status !== 'applied',
    );
    if (unresolvedAssignment) {
      throw new RadiologySessionError(
        422,
        'review_draft_field_unresolved',
        `Template field ${unresolvedAssignment.fieldId} is ${unresolvedAssignment.status} and must be resolved before approval`,
      );
    }
  }

  return {
    baseDraftSha256: input.baseDraftSha256 ?? null,
    acceptedTemplateSegmentIds: acceptedIds,
    reviewedResidualAtomIds: reviewedResidualIds,
    acceptedSegments,
    requireCompleteDraftEvidence: input.approved,
  };
}

/**
 * Removes only unchanged deterministic template fragments. The remaining text
 * is what the doctor dictated or manually added, so every critical entity in
 * it must be supported by the reviewed verbatim transcript.
 */
function feedbackComparableText(
  report: RadiologyArtifactReport | null,
  finalReport: string,
  review?: ValidatedReviewDraftFeedback,
): string {
  let comparable = finalReport.replace(/\r\n?/gu, '\n');
  if (!report) return comparable;

  comparable = removeOnce(comparable, report.title.replace(/\r\n?/gu, '\n'));
  if (report.reviewDraft) {
    const acceptedIds = new Set(
      review?.acceptedTemplateSegmentIds ?? [],
    );
    const orderedSegments = [...report.reviewDraft.segments]
      .sort((left, right) => left.start - right.start);
    const remainingSectionBodies = reviewSectionBodies(
      finalReport.replace(/\r\n?/gu, '\n'),
      report.reviewDraft,
    );
    const sectionCursors = new Map<string, number>();

    // First reserve every accepted or transcript-backed segment inside its
    // original section. A value copied under another organ can no longer
    // satisfy this check.
    for (const segment of orderedSegments) {
      const normalizedSegmentText = segment.text.replace(/\r\n?/gu, '\n');
      const requiredAcceptedSegment = (
        segment.confirmationRequired
        && acceptedIds.has(segment.id)
      );
      const requiredEvidenceSegment = (
        review?.requireCompleteDraftEvidence
        && segment.evidence.length > 0
      );
      if (!requiredAcceptedSegment && !requiredEvidenceSegment) continue;
      const sectionBody = remainingSectionBodies.get(segment.sectionId) ?? '';
      const cursor = sectionCursors.get(segment.sectionId) ?? 0;
      const relativeIndex = segmentIndex(
        sectionBody.slice(cursor),
        normalizedSegmentText,
      );
      if (relativeIndex < 0) {
        throw new RadiologySessionError(
          409,
          requiredAcceptedSegment
            ? 'accepted_template_segment_mismatch'
            : 'transcript_template_segment_mismatch',
          requiredAcceptedSegment
            ? `Accepted template segment ${segment.id} is absent, modified, or moved to another section`
            : `Transcript-backed template segment ${segment.id} is absent, modified, or moved to another section`,
        );
      }
      const absoluteIndex = cursor + relativeIndex;
      sectionCursors.set(
        segment.sectionId,
        absoluteIndex + normalizedSegmentText.length,
      );
      remainingSectionBodies.set(
        segment.sectionId,
        maskSegmentAt(sectionBody, absoluteIndex, normalizedSegmentText),
      );
    }

    // Accepted/required occurrences have been reserved. Any remaining exact
    // occurrence of an omitted template segment is therefore unapproved,
    // including when it was moved under a different organ.
    for (const segment of orderedSegments) {
      if (
        !segment.confirmationRequired
        || acceptedIds.has(segment.id)
      ) {
        continue;
      }
      const normalizedSegmentText = segment.text.replace(/\r\n?/gu, '\n');
      if (
        [...remainingSectionBodies.values()].some(
          (body) => segmentIndex(body, normalizedSegmentText) >= 0,
        )
      ) {
        throw new RadiologySessionError(
          422,
          'unaccepted_template_segment_present',
          `Unaccepted template segment ${segment.id} must be removed from finalReport`,
        );
      }
    }
    if (review?.requireCompleteDraftEvidence) {
      validateFinalFieldBindings(report, reviewSectionBodies(
        finalReport.replace(/\r\n?/gu, '\n'),
        report.reviewDraft,
      ), acceptedIds);
    }

    for (const section of report.reviewDraft.sections) {
      comparable = removeOnce(comparable, `${section.label}:`.replace(/\r\n?/gu, '\n'));
    }
    for (const segment of orderedSegments) {
      const normalizedSegmentText = segment.text.replace(/\r\n?/gu, '\n');
      const found = segmentIndex(comparable, normalizedSegmentText) >= 0;
      if (
        found
        && (
          !segment.confirmationRequired
          || acceptedIds.has(segment.id)
        )
      ) {
        comparable = removeSegmentOnce(comparable, normalizedSegmentText);
      }
    }
    // The immutable evidence text preserves organ/field association that is
    // intentionally absent from value-only template segments. Any text that
    // was added to or changed in the draft remains in `comparable` and is
    // checked as an additional physician-authored claim.
    return `${report.evidenceBackedText}\n${comparable}`;
  }

  for (const block of report.blocks) {
    // Section labels are presentation metadata, never acoustic evidence.
    comparable = removeOnce(comparable, `${block.label}:`.replace(/\r\n?/gu, '\n'));
    if (block.origin === 'template_default') {
      comparable = removeOnce(comparable, block.text.replace(/\r\n?/gu, '\n'));
    } else if (block.origin === 'generated_extract') {
      // The unchanged extractive conclusion was already claim-checked when
      // the immutable artifact was created. Any physician edit no longer
      // matches this exact fragment and remains in the feedback safety input.
      comparable = removeOnce(comparable, block.text.replace(/\r\n?/gu, '\n'));
    }
  }
  return comparable;
}

export class RadiologySessionService {
  private readonly sessions = new Map<string, ActiveRadiologySession>();
  private readonly sessionTtlMs: number;
  private readonly maxChunkBytes: number;
  private readonly maxChunksPerSession: number;
  private readonly maxTotalAudioBytes: number;
  private readonly maxActiveSessions: number;
  private readonly maxActiveAudioBytes: number;
  private readonly maxOwnerActiveSessions: number;
  private readonly maxOwnerActiveAudioBytes: number;
  private readonly maxPendingTranscriptions: number;
  private pendingTranscriptions = 0;
  private readonly now: () => Date;
  private readonly idFactory: () => string;
  private readonly allowUnownedSessions: boolean;
  private readonly allowAudioPersistence: boolean;

  constructor(private readonly options: RadiologySessionServiceOptions) {
    this.sessionTtlMs = options.sessionTtlMs ?? 30 * 60 * 1000;
    this.maxChunkBytes = options.maxChunkBytes ?? 32 * 1024 * 1024;
    this.maxChunksPerSession = options.maxChunksPerSession ?? 64;
    this.maxTotalAudioBytes = options.maxTotalAudioBytes ?? 64 * 1024 * 1024;
    this.maxActiveSessions = options.maxActiveSessions ?? 32;
    this.maxActiveAudioBytes = options.maxActiveAudioBytes ?? 256 * 1024 * 1024;
    this.maxOwnerActiveSessions = options.maxOwnerActiveSessions ?? 4;
    this.maxOwnerActiveAudioBytes =
      options.maxOwnerActiveAudioBytes ?? 64 * 1024 * 1024;
    this.maxPendingTranscriptions = options.maxPendingTranscriptions ?? 8;
    this.allowUnownedSessions = options.allowUnownedSessions ?? false;
    this.allowAudioPersistence = options.allowAudioPersistence ?? false;
    this.now = options.now ?? (() => new Date());
    this.idFactory = options.idFactory ?? randomUUID;
  }

  create(
    input: CreateRadiologySessionInput,
    actor?: RadiologySessionActor,
  ): CreateRadiologySessionResult {
    this.cleanupExpired();
    if (this.sessions.size >= this.maxActiveSessions) {
      throw new RadiologySessionError(
        503,
        'active_session_limit',
        'Too many active radiology sessions; retry after an existing session finishes or expires',
      );
    }
    const ownerDoctorId = this.ownerForCreate(actor);
    const ownerSessionCount = [...this.sessions.values()]
      .filter((session) => session.ownerDoctorId === ownerDoctorId)
      .length;
    if (ownerSessionCount >= this.maxOwnerActiveSessions) {
      throw new RadiologySessionError(
        429,
        'owner_active_session_limit',
        'This doctor has too many active radiology sessions',
      );
    }
    const id = this.idFactory();
    const now = this.now();
    const source = input.source ?? 'unknown';
    this.sessions.set(id, {
      id,
      ownerDoctorId,
      templateId: input.templateId,
      source,
      ...(input.mimeType ? { mimeType: input.mimeType } : {}),
      retainAudio: input.retainAudio === true && this.allowAudioPersistence,
      createdAt: now.toISOString(),
      lastActivityAtMs: now.getTime(),
      totalAudioBytes: 0,
      chunks: new Map(),
    });
    return {
      sessionId: id,
      mode: 'radiology',
      templateId: input.templateId,
      source,
      retainAudio: input.retainAudio === true && this.allowAudioPersistence,
      createdAt: now.toISOString(),
      chunkUrl: `/api/sessions/${id}/chunks`,
      finishUrl: `/api/sessions/${id}/finish`,
    };
  }

  async addChunk(
    sessionId: string,
    input: { audioBase64: string; chunkIndex?: number; mimeType?: string },
    actor?: RadiologySessionActor,
  ): Promise<{
    ok: true;
    chunkIndex: number;
    duplicate: boolean;
    retried?: boolean;
    audioSha256: string;
  }> {
    const session = this.requireActiveSession(sessionId, actor);
    if (!this.options.transcribeChunk) {
      throw new RadiologySessionError(
        503,
        'radiology_transcriber_unavailable',
        'Radiology transcription is not configured',
      );
    }
    if (session.finishPromise) {
      throw new RadiologySessionError(409, 'session_finishing', 'Session is already finishing');
    }
    if (!session.mimeType && input.mimeType?.trim()) {
      session.mimeType = input.mimeType.trim().slice(0, 200);
    }

    const index = input.chunkIndex ?? session.chunks.size;
    if (!Number.isSafeInteger(index) || index < 0 || index > 0xffff_ffff) {
      throw new RadiologySessionError(400, 'invalid_chunk_index', 'chunk_index must be an unsigned 32-bit integer');
    }
    const audio = decodeBase64Audio(input.audioBase64);
    if (audio.length > this.maxChunkBytes) {
      throw new RadiologySessionError(413, 'audio_chunk_too_large', 'Audio chunk exceeds the configured limit');
    }
    const audioSha256 = createHash('sha256').update(audio).digest('hex');
    const existing = session.chunks.get(index);
    if (existing) {
      if (existing.audioSha256 !== audioSha256) {
        throw new RadiologySessionError(
          409,
          'chunk_index_conflict',
          'A different audio chunk already uses this chunk_index',
        );
      }
      session.lastActivityAtMs = this.now().getTime();
      if (existing.transcriptionState === 'rejected') {
        this.assertTranscriptionCapacity();
        this.startChunkTranscription(session, existing);
        return {
          ok: true,
          chunkIndex: index,
          duplicate: true,
          retried: true,
          audioSha256,
        };
      }
      return { ok: true, chunkIndex: index, duplicate: true, retried: false, audioSha256 };
    }
    if (session.chunks.size >= this.maxChunksPerSession) {
      throw new RadiologySessionError(
        413,
        'session_chunk_limit',
        'Radiology session exceeds the configured chunk count limit',
      );
    }
    if (session.totalAudioBytes + audio.length > this.maxTotalAudioBytes) {
      throw new RadiologySessionError(
        413,
        'session_audio_too_large',
        'Radiology session exceeds the configured total audio limit',
      );
    }
    const activeAudioBytes = [...this.sessions.values()]
      .reduce((total, active) => total + active.totalAudioBytes, 0);
    if (activeAudioBytes + audio.length > this.maxActiveAudioBytes) {
      throw new RadiologySessionError(
        503,
        'active_audio_bytes_limit',
        'The global active radiology audio memory limit has been reached',
      );
    }
    const ownerActiveAudioBytes = [...this.sessions.values()]
      .filter((active) => active.ownerDoctorId === session.ownerDoctorId)
      .reduce((total, active) => total + active.totalAudioBytes, 0);
    if (ownerActiveAudioBytes + audio.length > this.maxOwnerActiveAudioBytes) {
      throw new RadiologySessionError(
        429,
        'owner_active_audio_bytes_limit',
        'This doctor reached the active radiology audio memory limit',
      );
    }
    this.assertTranscriptionCapacity();

    const chunk: SessionChunk = {
      index,
      audio,
      audioSha256,
      transcription: null,
      transcriptionState: 'pending',
    };
    this.startChunkTranscription(session, chunk);
    session.chunks.set(index, chunk);
    session.totalAudioBytes += audio.length;
    session.lastActivityAtMs = this.now().getTime();
    return { ok: true, chunkIndex: index, duplicate: false, audioSha256 };
  }

  private startChunkTranscription(
    session: ActiveRadiologySession,
    chunk: SessionChunk,
  ): void {
    this.pendingTranscriptions += 1;
    chunk.transcriptionState = 'pending';
    const transcription = Promise.resolve()
      .then(() => this.options.transcribeChunk!(
        chunk.audio.toString('base64'),
        {
          sessionId: session.id,
          templateId: session.templateId,
          chunkIndex: chunk.index,
        },
      ))
      .then(
        (result) => {
          chunk.transcriptionState = 'fulfilled';
          return normalizeChunkResult(result);
        },
        (error: unknown) => {
          chunk.transcriptionState = 'rejected';
          throw error;
        },
      )
      .finally(() => {
        this.pendingTranscriptions -= 1;
      });
    chunk.transcription = transcription;
    // Attach a handler immediately so an early ASR failure cannot become an
    // unhandled rejection while the client is still recording or retrying.
    void transcription.catch(() => undefined);
  }

  private assertTranscriptionCapacity(): void {
    if (this.pendingTranscriptions >= this.maxPendingTranscriptions) {
      throw new RadiologySessionError(
        503,
        'asr_backpressure',
        'Too many ASR transcriptions are pending; retry this chunk later',
      );
    }
  }

  async finish(
    sessionId: string,
    input?: { browserTranscript?: string },
    actor?: RadiologySessionActor,
  ): Promise<RadiologyTranscriptionArtifact> {
    this.cleanupExpired();
    const id = safeSessionId(sessionId);
    const session = this.sessions.get(id);
    if (!session) {
      const persisted = await this.options.store.getArtifact(id);
      if (persisted) {
        this.assertOwner(this.artifactOwner(persisted), actor);
        return persisted;
      }
      throw new RadiologySessionError(404, 'session_not_found', 'Session not found or expired');
    }
    this.assertOwner(session.ownerDoctorId, actor);
    session.lastActivityAtMs = this.now().getTime();
    if (session.finishPromise) return session.finishPromise;
    session.finishPromise = this.finishOnce(session, input);
    try {
      const artifact = await session.finishPromise;
      this.sessions.delete(sessionId);
      return artifact;
    } catch (error) {
      session.finishPromise = undefined;
      throw error;
    }
  }

  async getArtifact(
    sessionId: string,
    actor?: RadiologySessionActor,
  ): Promise<RadiologyTranscriptionArtifact | null> {
    const artifact = await this.options.store.getArtifact(sessionId);
    if (artifact) this.assertOwner(this.artifactOwner(artifact), actor);
    return artifact;
  }

  private async buildRecomposeRevision(
    artifact: RadiologyTranscriptionArtifact,
    verbatimTranscript: string,
  ): Promise<RadiologyRecomposeRevision> {
    const normalizationResult = denormalizeDetailed(verbatimTranscript);
    const normalization = artifactNormalization(normalizationResult, verbatimTranscript);
    const normalizationGate = rawToNormalizedStage(
      verbatimTranscript,
      normalization,
      true,
    );
    const structuredReport = this.options.structureTranscript && !normalizationGate.destructive
      ? await this.options.structureTranscript(
          artifact.templateId,
          normalization.text,
          {
            allowLLM: !normalizationGate.ambiguous,
            normalizationAmbiguous: normalizationGate.ambiguous,
            rawTranscript: verbatimTranscript,
            normalizationAlignment: normalizationResult.alignment,
          },
        )
      : null;
    const report = artifactReport(structuredReport);
    const evidenceBackedText = report?.evidenceBackedText ?? '';
    const normalizedToReportStage = structuredReport
      ? safetyStageFromReport(
          'normalized_to_report',
          normalization.text,
          evidenceBackedText,
          verifyRadiologySafety(normalization.text, evidenceBackedText),
        )
      : notRunSafetyStage(
          'normalized_to_report',
          normalization.sha256,
          null,
        );
    return {
      schemaVersion: 1,
      kind: 'radiology-recompose-revision',
      sessionId: artifact.sessionId,
      templateId: artifact.templateId,
      sourceArtifactSha256: sha256Text(canonicalJson(artifact)),
      verbatimTranscript: {
        text: verbatimTranscript,
        sha256: sha256Text(verbatimTranscript),
      },
      normalization,
      routing: routingForReport(structuredReport),
      report,
      safety: safetyResult(
        structuredReport,
        [normalizationGate.stage, normalizedToReportStage],
      ),
      components: artifact.components,
    };
  }

  private artifactWithRevision(
    artifact: RadiologyTranscriptionArtifact,
    revision: RadiologyRecomposeRevision,
  ): RadiologyTranscriptionArtifact {
    return {
      ...artifact,
      normalizedTranscript: {
        text: revision.normalization.text,
        sha256: revision.normalization.sha256,
      },
      normalization: revision.normalization,
      sections: revision.report?.sections ?? [],
      routing: revision.routing,
      unmatchedText: revision.report?.unmatched ?? '',
      report: revision.report,
      reportSha256: revision.report
        ? sha256Text(revision.report.fullText)
        : null,
      safety: revision.safety,
    };
  }

  async recompose(
    sessionId: string,
    input: RadiologyRecomposeInput,
    actor?: RadiologySessionActor,
  ): Promise<RadiologyRecomposeRevision> {
    const artifact = await this.options.store.getArtifact(sessionId);
    if (!artifact) {
      throw new RadiologySessionError(404, 'artifact_not_found', 'Radiology session artifact not found');
    }
    this.assertOwner(this.artifactOwner(artifact), actor);
    const submittedCorrections: StoredSpanCorrection[] = input.spanCorrections.map(
      (correction) => ({
        ...correction,
        confidence: correction.confidence ?? null,
        author: correction.author?.trim() ?? '',
      }),
    );
    validateArtifactCorrections(
      artifact,
      input.verbatimTranscript,
      submittedCorrections,
    );
    return this.buildRecomposeRevision(artifact, input.verbatimTranscript);
  }

  async saveFeedback(
    sessionId: string,
    input: RadiologyFeedbackInput,
    authenticatedAuthor?: string,
    actor?: RadiologySessionActor,
  ): Promise<SaveRadiologyFeedbackResult> {
    const artifact = await this.options.store.getArtifact(sessionId);
    if (!artifact) {
      throw new RadiologySessionError(404, 'artifact_not_found', 'Radiology session artifact not found');
    }
    this.assertOwner(this.artifactOwner(artifact), actor);
    if (
      input.approved
      && artifact.legacySchemaVersion === LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION
    ) {
      throw new RadiologySessionError(
        422,
        'legacy_artifact_approval_blocked',
        'Artifact v1 must be reprocessed through the v2 integrity pipeline before approval',
      );
    }
    if (
      input.approved
      && (
        artifact.longform.degraded
        || artifact.longform.seamConflicts.some((seam) => seam.critical)
      )
    ) {
      throw new RadiologySessionError(
        422,
        'longform_integrity_approval_blocked',
        'Degraded long-form decoding or a critical overlap conflict must be reprocessed before approval',
      );
    }
    const defaultAuthor = authenticatedAuthor?.trim() || input.author?.trim() || '';
    if (!defaultAuthor) {
      throw new RadiologySessionError(400, 'feedback_author_required', 'Feedback author is required');
    }
    const submittedCorrections = input.spanCorrections.map((correction) => ({
      ...correction,
      confidence: correction.confidence ?? null,
      author:
        authenticatedAuthor?.trim()
        || correction.author?.trim()
        || defaultAuthor,
    }));
    if (input.approved && (!input.verbatimTranscript.trim() || !input.finalReport.trim())) {
      throw new RadiologySessionError(
        400,
        'approved_feedback_incomplete',
        'Approved feedback requires both a verbatim transcript and a final report',
      );
    }
    const corrections = validateArtifactCorrections(
      artifact,
      input.verbatimTranscript,
      submittedCorrections,
    );
    const recomposeRevision = corrections.length > 0
      ? await this.buildRecomposeRevision(artifact, input.verbatimTranscript)
      : null;
    const effectiveArtifact = recomposeRevision
      ? this.artifactWithRevision(artifact, recomposeRevision)
      : artifact;
    if (input.approved && effectiveArtifact.report === null) {
      throw new RadiologySessionError(
        422,
        'artifact_report_unavailable',
        'The reviewed transcript has no integrity-safe report and must be reprocessed before approval',
      );
    }
    if (input.approved && effectiveArtifact.routing.unmatchedAtomIds.length > 0) {
      throw new RadiologySessionError(
        422,
        'unmatched_atoms_approval_blocked',
        'Every clinical transcript atom must be assigned before approval',
      );
    }
    const reviewDraftFeedback = validateReviewDraftFeedback(effectiveArtifact, input);
    const normalizationResolutions = validateNormalizationResolutions(
      effectiveArtifact,
      input,
      recomposeRevision ? artifact : undefined,
    );
    const reviewedNormalization = recomposeRevision?.normalization ?? artifactNormalization(
      denormalizeDetailed(input.verbatimTranscript),
      input.verbatimTranscript,
    );
    const reviewedNormalizationGate = rawToNormalizedStage(
      input.verbatimTranscript,
      reviewedNormalization,
      true,
    );
    if (input.approved && reviewedNormalizationGate.stage.status !== 'passed') {
      const details = reviewedNormalizationGate.stage.issues
        .map((issue) => issue.message)
        .join('; ');
      throw new RadiologySessionError(
        422,
        'normalization_approval_blocked',
        details
          ? `Verbatim transcript has unresolved normalization issues: ${details}`
          : 'Verbatim transcript has unresolved normalization issues',
      );
    }

    const effectiveContent = {
      sessionId,
      templateId: artifact.templateId,
      source: artifact.source.type,
      sourceArtifactSha256: recomposeRevision?.sourceArtifactSha256
        ?? sha256Text(canonicalJson(artifact)),
      author: defaultAuthor,
      verbatimTranscript: input.verbatimTranscript,
      finalReport: input.finalReport,
      spanCorrections: submittedCorrections,
      normalizationResolutions: input.normalizationResolutions ?? [],
      ...(effectiveArtifact.report?.reviewDraft
        ? {
            baseDraftSha256: reviewDraftFeedback.baseDraftSha256,
            acceptedTemplateSegmentIds: reviewDraftFeedback.acceptedTemplateSegmentIds,
            reviewedResidualAtomIds: reviewDraftFeedback.reviewedResidualAtomIds,
          }
        : {}),
      approved: input.approved,
    };
    const contentSha256 = sha256Text(canonicalJson(effectiveContent));
    const existingFeedback = await this.options.store.getFeedbackByIdempotencyKey(
      sessionId,
      input.idempotencyKey,
    );
    if (existingFeedback) {
      if (existingFeedback.contentSha256 !== contentSha256) {
        throw new RadiologySessionError(
          409,
          'feedback_idempotency_conflict',
          'idempotencyKey is already bound to a different feedback payload',
        );
      }
      return { feedback: existingFeedback, idempotentReplay: true };
    }

    // Re-run the checks over the text the doctor is actually approving. This
    // both lets a doctor repair a failed draft and prevents a safe draft from
    // becoming unsafe through edits made in the review form.
    const comparableFinalReport = feedbackComparableText(
      effectiveArtifact.report,
      input.finalReport,
      reviewDraftFeedback,
    );
    const reviewedSafety = verifyRadiologySafety(
      reviewedNormalization.text,
      comparableFinalReport,
    );
    const reviewedSafetyStage = safetyStageFromReport(
      'verbatim_to_final_report',
      reviewedNormalization.text,
      comparableFinalReport,
      reviewedSafety,
    );
    reviewedSafetyStage.sourceSha256 = sha256Text(input.verbatimTranscript);
    if (input.approved && !reviewedSafety.ok) {
      const details = reviewedSafety.issues.map((issue) => issue.message).join('; ');
      throw new RadiologySessionError(
        422,
        'safety_approval_blocked',
        details
          ? `Final report has unresolved critical safety issues: ${details}`
          : 'Final report has unresolved critical safety issues',
      );
    }

    const exclusionReasons = trainingExclusionReasons({
      source: artifact.source.type,
      rawAvailable: artifact.rawTranscript.rawAvailable,
      audioStored: artifact.audio.stored,
      asrChunks: artifact.asrChunks ?? [],
      model: artifact.model,
    });
    if (artifact.legacySchemaVersion === LEGACY_RADIOLOGY_ARTIFACT_SCHEMA_VERSION) {
      exclusionReasons.push('legacy_artifact_schema');
    }
    if (!input.approved) exclusionReasons.push('not_approved');
    if (reviewedNormalizationGate.stage.status !== 'passed') {
      exclusionReasons.push('feedback_normalization_failed');
    }
    if (!reviewedSafety.ok) exclusionReasons.push('feedback_safety_failed');
    if (!input.verbatimTranscript.trim()) exclusionReasons.push('verbatim_transcript_empty');
    return this.options.store.saveFeedback({
      schemaVersion: RADIOLOGY_FEEDBACK_SCHEMA_VERSION,
      kind: 'radiology-feedback',
      datasetVersion: 'radiology-feedback/v2',
      feedbackId: randomUUID(),
      idempotencyKey: input.idempotencyKey,
      contentSha256,
      sessionId,
      templateId: artifact.templateId,
      createdAt: this.now().toISOString(),
      source: artifact.source.type,
      author: defaultAuthor,
      verbatimTranscript: input.verbatimTranscript,
      verbatimTranscriptSha256: sha256Text(input.verbatimTranscript),
      finalReport: input.finalReport,
      finalReportSha256: sha256Text(input.finalReport),
      spanCorrections: corrections,
      normalizationResolutions,
      baseDraftSha256: reviewDraftFeedback.baseDraftSha256,
      acceptedTemplateSegmentIds: reviewDraftFeedback.acceptedTemplateSegmentIds,
      reviewedResidualAtomIds: reviewDraftFeedback.reviewedResidualAtomIds,
      recomposeRevision,
      approved: input.approved,
      safety: reviewedSafety,
      normalizationSafetyStage: reviewedNormalizationGate.stage,
      safetyStage: reviewedSafetyStage,
      training: {
        eligible: uniqueReasons(exclusionReasons).length === 0,
        exclusionReasons: uniqueReasons(exclusionReasons),
      },
    });
  }

  private ownerForCreate(actor: RadiologySessionActor | undefined): string | null {
    if (actor) {
      const doctorId = actor.doctorId?.trim();
      if (!doctorId) {
        throw new RadiologySessionError(
          403,
          'radiology_actor_identity_required',
          'Authenticated radiology sessions require a doctor id',
        );
      }
      return doctorId;
    }
    if (this.allowUnownedSessions) return null;
    throw new RadiologySessionError(
      401,
      'radiology_auth_required',
      'Authentication is required for radiology sessions',
    );
  }

  private artifactOwner(artifact: RadiologyTranscriptionArtifact): string | null {
    // Artifacts written before ownerDoctorId was introduced are intentionally
    // treated as unowned. Only an admin (or the explicit no-auth compatibility
    // mode) may read them; a regular authenticated doctor may not claim them.
    return typeof artifact.ownerDoctorId === 'string' && artifact.ownerDoctorId.trim()
      ? artifact.ownerDoctorId.trim()
      : null;
  }

  private assertOwner(
    ownerDoctorId: string | null,
    actor: RadiologySessionActor | undefined,
  ): void {
    if (actor?.role === 'admin') return;
    const actorDoctorId = actor?.doctorId?.trim();
    if (ownerDoctorId !== null && actorDoctorId === ownerDoctorId) return;
    if (ownerDoctorId === null && actor === undefined && this.allowUnownedSessions) return;
    throw new RadiologySessionError(
      403,
      'radiology_session_forbidden',
      'Radiology session belongs to another doctor',
    );
  }

  private requireActiveSession(
    sessionId: string,
    actor?: RadiologySessionActor,
  ): ActiveRadiologySession {
    this.cleanupExpired();
    const session = this.sessions.get(safeSessionId(sessionId));
    if (!session) {
      throw new RadiologySessionError(404, 'session_not_found', 'Session not found or expired');
    }
    this.assertOwner(session.ownerDoctorId, actor);
    return session;
  }

  private cleanupExpired(): void {
    const now = this.now().getTime();
    for (const [id, session] of this.sessions) {
      const ownsInFlightWork =
        Boolean(session.finishPromise)
        || [...session.chunks.values()]
          .some((chunk) => chunk.transcriptionState === 'pending');
      if (
        !ownsInFlightWork
        && now - session.lastActivityAtMs > this.sessionTtlMs
      ) {
        this.sessions.delete(id);
      }
    }
  }

  private async finishOnce(
    session: ActiveRadiologySession,
    input?: { browserTranscript?: string },
  ): Promise<RadiologyTranscriptionArtifact> {
    const ordered = [...session.chunks.values()].sort((a, b) => a.index - b.index);
    let chunks: RadiologyChunkTranscription[];
    if (ordered.length > 0) {
      try {
        chunks = await Promise.all(ordered.map((chunk) => {
          if (!chunk.transcription) {
            throw new RadiologySessionError(
              503,
              'chunk_transcription_unavailable',
              `Chunk ${chunk.index} has no active transcription`,
            );
          }
          return chunk.transcription;
        }));
      } catch (error) {
        if (error instanceof RadiologySessionError) throw error;
        throw new RadiologySessionError(
          503,
          'asr_transcription_failed',
          'The remote ASR service failed to transcribe this recording; retry the chunk or session',
        );
      }
    } else if (
      (session.source === 'browser' || session.source === 'manual')
      && input?.browserTranscript?.trim()
    ) {
      chunks = [{
        rawText: input.browserTranscript.trim(),
        normalizedText: input.browserTranscript.trim(),
        rawAvailable: false,
        language: 'ru',
        source: session.source,
        words: [],
      }];
    } else {
      throw new RadiologySessionError(422, 'no_audio_chunks', 'No audio chunks received in session');
    }

    const asrChunks = chunks.map((chunk, index) => artifactTranscriptionChunk(
      chunk,
      ordered[index]?.index ?? index,
      ordered[index]?.audioSha256,
    ));
    assertUniformASRProvenance(session.templateId, this.options.model, chunks, asrChunks);
    const rawText = chunks.length === 1
      ? (chunks[0].rawText ?? chunks[0].normalizedText)
      : chunks.map((chunk) => chunk.rawText?.trim()).filter(Boolean).join(' ');
    const rawAvailable = chunks.every((chunk) => chunk.rawAvailable === true);
    const normalizationInput = rawText
      || chunks.map((chunk) => chunk.normalizedText.trim()).filter(Boolean).join(' ');
    const normalizationResult = denormalizeDetailed(normalizationInput);
    const normalization = artifactNormalization(
      normalizationResult,
      normalizationInput,
    );
    const normalizedText = normalization.text;
    if (!normalizedText) {
      throw new RadiologySessionError(422, 'empty_transcription', 'All chunks failed transcription');
    }
    const normalizationGate = rawToNormalizedStage(
      rawText || normalizedText,
      normalization,
      rawAvailable,
    );
    const sourceCandidates = new Set(
      chunks.map((chunk) => chunk.source).filter((source): source is RadiologyTranscriptionSource => Boolean(source)),
    );
    const source = session.source === 'browser' || session.source === 'manual'
      ? session.source
      : sourceCandidates.size === 1
        ? [...sourceCandidates][0]
        : 'unknown';
    const words = chunks.flatMap((chunk, index) =>
      (chunk.words ?? []).map((word) => ({ ...word, chunkIndex: word.chunkIndex ?? ordered[index]?.index ?? index })),
    );
    const structuredReport = this.options.structureTranscript && !normalizationGate.destructive
      ? await this.options.structureTranscript(
          session.templateId,
          normalizedText,
          {
            allowLLM: !normalizationGate.ambiguous,
            normalizationAmbiguous: normalizationGate.ambiguous,
            rawTranscript: rawText || normalizationInput,
            normalizationAlignment: normalizationResult.alignment,
          },
        )
      : null;
    const report = artifactReport(structuredReport);
    const evidenceBackedText = report?.evidenceBackedText ?? '';
    const normalizedToReportStage = structuredReport
      ? safetyStageFromReport(
          'normalized_to_report',
          normalizedText,
          evidenceBackedText,
          verifyRadiologySafety(normalizedText, evidenceBackedText),
        )
      : notRunSafetyStage(
          'normalized_to_report',
          normalization.sha256,
          null,
        );
    const audioChunks = ordered.map((chunk) => ({ index: chunk.index, audio: chunk.audio }));
    const audioHash = audioArtifactHash(audioChunks);
    const retainAudio = session.retainAudio && audioChunks.length > 0;
    const resolvedModel = modelMetadata(session.templateId, this.options.model, chunks);
    const seamConflicts = chunks.flatMap((chunk, chunkIndex) =>
      (chunk.longform?.seams ?? [])
        .filter((seam) => seam.conflict)
        .map((seam) => ({
          chunkIndex,
          startMs: seam.startMs,
          endMs: seam.endMs,
          critical: seam.critical,
          ...(seam.leftText ? { leftText: seam.leftText } : {}),
          ...(seam.rightText ? { rightText: seam.rightText } : {}),
        })),
    );
    const longformDegraded = chunks.some((chunk) => chunk.longform?.degraded === true);
    const exclusionReasons = trainingExclusionReasons({
      source,
      rawAvailable,
      audioStored: retainAudio,
      asrChunks,
      model: resolvedModel,
    });
    if (normalizationGate.stage.status !== 'passed') {
      exclusionReasons.push('raw_to_normalized_safety_not_passed');
    }
    if (longformDegraded) exclusionReasons.push('longform_degraded');
    if (seamConflicts.length > 0) exclusionReasons.push('longform_seam_conflict');
    const artifactSafety = safetyResult(
      structuredReport,
      [normalizationGate.stage, normalizedToReportStage],
    );
    if (longformDegraded) {
      artifactSafety.issues.push({
        code: 'longform_degraded',
        severity: 'critical',
        message: 'Long-form transcription used a degraded runtime path and is not approvable.',
      });
    }
    for (const seam of seamConflicts.filter((candidate) => candidate.critical)) {
      artifactSafety.issues.push({
        code: 'overlap_seam_conflict',
        severity: 'critical',
        message: `Critical overlap conflict at ${seam.startMs}-${seam.endMs} ms requires physician review.`,
      });
    }
    if (longformDegraded || seamConflicts.some((seam) => seam.critical)) {
      artifactSafety.status = 'failed';
      artifactSafety.requiresReview = true;
      artifactSafety.approvalBlocked = true;
    }
    if (artifactSafety.status !== 'passed') exclusionReasons.push('clinical_safety_not_passed');
    if ((structuredReport?.routing.unmatchedAtomIds.length ?? 0) > 0) {
      exclusionReasons.push('unmatched_clinical_content');
    }

    const artifact: RadiologyTranscriptionArtifact = {
      schemaVersion: RADIOLOGY_ARTIFACT_SCHEMA_VERSION,
      kind: 'radiology-transcription',
      sessionId: session.id,
      ownerDoctorId: session.ownerDoctorId,
      templateId: session.templateId,
      createdAt: session.createdAt,
      completedAt: this.now().toISOString(),
      source: {
        type: source,
        audioSha256: audioHash.sha256,
      },
      audio: {
        sha256: audioHash.sha256,
        hashKind: audioHash.hashKind,
        bytes: audioChunks.reduce((total, chunk) => total + chunk.audio.length, 0),
        ...(session.mimeType ? { mimeType: session.mimeType } : {}),
        stored: retainAudio,
        chunks: ordered.map((chunk) => ({
          index: chunk.index,
          sha256: chunk.audioSha256,
          bytes: chunk.audio.length,
          stored: retainAudio,
        })),
      },
      rawTranscript: {
        text: rawText || normalizedText,
        sha256: sha256Text(rawText || normalizedText),
        language: chunks.find((chunk) => chunk.language)?.language || 'ru',
        rawAvailable,
        words,
      },
      normalizedTranscript: {
        text: normalizedText,
        sha256: sha256Text(normalizedText),
      },
      normalization,
      asrChunks,
      longform: {
        degraded: longformDegraded,
        seamConflicts,
      },
      sections: report?.sections ?? [],
      routing: routingForReport(structuredReport),
      unmatchedText: structuredReport?.unmatched ?? '',
      report,
      reportSha256: report ? sha256Text(report.fullText) : null,
      safety: artifactSafety,
      components: resolvedModel,
      model: resolvedModel,
      training: {
        eligible: exclusionReasons.length === 0,
        exclusionReasons,
      },
    };
    return this.options.store.saveArtifact(artifact, retainAudio ? audioChunks : null);
  }
}

export function defaultRadiologyDataDir(): string {
  return path.resolve(process.env.RADIOLOGY_DATA_DIR?.trim() || './data/radiology');
}
