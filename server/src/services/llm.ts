import type { MedicalDocument, RiskAssessment, StructureResult, LLMConfig } from '../types.js';
import { findExamTemplate, formatExamLine, parseExamValuesFromText, parseExamDate, examTemplates } from '../data/examTemplates.js';
import { findDietTemplate } from '../data/dietTemplates.js';
import { applyMedicalDictionary } from './medical-dictionary.js';
import { AnthropicProvider } from './anthropic-provider.js';
import { buildLabReferenceForPrompt } from '../data/labReference.js';

const DEFAULT_RISK_ASSESSMENT: RiskAssessment = {
  fallInLast3Months: 'нет',
  dizzinessOrWeakness: 'нет',
  needsEscort: 'нет',
  painScore: '0',
};

interface LlamaCompletionResponse {
  content: string;
}

type MedicalDocumentPatch = Partial<
  Omit<MedicalDocument, 'patient' | 'riskAssessment'> & {
    patient: Partial<MedicalDocument['patient']>;
    riskAssessment: Partial<RiskAssessment>;
  }
>;

type RewriteableField = keyof Omit<MedicalDocument, 'patient' | 'riskAssessment'>;

const ALL_TEXT_FIELDS: RewriteableField[] = [
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

export class LLMService {
  private config: LLMConfig;
  private _rescuedReferrals: string[] = [];
  private anthropic: AnthropicProvider | null = null;

  constructor(config: LLMConfig) {
    this.config = config;
    if (config.provider === 'anthropic') {
      if (!config.anthropic?.apiKey) {
        throw new Error('LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing in server/.env');
      }
      this.anthropic = new AnthropicProvider({
        apiKey: config.anthropic.apiKey,
        model: config.anthropic.model,
        maxTokens: config.anthropic.maxTokens,
        temperature: config.temperature,
        requestTimeoutMs: config.requestTimeoutMs,
        maxRetries: config.anthropic.maxRetries,
      });
      console.log(`[llm] provider=anthropic model=${config.anthropic.model}`);
    } else {
      console.log(`[llm] provider=llama serverUrl=${config.serverUrl} model=${config.model}`);
    }
  }

  async healthCheck(): Promise<boolean> {
    if (this.anthropic) {
      return this.anthropic.healthCheck();
    }
    try {
      const response = await this.fetchWithTimeout(`${this.config.serverUrl}/health`, {
        method: 'GET',
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async structureText(rawText: string): Promise<StructureResult> {
    const startTime = Date.now();

    try {
      const document = this.anthropic
        ? await this.structureWithAnthropic(rawText)
        : await this.structureWithLlamaCpp(rawText);
      return {
        document,
        rawText,
        processingTime: (Date.now() - startTime) / 1000,
      };
    } catch (error) {
      if (this.config.allowMockOnFailure) {
        console.warn('LLM failed, using mock structured document because ALLOW_MOCK_LLM=true', error);
        const mock = this.enrichPatientFromRawText(this.getMockStructuredDocument(rawText), rawText);
        return {
          document: mock,
          rawText,
          processingTime: (Date.now() - startTime) / 1000,
        };
      }

      throw error;
    }
  }

  async applyAddendum(document: MedicalDocument, addendumText: string): Promise<MedicalDocument> {
    const systemPrompt = `You are a medical assistant. Apply addendum to an existing structured medical document.
Rules:
1) Return ONLY changed fields as JSON patch (not the full document).
2) Include a field only if addendum explicitly changes it.
3) If addendum contradicts previous data, patch with addendum value.
4) Do NOT invent new information.
5) If no changes, return {}.
6) Output STRICT JSON only.
7) PRESERVE ALL dictated text — do not omit, summarize or shorten any information.
8) Each piece of information goes to EXACTLY ONE field. Do not duplicate between fields.
9) When the doctor names a section explicitly (e.g. "анамнез жизни:", "диета:"), put ALL following text into that section only.
10) Medications patient CURRENTLY takes (doctor says "амбулаторно принимает") → "conclusion". New prescriptions (doctor says "назначаю", "рекомендую") → "recommendations". Medications mentioned in context of life history/анамнез жизни → "clinicalCourse". Diet → as one of numbered items in "recommendations".
11) IMPORTANT: If addendum mentions exam results (ОАК, Б/х, ЭКГ, ЭхоКГ, etc.), put them in "outpatientExams".`;

    const userPrompt = `Current document (JSON):
${JSON.stringify(document, null, 2)}

Addendum:
${addendumText}

Return JSON patch only.`;

    if (this.anthropic) {
      const raw = await this.anthropic.completeJson({
        systemPrompt,
        userPrompt,
        jsonSchema: this.getAddendumPatchJsonSchema() as any,
        toolName: 'submit_document_patch',
        toolDescription: 'Submit a JSON patch with only the fields that changed.',
        maxTokens: 4096,
        temperature: 0,
        operation: 'applyAddendum',
      });
      console.log(`[llm anthropic] applyAddendum: raw ${raw.length} chars`);
      const patch = await this.parsePatchWithRepair(raw);
      const merged = this.mergeDocumentWithPatch(document, patch, addendumText);
      return this.validateAndCleanDocument(merged);
    }

    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: this.buildCompletionBody({
        prompt: `<|im_start|>system\n/no_think\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: Math.max(4096, this.config.maxTokens),
        temperature: 0,
        stop: ['<|im_end|>'],
        json_schema: this.getAddendumPatchJsonSchema(),
        stream: false,
      }),
    });

    if (!response.ok) {
      await this.throwLlmError(response, 'applyAddendum');
    }

    const data = (await response.json()) as LlamaCompletionResponse;
    const raw = this.stripThinkingBlocks(data.content);
    console.log(`[llm] applyAddendum: LLM returned ${raw.length} chars for addendum "${addendumText.substring(0, 80)}..."`);
    const patch = await this.parsePatchWithRepair(raw);
    const patchKeys = Object.keys(patch).filter(k => {
      const v = patch[k as keyof MedicalDocumentPatch];
      return v !== undefined && v !== '' && (typeof v !== 'object' || Object.keys(v as object).length > 0);
    });
    console.log(`[llm] applyAddendum: patch fields: [${patchKeys.join(', ')}]`);
    const merged = this.mergeDocumentWithPatch(document, patch, addendumText);
    return this.validateAndCleanDocument(merged);
  }

  async applyInstruction(document: MedicalDocument, instruction: string): Promise<MedicalDocument> {
    const normalizedInstruction = instruction.trim();
    if (!normalizedInstruction) return document;

    const directSectionUpdate = this.tryApplyDirectSectionInstruction(document, normalizedInstruction);
    if (directSectionUpdate) {
      return directSectionUpdate;
    }

    const systemPrompt = `You are a medical assistant editing an existing structured medical document by user instruction.
Rules:
1) Return ONLY changed fields as JSON patch (not the full document).
2) Include a field only if the instruction explicitly changes it.
3) If instruction says rewrite/fix a section, return the full final text for that section.
4) Preserve clinical facts; do not invent new facts.
5) If no changes are needed, return {}.
6) Output STRICT JSON only.
7) PRESERVE ALL text — do not omit or shorten any information.
8) Each piece of information goes to EXACTLY ONE field. Do not duplicate between fields.`;

    const userPrompt = `Current document (JSON):
${JSON.stringify(document, null, 2)}

Instruction:
${normalizedInstruction}

Return JSON patch only.`;

    if (this.anthropic) {
      const raw = await this.anthropic.completeJson({
        systemPrompt,
        userPrompt,
        jsonSchema: this.getAddendumPatchJsonSchema() as any,
        toolName: 'submit_document_patch',
        toolDescription: 'Submit a JSON patch with only the fields that changed.',
        maxTokens: 2048,
        temperature: 0,
        operation: 'applyInstruction',
      });
      const patch = await this.parsePatchWithRepair(raw);
      const merged = this.mergeDocumentWithInstructionPatch(document, patch);
      return this.validateAndCleanDocument(merged);
    }

    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: this.buildCompletionBody({
        prompt: `<|im_start|>system\n/no_think\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: Math.max(768, this.config.maxTokens),
        temperature: 0,
        stop: ['<|im_end|>'],
        json_schema: this.getAddendumPatchJsonSchema(),
        stream: false,
      }),
    });

    if (!response.ok) {
      await this.throwLlmError(response, 'applyInstruction');
    }

    const data = (await response.json()) as LlamaCompletionResponse;
    const raw = this.stripThinkingBlocks(data.content);
    const patch = await this.parsePatchWithRepair(raw);
    const merged = this.mergeDocumentWithInstructionPatch(document, patch);
    return this.validateAndCleanDocument(merged);
  }

  private getSectionMatchers(): Array<{ field: RewriteableField; pattern: string }> {
    return [
      // More specific patterns first to avoid false matches
      { field: 'outpatientExams', pattern: '(?:амбулаторн\\S*\\s+(?:данн\\S*|обследовани\\S*)|амбулаторн\\S*\\s+результат\\S*)' },
      { field: 'clinicalCourse', pattern: '(?:перенесённ\\S*\\s+заболевани\\S*|анамнез\\s+жизни)' },
      { field: 'allergyHistory', pattern: 'аллерг\\S*' },
      { field: 'neurologicalStatus', pattern: 'неврологическ\\S*' },
      { field: 'anamnesis', pattern: 'анамнез\\S*' },
      { field: 'complaints', pattern: 'жалоб\\S*' },
      { field: 'objectiveStatus', pattern: 'объектив\\S*' },
      { field: 'finalDiagnosis', pattern: '(?:заключительн\\S*\\s+диагноз\\S*|окончательн\\S*\\s+диагноз\\S*)' },
      { field: 'diagnosis', pattern: '(?:предварительн\\S*\\s+диагноз\\S*|диагноз\\S*)' },
      { field: 'conclusion', pattern: '(?:амбулаторн\\S*\\s+терапи\\S*|амбулаторно\\s+принимает|сопутствующ\\S*)' },
      { field: 'recommendations', pattern: '(?:рекомендац\\S*|план\\s+лечени\\S*)' },
      { field: 'doctorNotes', pattern: '(?:план\\s+обследовани\\S*|направлени\\S*|прочее|заметк\\S*)' },
    ];
  }

  private extractSectionPrefixedPayload(
    text: string
  ): { field: RewriteableField; payload: string } | null {
    for (const { field, pattern } of this.getSectionMatchers()) {
      const directPattern = new RegExp(
        `^(?:(?:пожалуйста)\\s+)?(?:(?:добав(?:ь|ьте)?|внес(?:и|ите)?|запиш(?:и|ите)?|укаж(?:и|ите)?|` +
          `измени(?:ть|те)?|исправ(?:ь|ьте)?|замени(?:ть|те)?|обнов(?:и|ите)?|перепиш(?:и|ите)?|` +
          `отредактируй(?:те)?)\\s+)?(?:в\\s+(?:раздел|поле)\\s+)?(?:${pattern})\\s*[:\\-.]?\\s*(.+)$`,
        'iu'
      );
      const match = text.match(directPattern);
      if (!match) continue;

      const payload = match[1]?.trim();
      if (!payload) return null;
      return { field, payload };
    }
    return null;
  }

  private tryApplyDirectSectionInstruction(
    document: MedicalDocument,
    instruction: string
  ): MedicalDocument | null {
    const direct = this.extractSectionPrefixedPayload(instruction);
    if (direct) {
      const { field, payload } = direct;
      const isReplaceInstruction = /(перепиш|исправ|замен|обнов|отредакт|измени)/iu.test(instruction);
      const currentValue = document[field].trim();
      const nextValue = isReplaceInstruction || !currentValue
        ? payload
        : currentValue.includes(payload)
          ? currentValue
          : `${currentValue}\n\n${payload}`.trim();

      return this.validateAndCleanDocument({
        ...document,
        [field]: nextValue,
      });
    }
    return null;
  }

  private async structureWithAnthropic(rawText: string): Promise<MedicalDocument> {
    if (!this.anthropic) throw new Error('structureWithAnthropic called but provider not initialized');

    const systemPrompt = this.getSystemPrompt();
    const userPrompt = this.getUserPrompt(rawText);
    const schema = this.getDocumentJsonSchema() as {
      type: 'object';
      properties: Record<string, unknown>;
      required?: string[];
      additionalProperties?: boolean;
    };

    const raw = await this.anthropic.completeJson({
      systemPrompt,
      userPrompt,
      jsonSchema: schema,
      toolName: 'submit_medical_document',
      toolDescription: 'Submit the structured medical consultation document extracted from the doctor dictation.',
      maxTokens: 8192,
      temperature: 0,
      operation: 'structureText',
    });

    console.log(`[LLM anthropic] structureText raw length: ${raw.length}`);
    const document = await this.parseDocumentWithRepair(raw, rawText);

    const LLM_FIELDS = ['complaints','anamnesis','clinicalCourse','allergyHistory','objectiveStatus',
      'diagnosis','finalDiagnosis','conclusion','recommendations','doctorNotes','outpatientExams'] as const;
    console.log('\n\x1b[35m━━━ [LLM RAW — anthropic] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m');
    for (const f of LLM_FIELDS) {
      const v = (document as any)[f];
      if (v && typeof v === 'string' && v.trim()) {
        console.log(`  \x1b[35m${f}:\x1b[0m ${v}`);
      }
    }
    console.log('\x1b[35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\n');

    const cleaned = this.validateAndCleanDocument(document);
    const enriched = this.enrichPatientFromRawText(cleaned, rawText);
    // rescueExamsFromRawText намеренно пропущен для Anthropic: Claude с tool-use
    // надёжно извлекает все лабораторные значения, а rescue создаёт дубли (пытается
    // добавить данные, которые уже корректно уложены в основном блоке).
    this.rescueDiagnosisFromRawText(enriched, rawText);
    this.rescueRecommendationsFromRawText(enriched, rawText);
    this.clearRiskAssessmentIfNotMentioned(enriched, rawText);

    // Детерминированные семантические пост-проходы (P1-P5)
    this.runSemanticRoutingPasses(enriched, rawText);

    const FINAL_FIELDS = ['complaints','anamnesis','clinicalCourse','allergyHistory','objectiveStatus',
      'diagnosis','finalDiagnosis','conclusion','recommendations','doctorNotes','outpatientExams'] as const;
    console.log('\n\x1b[32m━━━ [FINAL — anthropic] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m');
    for (const f of FINAL_FIELDS) {
      const v = (enriched as any)[f];
      if (v && typeof v === 'string' && v.trim()) {
        console.log(`  \x1b[32m${f}:\x1b[0m ${v}`);
      }
    }
    console.log('\x1b[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\n');

    return enriched;
  }

  private async structureWithLlamaCpp(rawText: string): Promise<MedicalDocument> {
    const systemPrompt = this.getSystemPrompt();
    const userPrompt = this.getUserPrompt(rawText);

    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: this.buildCompletionBody({
        prompt: `<|im_start|>system\n/no_think\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: Math.max(16384, this.config.maxTokens),
        temperature: 0,
        stop: ['<|im_end|>'],
        json_schema: this.getDocumentJsonSchema(),
        stream: false,
      }),
    });

    if (!response.ok) {
      await this.throwLlmError(response, 'structureText');
    }

    const data = (await response.json()) as LlamaCompletionResponse;
    console.log(`[LLM] response keys: ${Object.keys(data).join(', ')}, content type: ${typeof data.content}, content length: ${String(data.content ?? '').length}`);
    const raw = this.stripThinkingBlocks(data.content);
    const stoppedEos = (data as any).stop_type === 'eos' || (data as any).stopped_eos === true || (data as any).stop === 'eos';
    const truncated = !stoppedEos && raw.length > 0;
    const promptSize = systemPrompt.length + userPrompt.length;
    console.log(`LLM structureText: prompt=${promptSize} chars, response=${raw.length} chars, stopped_eos=${stoppedEos}, truncated=${truncated}`);
    if (truncated) {
      console.warn(`[LLM] WARNING: Response truncated (no EOS). Prompt=${promptSize} chars, input=${rawText.length} chars, output=${raw.length} chars. Check llama.cpp n_ctx setting.`);
    }
    const document = await this.parseDocumentWithRepair(raw, rawText);

    // Log raw LLM output BEFORE post-processing
    const LLM_FIELDS = ['complaints','anamnesis','clinicalCourse','allergyHistory','objectiveStatus',
      'diagnosis','finalDiagnosis','conclusion','recommendations','doctorNotes','outpatientExams'] as const;
    console.log('\n\x1b[35m━━━ [LLM RAW] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m');
    for (const f of LLM_FIELDS) {
      const v = (document as any)[f];
      if (v && typeof v === 'string' && v.trim()) {
        console.log(`  \x1b[35m${f}:\x1b[0m ${v}`);
      }
    }
    console.log('\x1b[35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\n');

    const cleaned = this.validateAndCleanDocument(document);
    const enriched = this.enrichPatientFromRawText(cleaned, rawText);

    // Пре-экстракция: спасаем данные обследований из исходного текста,
    // которые LLM мог потерять
    this.rescueExamsFromRawText(enriched, rawText);

    // Спасаем диагноз если LLM его обрезал
    this.rescueDiagnosisFromRawText(enriched, rawText);

    // Спасаем рекомендации из исходного текста если LLM их потерял
    this.rescueRecommendationsFromRawText(enriched, rawText);

    this.clearRiskAssessmentIfNotMentioned(enriched, rawText);

    // Log final result AFTER all post-processing
    const FINAL_FIELDS = ['complaints','anamnesis','clinicalCourse','allergyHistory','objectiveStatus',
      'diagnosis','finalDiagnosis','conclusion','recommendations','doctorNotes','outpatientExams'] as const;
    console.log('\n\x1b[32m━━━ [FINAL] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m');
    for (const f of FINAL_FIELDS) {
      const v = (enriched as any)[f];
      if (v && typeof v === 'string' && v.trim()) {
        console.log(`  \x1b[32m${f}:\x1b[0m ${v}`);
      }
    }
    console.log('\x1b[32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\n');

    return enriched;
  }

  async generateRecommendations(document: MedicalDocument): Promise<string> {
    const systemPrompt = `You are a clinical assistant for doctors.
Rules:
1) Always answer in Russian.
2) Provide concise practical recommendations in bullet points.
3) Do not invent diagnoses; if data is insufficient, explicitly state what is missing.
4) Keep tone professional and clinical.`;

    const userPrompt = `Medical document (JSON):
${JSON.stringify(document, null, 2)}

Return up to 6 practical recommendations for the doctor.
Answer in Russian and use bullet points.`;

    if (this.anthropic) {
      return this.anthropic.completeText({
        systemPrompt,
        userPrompt,
        maxTokens: 1024,
        temperature: 0.3,
        operation: 'generateRecommendations',
      });
    }

    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: this.buildCompletionBody({
        prompt: `<|im_start|>system\n/no_think\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: 512,
        // 0.3 — естественнее звучит, меньше зацикливания на повторах
        temperature: 0.3,
        stop: ['<|im_end|>'],
        stream: false,
      }),
    });

    if (!response.ok) {
      await this.throwLlmError(response, 'generateRecommendations');
    }

    const data = (await response.json()) as LlamaCompletionResponse;
    return this.stripThinkingBlocks(data.content);
  }

  async chat(
    question: string,
    history: Array<{ role: 'user' | 'assistant'; text: string }> = [],
    document?: MedicalDocument
  ): Promise<string> {
    const systemPrompt = `You are a medical assistant for doctors.
Rules:
1) Always answer in Russian.
2) You can provide practical medical workflow advice and differential suggestions.
3) If the user asks about specific facts not provided, explicitly say what data is missing.
4) Do not claim a definitive diagnosis without sufficient evidence.
5) Keep answers clear and clinically useful.`;

    const historyText = history
      .slice(-8)
      .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.text}`)
      .join('\n');

    const contextBlock = document
      ? `Current case document (JSON):\n${JSON.stringify(document, null, 2)}\n\n`
      : '';

    const userPrompt = `${contextBlock}${historyText ? `Conversation history:\n${historyText}\n\n` : ''}Question:\n${question}\n\nAnswer in Russian plain text.`;

    if (this.anthropic) {
      return this.anthropic.completeText({
        systemPrompt,
        userPrompt,
        maxTokens: 2048,
        temperature: 0.4,
        operation: 'chat',
      });
    }

    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: this.buildCompletionBody({
        prompt: `<|im_start|>system\n/no_think\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: 1024,
        // 0.4 — для диалога нужна вариативность, но в рамках клинической точности
        temperature: 0.4,
        stop: ['<|im_end|>'],
        stream: false,
      }),
    });

    if (!response.ok) {
      await this.throwLlmError(response, 'chat');
    }

    const data = (await response.json()) as LlamaCompletionResponse;
    return this.stripThinkingBlocks(data.content);
  }

  async rewriteDocumentField(field: RewriteableField, text: string): Promise<string> {
    const normalized = text.trim();
    if (!normalized) return '';

    const fieldLabel = this.getFieldLabel(field);
    const systemPrompt = `You are a medical editor.
Rules:
1) Keep original medical facts unchanged.
2) Fix grammar, punctuation, and obvious speech-to-text mistakes.
3) Keep concise clinical style.
4) Output plain Russian text only, no markdown, no explanations.`;

    const userPrompt = `Rewrite this section "${fieldLabel}" in Russian.
Do not add new facts.

Text:
${normalized}`;

    if (this.anthropic) {
      return this.anthropic.completeText({
        systemPrompt,
        userPrompt,
        maxTokens: 768,
        temperature: 0,
        operation: 'rewriteField',
      });
    }

    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: this.buildCompletionBody({
        prompt: `<|im_start|>system\n/no_think\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: Math.min(384, this.config.maxTokens),
        temperature: 0,
        stop: ['<|im_end|>'],
        stream: false,
      }),
    });

    if (!response.ok) {
      await this.throwLlmError(response, 'rewriteField');
    }

    const data = (await response.json()) as LlamaCompletionResponse;
    return this.stripThinkingBlocks(data.content);
  }

  private stripThinkingBlocks(text: unknown): string {
    const str = typeof text === 'string' ? text : String(text ?? '');
    return str
      .replace(/<think>[\s\S]*?<\/think>/gi, '')
      .replace(/<\/?think>/gi, '')
      .trim();
  }

  private async parsePatchWithRepair(content: string): Promise<MedicalDocumentPatch> {
    try {
      return this.parseLlmJson<MedicalDocumentPatch>(content);
    } catch {
      const repaired = await this.repairJsonWithLlm(content, this.getAddendumPatchJsonSchema());
      return this.parseLlmJson<MedicalDocumentPatch>(repaired);
    }
  }

  private mergeDocumentWithPatch(
    base: MedicalDocument,
    patch: MedicalDocumentPatch,
    addendumText: string
  ): MedicalDocument {
    const merged: MedicalDocument = {
      ...base,
      patient: {
        ...base.patient,
      },
      riskAssessment: {
        ...base.riskAssessment,
      },
    };

    let changed = false;

    if (patch.patient && typeof patch.patient === 'object') {
      for (const key of ['fullName', 'age', 'gender', 'complaintDate'] as const) {
        const value = patch.patient[key];
        if (typeof value === 'string' && value.trim().length > 0) {
          const nextValue = value.trim();
          if (nextValue !== merged.patient[key]) {
            merged.patient[key] = nextValue;
            changed = true;
          }
        }
      }
    }

    if (patch.riskAssessment && typeof patch.riskAssessment === 'object') {
      for (const key of ['fallInLast3Months', 'dizzinessOrWeakness', 'needsEscort', 'painScore'] as const) {
        const value = patch.riskAssessment[key];
        if (typeof value === 'string' && value.trim().length > 0) {
          merged.riskAssessment[key] = value.trim();
          changed = true;
        }
      }
    }

    for (const key of ALL_TEXT_FIELDS) {
      const value = patch[key];
      if (typeof value === 'string' && value.trim().length > 0) {
        const nextValue = value.trim();
        const prevValue = merged[key];
        if (prevValue.trim().length === 0) {
          merged[key] = nextValue;
          changed = true;
        } else if (!prevValue.includes(nextValue)) {
          merged[key] = `${prevValue}\n\n${nextValue}`.trim();
          changed = true;
        }
      }
    }

    if (!changed) {
      const fallbackText = addendumText.trim();
      if (fallbackText.length > 0) {
        const direct = this.extractSectionPrefixedPayload(fallbackText);
        if (direct) {
          const prevFieldValue = merged[direct.field].trim();
          merged[direct.field] = prevFieldValue
            ? `${prevFieldValue}\n\n${direct.payload}`.trim()
            : direct.payload;
        } else {
          const prevNotes = merged.doctorNotes.trim();
          merged.doctorNotes = prevNotes
            ? `${prevNotes}\n\n${fallbackText}`.trim()
            : fallbackText;
        }
      }
    }

    return merged;
  }

  private mergeDocumentWithInstructionPatch(
    base: MedicalDocument,
    patch: MedicalDocumentPatch
  ): MedicalDocument {
    const merged: MedicalDocument = {
      ...base,
      patient: {
        ...base.patient,
      },
      riskAssessment: {
        ...base.riskAssessment,
      },
    };

    if (patch.patient && typeof patch.patient === 'object') {
      for (const key of ['fullName', 'age', 'gender', 'complaintDate'] as const) {
        const value = patch.patient[key];
        if (typeof value === 'string') {
          merged.patient[key] = value.trim();
        }
      }
    }

    if (patch.riskAssessment && typeof patch.riskAssessment === 'object') {
      for (const key of ['fallInLast3Months', 'dizzinessOrWeakness', 'needsEscort', 'painScore'] as const) {
        const value = patch.riskAssessment[key];
        if (typeof value === 'string') {
          merged.riskAssessment[key] = value.trim();
        }
      }
    }

    for (const key of ALL_TEXT_FIELDS) {
      const value = patch[key];
      if (typeof value === 'string') {
        merged[key] = value.trim();
      }
    }

    return merged;
  }

  private async fetchWithTimeout(url: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.requestTimeoutMs);

    try {
      return await fetch(url, {
        ...init,
        signal: controller.signal,
      });
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`LLM request timeout after ${this.config.requestTimeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  private async throwLlmError(response: Response, context: string): Promise<never> {
    let detail = '';
    try {
      const body = await response.text();
      if (body) {
        detail = ` — ${body}`;
        console.error(`LLM error body [${context}]:`, body);
      }
    } catch {
      // ignore body read error
    }
    throw new Error(`LLM server error [${context}]: ${response.status} ${response.statusText}${detail}`);
  }

  private getSystemPrompt(): string {
    const labRef = buildLabReferenceForPrompt();
    return `Ты — медицинский ассистент. Твоя задача: структурировать диктовку врача в JSON-документ консультации. Это официальный медицинский документ — точность важнее краткости.

═══════════════════════════════════════════════════════
ГЛАВНЫЕ ПРАВИЛА (нарушение = ошибка):
═══════════════════════════════════════════════════════
A) ПЕРЕНОСИ ДИКТОВКУ ДОСЛОВНО. Ты инструмент структурирования, а НЕ редактор. Запрещено: сокращать, пересказывать, обобщать, выкидывать "неважное". Если врач произнёс фразу — она должна попасть в JSON.
Б) ФИКСИРОВАННЫЙ ПОРЯДОК БЛОКОВ. Врач диктует разделы в строгой последовательности:
   1. Жалобы (complaints)
   2. Анамнез заболевания (anamnesis)
   3. Анамнез жизни (clinicalCourse)
   4. Аллергологический анамнез (allergyHistory)
   5. Амбулаторные обследования (outpatientExams)
   6. Объективный статус (objectiveStatus)
   7. Неврологический статус (neurologicalStatus)
   8. Предварительный/клинический диагноз (diagnosis)
   9. План обследования (doctorNotes)
   10. Рекомендации / План лечения (recommendations) — ВКЛЮЧАЕТ ДИЕТУ отдельным пунктом списка.
В) СЕГМЕНТАЦИЯ ПО ГОЛОСОВЫМ ЗАГОЛОВКАМ. Если врач произнёс название блока (например «анамнез жизни»), весь следующий текст до следующего названного блока принадлежит ИСКЛЮЧИТЕЛЬНО этому блоку. ЗАПРЕЩЕНО переносить содержимое в другой блок по «смысловым» соображениям. Пример: если в блоке «анамнез жизни» врач упоминает текущие препараты — они остаются в clinicalCourse, а НЕ перемещаются в conclusion/recommendations.
Г) ЕСЛИ ЗАГОЛОВОК НЕ ОЗВУЧЕН — определяй блок по контексту и по соседним явно названным блокам (раз следующий произнесённый заголовок «Анамнез заболевания», значит предыдущий блок был «Жалобы»).
Д) КАЖДЫЙ ФАКТ РОВНО В ОДНО ПОЛЕ. Дублирование запрещено.
Е) НЕ ПРИДУМЫВАЙ. Если данных нет — пустая строка. Никаких "Hb - , Эр - , Тр - ..." с прочерками вместо значений. НИКОГДА не вставляй шаблонные плейсхолдеры с прочерками для непродиктованных показателей.

═══════════════════════════════════════════════════════
ОПИСАНИЕ ПОЛЕЙ:
═══════════════════════════════════════════════════════
1) complaints — жалобы пациента, дословно.
2) anamnesis — анамнез заболевания: хронология текущей болезни, когда началось, как протекало, предыдущие обследования/операции в связи с этим заболеванием.
3) clinicalCourse — АНАМНЕЗ ЖИЗНИ: перенесённые болезни (туберкулёз, гепатиты, Боткина), гемотрансфузии, операции, травмы, наследственность, вредные привычки, сопутствующая патология (СД, ХБП и т.д.), диспансерный учёт, препараты принимаемые пациентом РАНЕЕ или хронически, гинекологический анамнез у женщин. ВСЁ что врач произнёс в блоке «анамнез жизни» — ТОЛЬКО сюда.
4) allergyHistory — ТОЛЬКО аллергии и непереносимость. Никаких гинекологических данных, никакого анамнеза жизни.
5) outpatientExams — результаты УЖЕ ПРОВЕДЁННЫХ обследований, нумерованный список (\\n между пунктами). Формат пункта: "N. НАЗВАНИЕ от ДД.ММ.ГГГГг.: <значения>". Разделы обследований: ОАК, Б/х, коагулограмма, гормоны, ОАМ, ЭКГ, ЭхоКГ, Холтер, СМАД, УЗИ, УЗДГ, КАГ и т.д. См. раздел «ФОРМАТ ЛАБ-ПОКАЗАТЕЛЕЙ» ниже.
6) objectiveStatus — осмотр по системам: общее состояние, кожные покровы, дыхательная, сердечно-сосудистая (АД, ЧСС, тоны), пищеварения, мочевыделения.
7) neurologicalStatus — ТОЛЬКО неврологический статус. Если не произносился — пусто.
8) diagnosis — предварительный/клинический диагноз. Если врач упомянул диагноз внутри блока диагноза — всё попадает сюда дословно.
8a) finalDiagnosis — ЗАКЛЮЧЕНИЕ / клиническое обоснование случая: нарративный абзац ПОСЛЕ диагноза, резюмирующий приём (начинается с «Пациент(ка) обратил(а)ся…», «На основании жалоб…», «Проведено клинико-анамнестическое…», «Состояние отягощено коморбидной патологией…», заканчивается «направляется на госпитализацию…» и/или «Согласовано с руководителем…»). Также сюда идёт заключительный диагноз, если произнесён отдельно. Если ни того ни другого нет — пусто.
9) conclusion — СОПУТСТВУЮЩИЙ ДИАГНОЗ (если произнесён). Если не произносился — пусто.
10) doctorNotes — ПЛАН ОБСЛЕДОВАНИЯ: направления на анализы/исследования (контроль СМАД, КАГ в плановом порядке, повторный ТТГ и т.п.). Нумерованный список.
11) recommendations — РЕКОМЕНДАЦИИ / ПЛАН ЛЕЧЕНИЯ. СТРОГИЙ ФОРМАТ:
   • ОБЯЗАТЕЛЬНО нумерованный список. Каждый пункт начинается с "N. " (1., 2., 3. ...).
   • Пункты разделяются ТОЛЬКО символом переноса строки \\n. Внутри одного пункта — НИ ОДНОГО \\n.
   • Один пункт = одна рекомендация ИЛИ один препарат СО ВСЕМИ его атрибутами: название, доза, кратность, путь, время, длительность, контроль, побочные эффекты, последствия отмены. ВСЁ это склеивается в один сплошной абзац через ", " или ". " — БЕЗ переноса строки.
   • ЗАПРЕЩЕНО выносить "Побочные эффекты: ...", "При отмене: ...", "Под контролем: ..." в отдельные пункты или отдельные строки. Они — часть пункта препарата, к которому относятся.
   • Если врач диктует немедикаментозные рекомендации (самоконтроль АД, физ. активность, плавание, повторный осмотр) — каждая такая рекомендация тоже ОТДЕЛЬНЫЙ нумерованный пункт в одну строку.
   • ДИЕТА — ВСЕГДА отдельный пункт внутри recommendations. Если врач назвал номер стола («Стол №9», «Диета №10») — пункт содержит ТОЛЬКО номер/название, без расшифровки (шаблон раскроется автоматически). Если врач продиктовал конкретные ограничения (гипохолестериновая, гипонатриевая, ограничение соли до 5 г) — оформляй полностью как пункт списка.
   ПРАВИЛЬНЫЙ ПРИМЕР (обрати внимание: никаких \\n внутри пункта):
   "1. Самоконтроль АД утром и вечером с ведением дневника.\\n2. Дозированное увеличение физической активности, регулярная аэробная нагрузка (ходьба, бег трусцой, велосипед, плавание) не менее 30-40 минут в день, возможен дробный режим по 10-15 минут несколько раз в день.\\n3. Плавание 2-3 раза в неделю.\\n4. Таблетка Кадиован 80/12,5 мг — по 1 таблетке 1 раз в день внутрь утром до завтрака, длительно, под контролем АД, креатинина и калия крови. Побочные эффекты: избыточное снижение АД, головокружения, электролитные нарушения. При отмене препарата возможно повторное повышение АД.\\n5. Капсула Хептера — по 1 капсуле 1 раз в день внутрь во время еды, 4 недели, после завершения курса начать Урсосан, под контролем переносимости и биохимии печени. Побочные эффекты: дискомфорт в животе, тошнота, редко аллергические реакции. При отмене — замедление регресса жировой болезни печени.\\n6. Капсула Урсосан 500 мг — по 2 капсулы внутрь перед сном в течение 3 месяцев, далее контрольное УЗИ ОБП в динамике, под контролем биохимии печени. Побочные эффекты: послабление стула, дискомфорт в животе, редко аллергические реакции. При отмене — сохранение билиарного сладжа и застойных явлений в желчном пузыре."
   НЕПРАВИЛЬНО (так делать НЕЛЬЗЯ):
   "Таблетка Кадиован 80/12,5 мг — по 1 таблетке 1 раз в день утром.\\nПобочные эффекты: ...\\nПри отмене: ..." — здесь атрибуты одного препарата разорваны на 3 строки, нет нумерации.

═══════════════════════════════════════════════════════
ФОРМАТ ЛАБ-ПОКАЗАТЕЛЕЙ (КРИТИЧНО):
═══════════════════════════════════════════════════════
• Формат каждого показателя: "Полное название (ИНДЕКС) значение единица".
  Пример: "Гемоглобин (HGB) 142,00 г/л", "Тромбоциты (PLT) 260 10⁹/л", "Креатинин (CREA) 62,6 мкмоль/л".
• Пиши ТОЛЬКО те показатели, которые врач реально продиктовал. Если продиктовано 3 показателя — в JSON 3 показателя. НИКАКИХ прочерков "Hb - , Эр - , Тр - " для недиктованных.
• Если врач сказал только индекс («tr 260») — всё равно разверни до полного формата: "Тромбоциты (PLT) 260 10⁹/л". Справочник ниже поможет подставить полное название, индекс и единицу.
• Норма в скобках ТОЛЬКО если значение ВНЕ диапазона нормы. Формат: "Гемоглобин (HGB) 118,00 г/л (136 - 169)". Если значение в норме — норму не пиши.
• Определение пола для гендер-зависимых норм: берём из явной диктовки пола врачом; если пол не назван — анализируем контекст (склонения «пациент/пациентка», упоминание гинекологического анамнеза → женский; упоминание простаты/предстательной → мужской).
• Дата каждого обследования в формате "от ДД.ММ.ГГГГг." если врач назвал дату.
• Каждый тип обследования на каждую дату — только ОДИН РАЗ (не дублировать, даже если врач повторил).
• outpatientExams содержит ТОЛЬКО выполненные обследования. Направления типа «сделать КАГ», «контроль ТТГ» → doctorNotes.

СПРАВОЧНИК ЛАБ-ПОКАЗАТЕЛЕЙ (полное название, индекс, единица, норма):
${labRef}

═══════════════════════════════════════════════════════
ПРАВИЛА ДЛЯ РЕКОМЕНДАЦИЙ И ПРЕПАРАТОВ:
═══════════════════════════════════════════════════════
• Сохраняй формулировки врача ДОСЛОВНО: дозы, кратность, путь, длительность, контроль, побочные эффекты, последствия отмены.
• Название препарата пиши так, как есть у врача, но исправляй явные искажения распознавания: Пристилол → Престилол, Перендаприл → периндоприл, нальпазо → Нольпаза, пантопросол → пантопразол, Коронерография → Коронарография, Слофаст → slow-fast, диостиум → Dostium, ПОП LED → LAD.
• Если рекомендация упомянута ВНУТРИ блока обследований (например, «под контролем ФГДС не реже 1 раза в год») — НЕ вырывай её из контекста, оставляй в outpatientExams/recommendations там, где произнёс врач.
• Диета — всегда внутри recommendations отдельным пунктом. Отдельного поля diet нет.

═══════════════════════════════════════════════════════
ФОРМАТИРОВАНИЕ И КОРРЕКТУРА:
═══════════════════════════════════════════════════════
• Исправляй грамматику, пунктуацию, опечатки распознавания. Пробел после знаков препинания, заглавная после точки, никакой двойной пунктуации.
• Убирай слова-паразиты (ну, вот, значит, эээ), голосовые команды («точка», «запятая», «скобка открывается»).
• "мм рт.ст." сокращённо. ИМТ вместо "индекс массы тела", АД вместо "артериальное давление".
• Десятичные через запятую ("34 и 2" → "34,2"). Вес в кг, рост в см.
• Римские цифры для степени/стадии/класса/ФК: «третьей степени» → «III степени», «ФК 2 NYHA» → «ФК II (NYHA)». Арабские для типа/риска/баллов (СД 2 типа, риск 4).
• Даты обследований: "от ДД.ММ.ГГГГг." — "19 января 26 года" → "от 19.01.2026г.". В анамнезе сохраняй формулировку врача ("в 2009 году", "6 лет назад").
• Ничего не дописывай от себя. Пустое поле лучше выдуманного.

Верни ТОЛЬКО JSON, без дополнительного текста.`;
  }

  private getUserPrompt(rawText: string): string {
    return `Structure the following medical dictation into a consultation document.
IMPORTANT: Preserve ALL dictated text. Do NOT omit any details. Each fact goes to exactly ONE field — no duplication.

TEXT:
${rawText}

Return STRICT JSON (use empty strings if data is missing):
{
  "patient": {
    "fullName": "ФИО пациента или пусто",
    "age": "Возраст, например 45 лет, или пусто",
    "gender": "мужской | женский | пусто",
    "complaintDate": "YYYY-MM-DD или пусто"
  },
  "riskAssessment": {
    "fallInLast3Months": "да | нет (по шкале Морзе: падал ли в последние 3 месяца)",
    "dizzinessOrWeakness": "да | нет (головокружение или слабость на момент осмотра)",
    "needsEscort": "да | нет (нужно ли сопровождение)",
    "painScore": "число от 0 до 10 (оценка боли в баллах, по умолчанию 0)"
  },
  "complaints": "Жалобы пациента",
  "anamnesis": "Анамнез заболевания (история текущей болезни, хронология)",
  "outpatientExams": "Данные имеющихся дополнительных исследований — НУМЕРОВАННЫЙ СПИСОК. ВСЕ продиктованные показатели без пропусков. Формат: 1. ОАК от дата: Hb - значение г/л, Эр - значение *10¹²/л, Л - значение *10⁹/л, Тр - значение *10⁹/л, СОЭ - значение мм/ч. 2. Б/х анализ крови от дата: креатинин, глюкоза, АЛТ, АСТ, билирубин, ХС общий, ХС ЛПНП, ТГ, СРБ — ВСЁ что продиктовано. 3. ОАМ: отн. плотность, белок, Л в п/з, Эр в п/з.",
  "clinicalCourse": "Анамнез жизни (перенесённые заболевания: туберкулёз, гепатиты, операции, травмы, наследственность, вредные привычки, сопутствующая патология, препараты которые принимал РАНЕЕ)",
  "allergyHistory": "Аллергологический анамнез (непереносимость препаратов, пищевых продуктов)",
  "objectiveStatus": "Объективный статус — структурировать по системам органов: Общие данные (вес, рост, ИМТ, состояние, кожа, слизистые). Система органов дыхания (ЧДД, перкуссия, аускультация). Сердечно-сосудистая система (тоны, АД, ЧСС). Система органов пищеварения (язык, живот, печень, стул). Система мочевыделения (симптом поколачивания, мочеиспускание).",
  "neurologicalStatus": "Неврологический статус",
  "diagnosis": "Предварительный диагноз (с кодом МКБ-10 если озвучен)",
  "finalDiagnosis": "Заключение — клиническое обоснование / резюме случая: нарративный абзац после диагноза (начинается обычно с «Пациент(ка) обратил(а)ся…», «На основании…», «Проведено клинико-анамнестическое…», «Состояние отягощено…», заканчивается «направляется на госпитализацию…», «Согласовано с…»). Если нет — пусто.",
  "conclusion": "Амбулаторная терапия (ТОЛЬКО препараты которые пациент СЕЙЧАС принимает амбулаторно — НУМЕРОВАННЫЙ СПИСОК)",
  "doctorNotes": "План обследования (ТОЛЬКО лабораторные анализы и инструментальные исследования: КНТЖ, ТТГ, Холтер, СМАД и т.д.)",
  "recommendations": "Рекомендации / План лечения (ВСЕ рекомендации: препараты, питание, физ. активность, консультации специалистов, повторный осмотр — НУМЕРОВАННЫЙ СПИСОК). ДИЕТА — один из пунктов этого списка."
}

JSON:`;
  }

  private normalizeYesNo(value: string): string {
    const v = (value || '').trim().toLowerCase();
    if (/^(да|yes|true|1)$/i.test(v)) return 'да';
    return 'нет';
  }

  private normalizePainScore(value: string): string {
    const v = (value || '').trim().replace(/[бb]/gi, '');
    const num = parseInt(v, 10);
    if (isNaN(num) || num < 0) return '0';
    if (num > 10) return '10';
    return String(num);
  }

  private validateRiskAssessment(ra: Partial<RiskAssessment> | undefined): RiskAssessment {
    return {
      fallInLast3Months: this.normalizeYesNo(ra?.fallInLast3Months || ''),
      dizzinessOrWeakness: this.normalizeYesNo(ra?.dizzinessOrWeakness || ''),
      needsEscort: this.normalizeYesNo(ra?.needsEscort || ''),
      painScore: this.normalizePainScore(ra?.painScore || '0'),
    };
  }

  private unescapeLiteralNewlines(v: string): string {
    if (!v) return '';
    // Claude иногда возвращает литеральные "\n" (два символа) вместо переноса.
    // Также встречаются "\\n", "\r\n". Приводим всё к настоящему \n.
    return v
      .replace(/\\r\\n/g, '\n')
      .replace(/\\n/g, '\n')
      .replace(/\r\n/g, '\n');
  }

  private validateAndCleanDocument(doc: MedicalDocument): MedicalDocument {
    const rawAge = doc.patient?.age || '';
    const rawGender = doc.patient?.gender || '';
    // Разэкранируем литеральные "\n" во всех текстовых полях до любой обработки
    for (const f of ALL_TEXT_FIELDS) {
      if (typeof (doc as any)[f] === 'string') {
        (doc as any)[f] = this.unescapeLiteralNewlines((doc as any)[f]);
      }
    }
    const result: MedicalDocument = {
      patient: {
        fullName: doc.patient?.fullName || '',
        age: this.normalizeAge(rawAge),
        gender: this.normalizeGender(rawGender),
        complaintDate: doc.patient?.complaintDate || '',
      },
      riskAssessment: this.validateRiskAssessment(doc.riskAssessment),
      complaints: this.stripSectionPrefix('complaints', doc.complaints || ''),
      anamnesis: this.stripSectionPrefix('anamnesis', doc.anamnesis || ''),
      outpatientExams: this.config.provider === 'anthropic'
        ? this.stripSectionPrefix('outpatientExams', doc.outpatientExams || '')
        : this.postProcessOutpatientExams(this.stripSectionPrefix('outpatientExams', doc.outpatientExams || '')),
      clinicalCourse: this.stripSectionPrefix('clinicalCourse', doc.clinicalCourse || ''),
      allergyHistory: this.stripSectionPrefix('allergyHistory', doc.allergyHistory || ''),
      objectiveStatus: this.stripSectionPrefix('objectiveStatus', doc.objectiveStatus || ''),
      neurologicalStatus: this.stripSectionPrefix('neurologicalStatus', doc.neurologicalStatus || ''),
      diagnosis: this.stripSectionPrefix('diagnosis', doc.diagnosis || ''),
      finalDiagnosis: this.stripSectionPrefix('finalDiagnosis', doc.finalDiagnosis || ''),
      conclusion: this.splitInlineNumberedList(this.stripSectionPrefix('conclusion', doc.conclusion || '')),
      doctorNotes: this.formatDoctorNotesAsList(this.stripSectionPrefix('doctorNotes', doc.doctorNotes || '')),
      recommendations: this.groupRecommendations(this.splitInlineNumberedList(this.stripSectionPrefix('recommendations', doc.recommendations || ''))),
    };

    // Пост-обработка: очистка висячих кавычек (" ' „ “ » ) которые LLM иногда
    // оставляет на концах полей после обрыва фразы. Не трогаем парные кавычки.
    for (const field of ALL_TEXT_FIELDS) {
      const v = result[field];
      if (!v) continue;
      let cleaned = v;
      // Убираем одиночные висячие кавычки в конце
      cleaned = cleaned.replace(/\s*["'„“”«»]+\s*$/u, '');
      // Убираем одиночные висячие кавычки в начале (если их нет в паре)
      const quoteCount = (cleaned.match(/["'„“”«»]/gu) || []).length;
      if (quoteCount === 1) {
        cleaned = cleaned.replace(/^\s*["'„“”«»]\s*/u, '');
      }
      if (cleaned !== v) {
        result[field] = cleaned.trim();
      }
    }

    // Пост-обработка: очистка полей состоящих только из цифр/пунктуации
    for (const field of ALL_TEXT_FIELDS) {
      const v = result[field];
      if (v && /^\s*[\d\s.,;:!?-]{1,5}\s*$/.test(v)) {
        console.log(`[postprocess] Cleared garbage-only field ${field}: "${v.trim()}"`);
        result[field] = '';
      }
    }

    // Пост-обработка: вычищаем placeholder-фразы которые LLM иногда подставляет
    // в пустые поля ("Не указаны", "Нет данных" и т.п.) — должны быть пустыми строками.
    const PLACEHOLDER_RE = /^\s*(?:не\s+указан[ыоа]?|нет\s+данн[ыхое]+|отсутству[юет]+|не\s+предъявл[яет]+|не\s+выявлен[ыоа]?|не\s+отмечает|жалоб\s+нет|без\s+особенност[ейи]+|—|-)\s*\.?\s*$/iu;
    for (const field of ALL_TEXT_FIELDS) {
      const v = result[field];
      if (v && PLACEHOLDER_RE.test(v)) {
        console.log(`[postprocess] Cleared placeholder field ${field}: "${v.trim()}"`);
        result[field] = '';
      }
    }

    // Пост-обработка: удаление мусорных токенов в начале/конце полей
    this.cleanFieldGarbage(result);

    // Пост-обработка: перемещение контента с секционными маркерами в правильные поля
    this.redistributeMisplacedSections(result);

    // Пост-обработка: спасаем данные обследований из неправильных полей
    this.rescueExamData(result);

    // Пост-обработка: спасаем анамнез жизни из anamnesis (LLM часто смешивает)
    this.splitLifeHistoryFromAnamnesis(result);

    // Пост-обработка: убираем диагнозный хвост из clinicalCourse
    this.cleanDiagnosisTailFromClinicalCourse(result);

    // Пост-обработка: убираем аллергические хвосты из clinicalCourse
    this.cleanAllergyTailFromClinicalCourse(result);

    // Пост-обработка: убираем аллергические хвосты из objectiveStatus
    this.cleanAllergyTailFromObjectiveStatus(result);

    // Пост-обработка: чистим allergyHistory от не-аллергических данных
    this.cleanAllergyHistory(result);

    // Пост-обработка: раскрытие шаблона диеты ВНУТРИ recommendations
    this.expandDietTemplateInRecommendations(result);

    // Пост-обработка: чистка diagnosis от нерелевантных данных (анамнез жизни, обследования)
    this.cleanDiagnosis(result);

    // Пост-обработка: дедупликация внутри полей (LLM иногда дублирует весь блок)
    this.deduplicateFields(result);

    // Пост-обработка: чистка recommendations от мусора Whisper
    this.cleanRecommendations(result);

    // Пост-обработка: если objectiveStatus содержит diagnosis-данные в конце, удалить
    this.cleanObjectiveStatusTail(result);

    // Пост-обработка: если finalDiagnosis совпадает с diagnosis, очистить finalDiagnosis
    if (result.finalDiagnosis && result.diagnosis &&
        result.finalDiagnosis.trim() === result.diagnosis.trim()) {
      console.log('[postprocess] finalDiagnosis is identical to diagnosis, clearing');
      result.finalDiagnosis = '';
    }

    // Пост-обработка: перемещение направлений из outpatientExams в recommendations
    if (this._rescuedReferrals.length > 0) {
      const referralText = this._rescuedReferrals.join('\n');
      const existing = result.recommendations.trim();
      result.recommendations = existing
        ? `${existing}\n${referralText}`
        : referralText;
      console.log(`[postprocess] Moved ${this._rescuedReferrals.length} referral(s) from outpatientExams → recommendations`);
      this._rescuedReferrals = [];
    }

    // Пост-обработка: повторное применение медицинского словаря
    // LLM может отменить коррекции словаря при структурировании текста
    this.reapplyMedicalDictionary(result);

    return result;
  }

  /**
   * Чистит recommendations от мусора Whisper (non-medical hallucinations).
   */
  private cleanRecommendations(doc: MedicalDocument): void {
    const reco = doc.recommendations;
    if (!reco || reco.length < 30) return;

    // Split строго по строкам — groupRecommendations уже сгруппировал препараты,
    // ломать его split-ом по точкам нельзя.
    const items = reco.split(/\n+/).filter(s => s.trim());
    const clean: string[] = [];

    // Medical recommendation keywords
    const medicalKeywords = /(?:таблетк[аи]|капсул|мг\b|раз\s+в\s+день|внутрь|длительно|постоянно|контроль|ингибитор|назначен|неназначен|диет[аы]|стол\s+№?\d|ограничен|рекомендован|направлен|обследован|консультаци|анализ|ЭКГ|ЭХО|МРТ|КТ|рентген|осмотр|повторн|бисопролол|конкор|форсига|джардинс|верошпирон|эналаприл|лизиноприл|дигоксин|ксарелто|варфарин|курс\b|терапи|лечени|приём|принимать|памятк[аи]\s+ВТЭ)/iu;
    // Historical phrases that don't belong in recommendations
    const historicalKeywords = /(?:имплантаци\S+\s+(?:ЭКС|ИКД|CRT|кардио)|терапи\S*\s+по\s+схем|обратилась?\s+для\s+решения\s+вопроса|консультирован\S*\s+(?:аритмолог|кардиолог|кардиохирург)|выписан\S*\s+на\s+терапи|проведен\S*\s+(?:оперативн|КАГ|АКШ|баллон|ангиопластик))/iu;
    // Garbage patterns. ВАЖНО: «%» — валидный медицинский символ (5% массы тела, ФВ 58%),
    // поэтому в classkey НЕ включаем.
    const garbageKeywords = /(?:видео\s+в\s+компьютер|тема\s+от\s+\d|объёмы\s+карт|сфера|фоти|превраду|настройки\s+наиболее|[&@#$])/iu;

    for (const item of items) {
      const trimmed = item.trim();
      if (!trimmed || trimmed.length < 3) continue;

      // Remove obvious garbage
      if (garbageKeywords.test(trimmed)) {
        console.log(`[postprocess] Removed garbage from recommendations: "${trimmed.substring(0, 50)}..."`);
        continue;
      }

      // Remove historical phrases (past events, not new prescriptions)
      if (historicalKeywords.test(trimmed)) {
        console.log(`[postprocess] Removed historical phrase from recommendations: "${trimmed.substring(0, 60)}..."`);
        continue;
      }

      // Remove bare "Рекомендации" header
      if (/^рекомендации\.?\s*$/iu.test(trimmed)) continue;

      // Keep if medical or numbered recommendation
      if (medicalKeywords.test(trimmed) || /^\d+\.\s/.test(trimmed)) {
        clean.push(trimmed);
      } else if (trimmed.length > 100) {
        // Long non-medical sentence — likely garbage
        console.log(`[postprocess] Removed non-medical sentence from recommendations: "${trimmed.substring(0, 50)}..."`);
      } else {
        clean.push(trimmed); // Keep short ambiguous items
      }
    }

    // Перенумеровываем всегда — на входе строки могут уже иметь номера, либо
    // быть без них. Сначала срезаем существующие номера, затем нумеруем заново.
    const renumbered = clean
      .map(l => l.replace(/^\s*\d+[\.\)]\s*/, '').trim())
      .filter(l => l.length > 0)
      .map((l, i) => `${i + 1}. ${l}`)
      .join('\n');
    if (renumbered !== reco) {
      doc.recommendations = renumbered;
      if (clean.length < items.length) {
        console.log(`[postprocess] Cleaned recommendations: ${items.length} → ${clean.length} items`);
      }
    }
  }

  /**
   * Повторно применяет медицинский словарь ко всем текстовым полям документа.
   * LLM (Qwen3-8B) часто отменяет коррекции словаря при структурировании,
   * поэтому нужно применить их снова после LLM.
   */
  /**
   * Чистит diagnosis от данных, которые не являются диагнозом:
   * - анамнез жизни (перенесённые заболевания, операции)
   * - данные обследований (ЭХО-КГ, ФВ и т.д.)
   * - текущая терапия (амбулаторно принимает...)
   * Перемещает их в правильные поля.
   */
  private cleanObjectiveStatusTail(doc: MedicalDocument): void {
    const obj = doc.objectiveStatus;
    const diag = doc.diagnosis;
    if (!obj || !diag || obj.length < 100 || diag.length < 50) return;

    // Check if ANY diagnosis sentence appears at the end of objectiveStatus
    const diagSentences = diag.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 15);
    if (diagSentences.length < 2) return;

    // Find earliest diagnosis sentence in second half of objectiveStatus
    let earliestIdx = -1;
    for (const sent of diagSentences) {
      const needle = sent.trim().substring(0, 40);
      const idx = obj.indexOf(needle);
      if (idx > 0 && idx > obj.length * 0.4) {
        if (earliestIdx < 0 || idx < earliestIdx) {
          earliestIdx = idx;
        }
      }
    }

    if (earliestIdx > 0) {
      const cleaned = obj.substring(0, earliestIdx).replace(/[.,;:\s]+$/, '').trim();
      if (cleaned.length > 50) {
        console.log(`[postprocess] Removed diagnosis tail from objectiveStatus (${obj.length} → ${cleaned.length} chars)`);
        doc.objectiveStatus = cleaned;
      }
    }
  }

  private cleanDiagnosis(doc: MedicalDocument): void {
    const diag = doc.diagnosis;
    if (!diag || diag.length < 200) return;

    // First: split off "Рекомендации ..." block if embedded in diagnosis
    const recoBlockMatch = diag.match(/\s*Рекомендации\s+([\s\S]*)$/iu);
    let diagText = diag;
    if (recoBlockMatch) {
      const recoBlock = recoBlockMatch[1].trim();
      diagText = diag.substring(0, recoBlockMatch.index!).trim();
      console.log(`[postprocess] Extracted "Рекомендации" block from diagnosis (${recoBlock.length} chars)`);

      // Весь извлечённый блок идёт в recommendations — диета теперь внутри него.
      const recoItems = recoBlock.split(/(?=\d+\.\s)/).filter(s => s.trim());
      const recoParts: string[] = recoItems.map(s => s.trim()).filter(Boolean);

      // Append reco parts to doc.recommendations
      if (recoParts.length > 0) {
        const recoText = recoParts.join('\n').trim();
        const existingReco = (doc.recommendations || '').trim();
        // Replace empty/garbage recommendations
        if (!existingReco || existingReco.length < 10) {
          doc.recommendations = recoText;
        } else if (!existingReco.includes(recoText.substring(0, Math.min(30, recoText.length)))) {
          doc.recommendations = `${existingReco}\n${recoText}`.trim();
        }
        console.log(`[postprocess] Moved recommendations from diagnosis → recommendations (${recoText.length} chars)`);
      }
    }

    const sentences = diagText.split(/(?<=[.!?])\s+/).filter(s => s.trim());
    if (sentences.length < 3) {
      doc.diagnosis = diagText;
      return;
    }

    // Patterns that should NOT be in diagnosis (checked FIRST — higher priority)
    const anamnesisLifeKeywords = /(?:перенесённ\S*\s+заболевани|перенесенн\S*\s+заболевани|туберкулез|вирусн\S+\s+гепатит|ВИЧ\s+отрицает|болезнь\s+Боткина|гемотрансфузи|наследственност|операци\S*\s+и\s+травм|паховой\s+грыж|вредн\S+\s+привычк|аллергол\S*\s+анамнез|отрицает$)/iu;
    const currentTherapyKeywords = /(?:амбулаторно\s+принимает|таблетка\s+\S+\s+\d+\s*мг)/iu;
    const examDataKeywords = /(?:по\s+данным\s+ЭХО|по\s+данным\s+ЭКГ|по\s+результатам|выполнена\s+КАГ|терапию\s+по\s+схеме|обратился\s+в?\s*связи|осмотрен\s+аритмолог|направлен\s+к?\s*кардиолог|ВП\s+\d+%|режим\s+стимуляции|проведена\s+проверка)/iu;
    // Patterns that SHOULD be in diagnosis
    const diagnosisKeywords = /(?:кардиомиопати|фибрилляци\S+\s+предсерд|тахи.?бради|состояние\s+после|имплантаци|истощение\s+батаре|кардиоресинхронизирующ|кардиовертер|атеросклероз|недостаточност|гипертензи|гипертони|стенокардия|порок\s+сердца|ХСН|ФК|NYHA|EHRA|ХВН|варикозн|стадия|функциональный\s+класс|степен[ьи]\s+риска|риск\s+сердечн)/iu;

    const diagParts: string[] = [];
    const anamnesisParts: string[] = [];
    const clinicalParts: string[] = [];
    const conclusionParts: string[] = [];

    let inNonDiagBlock = false;
    for (const sent of sentences) {
      // Check non-diagnosis patterns FIRST (they take priority)
      if (anamnesisLifeKeywords.test(sent)) {
        clinicalParts.push(sent);
        inNonDiagBlock = true;
      } else if (currentTherapyKeywords.test(sent)) {
        conclusionParts.push(sent);
        inNonDiagBlock = true;
      } else if (examDataKeywords.test(sent)) {
        anamnesisParts.push(sent);
        inNonDiagBlock = true;
      } else if (diagnosisKeywords.test(sent)) {
        diagParts.push(sent);
        inNonDiagBlock = false;
      } else if (inNonDiagBlock) {
        // Continue accumulating non-diagnosis block
        if (anamnesisLifeKeywords.test(sent) || sent.match(/отрицает/iu)) {
          clinicalParts.push(sent);
        } else {
          anamnesisParts.push(sent);
        }
      } else {
        // Default: keep in diagnosis
        diagParts.push(sent);
      }
    }

    const totalMoved = anamnesisParts.length + clinicalParts.length + conclusionParts.length;
    if (totalMoved === 0 && !recoBlockMatch) return;

    // Move to correct fields
    if (anamnesisParts.length > 0) {
      const moved = anamnesisParts.join(' ');
      const existing = doc.anamnesis.trim();
      if (!existing.includes(moved.substring(0, Math.min(50, moved.length)))) {
        doc.anamnesis = existing ? `${existing} ${moved}`.trim() : moved;
      }
      console.log(`[postprocess] Moved ${anamnesisParts.length} sentences from diagnosis → anamnesis`);
    }
    if (clinicalParts.length > 0) {
      const moved = clinicalParts.join(' ');
      const existing = doc.clinicalCourse.trim();
      if (!existing.includes(moved.substring(0, Math.min(50, moved.length)))) {
        doc.clinicalCourse = existing ? `${existing} ${moved}`.trim() : moved;
      }
      console.log(`[postprocess] Moved ${clinicalParts.length} sentences from diagnosis → clinicalCourse`);
    }
    if (conclusionParts.length > 0) {
      const moved = conclusionParts.join(' ');
      const existing = doc.conclusion.trim();
      if (!existing.includes(moved.substring(0, Math.min(50, moved.length)))) {
        doc.conclusion = existing ? `${existing} ${moved}`.trim() : moved;
      }
      console.log(`[postprocess] Moved ${conclusionParts.length} sentences from diagnosis → conclusion`);
    }

    doc.diagnosis = diagParts.join(' ').trim();
    console.log(`[postprocess] Cleaned diagnosis: ${diag.length} → ${doc.diagnosis.length} chars (moved ${totalMoved} sentences)`);
  }

  /**
   * Удаляет дублирование внутри полей.
   * LLM (Qwen3-8B) иногда дублирует весь блок текста или отдельные предложения.
   */
  private deduplicateFields(doc: MedicalDocument): void {
    for (const field of ALL_TEXT_FIELDS) {
      const value = doc[field];
      if (!value || value.length < 10) continue;

      // Strategy 1: if the field is exactly doubled (first half === second half)
      const mid = Math.floor(value.length / 2);
      const firstHalf = value.substring(0, mid).trim();
      const secondHalf = value.substring(mid).trim();
      if (firstHalf.length > 5 && firstHalf === secondHalf) {
        doc[field] = firstHalf;
        console.log(`[postprocess] Removed exact duplicate in ${field} (${value.length} → ${firstHalf.length} chars)`);
        continue;
      }

      // Strategy 1b: fuzzy duplicate — second half starts with 60%+ of first half
      // Catches cases where LLM repeats the block with minor additions at the end
      if (firstHalf.length > 100) {
        const overlap = Math.floor(firstHalf.length * 0.6);
        if (secondHalf.startsWith(firstHalf.substring(0, overlap))) {
          doc[field] = firstHalf;
          console.log(`[postprocess] Removed fuzzy duplicate in ${field} (${value.length} → ${firstHalf.length} chars)`);
          continue;
        }
      }

      // Strategy 2: sentence-level dedup
      const sentences = value.split(/(?<=[.!?])\s+/).filter(s => s.trim());
      if (sentences.length < 2) continue;

      const seen = new Set<string>();
      const unique: string[] = [];
      for (const sent of sentences) {
        const normalized = sent.trim().toLowerCase().replace(/\s+/g, ' ');
        if (normalized.length > 3 && seen.has(normalized)) {
          console.log(`[postprocess] Removed duplicate sentence in ${field}: "${sent.substring(0, 40)}..."`);
          continue;
        }
        seen.add(normalized);
        unique.push(sent);
      }

      if (unique.length < sentences.length) {
        doc[field] = unique.join(' ').trim();
      }
    }
  }

  private reapplyMedicalDictionary(doc: MedicalDocument): void {
    for (const field of ALL_TEXT_FIELDS) {
      const value = doc[field];
      if (value && value.trim()) {
        doc[field] = applyMedicalDictionary(value);
      }
    }
  }

  /**
   * Разбивает инлайновый нумерованный список на строки.
   * "1. Бисопролол... 2. Эдарби-Кло..." → "1. Бисопролол...\n2. Эдарби-Кло..."
   */
  private splitInlineNumberedList(text: string): string {
    if (!text.trim()) return '';
    // Если уже содержит \n между номерами — не трогаем
    if (/\d+\.\s.*\n\s*\d+\./.test(text)) return text;
    // Разбиваем по номерам: "1. ... 2. ... 3. ..."
    const split = text.replace(/\s+(\d+)\.\s+/g, '\n$1. ');
    return split.trim();
  }

  /**
   * Группирует recommendations: строки с атрибутами препарата
   * ("Побочные эффекты:", "При отмене...", "Под контролем...") склеиваются
   * с предыдущим пунктом. Результат нумеруется.
   */
  private groupRecommendations(text: string): string {
    if (!text.trim()) return '';

    // Режем КАЖДУЮ строку по предложениям, чтобы достать препараты, спрятанные
    // внутри длинной строки рядом с другим пунктом.
    const sentenceSplit = (s: string) =>
      s.replace(/([.!?;])\s+(?=[А-ЯЁA-Z])/g, '$1\n').split(/\n+/);

    const rawLines = text
      .split(/\n+/)
      .flatMap(sentenceSplit)
      .map((l) => l.replace(/^\s*\d+[\.\)]\s*/, '').trim())
      .filter((l) => l.length > 0);

    if (rawLines.length === 0) return '';

    // Маркеры начала нового самостоятельного пункта:
    // 1) Лекарственная форма (Таблетка/Капсула/...)
    // 2) Знакомые рубрики режима/образа жизни.
    const drugFormRe = /^(?:таблетк|капсул|раствор|инъекц|ампул|сироп|свеч|мазь|крем|гель|порошок|суспенз|спрей|ингалятор|пластыр|драже|флакон|саше|капл)/iu;
    const lifestyleStartRe = /^(?:гипонатриев|гипохолестерин|сбалансированн|низкокалорийн|нормализаци|прекращени|постепенн|дозированн|регулярн|самоконтрол|контрол(?:ь|ировать)|вестибулярн|повторн|консультаци|диет|питани|физическ|увеличени|ограничени|памятк|пациент(?:ка|у)?\s|рекомендован|плавани|ходьб|массаж|гимнастик)/iu;
    // Явные продолжения предыдущего пункта.
    const continuationRe = /^(?:побочны[йе]\s+эффект|при\s+отмене|под\s+контролем|длительно|постоянно|в\s+течение\s+\d|после\s+завершения|далее\s+контрольн|по\s+(?:одн|дв|тр|пол|\d)|утром|вечером|на\s+ночь|до\s+(?:еды|завтрак)|после\s+(?:еды|ужин|завтрак)|во\s+время\s+ед)/iu;

    // Препарат без слова «Таблетка»: «Индапамид + тельмисартан 2,5 мг», «Бетасерк 24 мг».
    // Эвристика: начинается с заглавной кириллической/латинской буквы И содержит
    // дозу (число + мг|мкг|г|мл|МЕ|IU|ЕД) в первых ~80 символах.
    const drugByNameRe = /^[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z\-]+.{0,80}?\d+[.,]?\d*\s*(?:мг|мкг|г|мл|МЕ|IU|ЕД)\b/u;
    // Торговое название без единицы измерения в той же фразе:
    // «Индапамид плюс тельмисартан», «Илпио 2,5 на 80», «Ко-Диован 80/12,5».
    // Триггеры: начинается с заглавной (>=5 симв в корне), далее число / «плюс Имя» / «+» / «(...)».
    const drugTradeNameRe = /^[А-ЯЁA-Z][а-яёa-z\-]{4,}(?:\s+\d+(?:[.,]\d+)?|\s+плюс\s+[а-яёa-zА-ЯЁA-Z]|\s*\+\s*[А-ЯЁA-Zа-яёa-z]|\s*\([^)]{1,25}\))/u;

    const grouped: string[] = [];
    for (const line of rawLines) {
      const isContinuation = continuationRe.test(line);
      const startsNewItem = !isContinuation && (
        drugFormRe.test(line) ||
        lifestyleStartRe.test(line) ||
        drugByNameRe.test(line) ||
        drugTradeNameRe.test(line)
      );
      if (grouped.length === 0 || startsNewItem) {
        grouped.push(line);
      } else {
        const prev = grouped[grouped.length - 1];
        grouped[grouped.length - 1] = prev.replace(/\.\s*$/, '') + '. ' + line;
      }
    }

    return grouped.map((l, i) => `${i + 1}. ${l}`).join('\n');
  }

  /**
   * Форматирует план обследования (doctorNotes) как нумерованный список.
   * Разбивает текст через точку на отдельные пункты, каждый на новой строке.
   */
  private formatDoctorNotesAsList(text: string): string {
    if (!text.trim()) return '';

    // Если уже нумерованный список — фильтруем мусорные пункты и перенумеровываем
    if (/^\s*\d+[\.\)]/m.test(text)) {
      const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
      const items = lines.map(l => l.replace(/^\s*\d+[\.\)]\s*/, '').trim()).filter(l => l.length > 0);
      const cleaned = items.filter(item => !this.isGarbageItem(item));
      if (cleaned.length === 0) return text;
      return cleaned.map((s, i) => `${i + 1}. ${s}`).join('\n');
    }

    // Ключевые слова обследований для разбиения
    const examKeywords = /(?:^|(?<=\.\s*))(?=(?:общий\s+анализ|моча\s+на|рентген|узи|холтер|эхокардио|эхокг|уздг|уздс|мрт|кт|мскт|фгдс|ээг|экг|коагулограм|коронарограф|консультаци|направлени|каг|смад|анализ\s+крови|биохим|б\/х|оак|оам|кровь\s+на|гормон|ттг|липидн|д-димер|тропонин|bnp|nt-pro))/giu;

    // Пробуем разбить по ключевым словам обследований
    const parts: string[] = [];
    let lastIndex = 0;
    const matches = [...text.matchAll(examKeywords)];

    if (matches.length >= 2) {
      for (const match of matches) {
        if (match.index! > lastIndex) {
          const chunk = text.slice(lastIndex, match.index!).replace(/[.\s]+$/, '').trim();
          if (chunk) parts.push(chunk);
        }
        lastIndex = match.index!;
      }
      const last = text.slice(lastIndex).replace(/[.\s]+$/, '').trim();
      if (last) parts.push(last);
    }

    if (parts.length < 2) {
      // Fallback: разбиваем по точкам, где после точки идёт заглавная буква
      const sentences = text.split(/\.\s+(?=[А-ЯA-Z])/).map(s => s.replace(/\.$/, '').trim()).filter(s => s.length > 0);
      if (sentences.length >= 2) {
        return this.mergeAndNumberItems(sentences);
      }
      return text;
    }

    return this.mergeAndNumberItems(parts);
  }

  /**
   * Склеивает пункты-продолжения (начинающиеся с маленькой буквы или предлога)
   * с предыдущим пунктом, затем нумерует.
   */
  private mergeAndNumberItems(items: string[]): string {
    const merged: string[] = [];
    for (const item of items) {
      // Если начинается с маленькой буквы или фразы-продолжения — склеиваем
      const isContinuation = /^[а-яёa-z]/.test(item)
        || /^(?:На\s+предмет|По\s+поводу|Для\s+(?:исключения|оценки|уточнения)|С\s+целью|В\s+(?:целях|связи)|При\s+необходимости)/u.test(item);
      if (merged.length > 0 && isContinuation) {
        merged[merged.length - 1] += '. ' + item;
      } else {
        merged.push(item);
      }
    }
    const cleaned = merged.filter(item => !this.isGarbageItem(item));
    if (cleaned.length === 0) return merged.map((s, i) => `${i + 1}. ${s}`).join('\n');
    return cleaned.map((s, i) => `${i + 1}. ${s}`).join('\n');
  }

  /**
   * Определяет, является ли пункт плана мусором Whisper.
   * Мусор: короткие бессмысленные фразы, латинские слова не из медицины,
   * бытовые фразы без медицинского контекста.
   */
  private isGarbageItem(item: string): boolean {
    const trimmed = item.replace(/\.\s*\d*\s*$/, '').trim();

    // Слишком короткий текст без медицинских аббревиатур (1-2 слова, < 15 символов)
    const medAbbreviations = /(?:АД|ЧСС|ЭКГ|ЭхоКГ|ОАК|ОАМ|БАК|МРТ|КТ|УЗИ|СМАД|ХМЭКГ|КАГ|ОКС|ФВ|АПФ|ЛПНП|ЛПВП|NT-proBNP|BNP|HbA1c|СКФ|МНО|ТТГ|СОЭ)/i;

    // Содержит латинские слова, не являющиеся медицинскими терминами
    const nonMedLatin = /\b(?:shovel|disease|continue|hello|world|nice|make|done|please|sorry|thanks|good|like|just|right|every|where)\b/i;
    if (nonMedLatin.test(trimmed)) return true;

    // Бытовые/бессмысленные фразы (Whisper галлюцинации и неформальная речь врача)
    const garbagePhrases = /(?:делай\s+стрижк|подушечк|прикинь|избежать\s+опыт|всех\s+сторон|стрижка|подушка|пусть\s+у\s+вас|мы\s+вам\s+(?:всегда\s+)?благодарн|после\s+курса\s+на\s+культур|всё\s+в\s+порядке|все\s+в\s+порядке)/iu;
    if (garbagePhrases.test(trimmed)) return true;

    // Пункт из одного слова без медицинской аббревиатуры — подозрительный
    const words = trimmed.split(/\s+/);
    if (words.length === 1 && trimmed.length < 10 && !medAbbreviations.test(trimmed)) {
      // Разрешаем известные медицинские однословные пункты
      const allowedSingle = /^(?:Пульс|пульс|ЭКГ|ОАК|ОАМ|БАК|СМАД|ХМЭКГ|КАГ|МРТ|КТ|УЗИ|коагулограмма)$/i;
      if (!allowedSingle.test(trimmed)) return true;
    }

    return false;
  }

  /**
   * Пост-обработка поля outpatientExams:
   * - Распознаёт названия обследований (ОАК, Б/х, ОАМ, ЭКГ и т.д.)
   * - Подставляет единицы измерения для параметров, если LLM их пропустила
   * - Форматирует нумерованным списком
   */
  private postProcessOutpatientExams(text: string): string {
    if (!text.trim()) return '';

    // Разбиваем на строки/элементы списка
    const lines = text
      .split(/\n/)
      .map((l) => l.replace(/^\s*\d+[\.\)]\s*/, '').trim()) // убираем существующую нумерацию
      .filter((l) => l.length > 0)
      // Убираем строки, которые выглядят как рекомендации (ошибка LLM: попали не в то поле)
      .filter((l) => {
        const isRecommendationBullet = /^[-–•]\s+.{10,}/u.test(l) &&
          /провест|назначит|рекоменд|оценит|исключит|направит|контрол|наблюден|соблюда|избегат|проверит|получит|принима|лечени|терапи|обследовани/iu.test(l);
        return !isRecommendationBullet;
      })
      // Убираем направления (без даты и без значений — это назначения, не результаты)
      .filter((l) => {
        // Если строка содержит "УЗИ/маммография/консультация" БЕЗ даты "от DD.MM" — это направление
        const isReferral = /(?:УЗИ|маммограф|консультаци|направлени|скрининг)/iu.test(l) &&
          !/от\s+\d{2}[./]\d{2}/iu.test(l) &&
          !/\d{2}\.\d{2}\.\d{4}/u.test(l);
        if (isReferral) {
          // Сохраняем для перемещения в recommendations (будет добавлено ниже)
          this._rescuedReferrals = this._rescuedReferrals || [];
          this._rescuedReferrals.push(l);
        }
        return !isReferral;
      });

    if (lines.length === 0) return text;

    const processed = lines.map((line) => {
      // Пытаемся найти шаблон обследования по строке
      const template = findExamTemplate(line);
      if (!template || template.parameters.length === 0) {
        return line; // обследования без параметров (ЭКГ, УЗИ) оставляем как есть
      }

      // Извлекаем дату если есть (оба формата: цифры и текстовый месяц)
      const date = parseExamDate(line)
        || line.match(/от\s+([\d_.]+\s*(?:г\.?|года?)?)/iu)?.[1]
        || undefined;

      // Извлекаем значения параметров из текста
      const values: Record<string, string> = {};
      for (const param of template.parameters) {
        // Ищем паттерн: "paramName - value" или "paramName value"
        const escapedName = param.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const valuePattern = new RegExp(
          `${escapedName}\\s*[-–—:]\\s*([\\d.,]+)`,
          'iu'
        );
        const match = line.match(valuePattern);
        if (match) {
          values[param.name] = match[1].replace(/[,.]$/, ''); // strip trailing comma/dot
        }
      }

      // Если по сокращённым именам ничего не нашли — пробуем голосовые алиасы
      if (Object.keys(values).length === 0) {
        const voiceValues = parseExamValuesFromText(template.id, line);
        Object.assign(values, voiceValues);
      }

      return formatExamLine(template, values, date);
    });

    // Чистим артефакты форматирования (двойные/тройные двоеточия, точка-двоеточие)
    const cleaned = processed.map(line =>
      line.replace(/[:]{2,}/g, ':').replace(/\.\s*:/g, ':')
    );

    // Нумеруем
    return cleaned.map((line, i) => `${i + 1}. ${line}`).join('\n');
  }

  /**
   * Ищет в текстовых полях секционные маркеры, попавшие не в то поле.
   * Например, если в complaints есть "Анамнез жизни: ...", перемещает этот фрагмент в clinicalCourse.
   * Мутирует объект doc напрямую.
   */
  private redistributeMisplacedSections(doc: MedicalDocument): void {
    // Маркеры секций: regex → целевое поле
    const sectionMarkers: Array<{ pattern: RegExp; target: RewriteableField }> = [
      { pattern: /анамнез\s+жизни\s*[:.,-]?\s*/iu, target: 'clinicalCourse' },
      { pattern: /перенесённ\S*\s+заболевани\S*\s*[:.,-]?\s*/iu, target: 'clinicalCourse' },
      { pattern: /анамнез\s+заболевания\s*[:.,-]?\s*/iu, target: 'anamnesis' },
      { pattern: /аллерг\S*\s+анамнез\S*\s*[:.,-]?\s*/iu, target: 'allergyHistory' },
      { pattern: /объектив\S*\s+статус\S*\s*[:.,-]?\s*/iu, target: 'objectiveStatus' },
      { pattern: /данны\S*\s+объективн\S*\s+(?:исследовани|осмотр)\S*\s*[:.,-]?\s*/iu, target: 'objectiveStatus' },
      { pattern: /амбулаторн\S*\s+терапи\S*\s*[:.,-]?\s*/iu, target: 'conclusion' },
      { pattern: /амбулаторно\s+принимает\s*[:.,-]?\s*/iu, target: 'conclusion' },
      { pattern: /(?:рекомендаци\S*|рекомендован\S*|план\s+лечени\S*|диет\S*\s*(?:№?\s*\d+\S*)?)\s*[:.,-]?\s*/iu, target: 'recommendations' },
      { pattern: /(?:предварительн\S*\s+)?диагноз\S*\s*[:.,-]?\s*/iu, target: 'diagnosis' },
      { pattern: /план\s+обследовани\S*\s*[:.,-]?\s*/iu, target: 'doctorNotes' },
    ];

    // Многократно перебираем поля пока есть перемещения (маркеров может быть несколько)
    let moved = true;
    let iterations = 0;
    while (moved && iterations < 10) {
      moved = false;
      iterations++;

      for (const field of ALL_TEXT_FIELDS) {
        const text = doc[field];
        if (!text) continue;

        for (const { pattern, target } of sectionMarkers) {
          if (target === field) continue;

          // Ищем маркер: в начале строки ИЛИ после точки/двоеточия внутри текста
          const searchPattern = new RegExp(`(?:^|\\n|[.!?,;]\\s+)\\s*(${pattern.source})`, 'iu');
          const searchMatch = text.match(searchPattern);
          if (!searchMatch) continue;

          const markerIdx = text.indexOf(searchMatch[0]);
          if (markerIdx < 0) continue;

          // Определяем где кончается "до маркера" (без точки/перевода строки)
          const rawBefore = text.substring(0, markerIdx);
          const beforeMarker = rawBefore.trim();
          const fromMarker = text.substring(markerIdx).replace(/^[.!?\s]+/, '').trim();

          // Убираем сам маркер из перемещаемого текста
          const cleanedFragment = fromMarker.replace(pattern, '').trim();
          if (!cleanedFragment) continue;

          doc[field] = beforeMarker;
          const existing = doc[target].trim();
          doc[target] = existing
            ? `${existing}\n\n${cleanedFragment}`.trim()
            : cleanedFragment;

          console.log(`[postprocess] Moved "${cleanedFragment.substring(0, 50)}..." from ${field} → ${target}`);
          moved = true;
          break; // перезапускаем цикл чтобы работать с обновлённым текстом
        }
      }
    }
  }

  /**
   * Спасает данные обследований (ОАК, Б/х, ЭКГ, ЭхоКГ, etc.) из неправильных полей
   * и перемещает их в outpatientExams.
   */
  private rescueExamData(doc: MedicalDocument): void {
    // Паттерны для обнаружения данных обследований.
    // Включает стандартные и расширенные анализы.
    const examPatterns = [
      // ── Стандартные лабораторные ──
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ОАК|общий анализ крови)\s+(?:от\s+)?[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:Б\/х|биохимия|биохимический анализ)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ОАМ|общий анализ мочи|анализ мочи)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:коагулограмма|гемостазиограмма)\s+[^\n]*/gimu,

      // ── Инструментальные ──
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ЭКГ|электрокардиограмма)\s+(?:от\s+)?[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ЭхоКГ|ЭХОКГ|эхокардиография)\s+(?:от\s+)?[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ХМЭКГ|холтер\S*)\s+(?:от\s+)?[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ЧПЭхоКГ)\s+(?:от\s+)?[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:СМАД)\s+(?:от\s+)?[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:УЗДГ|УЗДС|дуплекс)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:УЗИ)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:рентген\S*)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:МРТ|КТ|МСКТ)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ФГДС|гастроскопия|фиброгастроскопия)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ЭЭГ|электроэнцефалограмма)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:проверка ЭКС)\s+[^\n]*/gimu,

      // ── Расширенные лабораторные ──
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:тропонин\S*)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:D-димер|д-димер|Д-димер)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:BNP|NT-proBNP|про-БНП)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:гормон\S*\s+щитовидной|тиреоидн\S*\s+профиль)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:липидн\S*\s+(?:профиль|спектр)|липидограмма)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:гликированный гемоглобин|HbA1c|гликогемоглобин)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:СРБ|С-реактивный белок)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:МНО|АЧТВ|ПТИ|протромбин\S*)\s+(?:от\s+|[-–—]\s*\d)[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:КФК|КФК-МБ|ЛДГ|миоглобин)\s+(?:от\s+|[-–—]\s*\d)[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ферритин|железо\s+сыворот\S*|КНТЖ|трансферрин)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:мочевая кислота|мочевина)\s+(?:от\s+|[-–—]\s*\d)[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:иммунограмма|иммунологическ\S*)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:посев\S*|бак\.\s*посев|бакпосев)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:ПСА|PSA)\s+[^\n]*/gimu,
      /(?:^|\n)\s*(?:\d+\.\s*)?(?:анализ\s+крови\s+на\s+\S+)\s+[^\n]*/gimu,

      // ── Универсальный: "НазваниеАнализа от ДД.ММ.ГГГГ: параметр - значение ед.изм."
      /(?:^|\n)\s*(?:\d+\.\s*)?\S+\s+от\s+\d{1,2}[.,]\d{2}[.,]\d{2,4}\S*\s*:?\s*\S+\s*[-–—]\s*[\d.,]+\s*(?:г\/л|ммоль\/л|мкмоль\/л|мл\/мин|мм\/ч|МЕ\/л|мкг|мг|%|нг\/мл|пмоль\/л|мкМЕ\/мл)[^\n]*/gimu,
    ];

    // Поля из которых спасаем (не ищем в outpatientExams и doctorNotes)
    const sourceFields: RewriteableField[] = [
      'complaints', 'anamnesis', 'clinicalCourse', 'objectiveStatus',
      'conclusion', 'recommendations', 'diagnosis',
    ];

    const rescued: string[] = [];

    for (const field of sourceFields) {
      let text = doc[field];
      if (!text) continue;

      for (const pattern of examPatterns) {
        // Reset lastIndex for global patterns
        pattern.lastIndex = 0;
        const matches = text.match(pattern);
        if (!matches) continue;

        for (const match of matches) {
          const cleaned = match.replace(/^\s*\d+\.\s*/, '').trim();
          if (cleaned.length > 5) {
            rescued.push(cleaned);
            text = text.replace(match, '\n').trim();
            console.log(`[postprocess] Rescued exam data "${cleaned.substring(0, 50)}..." from ${field} → outpatientExams`);
          }
        }
      }

      doc[field] = text.replace(/\n{3,}/g, '\n\n').trim();
    }

    if (rescued.length > 0) {
      const existing = doc.outpatientExams.trim();
      const newExams = rescued.join('\n');
      const combined = existing ? `${existing}\n${newExams}` : newExams;
      // Перезапускаем постобработку чтобы перенумеровать
      doc.outpatientExams = this.postProcessOutpatientExams(combined);
    }
  }

  /**
   * Чистит allergyHistory от не-аллергических данных.
   * LLM часто сваливает в allergyHistory гинекологию, объективный статус, анамнез жизни.
   * Переносит эти данные в правильные поля.
   */
  /**
   * Убирает аллергические фразы попавшие в конец clinicalCourse.
   * LLM часто диктует анамнез жизни и сразу аллергоанамнез без паузы,
   * и они оба попадают в clinicalCourse.
   */
  /**
   * Спасает анамнез жизни из anamnesis когда LLM смешал их.
   * Признаки анамнеза жизни: туберкулёз, гепатит, операции, вредные привычки,
   * наследственность — особенно после слов "анамнез жизни" или "анамнез заболевания закончен".
   */
  /**
   * Убирает диагнозный хвост из clinicalCourse.
   * LLM иногда дописывает диагноз в конец анамнеза жизни.
   */
  private cleanDiagnosisTailFromClinicalCourse(doc: MedicalDocument): void {
    const clinical = doc.clinicalCourse;
    const diag = doc.diagnosis;
    if (!clinical || clinical.length < 100 || !diag || diag.length < 50) return;

    // Маркеры начала диагнозного блока в clinicalCourse
    const diagStartPattern = /(?:ИБС\b|ХСН\b|ХБП\b|ФК\s+[IVX]+|стадия\s+[IVX]+|артериальная\s+гипертония|сахарный\s+диабет\s+\d+\s+типа|хроническая\s+сердечная\s+недостаточность|приобретённый\s+порок|ишемическая\s+кардиомиопатия)/iu;

    const sentences = clinical.split(/(?<=[.!?])\s+/).filter(s => s.trim());
    // Only look in second half
    let cutIdx = -1;
    for (let i = Math.floor(sentences.length * 0.6); i < sentences.length; i++) {
      if (diagStartPattern.test(sentences[i])) {
        // Confirm it's really a diagnosis block: check if it overlaps with doc.diagnosis
        const candidate = sentences[i].trim().substring(0, 40);
        if (diag.includes(candidate)) {
          cutIdx = i;
          break;
        }
      }
    }

    if (cutIdx > 0) {
      const kept = sentences.slice(0, cutIdx).join(' ').trim();
      if (kept.length > 20) {
        const removed = sentences.slice(cutIdx).join(' ').trim();
        doc.clinicalCourse = kept;
        console.log(`[postprocess] Removed diagnosis tail from clinicalCourse (${removed.length} chars)`);
      }
    }
  }

  private splitLifeHistoryFromAnamnesis(doc: MedicalDocument): void {
    const anamnesis = doc.anamnesis;
    if (!anamnesis || anamnesis.length < 100 || doc.clinicalCourse.trim()) return;

    // Маркеры начала анамнеза жизни в тексте
    const lifeHistoryStartPattern = /(?:туберкулез\S*|туберкулёз\S*|гепатит[аы]?\s+и\s+|болезн\S+\s+Боткина|гемотрансфузи|вредн\S+\s+привычк|наследственност|операций\s+и\s+травм)/iu;

    const sentences = anamnesis.split(/(?<=[.!?])\s+/).filter(s => s.trim());
    let splitIdx = -1;

    for (let i = Math.floor(sentences.length / 3); i < sentences.length; i++) {
      if (lifeHistoryStartPattern.test(sentences[i])) {
        splitIdx = i;
        break;
      }
    }

    if (splitIdx > 0) {
      const anamnesisText = sentences.slice(0, splitIdx).join(' ').trim();
      const lifeHistoryText = sentences.slice(splitIdx).join(' ').trim();
      if (anamnesisText.length > 30 && lifeHistoryText.length > 20) {
        doc.anamnesis = anamnesisText;
        doc.clinicalCourse = lifeHistoryText;
        console.log(`[postprocess] Split anamnesis → clinicalCourse (${lifeHistoryText.length} chars moved)`);
      }
    }
  }

  /**
   * Убирает аллергические фразы попавшие в конец objectiveStatus.
   */
  private cleanAllergyTailFromObjectiveStatus(doc: MedicalDocument): void {
    const obj = doc.objectiveStatus;
    if (!obj || obj.length < 50) return;

    const allergyPattern = /(?:аллергоанамнез|аллергологический\s+анамнез|на\s+ингибитор(?:ы)?\s+АПФ|непереносимост\S+\s+(?:на\s+)?(?:медикамент|препарат|ингибитор)|сухой\s+кашель|аллерг\S+\s+на\s+(?:препарат|медикамент))/iu;

    // Split by sentence endings OR by capital letter after lowercase (no punctuation boundary)
    const sentences = obj.split(/(?<=[.!?])\s+|(?<=\S)\s+(?=[А-ЯЁ])/).filter(s => s.trim());
    let cutIdx = -1;
    for (let i = Math.floor(sentences.length / 2); i < sentences.length; i++) {
      if (allergyPattern.test(sentences[i])) {
        cutIdx = i;
        break;
      }
    }

    if (cutIdx > 0) {
      const kept = sentences.slice(0, cutIdx).join(' ').trim();
      const moved = sentences.slice(cutIdx).join(' ').trim();
      if (kept.length > 30) {
        doc.objectiveStatus = kept;
        const existingAllergy = (doc.allergyHistory || '').trim();
        if (!existingAllergy || existingAllergy === 'Не отягощен.') {
          doc.allergyHistory = moved;
        } else if (!existingAllergy.includes(moved.substring(0, 30))) {
          doc.allergyHistory = `${existingAllergy} ${moved}`.trim();
        }
        console.log(`[postprocess] Removed allergy tail from objectiveStatus → allergyHistory`);
      }
    }
  }

  private cleanAllergyTailFromClinicalCourse(doc: MedicalDocument): void {
    const clinical = doc.clinicalCourse;
    if (!clinical || clinical.length < 50) return;

    const allergyMarkers = /(?:аллергоанамнез|аллергологический\s+анамнез|аллерг\S*\s+на\s+|непереносимост\S+\s+(?:на\s+)?(?:медикамент|препарат|ингибитор|АПФ)|на\s+ингибитор\s+АПФ|сухой\s+кашель\s+(?:на|от)\s+|реакц\S+\s+на\s+(?:препарат|медикамент))/iu;

    const sentences = clinical.split(/(?<=[.!?])\s+/).filter(s => s.trim());
    // Find first sentence with allergy marker — remove it and everything after
    let cutIdx = -1;
    for (let i = Math.floor(sentences.length / 2); i < sentences.length; i++) {
      if (allergyMarkers.test(sentences[i])) {
        cutIdx = i;
        break;
      }
    }

    if (cutIdx > 0) {
      const kept = sentences.slice(0, cutIdx).join(' ').trim();
      const moved = sentences.slice(cutIdx).join(' ').trim();
      if (kept.length > 30) {
        doc.clinicalCourse = kept;
        // Move to allergyHistory if not already there
        const existingAllergy = (doc.allergyHistory || '').trim();
        if (!existingAllergy || existingAllergy === 'Не отягощен.') {
          doc.allergyHistory = moved;
        } else if (!existingAllergy.includes(moved.substring(0, 30))) {
          doc.allergyHistory = `${existingAllergy} ${moved}`.trim();
        }
        console.log(`[postprocess] Moved allergy tail from clinicalCourse → allergyHistory (${moved.length} chars)`);
      }
    }
  }

  private cleanAllergyHistory(doc: MedicalDocument): void {
    const allergy = doc.allergyHistory;
    if (!allergy || allergy.length < 20) return;

    // Split into sentences
    const sentences = allergy.split(/(?<=[.!?])\s+/).filter(s => s.trim());

    // Keywords that BELONG in allergyHistory
    const allergyKeywords = /аллерг|непереносим|(?:на\s+)?медикамент\S*\s+отрицает|отмечает\s+реакци|крапивниц|отёк\s+квинке|анафилак|лекарствен\S*\s+(?:аллерг|непереносим)/iu;

    // Keywords that DON'T belong in allergyHistory
    const objectiveKeywords = /(?:общее\s+состояни|обусловлено\s+сердеч|кожные\s+покров|сознание\s+ясное|послеоперацион|варикозн|сыпей|вес\s+\d|рост\s+\d|ИМТ\s+\d|пастозность|[рп]остозность|щитовидная\s+желез|дыхание|аускультативно|хрипов|тоны\s+сердца|систолическ|шум\s+на|иррадиаци|АД\s+[-–—]?\s*\d|ЧСС\s*[-–—.]?\s*\d|\d+\s*уд\/мин|\d+\/\d+\s*мм\s*рт|\d+\s+на\s+\d+\s+мм|мм\s+рт\.?\s*ст|мочеиспускани|почки|стол\s+регулярн|стул\s+регулярн|пульс\s+\d|SpO₂|живот\s+мягк|печень\s+не|не\s+увеличен|ритмичн|степен[ьи]\s+тяжести|безболезненн|справа|слева|скорость\s+\d+\s+балл)/iu;
    // Dosage fragments that leaked from clinicalCourse into allergyHistory
    const dosageGarbagePattern = /^\d+\s*мг\s+в\s+сутки|^\d+\s*мг\s+\d+\s+раз|\d+\s*мг\s+в\s+сутки,\s*аллергоанамнез/iu;
    const gynecoKeywords = /(?:беременност|родов?\b|аборт|менструаци|менопауз|перименопауз|миом[аы]\s+матки|гинекологическ)/iu;
    const lifeHistoryKeywords = /(?:перенесённ|аппендэктоми|холецистэктоми|туберкулёз|гемотрансфузи|наследственност|операци[ия]\s+и\s+травм)/iu;
    const scaleKeywords = /(?:шкала\s+для\s+оценки|CHA₂DS₂|CHADS|HAS-BLED|баллов?\s+по)/iu;

    const allergyParts: string[] = [];
    const objectiveParts: string[] = [];
    const clinicalParts: string[] = [];

    // Keywords for life history that end up in allergyHistory
    const lifeHistoryInAllergyKeywords = /(?:туберкулез|вирусн\S+\s+гепатит|ВИЧ|болезнь\s+Боткина|гемотрансфузи|наследственност|сердечно-сосудист)/iu;

    for (const sent of sentences) {
      if (dosageGarbagePattern.test(sent)) {
        // Dosage fragment leaked from clinicalCourse — discard
        console.log(`[postprocess] Removed dosage garbage from allergyHistory: "${sent.substring(0, 50)}"`);
        continue;
      }
      if (allergyKeywords.test(sent)) {
        allergyParts.push(sent);
      } else if (objectiveKeywords.test(sent) || scaleKeywords.test(sent)) {
        objectiveParts.push(sent);
      } else if (gynecoKeywords.test(sent) || lifeHistoryKeywords.test(sent) || lifeHistoryInAllergyKeywords.test(sent)) {
        clinicalParts.push(sent);
      } else if (/^[мМ][мМ]\s+рт\.?\s*ст\.?\.?\s*$/u.test(sent.trim())) {
        // Standalone "Мм рт.ст." / "мм рт.ст." — orphan Whisper garbage, discard
      } else if (allergyParts.length > 0 && sent.length < 50 && !(/\d+\/\d+|\d+\s+на\s+\d+\s+мм|[мМ][мМ]\s*рт|мм\b\.?$|уд\/мин|ЧСС|АД|мг|кг|см\b|стол\s+регулярн|стул\s+регулярн/iu.test(sent))) {
        // Short continuation of allergy text (but not measurement data)
        allergyParts.push(sent);
      } else {
        // Default: if we already have allergy text, the rest is likely misplaced
        objectiveParts.push(sent);
      }
    }

    // Only modify if we actually found misplaced data
    if (objectiveParts.length === 0 && clinicalParts.length === 0) return;

    if (objectiveParts.length > 0) {
      const moved = objectiveParts.join(' ');
      const existing = doc.objectiveStatus.trim();
      // Only add if not already present (avoid duplication)
      if (!existing.includes(moved.substring(0, 50))) {
        doc.objectiveStatus = existing ? `${existing} ${moved}`.trim() : moved;
      }
      console.log(`[postprocess] Moved ${objectiveParts.length} sentences from allergyHistory → objectiveStatus`);
    }

    if (clinicalParts.length > 0) {
      const moved = clinicalParts.join(' ');
      const existing = doc.clinicalCourse.trim();
      if (!existing.includes(moved.substring(0, 50))) {
        doc.clinicalCourse = existing ? `${existing} ${moved}`.trim() : moved;
      }
      console.log(`[postprocess] Moved ${clinicalParts.length} sentences from allergyHistory → clinicalCourse`);
    }

    // Dedup allergy sentences
    const seenAllergy = new Set<string>();
    const uniqueAllergy = allergyParts.filter(s => {
      const norm = s.trim().toLowerCase().replace(/\s+/g, ' ');
      if (seenAllergy.has(norm)) return false;
      seenAllergy.add(norm);
      return true;
    });
    doc.allergyHistory = uniqueAllergy.join(' ').trim() || 'Не отягощен.';
  }

  /**
   * Спасает данные обследований из исходного текста (raw), которые LLM мог потерять.
   * Сравнивает найденные в raw-тексте обследования с тем, что LLM вернул в outpatientExams.
   * Если обследование есть в raw но отсутствует в результате — добавляет его.
   */
  private rescueExamsFromRawText(doc: MedicalDocument, rawText: string): void {
    // Паттерны для извлечения блоков обследований из сырого текста.
    // Для анализов с параметрами (ОАК, Б/х, ОАМ) захватываем расширенный блок
    // до следующего маркера секции, чтобы не терять отдельные показатели.
    const examBlockPatterns: Array<{ templateId: string; label: string; pattern: RegExp }> = [
      // Анализы с параметрами — захватываем всё до следующего обследования/секции
      { templateId: 'oak', label: 'ОАК', pattern: /(?:ОАК|общий анализ крови|клинический анализ крови)([\s\S]*?)(?=(?:Б\/х|биохими|ОАМ|анализ мочи|ЭКГ|ЭхоКГ|ЭХОКГ|УЗДГ|УЗИ|рентген|МРТ|КТ|СМАД|холтер|ХМЭКГ|коагулограмма|ФГДС|диагноз|рекоменд|назначен|план\s+обследован|объективн|жалоб|анамнез|заключени|$))/gimu },
      { templateId: 'biochem', label: 'Б/х', pattern: /(?:Б\/х|биохимическ\S+\s+анализ\s+крови|биохимия\s+крови|биохимия)([\s\S]*?)(?=(?:ОАК|общий анализ крови|ОАМ|анализ мочи|ЭКГ|ЭхоКГ|ЭХОКГ|УЗДГ|УЗИ|рентген|МРТ|КТ|СМАД|холтер|ХМЭКГ|коагулограмма|ФГДС|диагноз|рекоменд|назначен|план\s+обследован|объективн|жалоб|анамнез|заключени|$))/gimu },
      { templateId: 'oam', label: 'ОАМ', pattern: /(?:ОАМ|общий анализ мочи|анализ мочи)([\s\S]*?)(?=(?:ОАК|общий анализ крови|Б\/х|биохими|ЭКГ|ЭхоКГ|ЭХОКГ|УЗДГ|УЗИ|рентген|МРТ|КТ|СМАД|холтер|ХМЭКГ|коагулограмма|ФГДС|диагноз|рекоменд|назначен|план\s+обследован|объективн|жалоб|анамнез|заключени|$))/gimu },
      // Обследования без параметров — короткий захват до точки
      { templateId: '', label: 'коагулограмма', pattern: /(?:коагулограмма|гемостазиограмма)[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'гликированный', pattern: /(?:гликированный гемоглобин|гликозилированный гемоглобин|HbA1c)[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: '25-ОН', pattern: /(?:25-?ОН|витамин\s*[ДD]|вит\.?\s*[ДD])\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'ЭКГ', pattern: /(?:ЭКГ|электрокардиограмма)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'ЭхоКГ', pattern: /(?:ЭхоКГ|ЭХОКГ|эхокардиография)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'ХМЭКГ', pattern: /(?:ХМЭКГ|холтер\S*\s+мониторирование|холтер)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'СМАД', pattern: /(?:СМАД)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'УЗДГ', pattern: /(?:УЗДГ|УЗДС)\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'УЗИ', pattern: /(?:УЗИ)\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'рентген', pattern: /(?:рентген\S*)\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'МРТ', pattern: /(?:МРТ|КТ|МСКТ)\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'ФГДС', pattern: /(?:ФГДС|гастроскопия)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'тропонин', pattern: /(?:тропонин\S*)\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'D-димер', pattern: /(?:D-димер|д-димер)\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'BNP', pattern: /(?:BNP|NT-proBNP|про-БНП)\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'ферритин', pattern: /(?:ферритин)\s+[^.]*?(?:\.|$)/gimu },
      { templateId: '', label: 'ПСА', pattern: /(?:ПСА|PSA)\s+[^.]*?(?:\.|$)/gimu },
    ];

    const currentExams = doc.outpatientExams.toLowerCase();
    const rescued: string[] = [];

    for (const { templateId, label, pattern } of examBlockPatterns) {
      // Проверяем: есть ли этот тип обследования уже в результате?
      if (currentExams.includes(label.toLowerCase())) continue;

      // Ищем в сыром тексте
      pattern.lastIndex = 0;
      const matches = rawText.match(pattern);
      if (!matches) continue;

      for (const match of matches) {
        const fullBlock = match.trim();
        // Минимальная валидация: должно содержать числовое значение
        if (fullBlock.length < 10 || !/\d/.test(fullBlock)) continue;

        // Для шаблонов с параметрами — парсим значения из голосового ввода
        if (templateId) {
          const template = examTemplates.find(t => t.id === templateId);
          if (template && template.parameters.length > 0) {
            const values = parseExamValuesFromText(templateId, fullBlock);
            const date = parseExamDate(fullBlock);
            const valueCount = Object.keys(values).length;

            if (valueCount > 0) {
              const formatted = formatExamLine(template, values, date);
              rescued.push(formatted);
              console.log(`[postprocess] Rescued+formatted from raw text: "${formatted.substring(0, 80)}..." (${label}, ${valueCount} params)`);
              continue;
            }
          }
        }

        // Fallback: добавляем как есть (для обследований без параметров или если парсинг не нашёл значений)
        rescued.push(fullBlock);
        console.log(`[postprocess] Rescued from raw text (raw): "${fullBlock.substring(0, 60)}..." (${label})`);
      }
    }

    if (rescued.length > 0) {
      const existing = doc.outpatientExams.trim();
      const newExams = rescued.join('\n');
      const combined = existing ? `${existing}\n${newExams}` : newExams;
      doc.outpatientExams = this.postProcessOutpatientExams(combined);
      console.log(`[postprocess] rescueExamsFromRawText: added ${rescued.length} exam(s) from raw text`);
    }
  }

  /**
   * Если в диктовке нет ни одного маркера шкалы Морзе/падений/боли/сопровождения —
   * riskAssessment должен быть пустым (дефолтные «нет/0»). Иначе Claude галлюцинирует
   * и подставляет «да» в dizzinessOrWeakness/needsEscort без каких-либо оснований.
   */
  private clearRiskAssessmentIfNotMentioned(doc: MedicalDocument, rawText: string): void {
    const text = (rawText || '').toLowerCase();
    const triggers = [
      /морзе/u,
      /\bпада(?:л|ть|ет|ю|ли|ла|ло)\b/u,
      /\bпадени[яйею]/u,
      /\bупал[аи]?\b/u,
      /головокруж/u,
      /\bслабост/u,
      /сопровожд/u,
      /шкал\w*\s+бол/u,
      /\bбол[ьи]\s+по\s+шкал/u,
      /\b(?:оценк|балл\w*)\s+бол/u,
      /\bриск\s+падени/u,
    ];
    const hasTrigger = triggers.some((re) => re.test(text));
    if (hasTrigger) return;

    const before = JSON.stringify(doc.riskAssessment);
    doc.riskAssessment = { ...DEFAULT_RISK_ASSESSMENT };
    const after = JSON.stringify(doc.riskAssessment);
    if (before !== after) {
      console.log(`[postprocess] riskAssessment cleared (no Morse triggers in raw): ${before} → ${after}`);
    }
  }

  /**
   * Спасает диагноз из сырого текста, если LLM его обрезал.
   * Ищет блок "диагноз:" в raw-тексте и сравнивает длину с тем что вернул LLM.
   */
  private rescueDiagnosisFromRawText(doc: MedicalDocument, rawText: string): void {
    // Ищем блок диагноза в исходном тексте, останавливаясь на следующей секции
    const diagMatch = rawText.match(
      /(?:предварительный\s+)?диагноз\s*[:.]?\s*([\s\S]*?)(?=(?:[\n.])\s*(?:план\s+обследовани|рекомендо|назначени|диет\S*\s*(?:№|\d)|питани|объективн|анамнез|жалоб|аллерг|данные\s+объективного))/iu
    );
    if (!diagMatch) return;

    const rawDiag = diagMatch[1].trim()
      .replace(/\n{2,}/g, ' ')
      .replace(/\s{2,}/g, ' ')
      // Убираем маркер "предварительный" в начале если остался
      .replace(/^предварительн\S*\s*[:.,-]?\s*/iu, '');
    const currentDiag = doc.diagnosis.trim();

    // Rescue ONLY if LLM returned empty or very short diagnosis (< 50 chars)
    // Don't replace a real LLM diagnosis with raw text — raw often contains
    // anamnesis/therapy sections that follow the diagnosis block
    if (rawDiag.length > 20 && currentDiag.length < 50) {
      console.log(`[postprocess] Diagnosis rescued from raw text: LLM had ${currentDiag.length} chars, raw has ${rawDiag.length} chars`);
      doc.diagnosis = rawDiag;
    }
  }

  /**
   * Спасает рекомендации из исходного текста если LLM их потерял.
   */
  private rescueRecommendationsFromRawText(doc: MedicalDocument, rawText: string): void {
    // Only rescue if recommendations are empty or garbage
    if (doc.recommendations && doc.recommendations.trim().length > 10) return;

    // Look for "Рекомендации" block in raw text
    const recoMatch = rawText.match(
      /рекомендаци\S*\s*[:.]?\s*([\s\S]*?)(?=(?:[\n.])\s*(?:диагноз|объективн|анамнез|жалоб|аллерг|данные\s+объективного|$))/iu
    );

    if (recoMatch && recoMatch[1].trim().length > 20) {
      const rawReco = recoMatch[1].trim()
        .replace(/\n{2,}/g, '\n')
        .replace(/\s{2,}/g, ' ');
      doc.recommendations = rawReco;
      console.log(`[postprocess] Recommendations rescued from raw text: ${rawReco.length} chars`);
      return;
    }

    // Alternative: look for numbered medication items with "мг по 1 таблетке" pattern
    const medItems: string[] = [];
    const medPattern = /\d+\.\s*(?:таблетк[аи]\s+\S+.*?(?:длительно|постоянно|в\s+сутки|мг\s+в\s+сутки)[.!,]?)/giu;
    let match;
    while ((match = medPattern.exec(rawText)) !== null) {
      medItems.push(match[0].trim());
    }

    if (medItems.length > 0) {
      const existing = doc.recommendations.trim();
      if (!existing || existing.length < 10) {
        doc.recommendations = medItems.join('\n');
        console.log(`[postprocess] Recommendations rescued (${medItems.length} med items) from raw text`);
      }
    }
  }

  /**
   * Детерминированный набор финальных пост-проходов, исправляющих типовые ошибки
   * маршрутизации LLM: препараты для постоянного приёма уезжают в conclusion,
   * "продолжать" из recommendations → conclusion, дубли токенов, утерянный контент.
   * Вызывается после всех rescue-проходов, один раз на документ.
   */
  private runSemanticRoutingPasses(doc: MedicalDocument, rawText: string): void {
    // Подсказки покрытия для P5: исходные «сырые» фрагменты, которые P1/P2
    // уже растащили по другим полям. P5 не должен считать их утерянным
    // контентом и роутить обратно в recommendations.
    const coverageHints = new Set<string>();
    this.extractConstantMedsFromClinicalCourse(doc, coverageHints, rawText);
    this.extractContinuedDrugsFromRecommendations(doc, coverageHints);
    this.cleanupNonDrugConclusionItems(doc);
    // P5 должен идти до словаря, чтобы восстановленный текст прошёл нормализацию.
    this.recoverMissingContent(doc, rawText, coverageHints);
    // Словарь может ДОБАВЛЯТЬ токены (напр. «HbA1c» после «гликированный гемоглобин»),
    // поэтому применяем ПЕРЕД P4 — иначе P4 удалит дубль, а словарь его вернёт.
    this.reapplyMedicalDictionary(doc);
    this.fixRedundantTokens(doc);
    // Финальный дедуп: P5 мог вернуть в recommendations строку с препаратами,
    // которые уже перечислены в conclusion (Whisper-артефактный echo-хвост).
    // Запускаем в самом конце, чтобы убрать последний эхо-дубль.
    this.dedupRecommendationsAgainstConclusion(doc);
    this.filterGarbageRecommendationItems(doc);
    this.dedupOutpatientExamsAgainstDoctorNotes(doc);
  }

  /**
   * P1.7: фильтрует мусор-пункты в recommendations, которые Claude иногда
   * выдаёт в виде ALL-CAPS псевдо-заголовков («РЕКОМЕНДАЦИЯ ПИТАНИЕ …»,
   * «НАЗНАЧЕНИЕ ОБСЛЕДОВАНИЕ …») — эхо структуры промпта, попавшее в текст.
   */
  private filterGarbageRecommendationItems(doc: MedicalDocument): void {
    const reco = doc.recommendations;
    if (!reco) return;

    const lines = reco.split(/\n+/)
      .map(l => l.replace(/^\s*\d+[\.\)]\s*/, '').trim())
      .filter(Boolean);

    const kept: string[] = [];
    let dropped = 0;
    for (const body of lines) {
      // Псевдо-заголовок: строка начинается с двух ALL-CAPS слов подряд,
      // первое ≥6 букв («РЕКОМЕНДАЦИЯ …», «НАЗНАЧЕНИЕ …», «ОБСЛЕДОВАНИЕ …»).
      // Обычный drug-пункт начинается с одного Capitalized-слова, поэтому
      // паттерн ALL-CAPS+ALL-CAPS надёжно изолирует мусор.
      if (/^[А-ЯЁ]{6,}\s+[А-ЯЁ]{4,}/u.test(body)) {
        dropped++;
        continue;
      }
      kept.push(body);
    }
    if (dropped === 0) return;
    doc.recommendations = kept.map((l, i) => `${i + 1}. ${l}`).join('\n');
    console.log(`[postprocess] P1.7: dropped ${dropped} ALL-CAPS pseudo-header item(s) from recommendations`);
  }

  /**
   * P3: убирает из outpatientExams «плановые» обследования, которые уже
   * перечислены в doctorNotes (План обследования). В outpatientExams должны
   * остаться только РЕЗУЛЬТАТЫ (с датами/значениями), а не список названий.
   */
  private dedupOutpatientExamsAgainstDoctorNotes(doc: MedicalDocument): void {
    const exams = doc.outpatientExams;
    const notes = doc.doctorNotes;
    if (!exams || !notes) return;

    // Сигнатуры названий из doctorNotes: Capitalized-токены ≥3 букв,
    // аббревиатуры (ЭКГ, СМАД, ЭхоКГ, УЗДГБЦ, ОАК, ОАМ).
    const noteSignatures = new Set<string>();
    const tokens = (notes.match(/[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z\-]{2,}/gu) || []);
    for (const t of tokens) {
      noteSignatures.add(t.toLowerCase().replace(/ё/g, 'е'));
    }
    if (noteSignatures.size === 0) return;

    // Внутри-строчное усечение: Claude иногда приклеивает перечисление
    // планируемых обследований ХВОСТОМ к последнему «настоящему» обследованию
    // (напр. «Заключение: … хронический бронхит. С-реактивный белок ЭК, СМАД,
    // ЭХОКАГ, УЗДГБЦ, УЗИ почек.»). Такой хвост — запятая-разделённый список
    // имён без числовых значений, состоящий из сигнатур doctorNotes.
    const trimTail = (line: string): { text: string; trimmed: boolean } => {
      // Находим последнюю точку, после которой идёт только comma-separated список.
      const tailMatch = line.match(/\.\s+([А-ЯЁA-Z][^.]*?(?:,\s+[А-ЯЁA-Z][^.]*?){1,})\.?\s*$/u);
      if (!tailMatch) return { text: line, trimmed: false };
      const tail = tailMatch[1];
      // Хвост не должен содержать числовых значений (дата/доза/результат).
      if (/\d/u.test(tail)) return { text: line, trimmed: false };
      const tailItems = tail.split(/,\s+/u).map(s => s.trim()).filter(Boolean);
      if (tailItems.length < 2) return { text: line, trimmed: false };
      // Каждый пункт должен быть коротким (1-4 слова) и иметь хоть один
      // Capitalized-токен из сигнатур doctorNotes.
      let matched = 0;
      for (const item of tailItems) {
        const tokens = (item.match(/[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z\-]{2,}/gu) || [])
          .map(t => t.toLowerCase().replace(/ё/g, 'е'));
        if (tokens.some(t => noteSignatures.has(t))) matched++;
      }
      // Жёсткий порог: ≥ 70% пунктов хвоста — сигнатуры плана.
      if (matched / tailItems.length < 0.7) return { text: line, trimmed: false };
      const cut = line.slice(0, tailMatch.index! + 1).trim();
      return { text: cut, trimmed: true };
    };

    const lines = exams.split(/\n+/).map(l => l.trim()).filter(Boolean);
    const kept: string[] = [];
    let dropped = 0;
    let trimmed = 0;
    for (const body of lines) {
      // Пункт с «данными» (дата/значение/ед.изм.) — оставляем, но пробуем обрезать хвост.
      const hasData = /\d+[,.]?\d*\s*(?:мг|мкг|мкмоль|ммоль|г\/л|мм|мс|мл|%|л\/мин|удар|мм\s*рт|ед\b|ме\b)|\d{2}[\.\/]\d{2}[\.\/]\d{2,4}|(?:^|\s)(?:от|на)\s+\d/iu.test(body);
      if (hasData) {
        const { text, trimmed: wasTrimmed } = trimTail(body);
        if (wasTrimmed) trimmed++;
        kept.push(text);
        continue;
      }

      const lineTokens = (body.match(/[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z\-]{2,}/gu) || [])
        .map(t => t.toLowerCase().replace(/ё/g, 'е'));
      if (lineTokens.length === 0) { kept.push(body); continue; }
      const overlap = lineTokens.filter(t => noteSignatures.has(t)).length;
      // ≥2 совпадения ИЛИ полное вхождение (все токены) — это перечисление плана.
      if (overlap >= 2 || (overlap >= 1 && overlap === lineTokens.length)) {
        dropped++;
        continue;
      }
      kept.push(body);
    }
    if (dropped === 0 && trimmed === 0) return;
    doc.outpatientExams = kept.join('\n');
    if (dropped > 0) console.log(`[postprocess] P3: dropped ${dropped} plan-like item(s) from outpatientExams (duplicated in doctorNotes)`);
    if (trimmed > 0) console.log(`[postprocess] P3: trimmed exam-list tail from ${trimmed} line(s) in outpatientExams`);
  }

  /**
   * Нормализует существующий блок как нумерованный список:
   * - если уже есть "1. ..." — возвращает как есть;
   * - иначе оборачивает всё содержимое в один пункт "1. ...".
   * Гарантирует, что `appendNumberedItem` правильно считает следующий номер.
   */
  private ensureNumberedList(text: string): string {
    const trimmed = (text || '').trim();
    if (!trimmed) return '';
    if (/^\s*\d+[\.\)]\s/mu.test(trimmed)) return trimmed;
    return `1. ${trimmed.replace(/\n+/g, ' ').trim()}`;
  }

  /**
   * Безопасно добавляет item в нумерованное поле с корректной нумерацией.
   */
  private appendNumberedItem(existing: string, item: string): string {
    const normalized = this.ensureNumberedList(existing);
    if (!normalized) return `1. ${item}`;
    const count = (normalized.match(/^\s*\d+[\.\)]\s/gmu) || []).length;
    return `${normalized}\n${count + 1}. ${item}`;
  }

  /** Извлекает "голову" drug-name (первое слово с заглавной) из строки. */
  private extractDrugHead(s: string): string | null {
    const m = s.trim().match(/^([А-ЯЁA-Z][а-яёa-z\-]{2,})/u);
    return m ? m[1] : null;
  }

  /**
   * Сопоставляет drug-head'ы по префиксу — устойчив к хвостовым искажениям
   * Whisper (Симбикорд/Симбикорт) и словоформам (Бетасерк/Бетасерка): 5+ букв
   * общего префикса достаточно (drug names почти всегда уникальны в первых 5).
   */
  private sameDrugHead(a: string, b: string): boolean {
    const norm = (s: string) => s.toLowerCase().replace(/ё/g, 'е');
    const la = norm(a);
    const lb = norm(b);
    if (la === lb) return true;
    if (la.length >= 5 && lb.startsWith(la)) return true;
    if (lb.length >= 5 && la.startsWith(lb)) return true;
    // Общий префикс ≥5 символов (одинаковый корень с разным хвостом)
    const minLen = Math.min(la.length, lb.length);
    if (minLen >= 5 && la.slice(0, 5) === lb.slice(0, 5)) return true;
    return false;
  }

  /**
   * Добавляет препарат в conclusion с дедупом по drug-head. Если в conclusion
   * уже есть пункт на тот же препарат — побеждает более длинный (детальный).
   */
  private mergeConclusionDrug(doc: MedicalDocument, item: string): void {
    const head = this.extractDrugHead(item);
    const current = doc.conclusion || '';
    if (!head || !current) {
      doc.conclusion = this.appendNumberedItem(current, item);
      return;
    }
    const lines = current.split(/\n+/).map(l => l.replace(/^\s*\d+[\.\)]\s*/, '').trim()).filter(Boolean);
    let replaced = false;
    const merged = lines.map((body) => {
      const bodyHead = this.extractDrugHead(body);
      if (bodyHead && this.sameDrugHead(bodyHead, head)) {
        replaced = true;
        return item.length > body.length ? item : body;
      }
      return body;
    });
    if (!replaced) merged.push(item);
    doc.conclusion = merged.map((l, i) => `${i + 1}. ${l}`).join('\n');
  }

  /**
   * P1: "Постоянный прием медикаментов: X" → conclusion.
   * Сканирует clinicalCourse (LLM кладёт туда амбулаторную терапию из «анамнеза
   * жизни») и recommendations (иногда LLM дублирует туда тот же блок отдельным
   * пунктом). Дополнительно выполняет rescue по rawText: Whisper часто пишет «.»
   * вместо «:» после «Постоянный прием медикаментов», и Claude разносит
   * препараты отдельными пунктами в recommendations без маркера-префикса;
   * P1 находит drug-head'ы и перетаскивает соответствующие пункты в conclusion.
   */
  private extractConstantMedsFromClinicalCourse(doc: MedicalDocument, coverageHints?: Set<string>, rawText?: string): void {
    const splitDrugList = (drugs: string): string[] => {
      const cleaned = drugs.replace(/[,.\s]+$/, '');
      return cleaned
        .split(/,\s+(?=[А-ЯЁA-Z][а-яёa-z\-]{2,})/u)
        .map(s => s.trim())
        .filter(s => s.length >= 3);
    };
    const isBlacklisted = (drugs: string): boolean => {
      if (/отрица|не\s+принима|отсутств|нет\b/iu.test(drugs)) return true;
      // \b не работает с кириллицей в JS — используем lookbehind по буквам.
      if (/(?<![а-яёa-z])(?:алкогол|табак|спиртн|наркотик|токсин|вод[ауые]|пищ[ауе]|кофе|ча[йяею])/iu.test(drugs)) return true;
      return drugs.length < 3;
    };

    const extracted: string[] = [];

    // --- 1. clinicalCourse: inline-фраза посреди прозы. ---
    let cc = doc.clinicalCourse;
    if (cc) {
      const re = /(?<=^|[.\s])(?:Постоянн(?:ый|о)\s+прием\s+медикаментов|Принимает\s+постоянно|Амбулаторно\s+принима(?:ет|ю)|Постоянно\s+принима(?:ет|ю))\s*[:,—–-]?\s*([^.\n]+?)\.?(?=$|\n|\s{2,}|\.\s)/giu;
      const matches = Array.from(cc.matchAll(re));
      // Идём с конца, чтобы удаление не сбивало индексы
      for (let i = matches.length - 1; i >= 0; i--) {
        const m = matches[i];
        const drugs = m[1].trim();
        if (isBlacklisted(drugs)) continue;
        const parts = splitDrugList(drugs);
        for (let j = parts.length - 1; j >= 0; j--) {
          extracted.unshift(parts[j]);
        }
        // Регистрируем «сырой» фрагмент в подсказках покрытия для P5
        coverageHints?.add(m[0]);
        cc = cc.slice(0, m.index!) + cc.slice(m.index! + m[0].length);
      }
      doc.clinicalCourse = cc
        .replace(/\s+/g, ' ')
        .replace(/\s+\./g, '.')
        .replace(/\.{2,}/g, '.')
        .trim();
    }

    // --- 2. recommendations: целый numbered-пункт с префиксом-триггером. ---
    const reco = doc.recommendations;
    if (reco) {
      const reLine = /^(?:Постоянн(?:ый|о)\s+прием\s+медикаментов|Принимает\s+постоянно|Амбулаторно\s+принима(?:ет|ю)|Постоянно\s+принима(?:ет|ю))\s*[:,.\-—–]?\s*(.+?)\.?$/iu;
      const lines = reco.split(/\n+/).filter(l => l.trim());
      const keep: string[] = [];
      const movedFromReco: string[] = [];
      for (const line of lines) {
        const body = line.replace(/^\s*\d+[\.\)]\s*/, '').trim();
        const m = body.match(reLine);
        if (m && !isBlacklisted(m[1].trim())) {
          movedFromReco.push(...splitDrugList(m[1].trim()));
          coverageHints?.add(body);
        } else {
          keep.push(body);
        }
      }
      if (movedFromReco.length > 0) {
        extracted.push(...movedFromReco);
        doc.recommendations = keep
          .map((l, i) => `${i + 1}. ${l}`)
          .join('\n');
      }
    }

    // --- 3. Rescue по rawText: Whisper пишет «.» вместо «:», Claude режет ---
    //      препараты на отдельные numbered-пункты в recommendations БЕЗ префикса.
    //      Находим drug-head'ы в блоке после «Постоянный прием медикаментов.» и
    //      перетаскиваем пункты с совпадающими drug-head'ами из recommendations.
    if (rawText && doc.recommendations) {
      // Принимаем «.» как допустимый разделитель (ключевое отличие от шагов 1-2).
      // ВАЖНО: \w в JS-regex НЕ матчит кириллицу даже с /u, поэтому используем
      // явные классы [а-яёa-z0-9]* для суффиксов слов-терминаторов.
      const reRaw = /(?<=^|[.\s])(?:Постоянн(?:ый|о)\s+прием\s+медикаментов|Принимает\s+постоянно|Амбулаторно\s+принима(?:ет|ю)|Постоянно\s+принима(?:ет|ю))\s*[:,.\-—–]\s+([^]*?)(?=\n\s*\n|Аллерголог[а-яёa-z]*\s+анамнез|Объективн[а-яёa-z]*\s+статус|Неврологическ[а-яёa-z]*\s+статус|Предварительн[а-яёa-z]*\s+диагноз|План\s+обследован|Рекомендации|Данные\s+проведенн|Диагноз\s+предвари|$)/iu;
      const rawMatch = rawText.match(reRaw);
      if (rawMatch) {
        const block = rawMatch[1].trim();
        if (!isBlacklisted(block.slice(0, 80))) {
          const drugHeads: string[] = [];
          // Первое слово-заглавное каждого предложения блока — кандидат в drug-head.
          for (const sentence of block.split(/\.\s+|\n+/)) {
            const head = this.extractDrugHead(sentence);
            if (head && head.length >= 4) drugHeads.push(head);
          }
          // Inline: Whisper часто склеивает «Drug A … Drug B … Drug C» БЕЗ точек.
          // Ищем кандидатов по шаблону CAPS+lowers + дозировочный/требовательный маркер.
          // Паддим пробелом, чтобы lookbehind срабатывал и на позиции 0.
          const paddedBlock = ' ' + block;
          const inlineRe = /(?<=[.\s])([А-ЯЁ][а-яё\-]{3,})(?=\s+(?:[Пп]о\s+|\d+[,.]?\d*\s*(?:мг|мкг|мл|МЕ|ЕД|IU|г\b)|две\s+целых|один\s+раз|\d))/gu;
          let mInline: RegExpExecArray | null;
          while ((mInline = inlineRe.exec(paddedBlock)) !== null) {
            const cand = mInline[1];
            if (cand.length >= 4 && !drugHeads.some(dh => this.sameDrugHead(dh, cand))) {
              drugHeads.push(cand);
            }
          }
          if (drugHeads.length > 0) {
            // Pre-split: Claude иногда возвращает весь numbered-список одной строкой.
            let recLines = (doc.recommendations || '').split(/\n+/).filter(l => l.trim());
            if (recLines.length === 1) {
              const splitParts = recLines[0].split(/(?<=\.)\s+(?=\d+[\.\)]\s+[А-ЯЁA-Z])/u);
              if (splitParts.length > 1) recLines = splitParts;
            }
            const keep: string[] = [];
            const moved: string[] = [];
            for (const line of recLines) {
              const body = line.replace(/^\s*\d+[\.\)]\s*/, '').trim();
              const lineHead = this.extractDrugHead(body);
              if (lineHead && drugHeads.some(dh => this.sameDrugHead(dh, lineHead))) {
                // Разбиваем склеенный body «Drug A …. Drug B …. Drug C.» на отдельные пункты.
                const parts = body.split(/\.\s+(?=[А-ЯЁ][а-яё\-]{3,})/u)
                  .map(p => p.trim())
                  .filter(Boolean);
                for (const p of parts) {
                  const item = p.endsWith('.') ? p : p + '.';
                  // Госпитализация/Направляется — не препарат, возвращаем в recommendations.
                  if (/^(?:госпитализ|направ(?:ля|ить|ляется))/iu.test(p)) {
                    keep.push(item);
                    continue;
                  }
                  const pHead = this.extractDrugHead(p);
                  if (pHead) {
                    moved.push(item);
                    coverageHints?.add(p);
                  } else {
                    keep.push(item);
                  }
                }
              } else {
                keep.push(body);
              }
            }
            if (moved.length > 0) {
              extracted.push(...moved);
              doc.recommendations = keep.map((l, i) => `${i + 1}. ${l}`).join('\n');
              // Сам «Постоянный прием медикаментов…» тоже подсказка для P5
              coverageHints?.add(rawMatch[0]);
            }
          }
        }
      }
    }

    if (extracted.length === 0) return;

    // Дедуп с слиянием: «Бетасерк» + «Бетасерк 24 мг …» → побеждает длинный.
    for (const item of extracted) {
      this.mergeConclusionDrug(doc, item);
    }

    console.log(`[postprocess] P1: ${extracted.length} constant-med item(s) → conclusion`);
  }

  /**
   * P2: препараты с маркером "продолжать" переносятся recommendations → conclusion.
   * Только когда в пункте НЕТ маркеров нового назначения ("назначаю", "начать", "добавить" и т.п.).
   */
  private extractContinuedDrugsFromRecommendations(doc: MedicalDocument, coverageHints?: Set<string>): void {
    const reco = doc.recommendations;
    if (!reco) return;

    const lines = reco.split(/\n+/).filter(l => l.trim());
    if (lines.length === 0) return;

    const keep: string[] = [];
    const moved: string[] = [];

    for (const line of lines) {
      const body = line.replace(/^\s*\d+[\.\)]\s*/, '').trim();
      const hasDrugMarker = /(?:таблетк|капсул|раствор|инъекц|ампул|\s\d+\s*(?:мг|мкг|мл|МЕ|ЕД|IU)\b|раз\s+в\s+день|раз\s+в\s+сутки|по\s+\d+\s+(?:т\b|табл|капс))/iu.test(body);
      // ВНИМАНИЕ: \b в JS-regex работает только для ASCII-слов, для кириллицы
      // используем lookaround по кириллическим буквам.
      const hasContinueMarker = /(?<![а-яёa-z])продолжать(?![а-яёa-z])|продолжает\s+прием|продолжить\s+прием|постоянный\s+прием/iu.test(body);
      // Расширенный набор маркеров нового назначения. Если оба маркера
      // ("продолжать" + "начать/добавить") встречаются в одном пункте, значит
      // там коктейль препаратов — безопаснее не перетаскивать, пусть остаётся
      // в recommendations (врач разберёт вручную).
      const hasNewPrescription = /(?<![а-яёa-z])(?:назнача(?:ю|ть|ет)|назначить|рекомендую\s+препарат|начат\w*\s+(?:прием|терап)|начать\s+прием|начать\s+тера|добав(?:ить|ляю)\s+(?:к\s+|препарат|к\s+терап)|впервые\s+назнач|стартова|инициац|приступить|первичн\w*\s+назнач)(?![а-яёa-z])/iu.test(body);

      if (hasDrugMarker && hasContinueMarker && !hasNewPrescription) {
        moved.push(body);
        coverageHints?.add(body);
      } else {
        keep.push(body);
      }
    }

    if (moved.length === 0) return;

    doc.recommendations = keep
      .map(l => l.replace(/^\s*\d+[\.\)]\s*/, '').trim())
      .filter(l => l.length > 0)
      .map((l, i) => `${i + 1}. ${l}`)
      .join('\n');

    for (const item of moved) {
      this.mergeConclusionDrug(doc, item);
    }

    console.log(`[postprocess] P2: moved ${moved.length} "продолжать" drug(s) recommendations → conclusion`);
  }

  /**
   * P1.5: убирает из conclusion пункты без drug-маркеров, которые уже покрыты
   * clinicalCourse/diagnosis (LLM иногда дублирует «Сопутствующую патологию» или
   * сам диагноз в conclusion, хотя там должна быть только амбулаторная терапия).
   */
  private cleanupNonDrugConclusionItems(doc: MedicalDocument): void {
    const concl = doc.conclusion;
    if (!concl) return;

    const hasDrugMarker = (s: string): boolean =>
      /(?:\d+\s*(?:мг|мкг|мл|г|%|ме\b|ед\b|iu\b)|таблетк|капсул|вдох|капел\w|раствор|инъекц|ампул|раз\s+в\s+(?:день|сутки|неделю|месяц)|по\s+\d+\s+(?:т\b|табл|капс|мл|капел)|\bпо\s+\d+\s+вдох|продолжать(?![а-яёa-z])|постоянн\w+\s+прием)/iu.test(s);

    const other = [doc.clinicalCourse || '', doc.diagnosis || '', doc.finalDiagnosis || ''].join(' ').toLowerCase();

    const lines = concl.split(/\n+/)
      .map(l => l.replace(/^\s*\d+[\.\)]\s*/, '').trim())
      .filter(Boolean);

    const kept: string[] = [];
    const movedToFinal: string[] = [];
    let dropped = 0;
    let droppedNarrative = 0;
    for (const body of lines) {
      if (hasDrugMarker(body)) { kept.push(body); continue; }
      const key = body.toLowerCase().replace(/[^a-zа-я0-9\s]/gu, ' ').replace(/\s+/g, ' ').trim();
      const head = key.split(' ').slice(0, 4).join(' ');
      if (head.length >= 10 && other.includes(head)) {
        dropped++;
        continue;
      }
      // Длинный non-drug пункт в conclusion — почти наверняка синтезированный
      // Claude-ом эпикриз/резюме («Пациентка обратилась с жалобами...
      // направляется на госпитализацию»). Его место — в finalDiagnosis (секция
      // «Заключение» в PDF), но в conclusion (амб. терапия) он не нужен. Порог
      // 150 chars оставляет короткие технические пометки вроде «продолжать по схеме».
      if (body.length >= 150) {
        droppedNarrative++;
        movedToFinal.push(body);
        continue;
      }
      kept.push(body);
    }
    if (dropped === 0 && droppedNarrative === 0) return;
    doc.conclusion = kept.map((l, i) => `${i + 1}. ${l}`).join('\n');
    if (movedToFinal.length > 0) {
      const current = (doc.finalDiagnosis || '').trim();
      const addition = movedToFinal.join(' ').trim();
      doc.finalDiagnosis = current ? `${current}\n\n${addition}` : addition;
    }
    if (dropped > 0) {
      console.log(`[postprocess] P1.5: dropped ${dropped} non-drug item(s) from conclusion (covered in clinicalCourse/diagnosis)`);
    }
    if (droppedNarrative > 0) {
      console.log(`[postprocess] P1.5: moved ${droppedNarrative} narrative item(s) from conclusion → finalDiagnosis (клинич. обоснование)`);
    }
  }

  /**
   * P1.6: удаляет из recommendations пункты, которые являются пересказом уже
   * перечисленных в conclusion препаратов. Claude иногда складывает «хвост»
   * сырой диктовки (список амбулаторных лекарств) в recommendations отдельной
   * слепленной строкой с Whisper-артефактами («одну раз раз в день»). Такой
   * пункт нечитаем и дублирует conclusion — выкидываем его целиком, если
   * ≥2 drug-head'a уже присутствуют в conclusion.
   */
  private dedupRecommendationsAgainstConclusion(doc: MedicalDocument): void {
    const reco = doc.recommendations;
    const concl = doc.conclusion;
    if (!reco || !concl) return;

    const conclHeads: string[] = [];
    for (const line of concl.split(/\n+/)) {
      const body = line.replace(/^\s*\d+[\.\)]\s*/, '').trim();
      const head = this.extractDrugHead(body);
      if (head) conclHeads.push(head);
    }
    if (conclHeads.length === 0) return;

    const lines = reco.split(/\n+/)
      .map(l => l.replace(/^\s*\d+[\.\)]\s*/, '').trim())
      .filter(Boolean);

    const kept: string[] = [];
    let dropped = 0;
    for (const body of lines) {
      // Собираем все drug-head кандидаты из строки (слова с заглавной ≥5 букв).
      const candidates = (body.match(/[А-ЯЁA-Z][а-яёa-z\-]{4,}/gu) || []);
      let matches = 0;
      for (const cand of candidates) {
        if (conclHeads.some(h => this.sameDrugHead(h, cand))) matches++;
      }
      if (matches >= 2) {
        dropped++;
        continue;
      }
      kept.push(body);
    }
    if (dropped === 0) return;
    doc.recommendations = kept.map((l, i) => `${i + 1}. ${l}`).join('\n');
    console.log(`[postprocess] P1.6: dropped ${dropped} recommendation item(s) duplicating conclusion drugs`);
  }

  /**
   * P4: чистка избыточных токенов / Whisper-артефактов во всех текстовых полях.
   */
  private fixRedundantTokens(doc: MedicalDocument): void {
    const fields: (keyof MedicalDocument)[] = [
      'complaints', 'anamnesis', 'outpatientExams', 'clinicalCourse', 'allergyHistory',
      'objectiveStatus', 'neurologicalStatus', 'diagnosis', 'finalDiagnosis',
      'conclusion', 'doctorNotes', 'recommendations',
    ];
    let totalChanges = 0;
    for (const f of fields) {
      const v = (doc as any)[f];
      if (typeof v !== 'string' || v.length === 0) continue;
      let fixed = v;
      // «(HbA1c) (HbA1c) (HbA1c)» → «(HbA1c)» (любое число повторов).
      // Словарь применяет wordRule «гликированный гемоглобин» → «... (HbA1c)»
      // и при повторных проходах может продублировать токен; здесь схлопываем.
      fixed = fixed.replace(/(\([A-Za-zА-ЯЁа-яё0-9\-\/]+\))(?:\s+\1)+/gu, '$1');
      // «24 мга» → «24 мг»
      fixed = fixed.replace(/(\d+\s*)мга\b/gu, '$1мг');
      // «мм рт.ст. Мм аррт.ст.» → «мм рт.ст.»
      fixed = fixed.replace(/мм\s*рт\.?\s*ст\.?\s*\.?\s*мм\s*а?ррт\.?\s*ст\.?/giu, 'мм рт.ст.');
      // «мг/мга на ...» — внутри дозы
      fixed = fixed.replace(/\bмга\s+на\s+/giu, 'мг на ');
      if (fixed !== v) {
        (doc as any)[f] = fixed;
        totalChanges++;
      }
    }
    if (totalChanges > 0) {
      console.log(`[postprocess] P4: ${totalChanges} field(s) normalized`);
    }
  }

  /**
   * P5: coverage-проверка. Сравнивает raw Whisper с финальным контентом доктора
   * (нормализованная подстрока — отпечаток). Потерянные предложения логируются
   * и маршрутизируются в best-fit поле по эвристике.
   */
  private recoverMissingContent(doc: MedicalDocument, rawText: string, coverageHints?: Set<string>): void {
    if (!rawText || rawText.length < 50) return;

    const normalize = (s: string) => s
      .toLowerCase()
      .replace(/ё/g, 'е')
      .replace(/[^a-zа-я0-9\s]/gu, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    const docTextRaw = [
      doc.complaints, doc.anamnesis, doc.outpatientExams, doc.clinicalCourse,
      doc.allergyHistory, doc.objectiveStatus, doc.neurologicalStatus,
      doc.diagnosis, doc.finalDiagnosis, doc.conclusion, doc.doctorNotes,
      doc.recommendations,
      // Подсказки покрытия от P1/P2 — фрагменты, которые уже растащены в conclusion
      // в виде отдельных пунктов, но исходное предложение целиком в полях не лежит.
      ...(coverageHints ? Array.from(coverageHints) : []),
    ].join(' ');
    const normDoc = normalize(docTextRaw);

    const sentences = rawText
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(s => s.length >= 20);

    const missing: string[] = [];
    for (const s of sentences) {
      const ns = normalize(s);
      if (ns.length < 20) continue;
      const fp1 = ns.substring(0, 25);
      if (normDoc.includes(fp1)) continue;
      const midStart = Math.max(0, Math.floor(ns.length / 2) - 8);
      const fp2 = ns.substring(midStart, midStart + 15);
      if (fp2.length >= 12 && normDoc.includes(fp2)) continue;
      missing.push(s);
    }

    if (missing.length === 0) return;

    console.log(`[postprocess] P5: ${missing.length} sentence(s) not found in doc fields`);
    for (const m of missing) {
      console.log(`  [P5 MISSING] ${m.substring(0, 120)}`);
    }

    for (const m of missing) {
      const target = this.routeMissingSentence(m);
      if (!target) {
        console.log(`  [P5 UNROUTED] ${m.substring(0, 80)}`);
        continue;
      }
      const current = ((doc as any)[target] as string) || '';
      if (target === 'recommendations' || target === 'doctorNotes' || target === 'conclusion') {
        (doc as any)[target] = this.appendNumberedItem(current, m);
      } else {
        (doc as any)[target] = current ? `${current} ${m}` : m;
      }
      console.log(`  [P5 ROUTED → ${target}] ${m.substring(0, 80)}`);
    }
  }

  /**
   * Эвристический роутинг для P5. Возвращает имя поля или null (если непонятно).
   */
  private routeMissingSentence(s: string): keyof MedicalDocument | null {
    const trimmed = s.trim();
    if (trimmed.length < 15) return null;
    // Заголовки разделов — не маршрутизируем
    if (/^(?:жалобы|анамнез(?:\s+заболевани\w*|\s+жизни)?|диагноз(?:\s+предварительн\w*)?|план(?:\s+обследовани\w*|\s+лечени\w*)?|рекомендации|объективн\w*|аллерголог\w*|амбулаторн\w*|данные\s+проведенн\w*)[.:!]?\s*$/iu.test(trimmed)) {
      return null;
    }
    const low = trimmed.toLowerCase();
    // Физическая активность / образ жизни — даже если содержит «раз в день»
    // и «длительно», это НЕ препарат (частый ложный матч: «Заниматься 30 мин
    // по 10 минут один раз в день утром длительно»).
    const isLifestyle = /(?:заниматься|заним[ауе]|ходьб|\bбег\b|бег\s+трусц|плаван|велосип|аэробн|физическ\w*\s+(?:актив|нагруз)|минут\s+в\s+день|\d+\s*[–\-]\s*\d+\s*минут|постепенно\s+увелич)/iu.test(low);
    // Препараты / назначения
    if (!isLifestyle && /(?:таблетк|капсул|раствор|инъекц|ампул|\bмг\b|\bмкг\b|\bмл\b|раз\s+в\s+день|раз\s+в\s+сутки|по\s+\d+\s+(?:т\b|табл|капс))/iu.test(low)) {
      // «длительно», «постоянно» — маркеры долгосрочной терапии (conclusion).
      return /продолжать|принимает\s+постоянн|амбулаторно\s+принима|длительно|постоянн\w*\s+прием/iu.test(low) ? 'conclusion' : 'recommendations';
    }
    // Исследования / лаборатория
    if (/(?:\bоак\b|\bоам\b|б\/х|биохим\w*|\bэкг\b|эхо[\-\s]?кг|\bмрт\b|\bкт\b|\bузи\b|холтер\w*|\bсмад\b|рентген\w*|гликирован\w*|hba1c|узд[гс])/iu.test(low)) {
      return 'outpatientExams';
    }
    // Образ жизни / профилактика / наблюдение
    if (/(?:снижени\w*\s+(?:масс\w*|вес\w*)|\bвес\b|физическ\w*|нагруз\w*|диет\w*|питани\w*|ограничен\w*|курени\w*|алкогол\w*|гимнастик\w*|самоконтрол\w*|\bконтрол\w*|памятк\w*|повторн\w*\s+осмотр|консультаци\w*|направлен\w*|динамик\w*)/iu.test(low)) {
      return 'recommendations';
    }
    // Аллергия
    if (/аллерг|непереносим/iu.test(low)) return 'allergyHistory';
    return null;
  }

  /**
   * Раскрывает краткое упоминание диеты ("Диета №9", "Стол 10")
   * внутри recommendations до полного текста шаблона.
   */
  private expandDietTemplateInRecommendations(doc: MedicalDocument): void {
    const reco = doc.recommendations;
    if (!reco) return;

    const lines = reco.split(/\n+/);
    let changed = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const m = line.match(/^(\s*\d+[\.\)]\s*)?(.+)$/u);
      if (!m) continue;
      const prefix = m[1] || '';
      const body = m[2].trim();
      if (body.length > 40) continue;
      // Грубый фильтр: пункт должен содержать слово «диета» / «стол» / «гипо…»
      if (!/\b(?:диет\S*|стол\s*(?:№?\s*)?\d+|гипохолестерин\S*|гипонатриев\S*)\b/iu.test(body)) continue;

      const template = findDietTemplate(body);
      if (template) {
        lines[i] = prefix + template.description;
        changed = true;
        console.log(`[postprocess] Expanded diet template in recommendations: "${body}" → "${template.name}"`);
      }
    }

    if (changed) doc.recommendations = lines.join('\n');
  }

  /**
   * Удаляет мусорные токены из начала и конца текстовых полей:
   * - обрезанные слоги в начале (1-2 символа перед пробелом)
   * - "Точка", "точка", "Запятая", "запятая" в конце поля
   * - пунктуацию-мусор в начале поля
   */
  private cleanFieldGarbage(doc: MedicalDocument): void {
    for (const field of ALL_TEXT_FIELDS) {
      let value = doc[field];
      if (!value) continue;

      // Удаляем обрезанный слог в самом начале: 1-3 символа кирилл./латиница + пробел
      // Например: "ле препарат..." → "препарат..."
      value = value.replace(/^[а-яёА-ЯЁa-zA-Z]{1,3}\s+(?=[А-ЯЁA-Z])/u, '');

      // Удаляем пунктуацию-мусор в начале поля (запятая, точка с запятой и т.п.)
      value = value.replace(/^[,;.—–-]+\s*/u, '');

      // Удаляем словесные команды пунктуации в КОНЦЕ поля
      value = value.replace(/[,.]?\s*\b[Тт]очка\s*$/u, '');
      value = value.replace(/[,.]?\s*\b[Зз]апятая\s*$/u, '');

      // Удаляем словесные команды пунктуации в конце КАЖДОЙ строки (для многострочных полей)
      value = value.replace(/\s*\b[Тт]очка\s*$/gmu, '');
      value = value.replace(/\s*\b[Зз]апятая\s*$/gmu, '');

      // DEV-007: Удаляем дублированные фразы подряд (2+ повтора)
      // "Номер 5 номер 5" → "Номер 5", "контроль АД контроль АД" → "контроль АД"
      value = value.replace(
        /(\b[а-яёА-ЯЁa-zA-Z0-9][а-яёА-ЯЁa-zA-Z0-9\s.,/]{2,40}?)(?:[,.\s]+\1){1,}/giu,
        '$1'
      );

      doc[field] = value.trim();
    }
  }

  // Убирает слова-маркеры разделов с начала текста (например «Рекомендую», «Заключение.»)
  private stripSectionPrefix(field: RewriteableField, text: string): string {
    const patterns: Record<RewriteableField, RegExp[]> = {
      complaints: [
        /^жалоб[ыа]\s+на\s+/iu,            // «Жалобы на боль...»
        /^жалоб[ыа]\s*[:.,-]\s*/iu,         // «Жалобы: боль...»
      ],
      anamnesis: [
        /^анамнез\S*\s+(?:заболевания|жизни)\s*[:.,-]?\s*/iu,  // «Анамнез заболевания:»
        /^анамнез\S*\s*[:.,-]\s*/iu,                            // «Анамнез:»
        /^история\s+(?:болезни|заболевания)\s*[:.,-]?\s*/iu,    // «История болезни:»
      ],
      objectiveStatus: [
        /^объектив\S*\s+статус[а]?\s*[:.,-]?\s*/iu,   // «Объективный статус:»
        /^объектив\S*\s*[:.,-]\s*/iu,                  // «Объективно:»
        /^при\s+осмотре\s*[:.,-]?\s*/iu,               // «При осмотре:»
        /^осмотр\S*\s*[:.,-]\s*/iu,                    // «Осмотр:»
      ],
      diagnosis: [
        /^предварительн\S*\s+диагноз\S*\s*[:.,-]?\s*/iu, // «Предварительный диагноз:»
        /^диагноз\S*\s*[:.,-]\s*/iu,                       // «Диагноз:» / «Диагноз.»
      ],
      finalDiagnosis: [
        /^заключительн\S*\s+диагноз\S*\s*[:.,-]?\s*/iu,   // «Заключительный диагноз:»
        /^окончательн\S*\s+диагноз\S*\s*[:.,-]?\s*/iu,    // «Окончательный диагноз:»
        /^диагноз\S*\s*[:.,-]\s*/iu,                       // «Диагноз:» (fallback)
      ],
      outpatientExams: [
        /^амбулаторн\S*\s+данн\S*\s*[:.,-]?\s*/iu,             // «Амбулаторные данные:»
        /^амбулаторн\S*\s+обследовани\S*\s*[:.,-]?\s*/iu,      // «Амбулаторные обследования:»
        /^(?:данные|результаты)\s+обследовани\S*\s*[:.,-]?\s*/iu, // «Данные обследований:»
        /^(?:проведённ\S*|выполненн\S*)\s+(?:обследовани\S*|исследовани\S*)\s*[:.,-]?\s*/iu,
      ],
      clinicalCourse: [
        /^перенесённ\S*\s+заболевани\S*\s*[:.,-]?\s*/iu,  // «Перенесённые заболевания:»
        /^анамнез\s+жизни\s*[:.,-]?\s*/iu,                 // «Анамнез жизни:»
        /^(?:клиническое\s+)?течени\S*\s*[:.,-]\s*/iu,     // «Течение:» (старый маркер)
      ],
      allergyHistory: [
        /^аллергологическ\S*\s+анамнез\S*\s*[:.,-]?\s*/iu, // «Аллергологический анамнез:»
        /^аллерг\S*\s+на\s+/iu,                             // «Аллергия на ...»
        /^аллерг\S*\s*[:.,-]\s*/iu,                         // «Аллергии:»
      ],
      neurologicalStatus: [
        /^неврологическ\S*\s+статус\S*\s*[:.,-]?\s*/iu, // «Неврологический статус:»
        /^неврологическ\S*\s*[:.,-]\s*/iu,               // «Неврологически:»
      ],
      conclusion: [
        /^амбулаторн\S*\s+терапи\S*\s*[:.,-]?\s*/iu,                // «Амбулаторная терапия:»
        /^амбулаторно\s+принимает\s*[:.,-]?\s*/iu,                   // «Амбулаторно принимает:»
        /^текущ\S*\s+(?:терапи\S*|лечени\S*)\s*[:.,-]?\s*/iu,       // «Текущая терапия:»
        /^сопутствующ\S*\s+диагноз\S*\s*[:.,-]?\s*/iu,              // «Сопутствующий диагноз:» (старый)
      ],
      recommendations: [
        /^рекомендаци\S*\s*[:.,-]\s*/iu,                 // «Рекомендации:»
        /^план\s+лечени\S*\s*[:.,-]?\s*/iu,              // «План лечения:»
        /^(?:рекомендую|рекомендуется|рекомендует|рекомендуем)\s+/iu,
      ],
      doctorNotes: [
        /^план\s+обследовани\S*\s*[:.,-]?\s*/iu,                     // «План обследования:»
        /^направлени\S*\s+на\s+обследовани\S*\s*[:.,-]?\s*/iu,      // «Направления на обследования:»
        /^прочее\s*[:.,-]?\s*/iu,                                     // «Прочее:» (старый)
        /^(?:заметк[иа]\s+врача|заметк[иа]|примечани\S*)\s*[:.,-]\s*/iu,
      ],
    };

    const fieldPatterns = patterns[field] ?? [];
    let result = text.trim();

    for (const pattern of fieldPatterns) {
      const cleaned = result.replace(pattern, '');
      if (cleaned !== result) {
        result = cleaned.trim();
        break; // применяем только первый совпавший паттерн
      }
    }

    // Капитализируем первую букву после очистки
    if (result.length > 0) {
      result = result[0].toUpperCase() + result.slice(1);
    }

    return result;
  }

  private parseLlmJson<T>(rawContent: string): T {
    const content = rawContent.trim();
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No JSON object found in LLM response');
    }

    const rawJson = jsonMatch[0];
    try {
      return JSON.parse(rawJson) as T;
    } catch {
      let sanitized = rawJson
        .replace(/```json|```/gi, '')
        .replace(/[\u201C\u201D]/g, '"')
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/([{,]\s*)([A-Za-z_][\w]*)(\s*:)/g, '$1"$2"$3')
        .replace(/'([^'\\]*(?:\\.[^'\\]*)*)'/g, '"$1"')
        .replace(/,\s*([}\]])/g, '$1')
        .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, ' ');

      try {
        return JSON.parse(sanitized) as T;
      } catch {
        // Экранируем реальные переносы строк внутри JSON-строк
        sanitized = this.escapeNewlinesInJsonStrings(sanitized);
        try {
          return JSON.parse(sanitized) as T;
        } catch {
          // Экранируем unescaped кавычки внутри JSON string values
          sanitized = this.escapeUnquotedQuotesInJsonStrings(sanitized);
          return JSON.parse(sanitized) as T;
        }
      }
    }
  }

  /**
   * Экранирует неэкранированные кавычки внутри JSON string values.
   * Использует статeful-парсер: отслеживает, находимся ли мы внутри строки.
   */
  private escapeUnquotedQuotesInJsonStrings(json: string): string {
    let result = '';
    let inString = false;
    let escaped = false;

    for (let i = 0; i < json.length; i++) {
      const ch = json[i];

      if (escaped) {
        result += ch;
        escaped = false;
        continue;
      }

      if (ch === '\\' && inString) {
        result += ch;
        escaped = true;
        continue;
      }

      if (ch === '"') {
        if (!inString) {
          // Открываем строку
          inString = true;
          result += ch;
        } else {
          // Проверяем: это закрывающая кавычка (за ней пробел/: /,/}/])
          // или кавычка внутри значения?
          let j = i + 1;
          while (j < json.length && (json[j] === ' ' || json[j] === '\t' || json[j] === '\r' || json[j] === '\n')) j++;
          const nextMeaningful = json[j];
          const isClosing = nextMeaningful === ':' || nextMeaningful === ',' || nextMeaningful === '}' || nextMeaningful === ']' || j >= json.length;
          if (isClosing) {
            inString = false;
            result += ch;
          } else {
            // Unescaped quote inside string value — escape it
            result += '\\"';
          }
        }
        continue;
      }

      result += ch;
    }

    return result;
  }

  /**
   * Экранирует реальные символы \n и \r внутри JSON string values,
   * не трогая структурные пробелы между ключами/значениями.
   */
  private escapeNewlinesInJsonStrings(json: string): string {
    let result = '';
    let inString = false;
    let escaped = false;

    for (let i = 0; i < json.length; i++) {
      const ch = json[i];

      if (escaped) {
        result += ch;
        escaped = false;
        continue;
      }

      if (ch === '\\' && inString) {
        result += ch;
        escaped = true;
        continue;
      }

      if (ch === '"') {
        inString = !inString;
        result += ch;
        continue;
      }

      if (inString) {
        if (ch === '\n') {
          result += '\\n';
          continue;
        }
        if (ch === '\r') {
          result += '\\r';
          continue;
        }
        if (ch === '\t') {
          result += '\\t';
          continue;
        }
      }

      result += ch;
    }

    return result;
  }

  private async parseDocumentWithRepair(content: string, rawText?: string): Promise<MedicalDocument> {
    try {
      return this.parseLlmJson<MedicalDocument>(content);
    } catch (firstError) {
      console.warn(`[llm] JSON parse failed (${firstError}), raw snippet: ${content.substring(0, 300)}`);
      try {
        const repaired = await this.repairJsonWithLlm(content);
        return this.parseLlmJson<MedicalDocument>(repaired);
      } catch (repairError) {
        console.warn(`[llm] LLM repair failed: ${repairError}`);
        // Graceful degradation: чем потерять 7 минут аудио, лучше отдать
        // пустой документ с сырым текстом в complaints — врач сам разнесёт по полям.
        console.warn('[llm] Falling back to mock document with raw transcription in complaints');
        return this.getMockStructuredDocument(rawText ?? content);
      }
    }
  }

  private async repairJsonWithLlm(brokenJson: string, schema: object = this.getDocumentJsonSchema()): Promise<string> {
    if (this.anthropic) {
      return this.anthropic.completeJson({
        systemPrompt: 'Fix invalid JSON. Return only valid JSON. Do not add new facts.',
        userPrompt: `Make this JSON valid:\n${brokenJson}`,
        jsonSchema: schema as any,
        toolName: 'submit_repaired_json',
        toolDescription: 'Submit the repaired JSON object.',
        maxTokens: 8192,
        temperature: 0,
        operation: 'repairJson',
      });
    }
    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: this.buildCompletionBody({
        prompt:
          `<|im_start|>system\n/no_think\nFix invalid JSON. Return only valid JSON. Do not add new facts.<|im_end|>\n` +
          `<|im_start|>user\nMake this JSON valid:\n${brokenJson}\n<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: 8192,
        temperature: 0,
        stop: ['<|im_end|>'],
        json_schema: schema,
        stream: false,
      }),
    });

    if (!response.ok) {
      await this.throwLlmError(response, 'repairJson');
    }

    const data = (await response.json()) as LlamaCompletionResponse;
    return data.content.trim();
  }

  private buildCompletionBody(payload: Record<string, unknown>): string {
    const body: Record<string, unknown> = {
      ...payload,
      model: this.config.model,
    };

    return JSON.stringify(body);
  }

  private getMockStructuredDocument(rawText: string): MedicalDocument {
    return {
      patient: {
        fullName: '',
        age: '',
        gender: '',
        complaintDate: '',
      },
      riskAssessment: { ...DEFAULT_RISK_ASSESSMENT },
      complaints: rawText.trim(),
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
    };
  }

  private enrichPatientFromRawText(document: MedicalDocument, rawText: string): MedicalDocument {
    const enriched: MedicalDocument = {
      ...document,
      patient: {
        ...document.patient,
      },
      riskAssessment: {
        ...(document.riskAssessment || DEFAULT_RISK_ASSESSMENT),
      },
    };

    const normalizedGender = this.normalizeGender(enriched.patient.gender);
    enriched.patient.gender = normalizedGender || this.extractGenderFromText(rawText) || '';

    const normalizedAge = this.normalizeAge(enriched.patient.age);
    enriched.patient.age = normalizedAge || this.extractAgeFromText(rawText) || '';

    return enriched;
  }

  private normalizeGender(value: string): string {
    const raw = value.trim().toLowerCase();
    if (!raw) return '';

    if (/(^|\b)(муж|мужской|мужчина|male|man)(\b|$)/i.test(raw)) return 'мужской';
    if (/(^|\b)(жен|женский|женщина|female|woman)(\b|$)/i.test(raw)) return 'женский';
    return '';
  }

  private normalizeAge(value: string): string {
    const raw = value.trim();
    if (!raw) return '';

    const direct = raw.match(/\b(\d{1,3})\s*(лет|года|год)\b/u);
    if (direct) return `${direct[1]} лет`;

    const digitsOnly = raw.match(/^\d{1,3}$/);
    if (digitsOnly) return `${digitsOnly[0]} лет`;

    const short = raw.match(/\b(\d{1,3})\s*л(?:\.|\b)/iu);
    if (short) return `${short[1]} лет`;

    return raw;
  }

  private extractGenderFromText(rawText: string): string {
    const text = rawText.toLowerCase();

    const explicitMale = /пол\s*[:\-]?\s*(мужской|мужчина|male|man)\b/iu;
    const explicitFemale = /пол\s*[:\-]?\s*(женский|женщина|female|woman)\b/iu;
    if (explicitMale.test(text)) return 'мужской';
    if (explicitFemale.test(text)) return 'женский';

    if (/\b(мужской|мужчина)\b/iu.test(text)) return 'мужской';
    if (/\b(женский|женщина)\b/iu.test(text)) return 'женский';

    return '';
  }

  private extractAgeFromText(rawText: string): string {
    const text = rawText.toLowerCase();

    const direct = text.match(/\b(\d{1,3})\s*(лет|года|год)\b/u);
    if (direct) return `${direct[1]} лет`;

    const byLabel = text.match(/(?:возраст|возраста)\s*[:\-]?\s*(\d{1,3})\b/u);
    if (byLabel) return `${byLabel[1]} лет`;

    const short = text.match(/\b(\d{1,3})\s*л(?:\.|\b)/iu);
    if (short) return `${short[1]} лет`;

    return '';
  }

  private getFieldLabel(field: RewriteableField): string {
    const labels: Record<RewriteableField, string> = {
      complaints: 'Жалобы',
      anamnesis: 'Анамнез заболевания',
      outpatientExams: 'Амбулаторные обследования',
      clinicalCourse: 'Анамнез жизни',
      allergyHistory: 'Аллергологический анамнез',
      objectiveStatus: 'Объективный статус',
      neurologicalStatus: 'Неврологический статус',
      diagnosis: 'Предварительный диагноз',
      finalDiagnosis: 'Заключение',
      conclusion: 'Амбулаторная терапия',
      doctorNotes: 'План обследования',
      recommendations: 'Рекомендации / План лечения',
    };
    return labels[field];
  }

  private getRiskAssessmentJsonSchema(): object {
    return {
      type: 'object',
      properties: {
        fallInLast3Months: { type: 'string' },
        dizzinessOrWeakness: { type: 'string' },
        needsEscort: { type: 'string' },
        painScore: { type: 'string' },
      },
      required: ['fallInLast3Months', 'dizzinessOrWeakness', 'needsEscort', 'painScore'],
      additionalProperties: false,
    };
  }

  private getDocumentJsonSchema(): object {
    return {
      type: 'object',
      properties: {
        patient: {
          type: 'object',
          properties: {
            fullName: { type: 'string' },
            age: { type: 'string' },
            gender: { type: 'string' },
            complaintDate: { type: 'string' },
          },
          required: ['fullName', 'age', 'gender', 'complaintDate'],
          additionalProperties: false,
        },
        riskAssessment: this.getRiskAssessmentJsonSchema(),
        complaints: { type: 'string' },
        anamnesis: { type: 'string' },
        outpatientExams: { type: 'string' },
        clinicalCourse: { type: 'string' },
        allergyHistory: { type: 'string' },
        objectiveStatus: { type: 'string' },
        neurologicalStatus: { type: 'string' },
        diagnosis: { type: 'string' },
        finalDiagnosis: { type: 'string' },
        conclusion: { type: 'string' },
        doctorNotes: { type: 'string' },
        recommendations: { type: 'string' },
      },
      required: [
        'patient',
        'riskAssessment',
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
      ],
      additionalProperties: false,
    };
  }

  private getAddendumPatchJsonSchema(): object {
    return {
      type: 'object',
      properties: {
        patient: {
          type: 'object',
          properties: {
            fullName: { type: 'string' },
            age: { type: 'string' },
            gender: { type: 'string' },
            complaintDate: { type: 'string' },
          },
          additionalProperties: false,
        },
        riskAssessment: {
          type: 'object',
          properties: {
            fallInLast3Months: { type: 'string' },
            dizzinessOrWeakness: { type: 'string' },
            needsEscort: { type: 'string' },
            painScore: { type: 'string' },
          },
          additionalProperties: false,
        },
        complaints: { type: 'string' },
        anamnesis: { type: 'string' },
        outpatientExams: { type: 'string' },
        clinicalCourse: { type: 'string' },
        allergyHistory: { type: 'string' },
        objectiveStatus: { type: 'string' },
        neurologicalStatus: { type: 'string' },
        diagnosis: { type: 'string' },
        finalDiagnosis: { type: 'string' },
        conclusion: { type: 'string' },
        doctorNotes: { type: 'string' },
        recommendations: { type: 'string' },
        diet: { type: 'string' },
      },
      additionalProperties: false,
    };
  }
}
