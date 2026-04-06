import type { MedicalDocument, RiskAssessment, StructureResult, LLMConfig } from '../types.js';
import { findExamTemplate, formatExamLine, parseExamValuesFromText, parseExamDate, examTemplates } from '../data/examTemplates.js';
import { findDietTemplate } from '../data/dietTemplates.js';
import { applyMedicalDictionary } from './medical-dictionary.js';

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
  'diet',
];

export class LLMService {
  private config: LLMConfig;

  constructor(config: LLMConfig) {
    this.config = config;
  }

  async healthCheck(): Promise<boolean> {
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
      const document = await this.structureWithLlamaCpp(rawText);
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
10) Medications patient CURRENTLY takes (doctor says "амбулаторно принимает") → "conclusion". New prescriptions (doctor says "назначаю", "рекомендую") → "recommendations". Medications mentioned in context of life history/анамнез жизни → "clinicalCourse". Diet → "diet".
11) IMPORTANT: If addendum mentions exam results (ОАК, Б/х, ЭКГ, ЭхоКГ, etc.), put them in "outpatientExams".`;

    const userPrompt = `Current document (JSON):
${JSON.stringify(document, null, 2)}

Addendum:
${addendumText}

Return JSON patch only.`;

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
      { field: 'diet', pattern: '(?:диет\\S*|стол\\s+\\d+|питани\\S*)' },
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
    const document = await this.parseDocumentWithRepair(raw);

    // Log raw LLM output BEFORE post-processing
    const LLM_FIELDS = ['complaints','anamnesis','clinicalCourse','allergyHistory','objectiveStatus',
      'diagnosis','finalDiagnosis','conclusion','recommendations','diet','doctorNotes','outpatientExams'] as const;
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

    // Log final result AFTER all post-processing
    const FINAL_FIELDS = ['complaints','anamnesis','clinicalCourse','allergyHistory','objectiveStatus',
      'diagnosis','finalDiagnosis','conclusion','recommendations','diet','doctorNotes','outpatientExams'] as const;
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

    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: this.buildCompletionBody({
        prompt: `<|im_start|>system\n/no_think\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: 256,
        temperature: 0,
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

    const response = await this.fetchWithTimeout(`${this.config.serverUrl}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: this.buildCompletionBody({
        prompt: `<|im_start|>system\n/no_think\n${systemPrompt}<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`,
        n_predict: 384,
        temperature: 0,
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
    return `Ты — медицинский ассистент. Твоя задача: структурировать диктовку врача в JSON-документ консультации.

ГЛАВНЫЕ ПРАВИЛА:
А) СОХРАНЯЙ ВЕСЬ ПРОДИКТОВАННЫЙ ТЕКСТ без сокращений и пропусков. Ты инструмент структурирования, не редактор.
Б) СТРОГИЕ ГРАНИЦЫ СЕКЦИЙ. Когда врач называет раздел — весь последующий текст принадлежит ТОЛЬКО этому разделу до следующего названия.
В) Каждый факт идёт РОВНО В ОДНО поле — дублирование запрещено.

ПОЛЯ И ЧТО В НИХ КЛАСТЬ:
1) "complaints" — жалобы пациента
2) "anamnesis" — анамнез заболевания (история текущей болезни, хронология)
3) "outpatientExams" — результаты обследований и анализов: НУМЕРОВАННЫЙ СПИСОК, каждый пункт с новой строки (\\n). Формат: "1. ОАК от ДД.ММ.ГГГГ: Hb - значение г/л, Эр - значение *10¹²/л". Единицы: Hb г/л, Эр *10¹²/л, глюкоза ммоль/л, креатинин мкмоль/л, АЛТ/АСТ МЕ/л, ХС/ЛПНП/ЛПВП ммоль/л, СОЭ мм/ч. Только то что продиктовано.
4) "clinicalCourse" — анамнез жизни: перенесённые болезни (туберкулёз, гепатиты), операции, травмы, вредные привычки, наследственность, сопутствующая патология, препараты которые пациент принимал РАНЕЕ. Гинекологический анамнез (беременности, роды, аборты, менструация, миома) → ТОЛЬКО сюда, НИКОГДА не в allergyHistory.
5) "allergyHistory" — ТОЛЬКО аллергические реакции и непереносимость препаратов. Максимум 1-2 предложения. ЗАПРЕЩЕНО класть сюда: гинекологические данные, объективный статус, анамнез жизни или другие данные.
6) "objectiveStatus" — данные объективного осмотра: общее состояние, кожные покровы, АД, ЧСС, ЧДД, ИМТ, SpO2, аускультация, живот, печень, отёки, стул, диурез. ЗАПРЕЩЕНО дублировать в другие поля.
7) "neurologicalStatus" — неврологический статус. Лабораторные данные сюда не класть.
8) "diagnosis" — ТОЛЬКО текст диагноза с кодом МКБ-10 если указан. НЕ включать: давление, анамнез, препараты.
8a) "finalDiagnosis" — только если врач говорит "заключительный диагноз", иначе пусто.
9) "conclusion" — амбулаторная терапия: препараты которые пациент СЕЙЧАС принимает ("принимает", "амбулаторно принимает") — нумерованный список (\\n между пунктами). НЕ новые назначения.
10) "doctorNotes" — план обследования: направления на анализы, ЭКГ, ЭхоКГ, консультации специалистов. НЕ рекомендации и НЕ схема лечения.
11) "recommendations" — НОВЫЕ назначения врача: препараты, процедуры, явка на приём — нумерованный список (\\n между пунктами). Формат: "1. Таб. Название доза по X таб. X раз/день". КАЖДЫЙ пункт = один препарат/назначение с полной информацией. НИКОГДА не дублировать в conclusion, doctorNotes или diet.
12) "diet" — ТОЛЬКО диетические рекомендации (номер стола или описание). НЕ схемы лечения, НЕ контроль АД, НЕ общие рекомендации.

МАРШРУТИЗАЦИЯ ПРЕПАРАТОВ:
- "принимает / амбулаторно принимает" → conclusion
- "назначаю / рекомендую / назначен" → recommendations
- В контексте анамнеза жизни → clinicalCourse

ПРАВИЛА ДЛЯ АНАЛИЗОВ:
- Каждый результат — к своей точной дате. НЕ дублировать показатели разных дат.
- Если дата неизвестна — писать "без даты".

ОБЩИЕ ПРАВИЛА:
- Не добавлять информацию которой нет в диктовке. Пустые строки если данных нет.
- Убирать слова-паразиты (ну, вот, значит) и мусор распознавания.
- Сохранять названия и дозировки препаратов точно.
- Шкала Морзе: fallInLast3Months/dizzinessOrWeakness/needsEscort (да/нет), painScore (0-10).
- "мм рт.ст." всегда сокращённо.
- Голосовые команды → символы: "скобка открывается"→"(", "закрывается"→")", "двоеточие"→":", "точка"→"."
- Даты: ДД.ММ.ГГГГг. — "5 февраля 26 года"→"05.02.2026г."
- Сокращения: ИМТ (не "индекс массы тела"), АД (не "артериальное давление")
- Вес в кг, рост в см. "34 и 2"→"34,2"
- Римские цифры для степень/стадия/класс/ФК: "третьей степени"→"III степени", "ФК 2 NYHA"→"ФК II (NYHA)"
- Арабские для тип/риск/баллы: "СД 2 типа", "риск 4"

КОРРЕКТУРА — это официальный медицинский документ:
- Исправляй опечатки (фибриляция→фибрилляция), грамматику, пропущенные предлоги ("течение"→"в течение")
- Пунктуация: точки между предложениями, запятые в перечислениях, заглавная буква после точки
- Нет двойной пунктуации, нет запятой перед точкой, пробел после каждого знака препинания

ПРИМЕР МАРШРУТИЗАЦИИ:
Врач диктует: "...аллергоанамнез: на ингибиторы АПФ — сухой кашель. Объективно: общее состояние средней тяжести, АД 130/80 мм рт.ст., ЧСС 72 уд/мин. Амбулаторно принимает: бисопролол 5 мг, эналаприл 10 мг. Рекомендую: 1. Таб. Лозартан 50 мг 1 раз/день..."
→ allergyHistory: "На ингибиторы АПФ — сухой кашель."
→ objectiveStatus: "Общее состояние средней тяжести. АД 130/80 мм рт.ст., ЧСС 72 уд/мин."
→ conclusion: "1. Бисопролол 5 мг\n2. Эналаприл 10 мг"
→ recommendations: "1. Таб. Лозартан 50 мг 1 раз/день"

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
  "outpatientExams": "Данные имеющихся дополнительных исследований — НУМЕРОВАННЫЙ СПИСОК. Каждое обследование с датой и единицами измерения. Формат: 1. ОАК от дата: Hb - значение г/л, Эр - значение *10¹²/л, ... 2. Б/х анализ крови от дата: ... Всегда подставлять единицы измерения.",
  "clinicalCourse": "Анамнез жизни (перенесённые заболевания: туберкулёз, гепатиты, операции, травмы, наследственность, вредные привычки, сопутствующая патология, препараты которые принимал РАНЕЕ)",
  "allergyHistory": "Аллергологический анамнез (непереносимость препаратов, пищевых продуктов)",
  "objectiveStatus": "Объективный статус (осмотр, аускультация, пульс, АД, температура, ИМТ, SpO2)",
  "neurologicalStatus": "Неврологический статус",
  "diagnosis": "Предварительный диагноз (с кодом МКБ-10 если озвучен)",
  "finalDiagnosis": "Заключительный диагноз (если озвучен отдельно, иначе пусто)",
  "conclusion": "Амбулаторная терапия (ТОЛЬКО препараты которые пациент СЕЙЧАС принимает амбулаторно — НУМЕРОВАННЫЙ СПИСОК)",
  "doctorNotes": "План обследования (направления на анализы, ЭКГ, ЭхоКГ, консультации специалистов)",
  "recommendations": "Рекомендации / План лечения (НОВЫЕ назначения: НУМЕРОВАННЫЙ СПИСОК. 1. Таб.название дозировка по ... в день; 2. ...)",
  "diet": "Диета (номер диеты или описание диетических рекомендаций, если озвучены)"
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

  private validateAndCleanDocument(doc: MedicalDocument): MedicalDocument {
    const rawAge = doc.patient?.age || '';
    const rawGender = doc.patient?.gender || '';
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
      outpatientExams: this.postProcessOutpatientExams(this.stripSectionPrefix('outpatientExams', doc.outpatientExams || '')),
      clinicalCourse: this.stripSectionPrefix('clinicalCourse', doc.clinicalCourse || ''),
      allergyHistory: this.stripSectionPrefix('allergyHistory', doc.allergyHistory || ''),
      objectiveStatus: this.stripSectionPrefix('objectiveStatus', doc.objectiveStatus || ''),
      neurologicalStatus: this.stripSectionPrefix('neurologicalStatus', doc.neurologicalStatus || ''),
      diagnosis: this.stripSectionPrefix('diagnosis', doc.diagnosis || ''),
      finalDiagnosis: this.stripSectionPrefix('finalDiagnosis', doc.finalDiagnosis || ''),
      conclusion: this.splitInlineNumberedList(this.stripSectionPrefix('conclusion', doc.conclusion || '')),
      doctorNotes: this.formatDoctorNotesAsList(this.stripSectionPrefix('doctorNotes', doc.doctorNotes || '')),
      recommendations: this.splitInlineNumberedList(this.stripSectionPrefix('recommendations', doc.recommendations || '')),
      diet: this.stripSectionPrefix('diet', doc.diet || ''),
    };

    // Пост-обработка: очистка полей состоящих только из цифр/пунктуации
    for (const field of ALL_TEXT_FIELDS) {
      const v = result[field];
      if (v && /^\s*[\d\s.,;:!?-]{1,5}\s*$/.test(v)) {
        console.log(`[postprocess] Cleared garbage-only field ${field}: "${v.trim()}"`);
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

    // Пост-обработка: дедупликация диеты из recommendations/conclusion
    this.deduplicateDiet(result);

    // Пост-обработка: подстановка шаблона диеты по номеру
    this.expandDietTemplate(result);

    // Пост-обработка: чистка diagnosis от нерелевантных данных (анамнез жизни, обследования)
    this.cleanDiagnosis(result);

    // Пост-обработка: дедупликация внутри полей (LLM иногда дублирует весь блок)
    this.deduplicateFields(result);

    // Пост-обработка: чистка diet от данных рекомендаций
    this.cleanDietField(result);

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

    // Split by numbered items or sentences
    const items = reco.split(/(?=\d+\.\s)|(?<=[.!?])\s+/).filter(s => s.trim());
    const clean: string[] = [];

    // Medical recommendation keywords
    const medicalKeywords = /(?:таблетк[аи]|капсул|мг\b|раз\s+в\s+день|внутрь|длительно|постоянно|контроль|ингибитор|назначен|неназначен|диет[аы]|стол\s+№?\d|ограничен|рекомендован|направлен|обследован|консультаци|анализ|ЭКГ|ЭХО|МРТ|КТ|рентген|осмотр|повторн|бисопролол|конкор|форсига|джардинс|верошпирон|эналаприл|лизиноприл|дигоксин|ксарелто|варфарин|курс\b|терапи|лечени|приём|принимать|памятк[аи]\s+ВТЭ)/iu;
    // Historical phrases that don't belong in recommendations
    const historicalKeywords = /(?:имплантаци\S+\s+(?:ЭКС|ИКД|CRT|кардио)|терапи\S*\s+по\s+схем|обратилась?\s+для\s+решения\s+вопроса|консультирован\S*\s+(?:аритмолог|кардиолог|кардиохирург)|выписан\S*\s+на\s+терапи|проведен\S*\s+(?:оперативн|КАГ|АКШ|баллон|ангиопластик))/iu;
    // Garbage patterns
    const garbageKeywords = /(?:видео\s+в\s+компьютер|тема\s+от\s+\d|объёмы\s+карт|сфера|фоти|превраду|настройки\s+наиболее|[&@#$%])/iu;

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

    if (clean.length < items.length) {
      // Remove orphan numbers left after deleting numbered items (e.g. lone "2\n")
      const joined = clean.join('\n')
        .replace(/^\d+\s*$/gm, '')
        .replace(/\n{2,}/g, '\n')
        .trim();
      doc.recommendations = joined;
      console.log(`[postprocess] Cleaned recommendations: ${items.length} → ${clean.length} items`);
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

      // Parse recommendations block: split diet vs medication/other recs
      const dietParts: string[] = [];
      const recoParts: string[] = [];

      // Split by numbered items or sentence boundaries
      const recoItems = recoBlock.split(/(?=\d+\.\s)/).filter(s => s.trim());
      const dietKeywords = /(?:диет[аы]|стол\s*(?:№?\s*)?\d+|гипохолестерин|ограничен\S*\s+(?:соли|жидкости)|питание)/iu;
      const nonDietRecoKeywords = /(?:таблетк[аи]|капсул|мг\s+по|раз\s+в\s+день|внутрь|длительно|постоянно|контроль\s+(?:АД|ЧСС|пульса)|памятк[аи]\s+ВТЭ|ингибитор|назначен|неназначен|титров)/iu;

      for (const item of recoItems) {
        const trimmed = item.trim();
        if (!trimmed) continue;
        if (dietKeywords.test(trimmed) && !nonDietRecoKeywords.test(trimmed)) {
          dietParts.push(trimmed);
        } else {
          recoParts.push(trimmed);
        }
      }

      // Append diet parts to doc.diet
      if (dietParts.length > 0) {
        const dietText = dietParts.join(' ').trim();
        const existingDiet = (doc.diet || '').trim();
        if (!existingDiet || !existingDiet.includes(dietText.substring(0, Math.min(30, dietText.length)))) {
          doc.diet = existingDiet ? `${existingDiet} ${dietText}`.trim() : dietText;
        }
        console.log(`[postprocess] Moved diet from diagnosis recommendations → diet (${dietText.length} chars)`);
      }

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

  /**
   * Чистит diet от данных, которые не относятся к диете.
   * LLM иногда сваливает рекомендации, памятки и контроль АД в diet.
   */
  private cleanDietField(doc: MedicalDocument): void {
    let diet = doc.diet;
    if (!diet || diet.length < 30) return;

    // Step 0: Remove Whisper garbage — sentences containing Latin non-medical words
    const medicalLatinWords = /(?:CRTD|MRI|NYHA|EHRA|CRT|VVI|DDD|VVIR|BRAVO|Medtronic|CRT-D|SpO|HbA|BNP|DASH|PCI|SCORE)/i;
    diet = diet.split(/(?<=[.!?])\s+/).filter(sentence => {
      // Check for non-medical Latin words (3+ chars, lowercase or capitalized)
      const latinWords = sentence.match(/\b[a-zA-Z]{3,}\b/g) || [];
      const nonMedicalLatin = latinWords.filter(w => !medicalLatinWords.test(w));
      if (nonMedicalLatin.length >= 2) {
        console.log(`[postprocess] Removed Whisper garbage from diet: "${sentence.substring(0, 50)}..."`);
        return false;
      }
      // Remove sentences with special chars garbage (&, @, #, etc.)
      if (/[&@#$%]+/.test(sentence)) {
        console.log(`[postprocess] Removed garbage chars from diet: "${sentence.substring(0, 50)}..."`);
        return false;
      }
      return true;
    }).join(' ');

    // Split by newlines AND sentences (diet often comes as one block)
    const lines = diet.split(/\n|(?<=[.!?])\s+/).filter(l => l.trim());
    const dietLines: string[] = [];

    // Keywords that BELONG in diet
    const dietKeywords = /(?:диет[аы]|стол\s*(?:№?\s*)?\d+|гипохолестерин|ограничен\S*\s+(?:соли|жидкости|жиров)|исключить|калорийност|питание|рацион|продукт|овощи|фрукт|молочн|каш[иу]|хлеб|мясо|рыб[аыу]|жареное|солёное|копчён|номер\s+\d+|литр[аов]?\b|памятк\S*\s+на\s+руки|на\s+руки)/iu;

    // Keywords that DON'T belong in diet
    const nonDietKeywords = /(?:таблетк[аи]|капсул[аы]|инъекци|мг\s+по|раз\s+в\s+день|внутрь|длительно|постоянно|контроль\s+(?:АД|ЧСС|пульса)|памятк[аи]\s+ВТЭ|ингибитор|назначен|неназначен|титров)/iu;

    for (const line of lines) {
      const cleaned = line.replace(/^\d+\.\s*/, '').trim();
      if (!cleaned) continue;

      if (nonDietKeywords.test(cleaned) && !dietKeywords.test(cleaned)) {
        console.log(`[postprocess] Removed non-diet content from diet: "${cleaned.substring(0, 50)}..."`);
      } else if (!dietKeywords.test(cleaned) && cleaned.length > 5) {
        // No diet keywords at all — likely Whisper garbage
        console.log(`[postprocess] Removed non-diet garbage from diet: "${cleaned.substring(0, 50)}..."`);
      } else {
        dietLines.push(line);
      }
    }

    doc.diet = dietLines.join('\n').trim();

    // Remove number-sequence garbage: "1. 2. 3. 4. 5. 6. 7. 8. 9. 10."
    doc.diet = doc.diet.replace(/(?:\s*\d+\.\s*){4,}\s*$/gu, '').trim();
    // Remove "cycling/following" + garbage
    doc.diet = doc.diet.replace(/\s*(?:cycling|following|next)\s*[.\s]*(?:\d+\.\s*)*.*$/giu, '').trim();

    // Remove duplicate diet descriptions (exact and substring-level)
    if (doc.diet) {
      const parts = doc.diet.split(/[.\n]+/).filter(p => p.trim());
      const unique: string[] = [];
      for (const part of parts) {
        const norm = part.trim().toLowerCase().replace(/\s+/g, ' ');
        if (norm.length < 3) continue;
        // Check if this part is already contained in an existing part (substring dedup)
        const isDup = unique.some(existing => {
          const existNorm = existing.trim().toLowerCase().replace(/\s+/g, ' ');
          return existNorm === norm || existNorm.includes(norm) || norm.includes(existNorm);
        });
        if (isDup) {
          console.log(`[postprocess] Removed duplicate diet part: "${part.trim().substring(0, 40)}..."`);
          continue;
        }
        unique.push(part.trim());
      }
      doc.diet = unique.join('. ').trim();
      if (doc.diet && !doc.diet.endsWith('.')) doc.diet += '.';
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
      .filter((l) => l.length > 0);

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
          values[param.name] = match[1];
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
      { pattern: /(?:рекомендаци\S*|рекомендован\S*|план\s+лечени\S*)\s*[:.,-]?\s*/iu, target: 'recommendations' },
      { pattern: /диет\S*\s*(?:№?\s*\d+\S*)?\s*[:.,-]?\s*/iu, target: 'diet' },
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
   * Убирает дублирование диеты: если diet не пустое и та же информация есть
   * в recommendations или conclusion — убирает её оттуда.
   */
  private deduplicateDiet(doc: MedicalDocument): void {
    const diet = doc.diet.trim();
    if (!diet) return;

    // Паттерны для обнаружения диетной информации в других полях
    const dietPatterns = [
      /диет\S*\s*(?:№?\s*\d+\S*)?[^.]*\.?\s*/giu,
      /стол\s*(?:№?\s*)?\d+\S*[^.]*\.?\s*/giu,
    ];

    for (const field of ['recommendations', 'conclusion'] as RewriteableField[]) {
      let text = doc[field];
      if (!text) continue;

      for (const pattern of dietPatterns) {
        const before = text;
        text = text.replace(pattern, '').trim();
        if (text !== before) {
          console.log(`[postprocess] Removed duplicate diet info from ${field}`);
        }
      }

      // Убираем пустые строки оставшиеся после удаления
      doc[field] = text.replace(/\n{3,}/g, '\n\n').trim();
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
   * Подставляет шаблон диеты по номеру.
   * Если в поле diet указан только номер диеты (например "Стол 10", "Диета 5"),
   * заменяет его на полный текст шаблона.
   */
  private expandDietTemplate(doc: MedicalDocument): void {
    const diet = doc.diet.trim();
    if (!diet) return;

    // Только подставляем шаблон если врач продиктовал ТОЛЬКО номер диеты (без конкретных инструкций).
    // Если текст длиннее ~25 символов — это конкретные рекомендации врача, оставляем как есть.
    const isJustDietNumber = diet.length <= 25;
    if (!isJustDietNumber) return;

    const template = findDietTemplate(diet);
    if (template) {
      console.log(`[postprocess] Expanded diet template: "${diet}" → "${template.name}"`);
      doc.diet = template.description;
    }
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
      diet: [
        /^диет\S*\s*[:.,-]?\s*/iu,                                    // «Диета:» / «Диета 5:»
        /^стол\s*\d+\S*\s*[:.,-]?\s*/iu,                              // «Стол 5:»
        /^питани\S*\s*[:.,-]?\s*/iu,                                   // «Питание:»
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

  private async parseDocumentWithRepair(content: string): Promise<MedicalDocument> {
    try {
      return this.parseLlmJson<MedicalDocument>(content);
    } catch (firstError) {
      console.warn(`[llm] JSON parse failed (${firstError}), raw snippet: ${content.substring(0, 300)}`);
      try {
        const repaired = await this.repairJsonWithLlm(content);
        return this.parseLlmJson<MedicalDocument>(repaired);
      } catch (repairError) {
        console.warn(`[llm] LLM repair failed: ${repairError}`);
        throw new Error(`Failed to parse LLM JSON after repair attempt: ${firstError}`);
      }
    }
  }

  private async repairJsonWithLlm(brokenJson: string, schema: object = this.getDocumentJsonSchema()): Promise<string> {
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
      diet: '',
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
      finalDiagnosis: 'Заключительный диагноз',
      conclusion: 'Амбулаторная терапия',
      doctorNotes: 'План обследования',
      recommendations: 'Рекомендации / План лечения',
      diet: 'Диета',
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
        diet: { type: 'string' },
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
        'diet',
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
