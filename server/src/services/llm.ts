import type { MedicalDocument, RiskAssessment, StructureResult, LLMConfig } from '../types.js';
import { findExamTemplate, formatExamLine } from '../data/examTemplates.js';
import { findDietTemplate } from '../data/dietTemplates.js';

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
        temperature: this.config.temperature,
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
        temperature: this.config.temperature,
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
    const raw = this.stripThinkingBlocks(data.content);
    const stoppedEos = (data as any).stop_type === 'eos' || (data as any).stopped_eos === true;
    const truncated = !stoppedEos && raw.length > 0;
    console.log(`LLM structureText response: ${raw.length} chars, stopped_eos: ${stoppedEos}, truncated: ${truncated}`);
    if (truncated) {
      console.warn(`[LLM] WARNING: Response may be truncated (no EOS). Input was ${rawText.length} chars. Consider increasing n_predict.`);
    }
    const document = await this.parseDocumentWithRepair(raw);
    const cleaned = this.validateAndCleanDocument(document);
    const enriched = this.enrichPatientFromRawText(cleaned, rawText);

    // Пре-экстракция: спасаем данные обследований из исходного текста,
    // которые LLM мог потерять
    this.rescueExamsFromRawText(enriched, rawText);

    // Спасаем диагноз если LLM его обрезал
    this.rescueDiagnosisFromRawText(enriched, rawText);

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
        temperature: 0.1,
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
        temperature: 0.2,
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
        temperature: 0.1,
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

  private stripThinkingBlocks(text: string): string {
    return text
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
    return `You are a medical assistant who must STRICTLY structure the doctor's dictation into a medical consultation document.

CRITICAL RULES — NEVER VIOLATE:
A) PRESERVE ALL DICTATED TEXT. Do NOT omit, summarize, shorten, or rephrase any information the doctor said. Every fact, every detail, every sentence must appear in the output. If the doctor said it, it MUST be in the JSON. You are a transcription structuring tool, not an editor.
B) SECTION BOUNDARIES ARE STRICT. When the doctor explicitly names a section (e.g. "анамнез жизни:", "диагноз:", "диета:", "жалобы:", "амбулаторная терапия:"), ALL subsequent text belongs ONLY to that section until the doctor names a different section. Do NOT copy or duplicate text between sections.
C) EACH PIECE OF INFORMATION GOES TO EXACTLY ONE FIELD. Never put the same fact into multiple fields. Choose the most appropriate field based on section markers or clinical context.

Field assignment rules:
1) "complaints" — patient complaints (жалобы). Only what the patient complains about.
2) "anamnesis" — disease history (анамнез заболевания). Timeline and course of current illness.
3) "outpatientExams" — ALL outpatient exam results and lab tests. This includes standard exams (ОАК, Б/х, ОАМ, ЭКГ, ЭхоКГ, ХМЭКГ, ЧПЭхоКГ, проверка ЭКС, рентген, УЗИ, СМАД, УЗДГ БЦА) AND any non-standard or specialized tests (тропонин, D-димер, BNP, NT-proBNP, липидный профиль, гликированный гемоглобин, гормоны щитовидной железы, ферритин, ПСА, иммунограмма, посевы, etc.). ANY test result with a date and/or numeric values belongs here.
   FORMATTING RULES for outpatientExams:
   - Output as NUMBERED LIST (1. 2. 3. ...), each exam on a new line.
   - For lab tests with parameters, use format: "ExamName от DATE: param1 - value unit, param2 - value unit."
   - Include ONLY the parameters the doctor actually dictated. Do NOT add parameters that were not mentioned.
   - Add measurement units for the dictated parameters:
     ОАК units: Hb - г/л, Эр - *10¹²/л, Л - *10⁹/л, Тр - *10⁹/л, СОЭ - мм/ч.
     Б/х units: общий белок - г/л, креатинин - мкмоль/л, СКФ - мл/мин/1,73 м², глюкоза - ммоль/л, АЛТ - МЕ/л, АСТ - МЕ/л, билирубин - мкмоль/л, ХС - ммоль/л, ЛПВП - ммоль/л, ЛПНП - ммоль/л, ТГ - ммоль/л, мочевая кислота - мкмоль/л, калий - ммоль/л, натрий - ммоль/л, железо - мкмоль/л, ферритин - нг/мл, КНТЖ - %.
     ОАМ units: отн. плотность, белок - г/л, Л - в п/з, Эр - в п/з.
   - For exams WITHOUT parameters (ЭКГ, ЭХОКГ, рентген, УЗИ, Холтер, СМАД, УЗДГ БЦА): "ExamName от DATE: description"
   - If doctor says only partial data (e.g. "ОАК Hb 149, эритроциты 5.4"), output ONLY those parameters with units — do NOT fill in a full template with empty values.
4) "clinicalCourse" — past medical history / life history (анамнез жизни / перенесённые заболевания). Surgeries, TB, hepatitis, injuries, family history, habits, comorbidities. Medications the patient PREVIOUSLY took or USED TO take go here.
5) "allergyHistory" — allergy history. Drug/food allergies or "отрицает".
6) "objectiveStatus" — physical examination findings (осмотр, аускультация, пульс, АД, температура, ИМТ, SpO2).
7) "neurologicalStatus" — neurological examination findings.
8) "diagnosis" — preliminary diagnosis (предварительный диагноз) with ICD-10 code if mentioned. This is the initial diagnosis before investigations.
8a) "finalDiagnosis" — final diagnosis (заключительный диагноз). If the doctor explicitly says "заключительный диагноз" or "окончательный диагноз", put it here. Otherwise leave empty.
9) "conclusion" — outpatient therapy (амбулаторная терапия). Medications the patient CURRENTLY takes at home. Not what the doctor prescribes now — only what patient already takes.
   FORMATTING: List medications as a numbered list (1. 2. 3. ...) each on a new line.
10) "doctorNotes" — investigation plan (план обследования). Lab orders, imaging orders, specialist consultations the doctor orders NOW.
11) "recommendations" — treatment plan (рекомендации / план лечения). Medications, dosages, regimens the doctor PRESCRIBES NOW. New prescriptions go here, not into "conclusion".
   FORMATTING RULES for recommendations:
   - When medications are listed, output as NUMBERED LIST (1. 2. 3. ...), each medication on a new line.
   - Format: "1. Таб.название дозировка по X таб. X раз(а) в день внутрь в TIME, описание;"
   - Include dosage, frequency, route of administration, time of day if mentioned.
   - Preserve all details about side effects, risks of discontinuation, and monitoring instructions.
12) "diet" — diet recommendations. If the doctor mentions a diet number (e.g. "диета 1а", "стол 5", "диета при диабете") or describes dietary restrictions, put it here. ONLY diet-related information.

Distinguishing medications:
- Medications patient CURRENTLY takes (the doctor says "амбулаторно принимает", "принимает", "на постоянной основе") → "conclusion" (амбулаторная терапия)
- Medications doctor PRESCRIBES NOW (the doctor says "назначаю", "рекомендую", "план лечения") → "recommendations" (план лечения)
- Medications mentioned in the CONTEXT OF LIFE HISTORY / ANAMNESIS (the doctor says "анамнез жизни... принимал", "лечился", "ранее получал", or medications mentioned INSIDE the анамнез жизни section) → "clinicalCourse" (анамнез жизни)
CRITICAL: The SECTION CONTEXT determines the field. If medications are mentioned while the doctor is dictating "анамнез жизни", they go to "clinicalCourse" even if they sound like current medications. Only medications explicitly introduced with "амбулаторно принимает" or "амбулаторная терапия" go to "conclusion".
Do NOT mix these. If the doctor says "амбулаторно принимает X" → conclusion. If the doctor says "назначаю Y" or "рекомендую Z" → recommendations.

Section boundary rules:
D) When the doctor finishes a section with a period (full stop / "точка") followed by silence or a new topic, that section is CLOSED. Subsequent text belongs to the NEXT appropriate section, NOT to the previous one. For example: "Жалобы на головокружение, нестабильность АД." — after this period, the next dictated text should NOT go into "complaints" but into the next relevant section.
E) The unit "мм рт.ст." should ALWAYS be abbreviated as "мм рт.ст." — never write it out as "миллиметров ртутного столба" or other variations.

General rules:
13) Do NOT add any information not present in the dictation.
14) If data is missing, return empty strings.
15) Extract patient age and gender ONLY if explicitly present.
16) Do NOT invent dates, diagnoses, or treatment plans.
17) Remove filler words (ну, вот, значит, так) but keep ALL medical content exactly as spoken.
18) Preserve drug names, dosages, and medical abbreviations exactly.
19) Extract risk assessment (Morse fall scale): "fallInLast3Months" (да/нет), "dizzinessOrWeakness" (да/нет), "needsEscort" (да/нет), "painScore" (число 0-10). Default all to "нет"/"0" if not mentioned.
20) Fix punctuation: remove excessive commas from speech pauses. Keep only grammatically correct punctuation.
21) Fix spelling errors in medical terms (e.g. "фибриляция" → "фибрилляция", "гипертензея" → "гипертензия").
22) Use Roman numerals for: functional class (ФК I-IV), NYHA (I-IV), CCS (I-IV), EHRA (I-IV), hypertension stages (ГБ I-III), AV-block degrees (I-III), valve insufficiency degrees (I-IV). Use Arabic numerals for diabetes types: СД 1 типа, СД 2 типа.

Return ONLY JSON, no extra text.`;
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
      conclusion: this.stripSectionPrefix('conclusion', doc.conclusion || ''),
      doctorNotes: this.stripSectionPrefix('doctorNotes', doc.doctorNotes || ''),
      recommendations: this.stripSectionPrefix('recommendations', doc.recommendations || ''),
      diet: this.stripSectionPrefix('diet', doc.diet || ''),
    };

    // Пост-обработка: перемещение контента с секционными маркерами в правильные поля
    this.redistributeMisplacedSections(result);

    // Пост-обработка: спасаем данные обследований из неправильных полей
    this.rescueExamData(result);

    // Пост-обработка: дедупликация диеты из recommendations/conclusion
    this.deduplicateDiet(result);

    // Пост-обработка: подстановка шаблона диеты по номеру
    this.expandDietTemplate(result);

    return result;
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

      // Извлекаем дату если есть
      const dateMatch = line.match(/от\s+([\d_.]+\s*(?:г\.?|года?)?)/iu);
      const date = dateMatch ? dateMatch[1] : undefined;

      // Извлекаем значения параметров из текста
      const values: Record<string, string> = {};
      for (const param of template.parameters) {
        // Ищем паттерн: "paramName - value" или "paramName value"
        const escapedName = param.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        // Ищем значение: число (возможно с запятой/точкой), может быть отрицательным
        const valuePattern = new RegExp(
          `${escapedName}\\s*[-–—:]\\s*([\\d.,]+)`,
          'iu'
        );
        const match = line.match(valuePattern);
        if (match) {
          values[param.name] = match[1];
        }
      }

      return formatExamLine(template, values, date);
    });

    // Нумеруем
    return processed.map((line, i) => `${i + 1}. ${line}`).join('\n');
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
      { pattern: /амбулаторн\S*\s+терапи\S*\s*[:.,-]?\s*/iu, target: 'conclusion' },
      { pattern: /амбулаторно\s+принимает\s*[:.,-]?\s*/iu, target: 'conclusion' },
      { pattern: /(?:рекомендаци\S*|рекомендовано|план\s+лечени\S*)\s*[:.,-]?\s*/iu, target: 'recommendations' },
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
          const searchPattern = new RegExp(`(?:^|\\n|[.!?]\\s+)\\s*(${pattern.source})`, 'iu');
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
   * Спасает данные обследований из исходного текста (raw), которые LLM мог потерять.
   * Сравнивает найденные в raw-тексте обследования с тем, что LLM вернул в outpatientExams.
   * Если обследование есть в raw но отсутствует в результате — добавляет его.
   */
  private rescueExamsFromRawText(doc: MedicalDocument, rawText: string): void {
    // Паттерны для извлечения полных блоков обследований из сырого текста
    const examBlockPatterns: Array<{ label: string; pattern: RegExp }> = [
      { label: 'ОАК', pattern: /(?:ОАК|общий анализ крови)[^.]*?(?:\.|$)/gimu },
      { label: 'Б/х', pattern: /(?:Б\/х|биохими\S+\s+анализ\s+крови|биохимия)[^.]*?(?:\.|$)/gimu },
      { label: 'ОАМ', pattern: /(?:ОАМ|общий анализ мочи|анализ мочи)[^.]*?(?:\.|$)/gimu },
      { label: 'коагулограмма', pattern: /(?:коагулограмма|гемостазиограмма)[^.]*?(?:\.|$)/gimu },
      { label: 'гликированный', pattern: /(?:гликированный гемоглобин|гликозилированный гемоглобин|HbA1c)[^.]*?(?:\.|$)/gimu },
      { label: '25-ОН', pattern: /(?:25-?ОН|витамин\s*[ДD]|вит\.?\s*[ДD])\s+[^.]*?(?:\.|$)/gimu },
      { label: 'ЭКГ', pattern: /(?:ЭКГ|электрокардиограмма)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { label: 'ЭхоКГ', pattern: /(?:ЭхоКГ|ЭХОКГ|эхокардиография)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { label: 'ХМЭКГ', pattern: /(?:ХМЭКГ|холтер\S*\s+мониторирование|холтер)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { label: 'СМАД', pattern: /(?:СМАД)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { label: 'УЗДГ', pattern: /(?:УЗДГ|УЗДС)\s+[^.]*?(?:\.|$)/gimu },
      { label: 'УЗИ', pattern: /(?:УЗИ)\s+[^.]*?(?:\.|$)/gimu },
      { label: 'рентген', pattern: /(?:рентген\S*)\s+[^.]*?(?:\.|$)/gimu },
      { label: 'МРТ', pattern: /(?:МРТ|КТ|МСКТ)\s+[^.]*?(?:\.|$)/gimu },
      { label: 'ФГДС', pattern: /(?:ФГДС|гастроскопия)\s+(?:от\s+)?[^.]*?(?:\.|$)/gimu },
      { label: 'тропонин', pattern: /(?:тропонин\S*)\s+[^.]*?(?:\.|$)/gimu },
      { label: 'D-димер', pattern: /(?:D-димер|д-димер)\s+[^.]*?(?:\.|$)/gimu },
      { label: 'BNP', pattern: /(?:BNP|NT-proBNP|про-БНП)\s+[^.]*?(?:\.|$)/gimu },
      { label: 'ферритин', pattern: /(?:ферритин)\s+[^.]*?(?:\.|$)/gimu },
      { label: 'ПСА', pattern: /(?:ПСА|PSA)\s+[^.]*?(?:\.|$)/gimu },
    ];

    const currentExams = doc.outpatientExams.toLowerCase();
    const rescued: string[] = [];

    for (const { label, pattern } of examBlockPatterns) {
      // Проверяем: есть ли этот тип обследования уже в результате?
      if (currentExams.includes(label.toLowerCase())) continue;

      // Ищем в сыром тексте
      pattern.lastIndex = 0;
      const matches = rawText.match(pattern);
      if (!matches) continue;

      for (const match of matches) {
        const cleaned = match.trim();
        // Минимальная валидация: должно содержать дату или числовое значение
        if (cleaned.length > 10 && (/\d/.test(cleaned))) {
          rescued.push(cleaned);
          console.log(`[postprocess] Rescued from raw text: "${cleaned.substring(0, 60)}..." (${label})`);
        }
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
    // Не перезаписываем если redistributeMisplacedSections уже правильно разложил поля
    // (т.е. план обследования и рекомендации уже заполнены)
    if (doc.doctorNotes.trim().length > 20 && doc.recommendations.trim().length > 20) {
      return; // перераспределение сработало — диагноз уже чистый
    }

    // Ищем блок диагноза в исходном тексте, останавливаясь на следующей секции
    // (через \n или через ". " + название секции)
    const diagMatch = rawText.match(
      /(?:предварительный\s+)?диагноз\s*[:.]?\s*([\s\S]*?)(?=(?:[\n.])\s*(?:план\s+обследовани|рекомендо|назначени|диет\S*\s*(?:№|\d)|питани|объективн|анамнез|жалоб|аллерг|данные\s+объективного))/iu
    );
    if (!diagMatch) return;

    const rawDiag = diagMatch[1].trim().replace(/\n{2,}/g, ' ').replace(/\s{2,}/g, ' ');
    const currentDiag = doc.diagnosis.trim();

    // Если диагноз из raw значительно длиннее (LLM обрезал) — заменяем
    if (rawDiag.length > 20 && rawDiag.length > currentDiag.length * 1.5) {
      console.log(`[postprocess] Diagnosis rescued from raw text: LLM had ${currentDiag.length} chars, raw has ${rawDiag.length} chars`);
      doc.diagnosis = rawDiag;
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

    const template = findDietTemplate(diet);
    if (template) {
      console.log(`[postprocess] Expanded diet template: "${diet}" → "${template.name}"`);
      doc.diet = template.description;
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
        return JSON.parse(sanitized) as T;
      }
    }
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
      console.warn(`[llm] JSON parse failed, attempting LLM repair: ${firstError}`);
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
