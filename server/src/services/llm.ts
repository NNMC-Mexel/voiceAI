import type { MedicalDocument, StructureResult, LLMConfig } from '../types.js';

interface LlamaCompletionResponse {
  content: string;
}

type MedicalDocumentPatch = Partial<
  Omit<MedicalDocument, 'patient'> & {
    patient: Partial<MedicalDocument['patient']>;
  }
>;

type RewriteableField = keyof Omit<MedicalDocument, 'patient'>;

const ALL_TEXT_FIELDS: RewriteableField[] = [
  'complaints',
  'anamnesis',
  'clinicalCourse',
  'allergyHistory',
  'objectiveStatus',
  'neurologicalStatus',
  'diagnosis',
  'conclusion',
  'recommendations',
  'doctorNotes',
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
6) Output STRICT JSON only.`;

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
        n_predict: this.config.maxTokens,
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
    const patch = await this.parsePatchWithRepair(raw);
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
6) Output STRICT JSON only.`;

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
        n_predict: this.config.maxTokens,
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
      { field: 'clinicalCourse', pattern: 'анамнез\\s+жизни' },
      { field: 'allergyHistory', pattern: 'аллерг\\S*' },
      { field: 'neurologicalStatus', pattern: 'неврологическ\\S*' },
      { field: 'anamnesis', pattern: 'анамнез\\S*' },
      { field: 'complaints', pattern: 'жалоб\\S*' },
      { field: 'objectiveStatus', pattern: 'объектив\\S*' },
      { field: 'diagnosis', pattern: 'диагноз\\S*' },
      { field: 'conclusion', pattern: '(?:сопутствующ\\S*|заключени\\S*)' },
      { field: 'recommendations', pattern: '(?:план\\s+лечени\\S*|рекомендац\\S*)' },
      { field: 'doctorNotes', pattern: '(?:прочее|заметк\\S*\\s+врача|заметк\\S*|примечан\\S*)' },
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
        n_predict: this.config.maxTokens,
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
    const document = await this.parseDocumentWithRepair(data.content);
    const cleaned = this.validateAndCleanDocument(document);
    return this.enrichPatientFromRawText(cleaned, rawText);
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

    for (const key of [
      'complaints',
      'anamnesis',
      'clinicalCourse',
      'allergyHistory',
      'objectiveStatus',
      'neurologicalStatus',
      'diagnosis',
      'conclusion',
      'recommendations',
      'doctorNotes',
    ] as const) {
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
    };

    if (patch.patient && typeof patch.patient === 'object') {
      for (const key of ['fullName', 'age', 'gender', 'complaintDate'] as const) {
        const value = patch.patient[key];
        if (typeof value === 'string') {
          merged.patient[key] = value.trim();
        }
      }
    }

    for (const key of [
      'complaints',
      'anamnesis',
      'clinicalCourse',
      'allergyHistory',
      'objectiveStatus',
      'neurologicalStatus',
      'diagnosis',
      'conclusion',
      'recommendations',
      'doctorNotes',
    ] as const) {
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
    return `You are a medical assistant who must STRICTLY structure the doctor's dictation into a "Первичный осмотр" (initial examination) document.\n\nRules:\n1) Do NOT add any information not present in the dictation.\n2) If data is missing, return empty strings.\n3) Extract patient age and gender ONLY if explicitly present in the dictation.\n4) Do NOT invent dates, diagnoses, or treatment plans.\n5) Remove filler words but keep all medical terminology exactly as spoken.\n6) Convert numbers from words to digits only if explicitly said.\n7) Preserve drug names, dosages, and medical abbreviations exactly.\n\nReturn ONLY JSON, no extra text.`;
  }

  private getUserPrompt(rawText: string): string {
    return `Structure the following medical dictation into a "Первичный осмотр" document.\n\nTEXT:\n${rawText}\n\nReturn STRICT JSON (use empty strings if data is missing):\n{\n  "patient": {\n    "fullName": "ФИО пациента или пусто",\n    "age": "Возраст, например 45 лет, или пусто",\n    "gender": "мужской | женский | пусто",\n    "complaintDate": "YYYY-MM-DD или пусто"\n  },\n  "complaints": "Жалобы при поступлении",\n  "anamnesis": "Анамнез заболевания",\n  "clinicalCourse": "Анамнез жизни (перенесённые болезни, операции, хронические заболевания)",\n  "allergyHistory": "Аллергологический анамнез (непереносимость препаратов, пищевых продуктов)",\n  "objectiveStatus": "Объективные данные (осмотр, аускультация, пульс, АД, температура, SpO2)",\n  "neurologicalStatus": "Неврологический статус",\n  "diagnosis": "Предварительный диагноз (основной)",\n  "conclusion": "Сопутствующий диагноз",\n  "recommendations": "План лечения (назначенные препараты и процедуры)",\n  "doctorNotes": "Прочее (план обследования, консультации)"\n}\n\nJSON:`;
  }

  private validateAndCleanDocument(doc: MedicalDocument): MedicalDocument {
    const rawAge = doc.patient?.age || '';
    const rawGender = doc.patient?.gender || '';
    return {
      patient: {
        fullName: doc.patient?.fullName || '',
        age: this.normalizeAge(rawAge),
        gender: this.normalizeGender(rawGender),
        complaintDate: doc.patient?.complaintDate || '',
      },
      complaints: this.stripSectionPrefix('complaints', doc.complaints || ''),
      anamnesis: this.stripSectionPrefix('anamnesis', doc.anamnesis || ''),
      clinicalCourse: this.stripSectionPrefix('clinicalCourse', doc.clinicalCourse || ''),
      allergyHistory: this.stripSectionPrefix('allergyHistory', doc.allergyHistory || ''),
      objectiveStatus: this.stripSectionPrefix('objectiveStatus', doc.objectiveStatus || ''),
      neurologicalStatus: this.stripSectionPrefix('neurologicalStatus', doc.neurologicalStatus || ''),
      diagnosis: this.stripSectionPrefix('diagnosis', doc.diagnosis || ''),
      conclusion: this.stripSectionPrefix('conclusion', doc.conclusion || ''),
      recommendations: this.stripSectionPrefix('recommendations', doc.recommendations || ''),
      doctorNotes: this.stripSectionPrefix('doctorNotes', doc.doctorNotes || ''),
    };
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
        /^диагноз\S*\s*[:.,-]\s*/iu,        // «Диагноз:» / «Диагноз.»
      ],
      clinicalCourse: [
        /^анамнез\s+жизни\s*[:.,-]?\s*/iu,             // «Анамнез жизни:»
        /^(?:клиническое\s+)?течени\S*\s*[:.,-]\s*/iu, // «Течение:» / «Клиническое течение:» (старый маркер)
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
        /^сопутствующ\S*\s+диагноз\S*\s*[:.,-]?\s*/iu,             // «Сопутствующий диагноз:»
        /^заключени\S*\s*(?:врача|специалиста)?\s*[:.,-]\s*/iu,     // «Заключение:» (старый маркер)
      ],
      recommendations: [
        /^план\s+лечени\S*\s*[:.,-]?\s*/iu,              // «План лечения:»
        /^рекомендаци\S*\s*[:.,-]\s*/iu,                 // «Рекомендации:»
        /^(?:рекомендую|рекомендуется|рекомендует|рекомендуем)\s+/iu, // глагольные формы
      ],
      doctorNotes: [
        /^прочее\s*[:.,-]?\s*/iu,                                    // «Прочее:»
        /^(?:заметк[иа]\s+врача|заметк[иа]|примечани\S*)\s*[:.,-]\s*/iu, // «Заметки врача:»
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
      const sanitized = rawJson
        .replace(/```json|```/gi, '')
        .replace(/[“”]/g, '"')
        .replace(/[‘’]/g, "'")
        .replace(/([{,]\s*)([A-Za-z_][\w]*)(\s*:)/g, '$1"$2"$3')
        .replace(/'([^'\\]*(?:\\.[^'\\]*)*)'/g, '"$1"')
        .replace(/,\s*([}\]])/g, '$1')
        .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, ' ');

      return JSON.parse(sanitized) as T;
    }
  }

  private async parseDocumentWithRepair(content: string): Promise<MedicalDocument> {
    try {
      return this.parseLlmJson<MedicalDocument>(content);
    } catch {
      const repaired = await this.repairJsonWithLlm(content);
      return this.parseLlmJson<MedicalDocument>(repaired);
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
        n_predict: 512,
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
      complaints: rawText.trim(),
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
  }

  private enrichPatientFromRawText(document: MedicalDocument, rawText: string): MedicalDocument {
    const enriched: MedicalDocument = {
      ...document,
      patient: {
        ...document.patient,
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
    return labels[field];
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
        complaints: { type: 'string' },
        anamnesis: { type: 'string' },
        clinicalCourse: { type: 'string' },
        allergyHistory: { type: 'string' },
        objectiveStatus: { type: 'string' },
        neurologicalStatus: { type: 'string' },
        diagnosis: { type: 'string' },
        conclusion: { type: 'string' },
        recommendations: { type: 'string' },
        doctorNotes: { type: 'string' },
      },
      required: [
        'patient',
        'complaints',
        'anamnesis',
        'clinicalCourse',
        'allergyHistory',
        'objectiveStatus',
        'neurologicalStatus',
        'diagnosis',
        'conclusion',
        'recommendations',
        'doctorNotes',
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
        complaints: { type: 'string' },
        anamnesis: { type: 'string' },
        clinicalCourse: { type: 'string' },
        allergyHistory: { type: 'string' },
        objectiveStatus: { type: 'string' },
        neurologicalStatus: { type: 'string' },
        diagnosis: { type: 'string' },
        conclusion: { type: 'string' },
        recommendations: { type: 'string' },
        doctorNotes: { type: 'string' },
      },
      additionalProperties: false,
    };
  }
}
