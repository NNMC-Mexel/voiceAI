/**
 * DocumentExtractorService — извлекает текст из PDF, Word (.docx) и изображений.
 *
 * PDF/Word: локально, без внешних API.
 * Изображения: Claude Vision (требует ANTHROPIC_API_KEY).
 */

import Anthropic from '@anthropic-ai/sdk';

export type ExtractionMethod = 'pdf' | 'word' | 'vision' | 'text';
type VisionProvider = 'anthropic' | 'ollama';

interface OllamaVisionConfig {
  serverUrl: string;
  model: string;
  timeoutMs: number;
}

interface DocumentExtractorOptions {
  visionProvider?: VisionProvider;
  anthropicApiKey?: string;
  ollama?: OllamaVisionConfig;
}

export interface DocumentExtractionResult {
  text: string;
  extractionMethod: ExtractionMethod;
  pageCount?: number;
  /** Предупреждение, если PDF был отсканирован (мало текста) */
  warning?: string;
}

// MIME-типы, поддерживаемые каждым экстрактором
const PDF_MIMES = new Set(['application/pdf']);
const WORD_MIMES = new Set([
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/msword',
]);
const IMAGE_MIMES = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif', 'image/heic', 'image/heif']);

// Расширения файлов как fallback если MIME не определён браузером
function detectTypeByFilename(filename: string): ExtractionMethod | null {
  const lower = filename.toLowerCase();
  if (lower.endsWith('.pdf')) return 'pdf';
  if (lower.endsWith('.docx')) return 'word';
  if (lower.endsWith('.doc')) return 'word';
  if (/\.(jpe?g|png|webp|gif|heic|heif|bmp|tiff?)$/.test(lower)) return 'vision';
  return null;
}

function messageFromError(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  try {
    return JSON.stringify(error);
  } catch {
    return 'Unknown error';
  }
}

function isRecoverableVisionPdfError(message: string): boolean {
  return /unsupported|media_type|document|pdf|invalid.*image/i.test(message);
}

function toVisionUserError(error: unknown): Error {
  const message = messageFromError(error);

  if (/credit balance is too low|billing|purchase credits|insufficient/i.test(message)) {
    return new Error('Недостаточно средств на Anthropic API для распознавания фото/PDF. Пополните баланс Anthropic или загрузите текстовый PDF/Word.');
  }

  if (/invalid x-api-key|authentication|api key|unauthorized/i.test(message)) {
    return new Error('Anthropic API key недействителен. Проверьте ANTHROPIC_API_KEY в server/.env и перезапустите сервер.');
  }

  if (/rate limit|too many requests/i.test(message)) {
    return new Error('Anthropic API временно ограничил запросы. Повторите попытку позже.');
  }

  return new Error(`Ошибка Claude Vision: ${message}`);
}

export class DocumentExtractorService {
  private visionApiKey: string | undefined;
  private visionProvider: VisionProvider;
  private ollama: OllamaVisionConfig | undefined;

  constructor(options?: string | DocumentExtractorOptions) {
    if (typeof options === 'string' || options === undefined) {
      this.visionApiKey = options;
      this.visionProvider = 'anthropic';
      return;
    }

    this.visionApiKey = options.anthropicApiKey;
    this.visionProvider = options.visionProvider ?? (options.ollama ? 'ollama' : 'anthropic');
    this.ollama = options.ollama;
  }

  get canProcessImages(): boolean {
    return this.visionProvider === 'ollama' ? !!this.ollama : !!this.visionApiKey;
  }

  async extract(
    buffer: Buffer,
    mimetype: string,
    filename: string,
  ): Promise<DocumentExtractionResult> {
    const mime = mimetype.toLowerCase().split(';')[0].trim();

    if (PDF_MIMES.has(mime) || detectTypeByFilename(filename) === 'pdf') {
      return this.extractPdf(buffer, filename);
    }

    if (WORD_MIMES.has(mime) || detectTypeByFilename(filename) === 'word') {
      if (filename.toLowerCase().endsWith('.doc')) {
        throw new Error(
          'Формат .doc (старый Word) не поддерживается. Сохраните документ в формате .docx.',
        );
      }
      return this.extractWord(buffer);
    }

    if (IMAGE_MIMES.has(mime) || detectTypeByFilename(filename) === 'vision') {
      return this.extractImage(buffer, mime, filename);
    }

    throw new Error(
      `Неподдерживаемый формат файла: ${mimetype || filename}. ` +
      'Поддерживаются: PDF, Word (.docx), изображения (JPEG, PNG, WebP).',
    );
  }

  // ─── PDF ──────────────────────────────────────────────────────────────────

  private async extractPdf(buffer: Buffer, filename: string): Promise<DocumentExtractionResult> {
    const { PDFParse } = await import('pdf-parse');
    const parser = new PDFParse({ data: buffer });

    let data: Awaited<ReturnType<typeof parser.getText>>;
    try {
      data = await parser.getText();
    } catch (err) {
      throw new Error(
        `Не удалось прочитать PDF: ${err instanceof Error ? err.message : 'неизвестная ошибка'}`,
      );
    } finally {
      await parser.destroy().catch(() => {});
    }
    const rawText = (data.text ?? '').trim();

    // Эвристика: если менее 50 символов на страницу — скорее всего отсканированный PDF
    const pageCount = data.total || data.pages.length || 0;
    const charsPerPage = pageCount > 0 ? rawText.length / pageCount : rawText.length;
    const isScanned = charsPerPage < 50;

    if (isScanned && this.canProcessImages) {
      console.log(`[document] PDF "${filename}" looks scanned (${charsPerPage.toFixed(0)} chars/page), trying vision...`);
      // Попытка через Vision если ключ есть
      try {
        return await this.extractImage(buffer, 'application/pdf', filename);
      } catch (err) {
        const message = messageFromError(err);
        if (!isRecoverableVisionPdfError(message)) {
          throw err;
        }
        // Если Vision не поддерживает PDF — падаем обратно на текстовый слой.
      }
    }

    if (!rawText) {
      const hint = this.canProcessImages
        ? 'Это отсканированный PDF без текстового слоя. Попробуйте загрузить как изображение.'
        : 'PDF не содержит текста (возможно, это скан). Сохраните страницы как изображения JPEG/PNG.';
      throw new Error(hint);
    }

    return {
      text: rawText,
      extractionMethod: 'pdf',
      pageCount,
      warning: isScanned ? 'PDF содержит мало текста, возможно это скан — проверьте результат.' : undefined,
    };
  }

  // ─── Word ─────────────────────────────────────────────────────────────────

  private async extractWord(buffer: Buffer): Promise<DocumentExtractionResult> {
    const mammoth = await import('mammoth');

    const result = await mammoth.extractRawText({ buffer });

    if (result.messages.length > 0) {
      for (const msg of result.messages) {
        if (msg.type === 'warning' || msg.type === 'error') {
          console.warn(`[document] mammoth: ${msg.message}`);
        }
      }
    }

    const text = (result.value ?? '').trim();
    if (!text) {
      throw new Error('Word-документ не содержит текста.');
    }

    return { text, extractionMethod: 'word' };
  }

  // ─── Image / Vision OCR ───────────────────────────────────────────────────

  private async extractImage(
    buffer: Buffer,
    mimetype: string,
    filename: string,
  ): Promise<DocumentExtractionResult> {
    if (this.visionProvider === 'ollama') {
      return this.extractImageWithOllama(buffer, mimetype, filename);
    }

    return this.extractImageWithAnthropic(buffer, mimetype, filename);
  }

  private async extractImageWithAnthropic(
    buffer: Buffer,
    mimetype: string,
    filename: string,
  ): Promise<DocumentExtractionResult> {
    if (!this.visionApiKey) {
      throw new Error(
        'Распознавание изображений требует ANTHROPIC_API_KEY в .env. ' +
        'Добавьте ключ и перезапустите сервер.',
      );
    }

    const MAX_VISION_BYTES = 8 * 1024 * 1024; // 8 MB
    if (buffer.length > MAX_VISION_BYTES) {
      throw new Error(
        `Изображение слишком большое (${(buffer.length / 1024 / 1024).toFixed(1)} МБ). ` +
        'Максимум 8 МБ. Уменьшите разрешение или сожмите файл.',
      );
    }

    const client = new Anthropic({ apiKey: this.visionApiKey });
    const base64 = buffer.toString('base64');
    const isPdf = mimetype.toLowerCase().includes('pdf') || filename.toLowerCase().endsWith('.pdf');
    const mediaType = isPdf ? 'application/pdf' : this.resolveMediaType(mimetype, filename);

    console.log(`[document] Vision OCR: ${filename} (${(buffer.length / 1024).toFixed(0)} KB, ${mediaType})`);

    const sourceBlock = isPdf
      ? {
          type: 'document',
          source: { type: 'base64', media_type: 'application/pdf', data: base64 },
        }
      : {
          type: 'image',
          source: { type: 'base64', media_type: mediaType, data: base64 },
        };

    let response: Awaited<ReturnType<typeof client.messages.create>>;
    try {
      response = await client.messages.create({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 4096,
        messages: [
          {
            role: 'user',
            content: [
              sourceBlock as Anthropic.ContentBlockParam,
              {
                type: 'text',
                text:
                  'Это медицинский документ (анализы, выписка, направление или справка). ' +
                  'Задача: извлечь ВЕСЬ текст из документа максимально точно. ' +
                  'Правила:\n' +
                  '- Сохраняй все числа, единицы измерения и даты без изменений\n' +
                  '- Сохраняй структуру (заголовки, строки таблицы)\n' +
                  '- Не добавляй комментарии и пояснения — только текст документа\n' +
                  '- Если документ на русском — транскрибируй на русском\n' +
                  'Текст документа:',
              },
            ],
          },
        ],
      });
    } catch (err) {
      throw toVisionUserError(err);
    }

    const text = response.content
      .filter((c): c is Anthropic.TextBlock => c.type === 'text')
      .map((c) => c.text)
      .join('\n')
      .trim();

    if (!text) {
      throw new Error('Claude Vision не смог извлечь текст из изображения. Проверьте качество фото.');
    }

    return { text, extractionMethod: 'vision' };
  }

  private async extractImageWithOllama(
    buffer: Buffer,
    mimetype: string,
    filename: string,
  ): Promise<DocumentExtractionResult> {
    if (!this.ollama) {
      throw new Error('Распознавание изображений через Ollama не настроено.');
    }

    const isPdf = mimetype.toLowerCase().includes('pdf') || filename.toLowerCase().endsWith('.pdf');
    if (isPdf) {
      throw new Error('Ollama Vision принимает изображения, но не PDF. Загрузите скан как фото JPEG/PNG.');
    }

    const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
    if (buffer.length > MAX_IMAGE_BYTES) {
      throw new Error(
        `Изображение слишком большое (${(buffer.length / 1024 / 1024).toFixed(1)} МБ). ` +
        'Максимум 8 МБ. Уменьшите разрешение или сожмите файл.',
      );
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.ollama.timeoutMs);
    const startedAt = Date.now();
    const prompt =
      'Это медицинский документ (анализы, выписка, направление или справка). ' +
      'Извлеки весь видимый текст максимально точно. ' +
      'Сохраняй числа, единицы измерения, даты, строки таблицы и заголовки. ' +
      'Если фрагмент плохо читается, пометь его как [неразборчиво], не восстанавливай по смыслу. ' +
      'Не придумывай отсутствующие значения, диагнозы, жалобы или анамнез. ' +
      'Не рассуждай. Не добавляй комментарии, выводы или пояснения. Верни только текст документа.';

    console.log(`[document] Ollama Vision OCR: ${filename} (${(buffer.length / 1024).toFixed(0)} KB, model=${this.ollama.model})`);

    try {
      const response = await fetch(`${this.ollama.serverUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          model: this.ollama.model,
          stream: false,
          think: false,
          options: {
            temperature: 0,
            num_predict: 2048,
          },
          messages: [
            {
              role: 'user',
              content: prompt,
              images: [buffer.toString('base64')],
            },
          ],
        }),
      });

      const raw = await response.text();
      if (!response.ok) {
        throw new Error(raw || `HTTP ${response.status}`);
      }

      const parsed = JSON.parse(raw) as { message?: { content?: unknown }; response?: unknown; error?: unknown };
      if (parsed.error) throw new Error(String(parsed.error));

      const text = (typeof parsed.message?.content === 'string'
        ? parsed.message.content
        : typeof parsed.response === 'string'
          ? parsed.response
          : '').trim();

      if (!text) {
        throw new Error('Ollama Vision не вернул текст. Проверьте, что модель поддерживает изображения.');
      }

      console.log(`[document] Ollama Vision OCR done: ${filename} (${text.length} chars, ${Date.now() - startedAt} ms)`);
      return { text, extractionMethod: 'vision' };
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        throw new Error(`Ollama Vision не ответил за ${Math.round(this.ollama.timeoutMs / 1000)} секунд.`);
      }
      const message = messageFromError(err);
      if (/does not support images|vision|image|unsupported/i.test(message)) {
        throw new Error(`Модель Ollama ${this.ollama.model} не поддерживает изображения. Установите vision-модель, например qwen2.5vl, и задайте DOCUMENT_VISION_MODEL.`);
      }
      throw new Error(`Ошибка Ollama Vision: ${message}`);
    } finally {
      clearTimeout(timeout);
    }
  }

  private resolveMediaType(
    mimetype: string,
    filename: string,
  ): 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp' {
    const m = mimetype.toLowerCase();
    if (m.includes('png')) return 'image/png';
    if (m.includes('gif')) return 'image/gif';
    if (m.includes('webp')) return 'image/webp';
    // HEIC/HEIF — конвертируем через jpeg (Claude не принимает напрямую)
    // Для простоты принимаем как jpeg и надеемся на лучшее — в большинстве случаев работает
    const lower = filename.toLowerCase();
    if (lower.endsWith('.png')) return 'image/png';
    if (lower.endsWith('.webp')) return 'image/webp';
    if (lower.endsWith('.gif')) return 'image/gif';
    return 'image/jpeg';
  }
}
