import Anthropic, { APIError } from '@anthropic-ai/sdk';
import type { Usage } from '@anthropic-ai/sdk/resources/messages/messages';

export interface AnthropicProviderConfig {
  apiKey: string;
  model: string;
  maxTokens: number;
  temperature: number;
  requestTimeoutMs: number;
  maxRetries: number;
}

interface CompleteJsonOptions {
  systemPrompt: string;
  userPrompt: string;
  jsonSchema: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
    additionalProperties?: boolean;
  };
  toolName: string;
  toolDescription: string;
  maxTokens?: number;
  temperature?: number;
  operation: string;
}

interface CompleteTextOptions {
  systemPrompt: string;
  userPrompt: string;
  maxTokens?: number;
  temperature?: number;
  operation: string;
}

export class AnthropicProvider {
  private client: Anthropic;
  private config: AnthropicProviderConfig;

  constructor(config: AnthropicProviderConfig) {
    if (!config.apiKey) {
      throw new Error('AnthropicProvider: ANTHROPIC_API_KEY is required');
    }
    this.config = config;
    this.client = new Anthropic({
      apiKey: config.apiKey,
      timeout: config.requestTimeoutMs,
      maxRetries: 0,
    });
  }

  async healthCheck(): Promise<boolean> {
    return Boolean(this.config.apiKey);
  }

  async completeJson(opts: CompleteJsonOptions): Promise<string> {
    const run = async () => {
      const response = await this.client.messages.create({
        model: this.config.model,
        max_tokens: opts.maxTokens ?? this.config.maxTokens,
        temperature: opts.temperature ?? this.config.temperature,
        system: [
          {
            type: 'text',
            text: opts.systemPrompt,
            cache_control: { type: 'ephemeral' },
          },
        ],
        tools: [
          {
            name: opts.toolName,
            description: opts.toolDescription,
            input_schema: opts.jsonSchema,
          },
        ],
        tool_choice: { type: 'tool', name: opts.toolName },
        messages: [{ role: 'user', content: opts.userPrompt }],
      });

      this.logUsage(opts.operation, response.usage);

      const toolUse = response.content.find((block) => block.type === 'tool_use');
      if (!toolUse || toolUse.type !== 'tool_use') {
        throw new Error(
          `Anthropic [${opts.operation}]: expected tool_use block, got stop_reason=${response.stop_reason}`
        );
      }
      return JSON.stringify(toolUse.input);
    };

    return this.withRetry(opts.operation, run);
  }

  async completeText(opts: CompleteTextOptions): Promise<string> {
    const run = async () => {
      const response = await this.client.messages.create({
        model: this.config.model,
        max_tokens: opts.maxTokens ?? this.config.maxTokens,
        temperature: opts.temperature ?? this.config.temperature,
        system: [
          {
            type: 'text',
            text: opts.systemPrompt,
            cache_control: { type: 'ephemeral' },
          },
        ],
        messages: [{ role: 'user', content: opts.userPrompt }],
      });

      this.logUsage(opts.operation, response.usage);

      const parts: string[] = [];
      for (const block of response.content) {
        if (block.type === 'text') parts.push(block.text);
      }
      return parts.join('').trim();
    };

    return this.withRetry(opts.operation, run);
  }

  private async withRetry<T>(operation: string, fn: () => Promise<T>): Promise<T> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        if (!this.isRetryable(error) || attempt === this.config.maxRetries) {
          break;
        }
        const delayMs = 1000 * Math.pow(2, attempt);
        console.warn(
          `[anthropic ${operation}] attempt ${attempt + 1} failed (${this.describeError(error)}), retrying in ${delayMs}ms`
        );
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
    const reason = this.describeError(lastError);
    throw new Error(`Anthropic [${operation}] failed after retries: ${reason}`);
  }

  private isRetryable(error: unknown): boolean {
    if (error instanceof APIError) {
      const status = error.status;
      if (status === 429) return true;
      if (status === 529) return true;
      if (status && status >= 500) return true;
      return false;
    }
    const msg = error instanceof Error ? error.message : String(error);
    return /timeout|ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN/i.test(msg);
  }

  private describeError(error: unknown): string {
    if (error instanceof APIError) {
      return `${error.status ?? 'no-status'} ${error.message}`;
    }
    if (error instanceof Error) return error.message;
    return String(error);
  }

  private logUsage(operation: string, usage: Usage | undefined): void {
    if (!usage) return;
    const cacheRead = usage.cache_read_input_tokens ?? 0;
    const cacheCreate = usage.cache_creation_input_tokens ?? 0;
    const input = usage.input_tokens ?? 0;
    const output = usage.output_tokens ?? 0;
    console.log(
      `[anthropic ${operation}] model=${this.config.model} input=${input} output=${output} cache_read=${cacheRead} cache_create=${cacheCreate}`
    );
  }
}
