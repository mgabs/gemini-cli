/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type Content,
  type GenerateContentParameters,
  type GenerateContentResponse,
} from '@google/genai';
import type { Config } from '../config/config.js';
import { LoggingContentGenerator } from './loggingContentGenerator.js';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';
import { debugLogger } from '../utils/debugLogger.js';
import { toContents } from '../code_assist/converter.js';

interface OpenAiMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

/**
 * Maps Gemini-style Content to OpenAI-style Messages.
 */
function mapGeminiToOpenAi(contents: Content[]): OpenAiMessage[] {
  const messages: OpenAiMessage[] = [];
  for (const content of contents) {
    const role = content.role === 'model' ? 'assistant' : 'user';
    const parts = content.parts || [];
    const text = parts
      .map((p) => p.text)
      .filter(Boolean)
      .join('\n');
    if (text) {
      messages.push({ role, content: text });
    }
  }
  return messages;
}

/**
 * A ContentGenerator that talks to OpenAI-compatible servers (like mlx_lm.server).
 */
class OpenAiCompatibleContentGenerator implements ContentGenerator {
  constructor(
    private readonly config: Config,
    private readonly baseUrl: string,
  ) {}

  private getModelName(): string {
    const settings = this.config.getLocalMlxSettings();
    return settings.model || 'mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit';
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const model = this.getModelName();
    const contents = toContents(request.contents);
    const messages = mapGeminiToOpenAi(contents);
    const url = `${this.baseUrl.replace(/\/$/, '')}/v1/chat/completions`;

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        temperature: request.config?.temperature ?? 0.7,
        max_tokens: request.config?.maxOutputTokens,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const data = await response.json();
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
    const text = data.choices[0].message.content;

    return {
      candidates: [
        {
          content: { role: 'model', parts: [{ text }] },
          finishReason: 'STOP',
        },
      ],
      usageMetadata: {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
        promptTokenCount: data.usage?.prompt_tokens,
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
        candidatesTokenCount: data.usage?.completion_tokens,
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
        totalTokenCount: data.usage?.total_tokens,
      },
    } as GenerateContentResponse;
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const model = this.getModelName();
    const contents = toContents(request.contents);
    const messages = mapGeminiToOpenAi(contents);
    const url = `${this.baseUrl.replace(/\/$/, '')}/v1/chat/completions`;

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        temperature: request.config?.temperature ?? 0.7,
        max_tokens: request.config?.maxOutputTokens,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API error (${response.status}): ${errorText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Response body is null');

    async function* streamGenerator(): AsyncGenerator<GenerateContentResponse> {
      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader!.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const cleanLine = line.trim();
            if (!cleanLine || cleanLine === 'data: [DONE]') continue;
            if (cleanLine.startsWith('data: ')) {
              try {
                // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
                const json = JSON.parse(cleanLine.replace(/^data: /, ''));
                // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
                const text = json.choices[0]?.delta?.content || '';
                if (text) {
                  yield {
                    candidates: [
                      {
                        content: { role: 'model', parts: [{ text }] },
                      },
                    ],
                  } as GenerateContentResponse;
                }
              } catch (e) {
                debugLogger.error(
                  `[OpenAiSlm] Failed to parse stream chunk:`,
                  e,
                );
              }
            }
          }
        }
      } finally {
        reader!.releaseLock();
      }
    }

    return streamGenerator();
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async countTokens(): Promise<any> {
    return Promise.resolve({ totalTokens: 0 });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async embedContent(): Promise<any> {
    return Promise.reject(
      new Error('Embeddings not supported for local MLX yet.'),
    );
  }
}

/**
 * Creates a ContentGenerator for OpenAI-compatible local servers.
 */
export function createOpenAiCompatibleContentGenerator(
  config: Config,
): LoggingContentGenerator {
  const settings = config.getLocalMlxSettings();
  const generator = new OpenAiCompatibleContentGenerator(
    config,
    settings.host || 'http://localhost:8080',
  );
  return new LoggingContentGenerator(generator, config);
}
