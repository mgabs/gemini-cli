/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GoogleGenAI,
  type CountTokensParameters,
  type CountTokensResponse,
  type EmbedContentParameters,
  type EmbedContentResponse,
  type GenerateContentParameters,
  type GenerateContentResponse,
  type GenerativeModelList,
} from '@google/genai';
import type { Config } from '../config/config.js';
import { LoggingContentGenerator } from './loggingContentGenerator.js';
import { debugLogger } from '../utils/debugLogger.js';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

/**
 * A ContentGenerator that intercepts requests and replaces the model name with the local MLX model.
 */
class LocalMlxContentGenerator implements ContentGenerator {
  constructor(
    private readonly models: GenerativeModelList,
    private readonly config: Config,
  ) {}

  private getLocalModelName(): string {
    const settings = this.config.getLocalMlxSettings();
    const model =
      settings.model || 'mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit';
    // The GoogleGenAI SDK might expect 'models/' prefix, but local servers usually don't.
    // However, the SDK prepends it if not present. Let's see.
    return model;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const modelName = this.getLocalModelName();
    debugLogger.log(`[LocalSlm] generateContent using model: ${modelName}`);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const model = this.models.getGenerativeModel({ model: modelName });
    // Overwrite the model property in the request to ensure the local server
    // receives the correct name in the payload.
    const modifiedRequest = { ...request, model: modelName };
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call
    return model.generateContent(modifiedRequest);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const modelName = this.getLocalModelName();
    debugLogger.log(
      `[LocalSlm] generateContentStream using model: ${modelName}`,
    );
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const model = this.models.getGenerativeModel({ model: modelName });
    const modifiedRequest = { ...request, model: modelName };
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call
    const result = await model.generateContentStream(modifiedRequest);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access
    return result.stream;
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const modelName = this.getLocalModelName();
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const model = this.models.getGenerativeModel({ model: modelName });
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call
    return model.countTokens(request);
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const modelName = this.getLocalModelName();
    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
    const model = this.models.getGenerativeModel({ model: modelName });
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call
    return model.embedContent(request);
  }
}

/**
 * Creates a ContentGenerator for a local MLX server.
 * This generator uses the GoogleGenAI SDK but points to a local base URL.
 *
 * @param config The application configuration.
 * @returns A new LoggingContentGenerator wrapping a local MLX client.
 */
export function createLocalMlxContentGenerator(
  config: Config,
): LoggingContentGenerator {
  const mlxSettings = config.getLocalMlxSettings();
  const googleGenAI = new GoogleGenAI({
    apiKey: 'no-api-key-needed',
    httpOptions: {
      baseUrl: mlxSettings.host!,
      timeout: 60000,
    },
  });
  return new LoggingContentGenerator(
    new LocalMlxContentGenerator(googleGenAI.models, config),
    config,
  );
}
