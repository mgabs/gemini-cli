/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { spawn, type ChildProcess } from 'node:child_process';
import { debugLogger } from '../utils/debugLogger.js';
import type { Config } from '../config/config.js';
import { delay } from '../utils/delay.js';

/**
 * Service to manage the lifecycle of a local SLM process (e.g., mlx_lm.server).
 */
export class LocalSlmProcessService {
  private process: ChildProcess | null = null;
  private isStopping = false;

  constructor(private readonly config: Config) {}

  /**
   * Starts the local SLM process if a command is configured and it's not already running.
   */
  async start(): Promise<void> {
    const settings = this.config.getLocalMlxSettings();
    if (!settings.enabled || !settings.command || this.process) {
      return;
    }

    const model = settings.model || 'mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit';
    const rawCommand = settings.command.replace('{model}', model);
    
    // Split command into executable and args, handling simple spaces
    // For complex commands, we might need a better parser or use shell: true
    const parts = rawCommand.split(' ');
    const executable = parts[0];
    const args = parts.slice(1);

    debugLogger.log(`[LocalSlm] Starting local SLM with command: ${rawCommand}`);

    try {
      this.process = spawn(executable, args, {
        stdio: 'pipe',
        detached: false, // We want it to die with us
      });
    } catch (err) {
      debugLogger.error(`[LocalSlm] Failed to spawn process:`, err);
      return;
    }

    this.process.stdout?.on('data', (data) => {
      debugLogger.debug(`[LocalSlm][stdout] ${data.toString().trim()}`);
    });

    this.process.stderr?.on('data', (data) => {
      // Server logs often go to stderr
      debugLogger.log(`[LocalSlm][stderr] ${data.toString().trim()}`);
    });

    this.process.on('error', (err) => {
      debugLogger.error(`[LocalSlm] Process error:`, err);
      this.process = null;
    });

    this.process.on('exit', (code, signal) => {
      if (!this.isStopping) {
        debugLogger.warn(`[LocalSlm] Process exited unexpectedly with code ${code} and signal ${signal}`);
      } else {
        debugLogger.log(`[LocalSlm] Process exited with code ${code} and signal ${signal}`);
      }
      this.process = null;
    });

    // Wait for the server to be ready
    await this.waitForReady(settings.host || 'http://localhost:8080');
  }

  /**
   * Stops the local SLM process if it's running.
   */
  async stop(): Promise<void> {
    if (!this.process) {
      return;
    }

    debugLogger.log(`[LocalSlm] Stopping local SLM process...`);
    this.isStopping = true;
    
    return new Promise((resolve) => {
      if (!this.process) {
        resolve();
        return;
      }

      const timeout = setTimeout(() => {
        if (this.process) {
          debugLogger.warn(`[LocalSlm] Process did not exit in time, killing...`);
          this.process.kill('SIGKILL');
        }
        resolve();
      }, 5000);

      this.process.on('exit', () => {
        clearTimeout(timeout);
        resolve();
      });

      this.process.kill('SIGTERM');
    });
  }

  /**
   * Checks if the local SLM process is currently running.
   */
  isAlive(): boolean {
    return this.process !== null;
  }

  /**
   * Waits for the local SLM server to respond to a health check.
   */
  private async waitForReady(host: string): Promise<void> {
    const maxAttempts = 30;
    const intervalMs = 2000;
    
    // Ensure host ends with / if needed, or just use it as a base
    const healthUrl = host.endsWith('/') ? `${host}v1/models` : `${host}/v1/models`;

    debugLogger.log(`[LocalSlm] Waiting for server to be ready at ${healthUrl}...`);

    for (let i = 0; i < maxAttempts; i++) {
      try {
        debugLogger.debug(`[LocalSlm] Health check attempt ${i + 1}/${maxAttempts}...`);
        const response = await fetch(healthUrl, { signal: AbortSignal.timeout(1000) });
        if (response.ok) {
          debugLogger.log(`[LocalSlm] Server is ready!`);
          return;
        }
        debugLogger.debug(`[LocalSlm] Server responded with status: ${response.status}`);
      } catch (err) {
        debugLogger.debug(`[LocalSlm] Health check failed: ${err instanceof Error ? err.message : String(err)}`);
        // Expected while server is starting
      }
      await delay(intervalMs);
    }

    debugLogger.error(`[LocalSlm] Server failed to become ready after ${maxAttempts} attempts.`);
  }
}
