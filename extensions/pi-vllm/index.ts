import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

type ProxyStatus = {
  ok?: boolean;
  request_count?: number;
  last_event?: string;
  last_error?: string | null;
  job_id?: string | null;
  job_state?: string | null;
  job_nodes?: string | null;
  backend_base_url?: string | null;
};

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(EXTENSION_DIR, "../..");
const PROXY_SCRIPT = resolve(REPO_ROOT, "pi_vllm_proxy.py");
const SBATCH_SCRIPT = resolve(REPO_ROOT, "slurm/pi-vllm.sbatch");
const LISTEN_HOST = process.env.PI_VLLM_PROXY_HOST ?? "127.0.0.1";
const LISTEN_PORT = Number(process.env.PI_VLLM_PROXY_PORT ?? "8123");
const MODEL_ID = process.env.PI_VLLM_MODEL_ID ?? "Qwen3.6-27B-FP8";
const PROVIDER_NAME = process.env.PI_VLLM_PROVIDER_NAME ?? "hpc-vllm";
const BASE_URL = `http://${LISTEN_HOST}:${LISTEN_PORT}/v1`;

let proxyProcess: ChildProcessWithoutNullStreams | undefined;
let lastProxyLine = "proxy not started";

async function fetchJson<T>(url: string, timeoutMs = 2000): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

async function proxyIsAlive(): Promise<boolean> {
  try {
    await fetchJson(`http://${LISTEN_HOST}:${LISTEN_PORT}/healthz`);
    return true;
  } catch {
    return false;
  }
}

function startProxy(): void {
  if (proxyProcess && proxyProcess.exitCode === null) return;
  if (!existsSync(PROXY_SCRIPT)) {
    throw new Error(`Missing proxy script: ${PROXY_SCRIPT}`);
  }
  if (!existsSync(SBATCH_SCRIPT)) {
    throw new Error(`Missing Slurm script: ${SBATCH_SCRIPT}`);
  }

  proxyProcess = spawn(
    "python3",
    [
      PROXY_SCRIPT,
      "--listen-host",
      LISTEN_HOST,
      "--listen-port",
      String(LISTEN_PORT),
      "--model-id",
      MODEL_ID,
      "--sbatch",
      SBATCH_SCRIPT,
    ],
    {
      cwd: REPO_ROOT,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  proxyProcess.stdout.on("data", (chunk) => {
    lastProxyLine = chunk.toString().trim().split("\n").at(-1) ?? lastProxyLine;
  });
  proxyProcess.stderr.on("data", (chunk) => {
    lastProxyLine = chunk.toString().trim().split("\n").at(-1) ?? lastProxyLine;
  });
  proxyProcess.on("exit", (code, signal) => {
    lastProxyLine = `proxy exited code=${code} signal=${signal}`;
  });
}

async function ensureProxyStarted(): Promise<void> {
  if (await proxyIsAlive()) return;
  startProxy();

  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    if (await proxyIsAlive()) return;
    await new Promise((resolveTimer) => setTimeout(resolveTimer, 250));
  }
  throw new Error(`Proxy did not become ready: ${lastProxyLine}`);
}

async function getStatus(): Promise<ProxyStatus> {
  return fetchJson<ProxyStatus>(`http://${LISTEN_HOST}:${LISTEN_PORT}/status`, 5000);
}

export default function (pi: ExtensionAPI) {
  pi.registerProvider(PROVIDER_NAME, {
    baseUrl: BASE_URL,
    api: "openai-completions",
    apiKey: "local",
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsUsageInStreaming: false,
    },
    models: [
      {
        id: MODEL_ID,
        name: "Qwen3.6 27B FP8 on Slurm vLLM",
        reasoning: true,
        input: ["text"],
        contextWindow: 262144,
        maxTokens: 262144,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      },
    ],
  });

  pi.on("session_start", async (_event, ctx) => {
    try {
      await ensureProxyStarted();
      ctx.ui.setStatus("vllm", "vLLM proxy ready");
    } catch (error) {
      ctx.ui.setStatus("vllm", "vLLM proxy failed");
      ctx.ui.notify(`vLLM proxy failed: ${(error as Error).message}`, "error");
    }
  });

  pi.on("before_provider_request", async (event, ctx) => {
    const provider = (event as any).model?.provider ?? (event as any).provider;
    if (provider && provider !== PROVIDER_NAME) return;
    ctx.ui.setStatus("vllm", "vLLM starting");
    await ensureProxyStarted();
    ctx.ui.setStatus("vllm", "vLLM proxy ready");
  });

  pi.registerCommand("vllm-start", {
    description: "Start the local Pi vLLM proxy. The Slurm job starts on first model request.",
    handler: async (_args, ctx) => {
      await ensureProxyStarted();
      ctx.ui.notify(`vLLM proxy listening at ${BASE_URL}`, "info");
    },
  });

  pi.registerCommand("vllm-status", {
    description: "Show proxy and Slurm/vLLM status.",
    handler: async (_args, ctx) => {
      await ensureProxyStarted();
      const status = await getStatus();
      const lines = [
        `proxy: ${BASE_URL}`,
        `requests: ${status.request_count ?? 0}`,
        `event: ${status.last_event ?? "unknown"}`,
        `job: ${status.job_id ?? "none"} ${status.job_state ?? ""}`.trim(),
        `nodes: ${status.job_nodes ?? "none"}`,
        `backend: ${status.backend_base_url ?? "none"}`,
        status.last_error ? `error: ${status.last_error}` : "",
      ].filter(Boolean);
      ctx.ui.notify(lines.join("\n"), status.last_error ? "error" : "info");
    },
  });

  pi.registerCommand("vllm-stop", {
    description: "Stop the local Pi vLLM proxy process. This does not cancel the Slurm job.",
    handler: async (_args, ctx) => {
      if (proxyProcess && proxyProcess.exitCode === null) {
        proxyProcess.kill("SIGTERM");
        ctx.ui.notify("Stopped local vLLM proxy process", "info");
      } else {
        ctx.ui.notify("Local vLLM proxy is not running in this Pi process", "info");
      }
    },
  });

  pi.on("session_shutdown", () => {
    if (proxyProcess && proxyProcess.exitCode === null) {
      proxyProcess.kill("SIGTERM");
    }
  });
}
