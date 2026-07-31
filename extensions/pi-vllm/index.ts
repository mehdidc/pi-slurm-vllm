import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn, execFileSync, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { hostname } from "node:os";
import { basename, dirname, extname, resolve } from "node:path";
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
  cluster?: string;
  model?: string;
  sbatch?: string;
};

type DiscoveredModel = {
  id: string;
  name: string;
  reasoning: boolean;
  input: ["text"];
  contextWindow: number;
  maxTokens: number;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
};

const SUPPORTED_CLUSTERS = ["jureca", "jupiter"] as const;
type SupportedCluster = (typeof SUPPORTED_CLUSTERS)[number];

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(EXTENSION_DIR, "../..");
const SLURM_ROOT = resolve(REPO_ROOT, "slurm");
const PROXY_SCRIPT = resolve(REPO_ROOT, "vllm_proxy.py");
const LISTEN_HOST = process.env.PI_VLLM_PROXY_HOST ?? "127.0.0.1";
const LISTEN_PORT = Number(process.env.PI_VLLM_PROXY_PORT ?? "8123");
const PROVIDER_NAME = process.env.PI_VLLM_PROVIDER_NAME ?? "hpc-vllm";
const BASE_URL = `http://${LISTEN_HOST}:${LISTEN_PORT}/v1`;

function supportedCluster(value: string | undefined): SupportedCluster | undefined {
  const normalized = value?.trim().toLowerCase();
  return SUPPORTED_CLUSTERS.find((cluster) => cluster === normalized);
}

function clusterFromSlurmConfig(): SupportedCluster | undefined {
  try {
    const config = execFileSync("scontrol", ["show", "config"], {
      encoding: "utf8",
      timeout: 5000,
      stdio: ["ignore", "pipe", "ignore"],
    });
    return supportedCluster(config.match(/^\s*ClusterName\s*=\s*(\S+)/m)?.[1]);
  } catch {
    return undefined;
  }
}

function clusterFromHostname(): SupportedCluster | undefined {
  const host = hostname().toLowerCase();
  if (host.includes("jureca") || /^jr/.test(host)) return "jureca";
  if (host.includes("jupiter") || /^jp/.test(host)) return "jupiter";
  return undefined;
}

function detectCluster(): SupportedCluster {
  const detected =
    supportedCluster(process.env.SLURM_CLUSTER_NAME) ??
    clusterFromSlurmConfig() ??
    clusterFromHostname();
  if (detected) return detected;

  const available = SUPPORTED_CLUSTERS.filter((cluster) =>
    existsSync(resolve(SLURM_ROOT, cluster)),
  );
  if (available.length === 1) return available[0];

  throw new Error(
    "Could not detect a supported Slurm cluster. Expected jureca or jupiter " +
      "from SLURM_CLUSTER_NAME, `scontrol show config`, or the hostname.",
  );
}

function maxModelLength(source: string, runner: string): number {
  const activeSource = source
    .split("\n")
    .filter((line) => !line.trimStart().startsWith("#"))
    .join("\n");
  const match = activeSource.match(/--max-model-len(?:=|\s+)(\d+)\b/);
  if (!match) {
    throw new Error(
      `Runner ${runner} must pass a literal --max-model-len VALUE`,
    );
  }
  return Number(match[1]);
}

function discoverModels(cluster: SupportedCluster): DiscoveredModel[] {
  const runnerDirectory = resolve(SLURM_ROOT, cluster);
  if (!existsSync(runnerDirectory)) {
    throw new Error(`No supported model runners found for detected cluster ${cluster}`);
  }

  const models = readdirSync(runnerDirectory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && extname(entry.name) === ".sbatch")
    .map((entry) => {
      const id = basename(entry.name, ".sbatch");
      const runner = resolve(runnerDirectory, entry.name);
      const source = readFileSync(runner, "utf8");
      const contextWindow = maxModelLength(source, runner);
      return {
        id,
        name: `${id} on ${cluster} Slurm vLLM`,
        reasoning: /--reasoning-parser\b|^\s*REASONING_PARSER=/m.test(source),
        input: ["text"] as ["text"],
        contextWindow,
        maxTokens: contextWindow,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      };
    })
    .sort((left, right) => left.id.localeCompare(right.id));

  if (models.length === 0) {
    throw new Error(`No .sbatch model runners found in ${runnerDirectory}`);
  }
  return models;
}

const CLUSTER = detectCluster();
const MODELS = discoverModels(CLUSTER);
const MODEL_IDS = new Set(MODELS.map((model) => model.id));

let proxyProcess: ChildProcessWithoutNullStreams | undefined;
let lastProxyLine = "proxy not started";
let proxyTransition: Promise<void> = Promise.resolve();

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

async function getStatus(): Promise<ProxyStatus> {
  return fetchJson<ProxyStatus>(`http://${LISTEN_HOST}:${LISTEN_PORT}/status`, 5000);
}

async function runningProxyStatus(): Promise<ProxyStatus | undefined> {
  try {
    return await getStatus();
  } catch {
    return undefined;
  }
}

function runnerPath(model: string): string {
  return resolve(SLURM_ROOT, CLUSTER, `${model}.sbatch`);
}

function startProxy(model: string): void {
  if (!MODEL_IDS.has(model)) {
    throw new Error(`Unsupported model ${model} on detected cluster ${CLUSTER}`);
  }
  if (!existsSync(PROXY_SCRIPT)) {
    throw new Error(`Missing proxy script: ${PROXY_SCRIPT}`);
  }
  const sbatch = runnerPath(model);
  if (!existsSync(sbatch)) {
    throw new Error(`Missing Slurm model runner: ${sbatch}`);
  }

  const child = spawn(
    "python3",
    [
      PROXY_SCRIPT,
      "--listen-host",
      LISTEN_HOST,
      "--listen-port",
      String(LISTEN_PORT),
      "--cluster",
      CLUSTER,
      "--model",
      model,
    ],
    {
      cwd: REPO_ROOT,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  proxyProcess = child;

  child.stdout.on("data", (chunk) => {
    lastProxyLine = chunk.toString().trim().split("\n").at(-1) ?? lastProxyLine;
  });
  child.stderr.on("data", (chunk) => {
    lastProxyLine = chunk.toString().trim().split("\n").at(-1) ?? lastProxyLine;
  });
  child.on("exit", (code, signal) => {
    lastProxyLine = `proxy exited code=${code} signal=${signal}`;
    if (proxyProcess === child) proxyProcess = undefined;
  });
}

async function stopOwnedProxy(): Promise<boolean> {
  const child = proxyProcess;
  if (!child || child.exitCode !== null) {
    proxyProcess = undefined;
    return false;
  }

  child.kill("SIGTERM");
  await Promise.race([
    new Promise<void>((resolveExit) => child.once("exit", () => resolveExit())),
    new Promise<void>((resolveTimeout) => setTimeout(resolveTimeout, 3000)),
  ]);
  if (child.exitCode === null) {
    child.kill("SIGKILL");
    await Promise.race([
      new Promise<void>((resolveExit) => child.once("exit", () => resolveExit())),
      new Promise<void>((resolveTimeout) => setTimeout(resolveTimeout, 1000)),
    ]);
  }
  if (proxyProcess === child) proxyProcess = undefined;
  return true;
}

async function transitionProxy(model: string): Promise<void> {
  const status = await runningProxyStatus();
  if (status?.cluster === CLUSTER && status.model === model) return;

  if (status) {
    if (!proxyProcess || proxyProcess.exitCode !== null) {
      const selection =
        status.cluster && status.model
          ? `${status.cluster}/${status.model}`
          : "an unidentified configuration";
      throw new Error(
        `Port ${LISTEN_PORT} already hosts an external proxy for ${selection}`,
      );
    }
    await stopOwnedProxy();
  } else if (proxyProcess && proxyProcess.exitCode === null) {
    await stopOwnedProxy();
  }

  lastProxyLine = `starting ${CLUSTER}/${model}`;
  startProxy(model);
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    const started = await runningProxyStatus();
    if (started?.cluster === CLUSTER && started.model === model) return;
    await new Promise((resolveTimer) => setTimeout(resolveTimer, 250));
  }
  throw new Error(`Proxy did not become ready: ${lastProxyLine}`);
}

function ensureProxyForModel(model: string): Promise<void> {
  const transition = proxyTransition.then(() => transitionProxy(model));
  proxyTransition = transition.catch(() => undefined);
  return transition;
}

function stopProxy(): Promise<boolean> {
  const stop = proxyTransition.then(() => stopOwnedProxy());
  proxyTransition = stop.then(
    () => undefined,
    () => undefined,
  );
  return stop;
}

function selectedExtensionModel(
  ctx: { model?: { provider?: string; id?: string } },
): string | undefined {
  if (ctx.model?.provider !== PROVIDER_NAME) return undefined;
  return ctx.model.id && MODEL_IDS.has(ctx.model.id) ? ctx.model.id : undefined;
}

export default function (pi: ExtensionAPI) {
  pi.registerProvider(PROVIDER_NAME, {
    name: `HPC vLLM (${CLUSTER})`,
    baseUrl: BASE_URL,
    api: "openai-completions",
    apiKey: "local",
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsUsageInStreaming: false,
    },
    models: MODELS,
  });

  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.setStatus(
      "vllm",
      `${CLUSTER}: ${MODELS.length} model${MODELS.length === 1 ? "" : "s"}`,
    );
  });

  pi.on("before_provider_request", async (_event, ctx) => {
    const model = selectedExtensionModel(ctx);
    if (!model) return;
    ctx.ui.setStatus("vllm", `starting ${model}`);
    await ensureProxyForModel(model);
    ctx.ui.setStatus("vllm", `${model} ready`);
  });

  pi.registerCommand("vllm-start", {
    description: "Start the proxy for the currently selected Slurm vLLM model.",
    handler: async (_args, ctx) => {
      const model = selectedExtensionModel(ctx);
      if (!model) {
        ctx.ui.notify(`Select a ${PROVIDER_NAME} model first`, "error");
        return;
      }
      await ensureProxyForModel(model);
      ctx.ui.notify(`vLLM proxy for ${CLUSTER}/${model} listening at ${BASE_URL}`, "info");
    },
  });

  pi.registerCommand("vllm-status", {
    description: "Show detected cluster and proxy/Slurm/vLLM status.",
    handler: async (_args, ctx) => {
      const status = await runningProxyStatus();
      if (!status) {
        ctx.ui.notify(
          `cluster: ${CLUSTER}\nmodels: ${MODELS.map((model) => model.id).join(", ")}\nproxy: stopped`,
          "info",
        );
        return;
      }
      const lines = [
        `cluster: ${CLUSTER}`,
        `proxy: ${BASE_URL}`,
        `runner: ${status.cluster ?? "unknown"}/${status.model ?? "unknown"}`,
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
    description: "Stop the extension-owned proxy process. This does not cancel its Slurm job.",
    handler: async (_args, ctx) => {
      const stopped = await stopProxy();
      ctx.ui.notify(
        stopped
          ? "Stopped local vLLM proxy process"
          : "No extension-owned vLLM proxy process is running",
        "info",
      );
    },
  });

  pi.on("session_shutdown", async () => {
    await stopProxy();
  });
}
