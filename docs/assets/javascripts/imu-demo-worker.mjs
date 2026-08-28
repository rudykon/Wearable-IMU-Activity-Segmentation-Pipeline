import {
  DemoInputError,
  MODEL_SPECS,
  parseTsv,
  runBrowserPipeline,
} from "./imu-demo-core.mjs";

const RUNTIME_BASE = new URL("../vendor/onnxruntime/", import.meta.url).href;
const RUNTIME_MODULE = new URL(
  "../vendor/onnxruntime/ort.webgpu.min.mjs",
  import.meta.url,
).href;
const DATABASE_NAME = "wearable-imu-model-cache";
const STORE_NAME = "onnx-models";

let runtimePromise;
let sessionPromise;
let activeJob = null;

function postProgress(jobId, phase, progress, details = {}) {
  self.postMessage({ type: "progress", jobId, phase, progress, ...details });
}

async function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = import(RUNTIME_MODULE).then((ort) => {
      ort.env.wasm.wasmPaths = RUNTIME_BASE;
      // GitHub Pages does not emit cross-origin-isolation headers, so a
      // single-threaded WASM fallback is the portable choice.
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      return ort;
    });
  }
  return runtimePromise;
}

function openDatabase() {
  if (!("indexedDB" in self)) return Promise.resolve(null);
  return new Promise((resolve) => {
    const request = indexedDB.open(DATABASE_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
  });
}

async function readCachedModel(key) {
  const database = await openDatabase();
  if (!database) return null;
  return new Promise((resolve) => {
    const transaction = database.transaction(STORE_NAME, "readonly");
    const request = transaction.objectStore(STORE_NAME).get(key);
    request.onsuccess = () => resolve(request.result instanceof ArrayBuffer ? request.result : null);
    request.onerror = () => resolve(null);
    transaction.oncomplete = () => database.close();
    transaction.onerror = () => database.close();
  });
}

async function writeCachedModel(key, bytes) {
  const database = await openDatabase();
  if (!database) return;
  await new Promise((resolve) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).put(bytes, key);
    transaction.oncomplete = resolve;
    transaction.onerror = resolve;
    transaction.onabort = resolve;
  });
  database.close();
}

async function deleteCachedModel(key) {
  const database = await openDatabase();
  if (!database) return;
  await new Promise((resolve) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).delete(key);
    transaction.oncomplete = resolve;
    transaction.onerror = resolve;
  });
  database.close();
}

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

async function fetchBytes(spec, onChunk) {
  const response = await fetch(spec.url, { mode: "cors", credentials: "omit" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while fetching ${spec.filename}`);
  }
  if (!response.body) {
    const bytes = await response.arrayBuffer();
    onChunk(bytes.byteLength);
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    length += value.byteLength;
    onChunk(value.byteLength);
  }
  const combined = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return combined.buffer;
}

async function loadModelBytes(spec, onChunk) {
  const cacheKey = `${spec.sha256}:${spec.size}`;
  const cached = await readCachedModel(cacheKey);
  if (cached && cached.byteLength === spec.size && (await sha256(cached)) === spec.sha256) {
    onChunk(spec.size, true);
    return { bytes: cached, cached: true };
  }
  if (cached) await deleteCachedModel(cacheKey);
  const bytes = await fetchBytes(spec, (amount) => onChunk(amount, false));
  if (bytes.byteLength !== spec.size || (await sha256(bytes)) !== spec.sha256) {
    throw new DemoInputError("model_hash_mismatch", { filename: spec.filename });
  }
  await writeCachedModel(cacheKey, bytes);
  return { bytes, cached: false };
}

async function createSessions(ort, models, provider) {
  const sessions = new Map();
  try {
    for (const spec of MODEL_SPECS) {
      const session = await ort.InferenceSession.create(models.get(spec.suffix), {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });
      sessions.set(spec.suffix, session);
    }
    return sessions;
  } catch (error) {
    for (const session of sessions.values()) await session.release?.();
    throw error;
  }
}

async function warmupSessions(ort, sessions) {
  for (const spec of MODEL_SPECS) {
    const session = sessions.get(spec.suffix);
    const input = new ort.Tensor(
      "float32",
      new Float32Array(spec.windowSize * 6),
      [1, spec.windowSize, 6],
    );
    const outputs = await session.run({ [session.inputNames[0]]: input });
    for (const output of Object.values(outputs)) output.dispose?.();
    input.dispose?.();
  }
}

async function prepareSessions(jobId) {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    const totalBytes = MODEL_SPECS.reduce((sum, spec) => sum + spec.size, 0);
    let loadedBytes = 0;
    let cachedModels = 0;
    const modelEntries = await Promise.all(
      MODEL_SPECS.map(async (spec) => {
        const loaded = await loadModelBytes(spec, (amount, fromCache) => {
          loadedBytes += amount;
          if (fromCache) cachedModels += 1;
          postProgress(jobId, "models", 0.12 + 0.23 * (loadedBytes / totalBytes), {
            loadedBytes,
            totalBytes,
            filename: spec.filename,
          });
        });
        return [spec.suffix, loaded.bytes];
      }),
    );
    const models = new Map(modelEntries);
    postProgress(jobId, "runtime", 0.37, { cachedModels });
    const ort = await getRuntime();

    if (self.navigator?.gpu) {
      let gpuSessions;
      try {
        gpuSessions = await createSessions(ort, models, "webgpu");
        await warmupSessions(ort, gpuSessions);
        return { ort, sessions: gpuSessions, backend: "WebGPU", cachedModels };
      } catch (error) {
        if (gpuSessions) {
          for (const session of gpuSessions.values()) await session.release?.();
        }
        postProgress(jobId, "fallback", 0.42, { reason: String(error) });
      }
    }
    const sessions = await createSessions(ort, models, "wasm");
    await warmupSessions(ort, sessions);
    return { ort, sessions, backend: "WASM", cachedModels };
  })().catch((error) => {
    sessionPromise = null;
    throw error;
  });
  return sessionPromise;
}

function pipelineProgress(jobId, update) {
  if (update.phase === "filtering") {
    postProgress(jobId, "filtering", 0.08);
    return;
  }
  if (update.phase === "inference") {
    const fraction = update.total ? update.completed / update.total : 0;
    const progress = 0.45 + (update.scaleIndex + fraction) * 0.15;
    postProgress(jobId, "inference", Math.min(progress, 0.9), {
      suffix: update.suffix,
      completed: update.completed,
      total: update.total,
    });
    return;
  }
  if (update.phase === "postprocessing") {
    postProgress(jobId, "postprocessing", 0.93);
  }
}

function transferableResult(result) {
  const transfer = [
    result.rawPreview.times.buffer,
    result.timeline.times.buffer,
    result.timeline.probabilities.buffer,
    result.timeline.decodedPath.buffer,
  ];
  for (const channel of result.rawPreview.channels) transfer.push(channel.buffer);
  return transfer;
}

self.addEventListener("message", async (event) => {
  const message = event.data;
  if (!message || message.type !== "run") return;
  if (activeJob !== null) {
    self.postMessage({ type: "error", jobId: message.jobId, code: "busy", details: {} });
    return;
  }
  activeJob = message.jobId;
  const started = performance.now();
  try {
    postProgress(message.jobId, "validating", 0.02);
    const recording = parseTsv(message.bytes, message.filename);
    postProgress(message.jobId, "validated", 0.06, {
      sampleCount: recording.sampleCount,
      durationSec: recording.durationSec,
    });
    const runtime = await prepareSessions(message.jobId);
    postProgress(message.jobId, "inference", 0.45, { backend: runtime.backend });
    const result = await runBrowserPipeline({
      ort: runtime.ort,
      sessions: runtime.sessions,
      recording,
      options: message.options,
      onProgress: (update) => pipelineProgress(message.jobId, update),
    });
    result.backend = runtime.backend;
    result.cachedModels = runtime.cachedModels;
    result.elapsedMs = performance.now() - started;
    postProgress(message.jobId, "complete", 1);
    self.postMessage(
      { type: "result", jobId: message.jobId, result },
      transferableResult(result),
    );
  } catch (error) {
    const code = error instanceof DemoInputError ? error.code : "runtime_error";
    self.postMessage({
      type: "error",
      jobId: message.jobId,
      code,
      details: error.details || {},
      message: String(error?.message || error),
    });
  } finally {
    activeJob = null;
  }
});
