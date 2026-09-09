import { CLASS_KEYS, MODEL_SPECS } from "./imu-demo-core.mjs";

async function releaseSessions(sessions) {
  await Promise.allSettled(
    [...(sessions?.values() || [])].map((session) => session.release?.()),
  );
}

async function createSessions(ort, models, provider) {
  const sessions = new Map();
  try {
    for (const spec of MODEL_SPECS) {
      sessions.set(spec.suffix, await ort.InferenceSession.create(models.get(spec.suffix), {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      }));
    }
    return sessions;
  } catch (error) {
    await releaseSessions(sessions);
    throw error;
  }
}

function probeInput(windowSize, probe) {
  // Nonzero, channel-asymmetric signals exercise the convolution layout. An
  // all-zero warmup can finish successfully even when this path is incorrect.
  const data = new Float32Array(windowSize * 6);
  let state = 42 + probe;
  for (let t = 0; t < windowSize; t += 1) {
    for (let channel = 0; channel < 6; channel += 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      data[t * 6 + channel] =
        Math.sin(t * (channel + 1) * 0.03 + probe) * (0.3 + channel * 0.2)
        + (state / 0x100000000 - 0.5) * (probe + 0.25);
    }
  }
  return data;
}

async function runProbe(ort, session, spec, probe) {
  const input = new ort.Tensor("float32", probeInput(spec.windowSize, probe), [1, spec.windowSize, 6]);
  let outputs;
  try {
    outputs = await session.run({ [session.inputNames[0]]: input });
    const data = outputs[session.outputNames[0]]?.data;
    if (!data || data.length !== CLASS_KEYS.length || !data.every(Number.isFinite)) {
      throw new Error(`${spec.suffix} backend check returned invalid logits`);
    }
    return Array.from(data);
  } finally {
    for (const output of Object.values(outputs || {})) output.dispose?.();
    input.dispose?.();
  }
}

async function warmupSessions(ort, sessions) {
  for (const spec of MODEL_SPECS) {
    await runProbe(ort, sessions.get(spec.suffix), spec, 0);
  }
}

async function validateWebGpu(ort, wasmSessions, gpuSessions) {
  for (const spec of MODEL_SPECS) {
    for (let probe = 0; probe < 2; probe += 1) {
      const reference = await runProbe(ort, wasmSessions.get(spec.suffix), spec, probe);
      const actual = await runProbe(ort, gpuSessions.get(spec.suffix), spec, probe);
      for (let i = 0; i < reference.length; i += 1) {
        // Allow normal float32 rounding, but reject the much larger Conv/layout
        // errors that can change the decoded activity sequence without throwing.
        if (Math.abs(actual[i] - reference[i]) > 1e-4 + 1e-4 * Math.abs(reference[i])) {
          throw new Error(`${spec.suffix} WebGPU numerical check failed (probe ${probe}, logit ${i})`);
        }
      }
    }
  }
}

export async function createValidatedSessions(ort, models, {
  preferWebGpu = false,
  onFallback = () => {},
} = {}) {
  const wasmSessions = await createSessions(ort, models, "wasm");
  let gpuSessions;
  try {
    if (preferWebGpu) {
      try {
        // ORT Web 1.29.0's default layout produces incorrect Conv outputs for
        // these models on WebGPU. Keep the exported NCHW convolution layout.
        gpuSessions = await createSessions(ort, models, { name: "webgpu", preferredLayout: "NCHW" });
        await validateWebGpu(ort, wasmSessions, gpuSessions);
      } catch (error) {
        await releaseSessions(gpuSessions);
        gpuSessions = null;
        onFallback(error);
      }
      if (gpuSessions) {
        await releaseSessions(wasmSessions);
        return { sessions: gpuSessions, backend: "WebGPU" };
      }
    }
    await warmupSessions(ort, wasmSessions);
    return { sessions: wasmSessions, backend: "WASM" };
  } catch (error) {
    await releaseSessions(gpuSessions);
    await releaseSessions(wasmSessions);
    throw error;
  }
}
