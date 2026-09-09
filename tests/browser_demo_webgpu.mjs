// Real ORT WebGPU compute through Dawn. CI supplies Chrome's SwiftShader Vulkan
// ICD so the regression runs even on hosts without a physical GPU.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { create, globals } from "webgpu";

import { MODEL_SPECS, parseTsv, runBrowserPipeline } from "../docs/assets/javascripts/imu-demo-core.mjs";
import { createValidatedSessions } from "../docs/assets/javascripts/imu-demo-runtime.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const modelRoot = process.env.IMU_BROWSER_MODEL_DIR
  || join(root, "android_realtime_app", "app", "src", "main", "assets");

async function main() {
  Object.assign(globalThis, globals);
  globalThis.navigator.gpu = create(process.platform === "linux" ? ["backend=vulkan"] : []);
  const ort = await import("onnxruntime-web/webgpu");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = pathToFileURL(join(root, "node_modules", "onnxruntime-web", "dist", "/")).href;

  const adapter = await navigator.gpu.requestAdapter();
  assert.ok(adapter, "WebGPU adapter is required; this regression must not silently skip");
  console.log("WebGPU adapter:", adapter.info.description);
  const models = new Map();
  for (const spec of MODEL_SPECS) {
    const bytes = await readFile(join(modelRoot, spec.filename));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), spec.sha256);
    models.set(spec.suffix, bytes);
  }
  const recording = parseTsv(
    await readFile(join(root, "demo", "examples", "synthetic_activity_imu.tsv")),
    "synthetic_activity_imu.tsv",
  );
  const runtimes = [];
  const results = [];
  try {
    for (const preferWebGpu of [true, false]) {
      const runtime = await createValidatedSessions(ort, models, { preferWebGpu });
      runtimes.push(runtime);
      assert.equal(runtime.backend, preferWebGpu ? "WebGPU" : "WASM");
      const result = await runBrowserPipeline({
        ort, sessions: runtime.sessions, recording,
        options: { fusionMode: "local_boundary", minDurationSec: 5, confidenceMin: 0.3, topK: 5 },
        onProgress: (update) => {
          if (update.phase === "inference" && update.completed === update.total) {
            console.log(`${runtime.backend}: completed ${update.suffix} (${update.total} windows)`);
          }
        },
      });
      assert.equal(result.timeline.rows, 118);
      assert.deepEqual(result.segments.map((segment) => segment.activityKey), ["fly", "running"]);
      assert.deepEqual(result.segments.map((segment) => [segment.startMs, segment.endMs]), [
        [1_700_000_029_840, 1_700_000_073_150],
        [1_700_000_076_060, 1_700_000_098_240],
      ]);
      assert.ok(Math.abs(result.segments[0].confidence - 0.4038439) < 1e-5);
      assert.ok(Math.abs(result.segments[1].confidence - 0.3185621) < 1e-5);
      results.push(result);
      console.log(runtime.backend, JSON.stringify(result.segments));
    }
    assert.deepEqual(results[0].timeline.decodedPath, results[1].timeline.decodedPath);
    assert.equal(results[0].timeline.probabilities.length, results[1].timeline.probabilities.length);
    let maxError = 0;
    results[0].timeline.probabilities.forEach((value, index) => {
      const error = Math.abs(value - results[1].timeline.probabilities[index]);
      assert.ok(Number.isFinite(error) && error < 1e-5, `posterior ${index} differs by ${error}`);
      maxError = Math.max(maxError, error);
    });
    console.log(`Full WebGPU/WASM regression passed; maximum posterior error: ${maxError}`);
  } finally {
    for (const runtime of runtimes) {
      await Promise.all([...runtime.sessions.values()].map((session) => session.release()));
    }
    delete globalThis.navigator.gpu;
  }
}

try {
  await main();
  // ORT retains its Dawn device; end the standalone compute test after cleanup.
  process.exit(0);
} catch (error) {
  console.error(error);
  process.exit(1);
}
