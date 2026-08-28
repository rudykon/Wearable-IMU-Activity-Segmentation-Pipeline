import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { constants } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import * as ort from "onnxruntime-web/wasm";

import {
  MODEL_SPECS,
  parseTsv,
  runBrowserPipeline,
} from "../docs/assets/javascripts/imu-demo-core.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const defaultModelRoot = join(
  root,
  "android_realtime_app",
  "app",
  "src",
  "main",
  "assets",
);
const modelRoot = process.env.IMU_BROWSER_MODEL_DIR || defaultModelRoot;

async function modelsAvailable() {
  try {
    await Promise.all(
      MODEL_SPECS.map((spec) => access(join(modelRoot, spec.filename), constants.R_OK)),
    );
    return true;
  } catch {
    return false;
  }
}

test("browser-local ONNX inference reproduces the public sample output", async (context) => {
  if (!(await modelsAvailable())) {
    context.skip("verified ONNX assets are not present in this checkout");
    return;
  }

  const runtimeDirectory = join(root, "node_modules", "onnxruntime-web", "dist", "/");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = pathToFileURL(runtimeDirectory).href;
  const sessions = new Map();
  for (const spec of MODEL_SPECS) {
    const bytes = await readFile(join(modelRoot, spec.filename));
    sessions.set(
      spec.suffix,
      await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
    );
  }

  const recording = parseTsv(
    await readFile(join(root, "demo", "examples", "synthetic_activity_imu.tsv")),
    "synthetic_activity_imu.tsv",
  );
  const result = await runBrowserPipeline({
    ort,
    sessions,
    recording,
    options: {
      fusionMode: "local_boundary",
      minDurationSec: 5,
      confidenceMin: 0.3,
      topK: 5,
    },
  });

  assert.deepEqual(result.modelScales, ["3s", "5s", "8s"]);
  assert.equal(result.timeline.rows, 118);
  assert.equal(result.timeline.decodedPath.length, 118);
  assert.equal(result.segments.length, 2);
  assert.deepEqual(
    result.segments.map((segment) => segment.activityKey),
    ["fly", "running"],
  );
  assert.deepEqual(
    result.segments.map((segment) => [segment.startMs, segment.endMs]),
    [
      [1_700_000_029_840, 1_700_000_073_150],
      [1_700_000_076_060, 1_700_000_098_240],
    ],
  );
  assert.ok(Math.abs(result.segments[0].confidence - 0.4038439) < 1e-5);
  assert.ok(Math.abs(result.segments[1].confidence - 0.3185621) < 1e-5);

  await Promise.all([...sessions.values()].map((session) => session.release?.()));
});
