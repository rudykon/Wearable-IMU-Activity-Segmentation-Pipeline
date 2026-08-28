import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  MODEL_REVISION,
  MODEL_SPECS,
  alignScaleProbabilities,
  butterworthFilter,
  fuseProbabilities,
  parseTsv,
  smoothProbabilities,
  viterbiDecode,
} from "../docs/assets/javascripts/imu-demo-core.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const samplePath = join(root, "demo", "examples", "synthetic_activity_imu.tsv");

test("browser Demo model contract is pinned to the published manifest", async () => {
  const manifest = JSON.parse(await readFile(join(root, "model-assets.json"), "utf8"));
  assert.match(MODEL_REVISION, /^[a-f0-9]{40}$/);
  assert.notEqual(MODEL_REVISION, "main");
  for (const spec of MODEL_SPECS) {
    const asset = manifest.assets.find((candidate) => candidate.path.endsWith(`/${spec.filename}`));
    assert.ok(asset, `${spec.filename} is present in model-assets.json`);
    assert.equal(spec.sha256, asset.sha256);
    assert.equal(spec.size, asset.size);
    assert.match(spec.url, new RegExp(`/resolve/${MODEL_REVISION}/`));
  }
});

test("the bundled recording parses to the documented 100 Hz contract", async () => {
  const recording = parseTsv(await readFile(samplePath), "synthetic_activity_imu.tsv");
  assert.equal(recording.sampleCount, 12_000);
  assert.equal(recording.medianIntervalMs, 10);
  assert.equal(recording.sampleRateHz, 100);
  assert.equal(recording.durationSec, 119.99);
  assert.equal(recording.userId, "synthetic_activity_imu");
});

test("the browser Butterworth port preserves the reference filtered values", async () => {
  const recording = parseTsv(await readFile(samplePath), "synthetic_activity_imu.tsv");
  const filtered = butterworthFilter(recording.channels);
  const expected = [1917.030029, -2208.036133, -38.013153, -68.967209, -41.972881, 15.013496];
  for (let channel = 0; channel < expected.length; channel += 1) {
    assert.ok(Math.abs(filtered[channel][0] - expected[channel]) < 1e-3);
  }
});

test("TSV validation reports missing fields and an incompatible sample rate", () => {
  assert.throws(
    () => parseTsv("ACC_TIME\tACC_X\n1\t2\n", "bad.tsv"),
    (error) => error.code === "missing_columns" && error.details.missing.includes("GYRO_Z"),
  );

  const rows = ["ACC_TIME\tACC_X\tACC_Y\tACC_Z\tGYRO_X\tGYRO_Y\tGYRO_Z"];
  for (let index = 0; index < 800; index += 1) {
    rows.push(`${1_700_000_000_000 + index * 20}\t1\t2\t3\t4\t5\t6`);
  }
  assert.throws(
    () => parseTsv(rows.join("\n"), "slow.tsv"),
    (error) => error.code === "bad_sample_rate" && error.details.medianIntervalMs === 20,
  );
});

test("scale alignment, fusion, smoothing, and Viterbi decoding stay normalized", () => {
  const scale3s = {
    suffix: "3s",
    timestamps: Float64Array.of(1_000, 2_000, 3_000),
    probabilities: {
      rows: 3,
      data: Float32Array.from([
        0.8, 0.2, 0, 0, 0, 0,
        0.1, 0.9, 0, 0, 0, 0,
        0.8, 0.2, 0, 0, 0, 0,
      ]),
    },
  };
  const scale5s = {
    suffix: "5s",
    timestamps: Float64Array.of(1_000, 3_000),
    probabilities: {
      rows: 2,
      data: Float32Array.from([
        0.6, 0.4, 0, 0, 0, 0,
        0.4, 0.6, 0, 0, 0, 0,
      ]),
    },
  };
  const { timestamps, aligned } = alignScaleProbabilities([scale3s, scale5s]);
  assert.deepEqual(Array.from(timestamps), [1_000, 2_000, 3_000]);
  assert.ok(Math.abs(aligned["5s"].data[6] - 0.5) < 1e-6);
  const fused = fuseProbabilities(aligned, "local_boundary");
  for (let row = 0; row < fused.rows; row += 1) {
    const total = fused.data.subarray(row * 6, row * 6 + 6).reduce((sum, value) => sum + value, 0);
    assert.ok(Math.abs(total - 1) < 1e-6);
  }
  const smoothed = smoothProbabilities(fused);
  const decoded = viterbiDecode(smoothed);
  assert.equal(decoded.length, 3);
  assert.ok(Array.from(decoded).every((state) => state >= 0 && state < 6));
});
