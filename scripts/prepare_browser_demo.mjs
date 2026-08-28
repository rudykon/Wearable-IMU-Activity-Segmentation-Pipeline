import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "node_modules", "onnxruntime-web", "dist");
const vendor = join(root, "docs", "assets", "vendor", "onnxruntime");
const data = join(root, "docs", "assets", "data");

const runtimeFiles = [
  "ort.webgpu.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
];

await mkdir(vendor, { recursive: true });
await mkdir(data, { recursive: true });
await Promise.all(
  runtimeFiles.map((name) => copyFile(join(dist, name), join(vendor, name))),
);
await copyFile(
  join(root, "demo", "examples", "synthetic_activity_imu.tsv"),
  join(data, "synthetic_activity_imu.tsv"),
);

console.log(
  `Prepared the browser Demo (${runtimeFiles.length} runtime files and one sample).`,
);
