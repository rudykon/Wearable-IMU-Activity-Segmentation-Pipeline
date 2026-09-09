import assert from "node:assert/strict";
import test from "node:test";

import { MODEL_SPECS } from "../docs/assets/javascripts/imu-demo-core.mjs";
import { createValidatedSessions } from "../docs/assets/javascripts/imu-demo-runtime.mjs";

const models = new Map(MODEL_SPECS.map((spec) => [spec.suffix, spec.suffix]));

function fakeRuntime({ gpuOutput = (data) => data, failCreate, failRun } = {}) {
  const sessions = [];
  const tensors = [];
  const ort = {
    Tensor: class {
      constructor(_type, data, dims) {
        Object.assign(this, { data, dims, disposed: false });
        tensors.push(this);
      }
      dispose() { this.disposed = true; }
    },
    InferenceSession: {
      async create(suffix, options) {
        const provider = options.executionProviders[0];
        const backend = typeof provider === "string" ? provider : provider.name;
        if (failCreate?.(backend, suffix)) throw new Error("creation failed");
        const session = {
          backend, suffix, provider, runs: 0, released: false,
          inputNames: ["input"], outputNames: ["logits"],
          async run({ input }) {
            this.runs += 1;
            if (failRun?.(backend, suffix)) throw new Error("execution failed");
            assert.ok(input.data.some((value) => value !== 0));
            assert.ok(input.data[0] !== input.data[1], "probe channels must differ");
            let data = Float32Array.from([-1.5, 0.1, 0.5, 1.2, -0.2, 0.7]);
            if (backend === "webgpu") data = gpuOutput(data, suffix, this.runs);
            return { logits: new ort.Tensor("float32", data, [1, data.length]) };
          },
          async release() { this.released = true; },
        };
        sessions.push(session);
        return session;
      },
    },
  };
  return { ort, sessions, tensors };
}

test("verified WebGPU uses NCHW and releases the WASM reference and probe tensors", async () => {
  const { ort, sessions, tensors } = fakeRuntime({
    gpuOutput: (data) => Float32Array.from(data, (value) => value + 5e-6),
  });
  const result = await createValidatedSessions(ort, models, { preferWebGpu: true });
  assert.equal(result.backend, "WebGPU");
  for (const session of sessions) {
    assert.equal(session.runs, 2);
    assert.equal(session.released, session.backend === "wasm");
    if (session.backend === "webgpu") assert.equal(session.provider.preferredLayout, "NCHW");
  }
  assert.ok(tensors.every((tensor) => tensor.disposed));
});

const failures = [
  ["silent drift on the second probe of the 8s model", {
    gpuOutput: (data, suffix, run) => {
      if (suffix === "8s" && run === 2) data[3] += 0.1;
      return data;
    },
  }],
  ["non-finite logits", { gpuOutput: (data) => { data[0] = NaN; return data; } }],
  ["wrong output size", { gpuOutput: (data) => data.slice(0, 5) }],
  ["GPU execution failure", { failRun: (backend) => backend === "webgpu" }],
  ["partial GPU initialization failure", {
    failCreate: (backend, suffix) => backend === "webgpu" && suffix === "5s",
  }],
];

for (const [name, behavior] of failures) {
  test(`${name} falls back to WASM and releases failed GPU resources`, async () => {
    const { ort, sessions, tensors } = fakeRuntime(behavior);
    const errors = [];
    const result = await createValidatedSessions(ort, models, {
      preferWebGpu: true,
      onFallback: (error) => errors.push(error),
    });
    assert.equal(result.backend, "WASM");
    assert.equal(errors.length, 1);
    assert.equal(result.sessions.size, 3);
    for (const session of sessions) {
      assert.equal(session.released, session.backend === "webgpu");
    }
    assert.ok(tensors.every((tensor) => tensor.disposed));
  });
}

test("devices without WebGPU warm up WASM without creating GPU sessions", async () => {
  const { ort, sessions, tensors } = fakeRuntime();
  const result = await createValidatedSessions(ort, models);
  assert.equal(result.backend, "WASM");
  assert.equal(sessions.length, 3);
  assert.ok(sessions.every((session) => session.backend === "wasm" && session.runs === 1));
  assert.ok(tensors.every((tensor) => tensor.disposed));
});

test("an unusable WASM backend rejects and releases its sessions", async () => {
  const { ort, sessions, tensors } = fakeRuntime({ failRun: () => true });
  await assert.rejects(createValidatedSessions(ort, models), /execution failed/);
  assert.ok(sessions.every((session) => session.released));
  assert.ok(tensors.every((tensor) => tensor.disposed));
});
