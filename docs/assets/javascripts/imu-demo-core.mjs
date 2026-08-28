export const REQUIRED_COLUMNS = [
  "ACC_TIME",
  "ACC_X",
  "ACC_Y",
  "ACC_Z",
  "GYRO_X",
  "GYRO_Y",
  "GYRO_Z",
];

export const CLASS_KEYS = [
  "background",
  "badminton",
  "jump_rope",
  "fly",
  "running",
  "table_tennis",
];

export const MODEL_REPOSITORY =
  "config-h/Wearable-IMU-Activity-Segmentation-Pipeline";
export const MODEL_REVISION = "e0f89bb6c779e9b974cd159eec7ecb3344de9ba7";

const MODEL_ROOT = `https://huggingface.co/${MODEL_REPOSITORY}/resolve/${MODEL_REVISION}/android_realtime_app/app/src/main/assets`;

export const MODEL_SPECS = [
  {
    suffix: "3s",
    windowSize: 300,
    filename: "combined_model_3s_seed42.onnx",
    size: 5_647_794,
    sha256: "5adb8807bfc737e11ee40cca0af0690c22fdd8d29b5c5aaa35c3e83f9f646839",
    mean: [
      1915.136445926244,
      -2222.7754116186634,
      -92.94102150871045,
      -69.23744962489933,
      -48.88914174075672,
      -0.9113457644647055,
    ],
    std: [
      3189.904663188159,
      2601.0831826462727,
      2464.1609405096888,
      1879.8900739939795,
      1536.0500336551013,
      1561.1619017886628,
    ],
  },
  {
    suffix: "5s",
    windowSize: 500,
    filename: "combined_model_5s_seed123.onnx",
    size: 5_647_794,
    sha256: "d812c5fc04df6c1ca249e3cec8c977a28c2de8036736c134210b3173490f2681",
    mean: [
      1916.0625992329878,
      -2223.3709419251018,
      -93.50274540433607,
      -69.27944811079628,
      -48.904664110408355,
      -0.9124877753603922,
    ],
    std: [
      3190.1034052965156,
      2601.1312245606523,
      2464.074362383159,
      1880.2410774988496,
      1536.3758970280326,
      1561.485755909182,
    ],
  },
  {
    suffix: "8s",
    windowSize: 800,
    filename: "combined_model_8s_seed123.onnx",
    size: 5_647_794,
    sha256: "75fb093eca823003c6f5e44b0a0363f24b04659ab2ae08ddb08c334481087f8d",
    mean: [
      1917.247802734375,
      -2223.91259765625,
      -94.01901245117188,
      -69.32743072509766,
      -48.92448425292969,
      -0.9227896928787231,
    ],
    std: [
      3191.30126953125,
      2603.224609375,
      2465.12255859375,
      1881.6754150390625,
      1537.2916259765625,
      1562.2393798828125,
    ],
  },
].map((spec) => ({ ...spec, url: `${MODEL_ROOT}/${spec.filename}` }));

export const FUSION_MODES = new Set([
  "average",
  "dynamic_boundary",
  "local_boundary",
  "confident_conflict",
  "weighted_long",
  "weighted_balanced",
]);

const STEP_SIZE = 100;
const CHANNEL_COUNT = 6;
const CLASS_COUNT = 6;
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
const MIN_SAMPLES = 800;
const MAX_SAMPLES = 60_000;
const FILTER_B = [
  0.43284664499029174,
  1.731386579961167,
  2.5970798699417503,
  1.731386579961167,
  0.43284664499029174,
];
const FILTER_A = [
  1,
  2.3695130071820376,
  2.31398841441588,
  1.0546654058785678,
  0.18737949236818488,
];
const FILTER_ZI = [
  0.5671533550097082,
  1.205279782229443,
  0.9221883267051434,
  0.24546715262187573,
];
const FILTER_PAD = 15;

export class DemoInputError extends Error {
  constructor(code, details = {}) {
    super(code);
    this.name = "DemoInputError";
    this.code = code;
    this.details = details;
  }
}

function decodeUpload(payload) {
  if (typeof payload === "string") return payload;
  try {
    const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new DemoInputError("bad_encoding", { cause: String(error) });
  }
}

function safeRecordingId(filename) {
  const stem = String(filename || "recording").replace(/\.[^.]+$/, "");
  return stem.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^_+|_+$/g, "") || "recording";
}

function median(values) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function parseTsv(payload, filename = "recording.tsv") {
  const byteLength =
    typeof payload === "string"
      ? new TextEncoder().encode(payload).byteLength
      : payload.byteLength;
  if (byteLength > MAX_UPLOAD_BYTES) {
    throw new DemoInputError("file_too_large", { byteLength });
  }

  const text = decodeUpload(payload).replace(/^\uFEFF/, "");
  const lines = text.split(/\r?\n/);
  const headerIndex = lines.findIndex((line) => line.trim().length > 0);
  if (headerIndex < 0) throw new DemoInputError("missing_header");

  const header = lines[headerIndex].split("\t").map((value) => value.trim());
  const gyroAlias = header.indexOf("GYRO_");
  if (!header.includes("GYRO_Z") && gyroAlias >= 0) header[gyroAlias] = "GYRO_Z";
  const indices = REQUIRED_COLUMNS.map((column) => header.indexOf(column));
  const missing = REQUIRED_COLUMNS.filter((_, index) => indices[index] < 0);
  if (missing.length) throw new DemoInputError("missing_columns", { missing });

  const rows = [];
  for (let lineIndex = headerIndex + 1; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    if (!line.trim()) continue;
    const fields = line.split("\t");
    const values = new Array(REQUIRED_COLUMNS.length);
    let valid = true;
    for (let columnIndex = 0; columnIndex < indices.length; columnIndex += 1) {
      const raw = fields[indices[columnIndex]];
      if (raw === undefined || raw.trim() === "") {
        valid = false;
        break;
      }
      const value = Number(raw);
      if (!Number.isFinite(value)) {
        valid = false;
        break;
      }
      values[columnIndex] = value;
    }
    if (valid && values[0] > 0) rows.push(values);
  }

  if (rows.length < MIN_SAMPLES) {
    throw new DemoInputError("too_few_samples", { count: rows.length });
  }
  if (rows.length > MAX_SAMPLES) {
    throw new DemoInputError("too_many_samples", { count: rows.length });
  }

  rows.sort((left, right) => left[0] - right[0]);
  const timestamps = new Float64Array(rows.length);
  const channels = Array.from(
    { length: CHANNEL_COUNT },
    () => new Float32Array(rows.length),
  );
  const intervals = new Float64Array(rows.length - 1);
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    timestamps[rowIndex] = row[0];
    for (let channel = 0; channel < CHANNEL_COUNT; channel += 1) {
      channels[channel][rowIndex] = row[channel + 1];
    }
    if (rowIndex > 0) {
      const interval = timestamps[rowIndex] - timestamps[rowIndex - 1];
      if (interval <= 0) {
        throw new DemoInputError("timestamps_not_unique");
      }
      intervals[rowIndex - 1] = interval;
    }
  }

  const medianIntervalMs = median(intervals);
  if (medianIntervalMs < 8 || medianIntervalMs > 12) {
    throw new DemoInputError("bad_sample_rate", { medianIntervalMs });
  }

  return {
    filename,
    userId: safeRecordingId(filename),
    timestamps,
    channels,
    sampleCount: rows.length,
    medianIntervalMs,
    sampleRateHz: 1000 / medianIntervalMs,
    durationSec: (timestamps[timestamps.length - 1] - timestamps[0]) / 1000,
  };
}

function lfilter(values, initialScale) {
  const state = FILTER_ZI.map((value) => value * initialScale);
  const output = new Float64Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const input = values[index];
    const current = FILTER_B[0] * input + state[0];
    for (let order = 1; order < state.length; order += 1) {
      state[order - 1] =
        FILTER_B[order] * input + state[order] - FILTER_A[order] * current;
    }
    state[state.length - 1] =
      FILTER_B[FILTER_B.length - 1] * input -
      FILTER_A[FILTER_A.length - 1] * current;
    output[index] = current;
  }
  return output;
}

function reverseInPlace(values) {
  for (let left = 0, right = values.length - 1; left < right; left += 1, right -= 1) {
    const value = values[left];
    values[left] = values[right];
    values[right] = value;
  }
  return values;
}

function filtfiltChannel(values) {
  const pad = Math.min(FILTER_PAD, values.length - 1);
  const extended = new Float64Array(values.length + 2 * pad);
  for (let index = 0; index < pad; index += 1) {
    extended[index] = 2 * values[0] - values[pad - index];
  }
  extended.set(values, pad);
  for (let index = 0; index < pad; index += 1) {
    extended[pad + values.length + index] =
      2 * values[values.length - 1] - values[values.length - 2 - index];
  }
  const forward = reverseInPlace(lfilter(extended, extended[0]));
  const backward = reverseInPlace(lfilter(forward, forward[0]));
  return Float32Array.from(backward.subarray(pad, pad + values.length));
}

export function butterworthFilter(channels) {
  if (!channels.length || channels[0].length <= 100) {
    return channels.map((values) => Float32Array.from(values));
  }
  return channels.map(filtfiltChannel);
}

function matrix(rows, data = new Float32Array(rows * CLASS_COUNT)) {
  return { rows, data };
}

function matrixValue(probabilities, row, column) {
  return probabilities.data[row * CLASS_COUNT + column];
}

function argMaxRow(probabilities, row) {
  const offset = row * CLASS_COUNT;
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let column = 0; column < CLASS_COUNT; column += 1) {
    const value = probabilities.data[offset + column];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = column;
    }
  }
  return bestIndex;
}

function topTwo(probabilities, row) {
  const offset = row * CLASS_COUNT;
  let first = Number.NEGATIVE_INFINITY;
  let second = Number.NEGATIVE_INFINITY;
  for (let column = 0; column < CLASS_COUNT; column += 1) {
    const value = probabilities.data[offset + column];
    if (value > first) {
      second = first;
      first = value;
    } else if (value > second) {
      second = value;
    }
  }
  return [first, second];
}

export function uniformNearest(values, size, Output = Float32Array) {
  if (size <= 1) return Output.from(values);
  const output = new Output(values.length);
  const left = Math.floor(size / 2);
  const right = size - left - 1;
  for (let index = 0; index < values.length; index += 1) {
    let total = 0;
    for (let delta = -left; delta <= right; delta += 1) {
      const source = Math.max(0, Math.min(values.length - 1, index + delta));
      total += values[source];
    }
    output[index] = total / size;
  }
  return output;
}

function interpolateScale(scale, referenceTimestamps) {
  if (scale.timestamps.length === 1) {
    const output = matrix(referenceTimestamps.length);
    for (let row = 0; row < output.rows; row += 1) {
      output.data.set(scale.probabilities.data.subarray(0, CLASS_COUNT), row * CLASS_COUNT);
    }
    return output;
  }

  const output = matrix(referenceTimestamps.length);
  let sourceIndex = 0;
  for (let row = 0; row < referenceTimestamps.length; row += 1) {
    const timestamp = referenceTimestamps[row];
    while (
      sourceIndex < scale.timestamps.length - 2 &&
      scale.timestamps[sourceIndex + 1] < timestamp
    ) {
      sourceIndex += 1;
    }
    if (timestamp <= scale.timestamps[0]) {
      output.data.set(scale.probabilities.data.subarray(0, CLASS_COUNT), row * CLASS_COUNT);
      continue;
    }
    if (timestamp >= scale.timestamps[scale.timestamps.length - 1]) {
      const source = (scale.timestamps.length - 1) * CLASS_COUNT;
      output.data.set(
        scale.probabilities.data.subarray(source, source + CLASS_COUNT),
        row * CLASS_COUNT,
      );
      continue;
    }
    const leftTime = scale.timestamps[sourceIndex];
    const rightTime = scale.timestamps[sourceIndex + 1];
    const alpha = (timestamp - leftTime) / Math.max(1, rightTime - leftTime);
    for (let column = 0; column < CLASS_COUNT; column += 1) {
      output.data[row * CLASS_COUNT + column] =
        matrixValue(scale.probabilities, sourceIndex, column) * (1 - alpha) +
        matrixValue(scale.probabilities, sourceIndex + 1, column) * alpha;
    }
  }
  return output;
}

export function alignScaleProbabilities(scales) {
  if (!scales.length) return { timestamps: new Float64Array(), aligned: {} };
  const reference = scales.reduce((best, current) =>
    current.timestamps.length > best.timestamps.length ? current : best,
  );
  const aligned = {};
  for (const scale of [...scales].sort(
    (left, right) => MODEL_SPECS.findIndex((spec) => spec.suffix === left.suffix) -
      MODEL_SPECS.findIndex((spec) => spec.suffix === right.suffix),
  )) {
    let exact = scale.timestamps.length === reference.timestamps.length;
    for (let index = 0; exact && index < scale.timestamps.length; index += 1) {
      exact = scale.timestamps[index] === reference.timestamps[index];
    }
    aligned[scale.suffix] = exact
      ? matrix(scale.probabilities.rows, Float32Array.from(scale.probabilities.data))
      : interpolateScale(scale, reference.timestamps);
  }
  return { timestamps: Float64Array.from(reference.timestamps), aligned };
}

function presentSuffixes(aligned) {
  return ["3s", "5s", "8s"].filter((suffix) => aligned[suffix]);
}

function fixedFusion(aligned, weightMap) {
  const suffixes = presentSuffixes(aligned);
  if (suffixes.length === 1) return matrix(aligned[suffixes[0]].rows, Float32Array.from(aligned[suffixes[0]].data));
  const output = matrix(aligned[suffixes[0]].rows);
  const totalWeight = suffixes.reduce((sum, suffix) => sum + weightMap[suffix], 0);
  for (const suffix of suffixes) {
    const weight = weightMap[suffix] / totalWeight;
    const source = aligned[suffix].data;
    for (let index = 0; index < output.data.length; index += 1) {
      output.data[index] += source[index] * weight;
    }
  }
  return output;
}

function averageFusion(aligned) {
  const suffixes = presentSuffixes(aligned);
  return fixedFusion(aligned, Object.fromEntries(suffixes.map((suffix) => [suffix, 1])));
}

function weightedRowsFusion(aligned, rowWeight) {
  const suffixes = presentSuffixes(aligned);
  const rows = aligned[suffixes[0]].rows;
  const output = matrix(rows);
  for (let row = 0; row < rows; row += 1) {
    const weights = rowWeight(row, suffixes);
    let total = 0;
    for (let index = 0; index < weights.length; index += 1) {
      weights[index] = Math.max(0.05, weights[index]);
      total += weights[index];
    }
    for (let suffixIndex = 0; suffixIndex < suffixes.length; suffixIndex += 1) {
      const source = aligned[suffixes[suffixIndex]].data;
      const weight = weights[suffixIndex] / total;
      for (let column = 0; column < CLASS_COUNT; column += 1) {
        output.data[row * CLASS_COUNT + column] +=
          source[row * CLASS_COUNT + column] * weight;
      }
    }
  }
  return output;
}

function localBoundaryFusion(aligned) {
  const suffixes = presentSuffixes(aligned);
  if (suffixes.length === 1 || !aligned["3s"]) return averageFusion(aligned);
  const rows = aligned[suffixes[0]].rows;
  const predictions = new Uint8Array(rows);
  for (let row = 0; row < rows; row += 1) predictions[row] = argMaxRow(aligned["3s"], row);
  const mask = new Float32Array(rows);
  for (let row = 1; row < rows; row += 1) {
    if (predictions[row] === predictions[row - 1]) continue;
    for (let index = Math.max(0, row - 3); index <= Math.min(rows - 1, row + 3); index += 1) {
      mask[index] = 1;
    }
  }
  const boundary = uniformNearest(mask, 3);
  const base = { "3s": 0.2, "5s": 0.35, "8s": 0.45 };
  return weightedRowsFusion(aligned, (row, ordered) =>
    ordered.map((suffix) => {
      if (suffix === "3s") return base[suffix] + 0.3 * boundary[row];
      if (suffix === "5s") return base[suffix] - 0.08 * boundary[row];
      return base[suffix] - 0.22 * boundary[row];
    }),
  );
}

function dynamicBoundaryFusion(aligned) {
  const suffixes = presentSuffixes(aligned);
  if (suffixes.length === 1) return averageFusion(aligned);
  const rows = aligned[suffixes[0]].rows;
  const boundary = new Float32Array(rows);
  if (aligned["3s"]) {
    const prediction = new Uint8Array(rows);
    const margin = new Float32Array(rows);
    for (let row = 0; row < rows; row += 1) {
      prediction[row] = argMaxRow(aligned["3s"], row);
      const [first, second] = topTwo(aligned["3s"], row);
      margin[row] = first - second;
    }
    const delta = new Float32Array(rows);
    for (let row = 1; row < rows; row += 1) {
      let total = 0;
      for (let column = 0; column < CLASS_COUNT; column += 1) {
        total += Math.abs(
          matrixValue(aligned["3s"], row, column) -
            matrixValue(aligned["3s"], row - 1, column),
        );
      }
      delta[row] = total / CLASS_COUNT;
    }
    const smoothDelta = uniformNearest(delta, 5);
    for (let row = 0; row < rows; row += 1) {
      const classChange =
        (row > 0 && prediction[row] !== prediction[row - 1]) ||
        (row + 1 < rows && prediction[row] !== prediction[row + 1])
          ? 1
          : 0;
      const ambiguity = Math.max(0, Math.min(1, (0.18 - margin[row]) / 0.18));
      boundary[row] = Math.max(classChange, ambiguity, Math.min(1, smoothDelta[row] / 0.12));
    }
  }
  if (suffixes.length > 1) {
    for (let row = 0; row < rows; row += 1) {
      const counts = new Uint8Array(CLASS_COUNT);
      for (const suffix of suffixes) counts[argMaxRow(aligned[suffix], row)] += 1;
      let majority = 0;
      for (const count of counts) majority = Math.max(majority, count);
      boundary[row] = Math.max(boundary[row], 1 - majority / suffixes.length);
    }
  }
  const base = { "3s": 0.2, "5s": 0.35, "8s": 0.45 };
  return weightedRowsFusion(aligned, (row, ordered) =>
    ordered.map((suffix) => {
      if (suffix === "3s") return base[suffix] + 0.35 * boundary[row];
      if (suffix === "8s") return base[suffix] - 0.3 * boundary[row];
      return base[suffix] + 0.05 * (1 - Math.abs(2 * boundary[row] - 1));
    }),
  );
}

function confidentConflictFusion(aligned) {
  const suffixes = presentSuffixes(aligned);
  if (suffixes.length === 1 || !aligned["3s"]) return averageFusion(aligned);
  const rows = aligned[suffixes[0]].rows;
  const gate = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const prediction3s = argMaxRow(aligned["3s"], row);
    const [top3s, second3s] = topTwo(aligned["3s"], row);
    const margin3s = top3s - second3s;
    let disagreement = false;
    let marginAdvantage = 0;
    let topAdvantage = 0;
    for (const suffix of ["5s", "8s"]) {
      if (!aligned[suffix]) continue;
      disagreement ||= prediction3s !== argMaxRow(aligned[suffix], row);
      const [topOther, secondOther] = topTwo(aligned[suffix], row);
      marginAdvantage = Math.max(marginAdvantage, margin3s - (topOther - secondOther));
      topAdvantage = Math.max(topAdvantage, top3s - topOther);
    }
    gate[row] =
      disagreement && margin3s >= 0.12 && marginAdvantage >= 0.05 && topAdvantage >= 0.03
        ? 1
        : 0;
  }
  const smoothGate = uniformNearest(gate, 3);
  const base = { "3s": 0.2, "5s": 0.35, "8s": 0.45 };
  return weightedRowsFusion(aligned, (row, ordered) =>
    ordered.map((suffix) => {
      if (suffix === "3s") return base[suffix] + 0.28 * smoothGate[row];
      if (suffix === "8s") return base[suffix] - 0.2 * smoothGate[row];
      return base[suffix] - 0.08 * smoothGate[row];
    }),
  );
}

export function fuseProbabilities(aligned, mode = "local_boundary") {
  if (!FUSION_MODES.has(mode)) throw new DemoInputError("bad_fusion", { mode });
  if (mode === "local_boundary") return localBoundaryFusion(aligned);
  if (mode === "dynamic_boundary") return dynamicBoundaryFusion(aligned);
  if (mode === "confident_conflict") return confidentConflictFusion(aligned);
  if (mode === "weighted_long") {
    return fixedFusion(aligned, { "3s": 0.15, "5s": 0.3, "8s": 0.55 });
  }
  if (mode === "weighted_balanced") {
    return fixedFusion(aligned, { "3s": 0.25, "5s": 0.35, "8s": 0.4 });
  }
  return averageFusion(aligned);
}

export function smoothProbabilities(probabilities) {
  const uniform = matrix(probabilities.rows);
  for (let column = 0; column < CLASS_COUNT; column += 1) {
    const values = new Float32Array(probabilities.rows);
    for (let row = 0; row < probabilities.rows; row += 1) {
      values[row] = matrixValue(probabilities, row, column);
    }
    const smoothed = uniformNearest(values, 7);
    for (let row = 0; row < probabilities.rows; row += 1) {
      uniform.data[row * CLASS_COUNT + column] = smoothed[row];
    }
  }

  const output = matrix(probabilities.rows);
  const half = 2;
  for (let row = 0; row < probabilities.rows; row += 1) {
    for (let column = 0; column < CLASS_COUNT; column += 1) {
      const values = [];
      for (let delta = -half; delta <= half; delta += 1) {
        const source = row + delta;
        values.push(
          source >= 0 && source < probabilities.rows
            ? matrixValue(uniform, source, column)
            : 0,
        );
      }
      values.sort((left, right) => left - right);
      output.data[row * CLASS_COUNT + column] = values[half];
    }
  }
  return output;
}

export function viterbiDecode(probabilities) {
  const rows = probabilities.rows;
  if (!rows) return new Uint8Array();
  const transition = Array.from({ length: CLASS_COUNT }, () =>
    new Float64Array(CLASS_COUNT).fill(0.001),
  );
  for (let state = 0; state < CLASS_COUNT; state += 1) transition[state][state] = 0.97;
  for (let state = 1; state < CLASS_COUNT; state += 1) {
    transition[0][state] = 0.01;
    transition[state][0] = 0.05;
  }
  const logTransition = transition.map((values) => {
    const total = values.reduce((sum, value) => sum + value, 0);
    return Float64Array.from(values, (value) => Math.log(value / total + 1e-10));
  });
  const scores = new Float64Array(rows * CLASS_COUNT);
  const backpointer = new Uint8Array(rows * CLASS_COUNT);
  const logInitial = Math.log(1 / CLASS_COUNT);
  for (let state = 0; state < CLASS_COUNT; state += 1) {
    scores[state] = logInitial + Math.log(matrixValue(probabilities, 0, state) + 1e-10);
  }
  for (let row = 1; row < rows; row += 1) {
    for (let state = 0; state < CLASS_COUNT; state += 1) {
      let best = Number.NEGATIVE_INFINITY;
      let bestPrevious = 0;
      for (let previous = 0; previous < CLASS_COUNT; previous += 1) {
        const candidate =
          scores[(row - 1) * CLASS_COUNT + previous] + logTransition[previous][state];
        if (candidate > best) {
          best = candidate;
          bestPrevious = previous;
        }
      }
      scores[row * CLASS_COUNT + state] =
        best + Math.log(matrixValue(probabilities, row, state) + 1e-10);
      backpointer[row * CLASS_COUNT + state] = bestPrevious;
    }
  }
  const path = new Uint8Array(rows);
  let finalState = 0;
  let finalScore = Number.NEGATIVE_INFINITY;
  for (let state = 0; state < CLASS_COUNT; state += 1) {
    const value = scores[(rows - 1) * CLASS_COUNT + state];
    if (value > finalScore) {
      finalScore = value;
      finalState = state;
    }
  }
  path[rows - 1] = finalState;
  for (let row = rows - 2; row >= 0; row -= 1) {
    path[row] = backpointer[(row + 1) * CLASS_COUNT + path[row + 1]];
  }
  return path;
}

function extractSegments(path, timestamps, probabilities) {
  if (!path.length) return [];
  const segments = [];
  let currentClass = path[0];
  let startIndex = 0;
  for (let index = 1; index <= path.length; index += 1) {
    if (index < path.length && path[index] === currentClass) continue;
    if (currentClass > 0) {
      const endIndex = index - 1;
      let confidence = 0;
      for (let row = startIndex; row <= endIndex; row += 1) {
        confidence += matrixValue(probabilities, row, currentClass);
      }
      confidence /= endIndex - startIndex + 1;
      const startMs = timestamps[startIndex] - 1500;
      const endMs = timestamps[endIndex] + 1500;
      segments.push({
        classIndex: currentClass,
        activityIndex: currentClass - 1,
        activityKey: CLASS_KEYS[currentClass],
        startMs,
        endMs,
        durationSec: (endMs - startMs) / 1000,
        confidence,
        startWindowIndex: startIndex,
        endWindowIndex: endIndex,
      });
    }
    if (index < path.length) {
      currentClass = path[index];
      startIndex = index;
    }
  }
  return segments;
}

function mergeSameClassSegments(segments) {
  if (segments.length <= 1) return segments.map((segment) => ({ ...segment }));
  const merged = [];
  for (const source of [...segments].sort((a, b) => a.startMs - b.startMs)) {
    const segment = { ...source };
    const previous = merged[merged.length - 1];
    const gap = previous ? (segment.startMs - previous.endMs) / 1000 : Number.POSITIVE_INFINITY;
    if (previous && segment.classIndex === previous.classIndex && gap < 60) {
      previous.endMs = segment.endMs;
      previous.endWindowIndex = segment.endWindowIndex;
      previous.durationSec = (previous.endMs - previous.startMs) / 1000;
      previous.confidence = (previous.confidence + segment.confidence) / 2;
    } else {
      merged.push(segment);
    }
  }
  return merged;
}

function lowerBound(values, target) {
  let left = 0;
  let right = values.length;
  while (left < right) {
    const middle = (left + right) >> 1;
    if (values[middle] < target) left = middle + 1;
    else right = middle;
  }
  return left;
}

function upperBound(values, target) {
  let left = 0;
  let right = values.length;
  while (left < right) {
    const middle = (left + right) >> 1;
    if (values[middle] <= target) left = middle + 1;
    else right = middle;
  }
  return left;
}

function refineBoundaries(segments, recording, filteredChannels) {
  if (!segments.length || !recording.timestamps.length) return segments;
  const energy = new Float64Array(recording.sampleCount);
  for (let index = 0; index < energy.length; index += 1) {
    energy[index] = Math.hypot(
      filteredChannels[0][index],
      filteredChannels[1][index],
      filteredChannels[2][index],
    );
  }
  const smoothedEnergy = uniformNearest(energy, 200, Float64Array);
  const timestamps = recording.timestamps;

  return segments.map((source) => {
    const segment = { ...source };
    const originalStart = segment.startMs;
    const originalEnd = segment.endMs;
    for (const boundary of ["start", "end"]) {
      const center = boundary === "start" ? segment.startMs : segment.endMs;
      const begin = lowerBound(timestamps, center - 15_000);
      const end = upperBound(timestamps, center + 15_000);
      if (end - begin <= 100) continue;
      let bestIndex = begin;
      let bestScore = boundary === "start" ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
      for (let index = begin; index < end; index += 1) {
        let gradient;
        if (index === begin) gradient = smoothedEnergy[index + 1] - smoothedEnergy[index];
        else if (index === end - 1) gradient = smoothedEnergy[index] - smoothedEnergy[index - 1];
        else gradient = (smoothedEnergy[index + 1] - smoothedEnergy[index - 1]) / 2;
        const score = boundary === "start" ? Math.abs(gradient) : gradient;
        if (
          (boundary === "start" && score > bestScore) ||
          (boundary === "end" && score < bestScore)
        ) {
          bestScore = score;
          bestIndex = index;
        }
      }
      const candidate = timestamps[bestIndex];
      if (Math.abs(candidate - center) < 15_000) {
        if (boundary === "start") segment.startMs = candidate;
        else segment.endMs = candidate;
      }
    }
    if (segment.startMs >= segment.endMs) {
      segment.startMs = originalStart;
      segment.endMs = originalEnd;
    }
    segment.durationSec = (segment.endMs - segment.startMs) / 1000;
    return segment;
  });
}

function resolveOverlaps(segments) {
  if (segments.length <= 1) return segments.map((segment) => ({ ...segment }));
  const resolved = [];
  for (const source of [...segments].sort((a, b) => a.startMs - b.startMs)) {
    const segment = { ...source };
    const previous = resolved[resolved.length - 1];
    if (!previous || segment.startMs >= previous.endMs) {
      resolved.push(segment);
      continue;
    }
    if (segment.classIndex === previous.classIndex) {
      previous.endMs = Math.max(previous.endMs, segment.endMs);
      previous.durationSec = (previous.endMs - previous.startMs) / 1000;
      continue;
    }
    if (segment.confidence > previous.confidence) {
      const middle = Math.floor((segment.startMs + previous.endMs) / 2);
      previous.endMs = middle;
      previous.durationSec = (previous.endMs - previous.startMs) / 1000;
      segment.startMs = middle;
    } else {
      segment.startMs = previous.endMs;
    }
    segment.durationSec = (segment.endMs - segment.startMs) / 1000;
    if (segment.durationSec > 0) resolved.push(segment);
  }
  return resolved;
}

function selectTopK(segments, topK) {
  if (!topK || segments.length <= topK) return segments;
  const byClass = new Map();
  for (const segment of segments) {
    const existing = byClass.get(segment.classIndex);
    if (!existing || segment.confidence > existing.confidence) {
      byClass.set(segment.classIndex, segment);
    }
  }
  const selected = [...byClass.values()];
  if (selected.length >= topK) {
    return selected
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, topK)
      .sort((a, b) => a.startMs - b.startMs);
  }
  const selectedSet = new Set(selected);
  const remaining = segments
    .filter((segment) => !selectedSet.has(segment))
    .sort((a, b) => b.confidence - a.confidence);
  while (selected.length < topK && remaining.length) selected.push(remaining.shift());
  return selected.sort((a, b) => a.startMs - b.startMs);
}

export function postprocessSegments(
  decodedPath,
  timestamps,
  probabilities,
  recording,
  filteredChannels,
  options,
) {
  let segments = extractSegments(decodedPath, timestamps, probabilities);
  segments = mergeSameClassSegments(segments);
  segments = refineBoundaries(segments, recording, filteredChannels);
  segments = resolveOverlaps(segments);
  segments = segments.filter((segment) => segment.durationSec >= options.minDurationSec);
  if (options.topK > 0 && segments.length > options.topK) {
    segments = selectTopK(segments, options.topK);
  }
  if (options.confidenceMin > 0) {
    segments = segments.filter((segment) => segment.confidence >= options.confidenceMin);
  }
  return segments.sort((a, b) => a.startMs - b.startMs);
}

function softmax(logits) {
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of logits) maximum = Math.max(maximum, value);
  const values = new Float32Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    values[index] = Math.exp(logits[index] - maximum);
    total += values[index];
  }
  for (let index = 0; index < values.length; index += 1) values[index] /= Math.max(total, 1e-12);
  return values;
}

export async function predictScale(
  ort,
  session,
  spec,
  filteredChannels,
  timestamps,
  onProgress = () => {},
) {
  const count = Math.floor((timestamps.length - spec.windowSize) / STEP_SIZE) + 1;
  if (count <= 0) return null;
  const scaleTimestamps = new Float64Array(count);
  const probabilities = matrix(count);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  for (let windowIndex = 0; windowIndex < count; windowIndex += 1) {
    const start = windowIndex * STEP_SIZE;
    const values = new Float32Array(spec.windowSize * CHANNEL_COUNT);
    for (let sample = 0; sample < spec.windowSize; sample += 1) {
      for (let channel = 0; channel < CHANNEL_COUNT; channel += 1) {
        values[sample * CHANNEL_COUNT + channel] =
          (filteredChannels[channel][start + sample] - spec.mean[channel]) / spec.std[channel];
      }
    }
    scaleTimestamps[windowIndex] = timestamps[start + Math.floor(spec.windowSize / 2)];
    const input = new ort.Tensor("float32", values, [1, spec.windowSize, CHANNEL_COUNT]);
    const outputMap = await session.run({ [inputName]: input });
    const output = outputMap[outputName];
    const probabilitiesRow = softmax(output.data);
    probabilities.data.set(probabilitiesRow, windowIndex * CLASS_COUNT);
    output.dispose?.();
    input.dispose?.();
    if (windowIndex === count - 1 || windowIndex % 4 === 0) {
      onProgress({ suffix: spec.suffix, completed: windowIndex + 1, total: count });
    }
  }
  return { suffix: spec.suffix, timestamps: scaleTimestamps, probabilities };
}

function validateOptions(options) {
  const normalized = {
    fusionMode: options.fusionMode || "local_boundary",
    minDurationSec: Number(options.minDurationSec ?? 5),
    confidenceMin: Number(options.confidenceMin ?? 0.3),
    topK: Number(options.topK ?? 5),
  };
  if (!FUSION_MODES.has(normalized.fusionMode)) throw new DemoInputError("bad_fusion");
  if (normalized.minDurationSec < 1 || normalized.minDurationSec > 180) {
    throw new DemoInputError("bad_min_duration");
  }
  if (normalized.confidenceMin < 0 || normalized.confidenceMin > 1) {
    throw new DemoInputError("bad_confidence");
  }
  if (!Number.isInteger(normalized.topK) || normalized.topK < 0 || normalized.topK > 10) {
    throw new DemoInputError("bad_top_k");
  }
  return normalized;
}

function downsampleRecording(recording, maximumPoints = 3_000) {
  const step = Math.max(1, Math.ceil(recording.sampleCount / maximumPoints));
  const indices = [];
  for (let index = 0; index < recording.sampleCount; index += step) indices.push(index);
  if (indices[indices.length - 1] !== recording.sampleCount - 1) indices.push(recording.sampleCount - 1);
  const times = new Float32Array(indices.length);
  const channels = Array.from({ length: CHANNEL_COUNT }, () => new Float32Array(indices.length));
  const origin = recording.timestamps[0];
  for (let outputIndex = 0; outputIndex < indices.length; outputIndex += 1) {
    const sourceIndex = indices[outputIndex];
    times[outputIndex] = (recording.timestamps[sourceIndex] - origin) / 1000;
    for (let channel = 0; channel < CHANNEL_COUNT; channel += 1) {
      channels[channel][outputIndex] = recording.channels[channel][sourceIndex];
    }
  }
  return { times, channels };
}

export async function runBrowserPipeline({
  ort,
  sessions,
  recording,
  options = {},
  onProgress = () => {},
}) {
  const settings = validateOptions(options);
  onProgress({ phase: "filtering" });
  const filteredChannels = butterworthFilter(recording.channels);
  const scales = [];
  for (let index = 0; index < MODEL_SPECS.length; index += 1) {
    const spec = MODEL_SPECS[index];
    const session = sessions.get(spec.suffix);
    if (!session || recording.sampleCount < spec.windowSize) continue;
    const result = await predictScale(
      ort,
      session,
      spec,
      filteredChannels,
      recording.timestamps,
      (progress) => onProgress({ phase: "inference", scaleIndex: index, ...progress }),
    );
    if (result) scales.push(result);
  }
  if (!scales.length) throw new DemoInputError("no_windows");
  onProgress({ phase: "postprocessing" });
  const { timestamps, aligned } = alignScaleProbabilities(scales);
  const fused = fuseProbabilities(aligned, settings.fusionMode);
  const smoothed = smoothProbabilities(fused);
  const decodedPath = viterbiDecode(smoothed);
  const segments = postprocessSegments(
    decodedPath,
    timestamps,
    smoothed,
    recording,
    filteredChannels,
    settings,
  );
  const origin = recording.timestamps[0];
  return {
    recording: {
      filename: recording.filename,
      userId: recording.userId,
      sampleCount: recording.sampleCount,
      sampleRateHz: recording.sampleRateHz,
      durationSec: recording.durationSec,
      originMs: origin,
    },
    rawPreview: downsampleRecording(recording),
    timeline: {
      times: Float32Array.from(timestamps, (timestamp) => (timestamp - origin) / 1000),
      probabilities: smoothed.data,
      decodedPath,
      rows: smoothed.rows,
    },
    segments,
    modelScales: scales.map((scale) => scale.suffix),
    settings,
  };
}
