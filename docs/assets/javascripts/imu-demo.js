(function () {
  "use strict";

  const root = document.getElementById("imu-browser-demo");
  if (!root) return;

  const locale = root.dataset.locale === "zh" ? "zh" : "en";
  const ownScript = Array.from(document.scripts).find((script) =>
    /\/imu-demo\.js(?:\?|$)/.test(script.src),
  );
  if (!ownScript) return;
  const workerUrl = new URL("imu-demo-worker.mjs", ownScript.src);
  const sampleUrl = new URL("../data/synthetic_activity_imu.tsv", ownScript.src);

  const COPY = {
    en: {
      sampleName: "synthetic_activity_imu.tsv · 12,000 samples · 120 s",
      readyTitle: "Ready for local inference",
      readyDetail: "Choose Run to execute the real 3 s, 5 s, and 8 s ONNX models in this browser.",
      reading: "Reading the recording",
      validating: "Checking columns, timestamps, and sample rate",
      validated: "Recording validated",
      filtering: "Applying the zero-phase Butterworth filter",
      models: "Downloading and verifying public ONNX models",
      runtime: "Starting the browser inference runtime",
      fallback: "WebGPU is unavailable for this model; switching to WASM",
      inference: (suffix, completed, total) =>
        `Running the ${suffix || "multi-scale"} model${total ? ` · ${completed}/${total} windows` : ""}`,
      postprocessing: "Fusing scales and applying the Temporal Record Layer",
      complete: "Local inference complete",
      runningDetail: "The file remains on this device. Only public model files are downloaded.",
      resultDetail: (result) =>
        `${result.recording.sampleCount.toLocaleString()} samples · ${result.timeline.rows} timeline points · ${result.segments.length} records · ${(result.elapsedMs / 1000).toFixed(1)} s`,
      selected: (name, size) => `${name} · ${(size / 1024).toFixed(1)} KB`,
      noRecords: "No activity records passed the current duration and confidence filters.",
      csv: "Download result CSV",
      backend: (backend) => `${backend} · local`,
      cache: (count) => (count === 3 ? "Models loaded from browser cache" : "Models verified and cached locally"),
      errors: {
        file_too_large: "The recording exceeds the 20 MB browser Demo limit.",
        bad_encoding: "The file is not valid UTF-8 text.",
        missing_header: "The recording does not contain a TSV header.",
        missing_columns: "Required columns are missing",
        too_few_samples: "At least 800 valid samples (8 seconds at 100 Hz) are required.",
        too_many_samples: "At most 60,000 valid samples (10 minutes at 100 Hz) are accepted.",
        timestamps_not_unique: "ACC_TIME must contain unique, strictly increasing timestamps.",
        bad_sample_rate: "The median sample interval must be 8–12 ms (approximately 100 Hz).",
        model_hash_mismatch: "A downloaded model failed its SHA-256 integrity check.",
        no_windows: "The recording did not produce any model windows.",
        busy: "The browser worker is already running another recording.",
        runtime_error: "Browser inference could not be completed.",
      },
      errorTitle: "Could not run this recording",
      table: ["Activity", "Start (s)", "End (s)", "Duration (s)", "Confidence"],
      csvHeader: [
        "recording",
        "activity",
        "start_ms",
        "end_ms",
        "start_relative_sec",
        "end_relative_sec",
        "duration_sec",
        "confidence",
      ],
      activities: {
        background: "Background",
        badminton: "Badminton",
        jump_rope: "Jump rope",
        fly: "Fly",
        running: "Running",
        table_tennis: "Table tennis",
      },
      time: "Time (s)",
      acceleration: "Acceleration · aₓ, aᵧ, a_z",
      gyroscope: "Angular velocity · ωₓ, ωᵧ, ω_z",
      likelihood: "Smoothed class likelihood",
      decoded: "Decoded activity timeline",
      fileReadError: "The selected recording could not be read.",
    },
    zh: {
      sampleName: "synthetic_activity_imu.tsv · 12,000 个样本 · 120 秒",
      readyTitle: "可以开始本地推理",
      readyDetail: "点击“运行”，浏览器将真实执行 3 秒、5 秒和 8 秒三份 ONNX 模型。",
      reading: "正在读取记录",
      validating: "正在检查列名、时间戳和采样率",
      validated: "记录校验通过",
      filtering: "正在执行零相位 Butterworth 滤波",
      models: "正在下载并校验公开 ONNX 模型",
      runtime: "正在启动浏览器推理运行时",
      fallback: "该模型无法使用 WebGPU，正在切换到 WASM",
      inference: (suffix, completed, total) =>
        `正在运行 ${suffix || "多尺度"} 模型${total ? ` · ${completed}/${total} 个窗口` : ""}`,
      postprocessing: "正在融合三尺度结果并执行时间记录层",
      complete: "本地推理完成",
      runningDetail: "记录始终留在本机；浏览器只会下载公开模型文件。",
      resultDetail: (result) =>
        `${result.recording.sampleCount.toLocaleString()} 个样本 · ${result.timeline.rows} 个时间线点 · ${result.segments.length} 条记录 · ${(result.elapsedMs / 1000).toFixed(1)} 秒`,
      selected: (name, size) => `${name} · ${(size / 1024).toFixed(1)} KB`,
      noRecords: "当前时长和置信度筛选条件下没有保留下来的活动记录。",
      csv: "下载结果 CSV",
      backend: (backend) => `${backend} · 本地计算`,
      cache: (count) => (count === 3 ? "模型来自浏览器缓存" : "模型已校验并缓存在本机"),
      errors: {
        file_too_large: "记录超过浏览器 Demo 的 20 MB 限制。",
        bad_encoding: "文件不是有效的 UTF-8 文本。",
        missing_header: "记录中没有 TSV 表头。",
        missing_columns: "缺少必需列",
        too_few_samples: "至少需要 800 个有效样本（100 Hz 下为 8 秒）。",
        too_many_samples: "最多接受 60,000 个有效样本（100 Hz 下为 10 分钟）。",
        timestamps_not_unique: "ACC_TIME 必须是唯一且严格递增的时间戳。",
        bad_sample_rate: "采样间隔中位数必须为 8–12 毫秒（约 100 Hz）。",
        model_hash_mismatch: "下载的模型未通过 SHA-256 完整性校验。",
        no_windows: "这份记录未能生成任何模型窗口。",
        busy: "浏览器 Worker 正在处理另一份记录。",
        runtime_error: "浏览器未能完成本次推理。",
      },
      errorTitle: "无法运行这份记录",
      table: ["活动", "开始（秒）", "结束（秒）", "时长（秒）", "置信度"],
      csvHeader: [
        "记录标识",
        "活动",
        "开始时间戳（毫秒）",
        "结束时间戳（毫秒）",
        "相对开始时间（秒）",
        "相对结束时间（秒）",
        "时长（秒）",
        "置信度",
      ],
      activities: {
        background: "无活动",
        badminton: "羽毛球",
        jump_rope: "跳绳",
        fly: "飞鸟",
        running: "跑步",
        table_tennis: "乒乓球",
      },
      time: "时间（秒）",
      acceleration: "加速度 · aₓ, aᵧ, a_z",
      gyroscope: "角速度 · ωₓ, ωᵧ, ω_z",
      likelihood: "平滑后的类别概率",
      decoded: "最终解码活动时间线",
      fileReadError: "无法读取所选记录。",
    },
  }[locale];

  const COLORS = ["#7b8794", "#8e5cc2", "#de7a22", "#168c7e", "#d34b65", "#3d6fb6"];
  const CHANNEL_COLORS = ["#3d6fb6", "#d34b65", "#168c7e", "#8e5cc2", "#de7a22", "#32a2a8"];
  const CHANNEL_LABELS = ["aₓ", "aᵧ", "a_z", "ωₓ", "ωᵧ", "ω_z"];

  const elements = Object.fromEntries(
    Array.from(root.querySelectorAll("[data-demo-id]")).map((element) => [
      element.dataset.demoId,
      element,
    ]),
  );
  let selectedFile = null;
  let latestResult = null;
  let currentJob = 0;
  let csvUrl = null;
  let running = false;

  const worker = new Worker(workerUrl, { type: "module", name: "imu-local-inference" });

  function setStatus(kind, title, detail, progress) {
    elements.status.dataset.kind = kind;
    elements.statusTitle.textContent = title;
    elements.statusDetail.textContent = detail || "";
    if (typeof progress === "number") {
      elements.progress.hidden = false;
      elements.progress.value = Math.max(0, Math.min(1, progress));
    } else {
      elements.progress.hidden = true;
      elements.progress.removeAttribute("value");
    }
  }

  function setRunning(value) {
    running = value;
    for (const control of root.querySelectorAll("button, input, select")) {
      control.disabled = value;
    }
    root.dataset.running = value ? "true" : "false";
  }

  function resetDemo() {
    if (running) return;
    selectedFile = null;
    elements.file.value = "";
    elements.fileName.textContent = COPY.sampleName;
    elements.fusion.value = "local_boundary";
    elements.duration.value = "5";
    elements.durationOutput.textContent = "5";
    elements.confidence.value = "0.30";
    elements.confidenceOutput.textContent = "0.30";
    elements.topK.value = "5";
    elements.topKOutput.textContent = "5";
    setStatus("ready", COPY.readyTitle, COPY.readyDetail);
  }

  function selectFile(file) {
    if (!file || running) return;
    selectedFile = file;
    elements.fileName.textContent = COPY.selected(file.name, file.size);
    setStatus("ready", COPY.readyTitle, COPY.readyDetail);
  }

  async function readCurrentRecording() {
    if (selectedFile) return selectedFile.arrayBuffer();
    const response = await fetch(sampleUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.arrayBuffer();
  }

  async function run() {
    if (running) return;
    setRunning(true);
    currentJob += 1;
    const jobId = currentJob;
    latestResult = null;
    elements.results.hidden = true;
    setStatus("running", COPY.reading, COPY.runningDetail, 0.01);
    try {
      const bytes = await readCurrentRecording();
      const filename = selectedFile ? selectedFile.name : "synthetic_activity_imu.tsv";
      worker.postMessage(
        {
          type: "run",
          jobId,
          filename,
          bytes,
          options: {
            fusionMode: elements.fusion.value,
            minDurationSec: Number(elements.duration.value),
            confidenceMin: Number(elements.confidence.value),
            topK: Number(elements.topK.value),
          },
        },
        [bytes],
      );
    } catch (error) {
      setRunning(false);
      setStatus("error", COPY.errorTitle, `${COPY.fileReadError} ${String(error.message || error)}`);
    }
  }

  function phaseText(message) {
    if (message.phase === "validating") return COPY.validating;
    if (message.phase === "validated") return COPY.validated;
    if (message.phase === "filtering") return COPY.filtering;
    if (message.phase === "models") return COPY.models;
    if (message.phase === "runtime") return COPY.runtime;
    if (message.phase === "fallback") return COPY.fallback;
    if (message.phase === "inference") {
      return COPY.inference(message.suffix, message.completed, message.total);
    }
    if (message.phase === "postprocessing") return COPY.postprocessing;
    return COPY.complete;
  }

  function errorDetail(message) {
    let detail = COPY.errors[message.code] || COPY.errors.runtime_error;
    if (message.code === "missing_columns" && message.details?.missing?.length) {
      detail += `: ${message.details.missing.join(", ")}.`;
    }
    if (message.code === "bad_sample_rate" && message.details?.medianIntervalMs) {
      detail += ` (${Number(message.details.medianIntervalMs).toFixed(2)} ms)`;
    }
    return detail;
  }

  worker.addEventListener("message", (event) => {
    const message = event.data;
    if (!message || message.jobId !== currentJob) return;
    if (message.type === "progress") {
      setStatus("running", phaseText(message), COPY.runningDetail, message.progress);
      return;
    }
    if (message.type === "error") {
      setRunning(false);
      setStatus("error", COPY.errorTitle, errorDetail(message));
      return;
    }
    if (message.type === "result") {
      setRunning(false);
      latestResult = message.result;
      renderResult(latestResult);
      setStatus("success", COPY.complete, COPY.resultDetail(latestResult));
    }
  });

  worker.addEventListener("error", (event) => {
    setRunning(false);
    setStatus("error", COPY.errorTitle, `${COPY.errors.runtime_error} ${event.message || ""}`);
  });

  function formatNumber(value, digits) {
    return Number(value).toFixed(digits);
  }

  function renderResult(result) {
    elements.results.hidden = false;
    elements.samples.textContent = result.recording.sampleCount.toLocaleString();
    elements.recordingDuration.textContent = `${formatNumber(result.recording.durationSec, 1)} s`;
    elements.points.textContent = String(result.timeline.rows);
    elements.records.textContent = String(result.segments.length);
    elements.backend.textContent = COPY.backend(result.backend);
    elements.cache.textContent = COPY.cache(result.cachedModels);
    renderTable(result);
    prepareCsv(result);
    activateTab("signal");
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderTable(result) {
    elements.tableHead.innerHTML = "";
    const headRow = document.createElement("tr");
    for (const label of COPY.table) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      headRow.append(cell);
    }
    elements.tableHead.append(headRow);
    elements.tableBody.innerHTML = "";
    if (!result.segments.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.className = "imu-demo-empty";
      cell.textContent = COPY.noRecords;
      row.append(cell);
      elements.tableBody.append(row);
      return;
    }
    for (const segment of result.segments) {
      const values = [
        COPY.activities[segment.activityKey],
        formatNumber((segment.startMs - result.recording.originMs) / 1000, 2),
        formatNumber((segment.endMs - result.recording.originMs) / 1000, 2),
        formatNumber(segment.durationSec, 2),
        formatNumber(segment.confidence, 4),
      ];
      const row = document.createElement("tr");
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      elements.tableBody.append(row);
    }
  }

  function csvCell(value) {
    const text = String(value ?? "");
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  }

  function prepareCsv(result) {
    const rows = [COPY.csvHeader];
    for (const segment of result.segments) {
      rows.push([
        result.recording.userId,
        COPY.activities[segment.activityKey],
        Math.round(segment.startMs),
        Math.round(segment.endMs),
        formatNumber((segment.startMs - result.recording.originMs) / 1000, 3),
        formatNumber((segment.endMs - result.recording.originMs) / 1000, 3),
        formatNumber(segment.durationSec, 3),
        formatNumber(segment.confidence, 6),
      ]);
    }
    const csv = `\uFEFF${rows.map((row) => row.map(csvCell).join(",")).join("\r\n")}\r\n`;
    if (csvUrl) URL.revokeObjectURL(csvUrl);
    csvUrl = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    elements.download.href = csvUrl;
    elements.download.download = `${result.recording.userId}_segments_${locale}.csv`;
    elements.download.textContent = COPY.csv;
  }

  function canvasContext(canvas, height) {
    const width = Math.max(320, canvas.clientWidth || 900);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    const styles = getComputedStyle(root);
    return {
      context,
      width,
      height,
      text: styles.getPropertyValue("--research-text").trim() || "#172b4d",
      muted: styles.getPropertyValue("--research-muted").trim() || "#5b6573",
      border: styles.getPropertyValue("--research-border").trim() || "#d9e1e8",
      surface: styles.getPropertyValue("--research-surface").trim() || "#ffffff",
    };
  }

  function drawAxes(chart, area, xMaximum, yMinimum, yMaximum, yLabel) {
    const { context, muted, border, text } = chart;
    context.save();
    context.strokeStyle = border;
    context.fillStyle = muted;
    context.lineWidth = 1;
    context.font = "12px Inter, Segoe UI, sans-serif";
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (let tick = 0; tick <= 4; tick += 1) {
      const ratio = tick / 4;
      const y = area.y + area.height * ratio;
      context.beginPath();
      context.moveTo(area.x, y);
      context.lineTo(area.x + area.width, y);
      context.stroke();
      const value = yMaximum - ratio * (yMaximum - yMinimum);
      context.fillText(value.toFixed(Math.abs(value) < 2 ? 1 : 0), area.x - 7, y);
    }
    context.textAlign = "center";
    context.textBaseline = "top";
    for (let tick = 0; tick <= 4; tick += 1) {
      const ratio = tick / 4;
      const x = area.x + area.width * ratio;
      context.fillText((xMaximum * ratio).toFixed(0), x, area.y + area.height + 6);
    }
    context.fillStyle = text;
    context.font = "600 12px Inter, Segoe UI, sans-serif";
    context.textAlign = "left";
    context.textBaseline = "bottom";
    context.fillText(yLabel, area.x, area.y - 8);
    context.restore();
  }

  function drawLegend(chart, labels, colors, y) {
    const { context, width, muted } = chart;
    context.save();
    context.font = "11px Inter, Segoe UI, sans-serif";
    context.textBaseline = "middle";
    let x = 62;
    for (let index = 0; index < labels.length; index += 1) {
      const label = labels[index];
      const itemWidth = context.measureText(label).width + 29;
      if (x + itemWidth > width - 20) {
        x = 62;
        y += 17;
      }
      context.strokeStyle = colors[index];
      context.lineWidth = 2.5;
      context.beginPath();
      context.moveTo(x, y);
      context.lineTo(x + 13, y);
      context.stroke();
      context.fillStyle = muted;
      context.fillText(label, x + 18, y);
      x += itemWidth;
    }
    context.restore();
  }

  function drawRawSignals(result) {
    const chart = canvasContext(elements.signalCanvas, 500);
    const { context, width, height, surface } = chart;
    context.fillStyle = surface;
    context.fillRect(0, 0, width, height);
    const times = result.rawPreview.times;
    const xMaximum = times[times.length - 1] || 1;
    const groups = [
      { channels: [0, 1, 2], title: COPY.acceleration, y: 42 },
      { channels: [3, 4, 5], title: COPY.gyroscope, y: 282 },
    ];
    for (const group of groups) {
      const area = { x: 58, y: group.y, width: width - 78, height: 168 };
      let minimum = Number.POSITIVE_INFINITY;
      let maximum = Number.NEGATIVE_INFINITY;
      for (const channel of group.channels) {
        for (const value of result.rawPreview.channels[channel]) {
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
        }
      }
      const padding = Math.max(1, (maximum - minimum) * 0.08);
      minimum -= padding;
      maximum += padding;
      drawAxes(chart, area, xMaximum, minimum, maximum, group.title);
      for (const channel of group.channels) {
        const values = result.rawPreview.channels[channel];
        context.strokeStyle = CHANNEL_COLORS[channel];
        context.lineWidth = 1.2;
        context.beginPath();
        for (let index = 0; index < times.length; index += 1) {
          const x = area.x + (times[index] / xMaximum) * area.width;
          const y = area.y + ((maximum - values[index]) / (maximum - minimum)) * area.height;
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
        context.stroke();
      }
      drawLegend(
        chart,
        group.channels.map((channel) => CHANNEL_LABELS[channel]),
        group.channels.map((channel) => CHANNEL_COLORS[channel]),
        group.y + 192,
      );
    }
    context.fillStyle = chart.muted;
    context.font = "12px Inter, Segoe UI, sans-serif";
    context.textAlign = "center";
    context.fillText(COPY.time, width / 2, height - 5);
  }

  function drawTimeline(result) {
    const chart = canvasContext(elements.timelineCanvas, 455);
    const { context, width, height, surface } = chart;
    context.fillStyle = surface;
    context.fillRect(0, 0, width, height);
    const times = result.timeline.times;
    const xMaximum = times[times.length - 1] || 1;
    const area = { x: 58, y: 42, width: width - 78, height: 235 };
    drawAxes(chart, area, xMaximum, 0, 1, COPY.likelihood);
    for (let classIndex = 0; classIndex < 6; classIndex += 1) {
      context.strokeStyle = COLORS[classIndex];
      context.lineWidth = classIndex === 0 ? 1.2 : 1.7;
      context.beginPath();
      for (let row = 0; row < result.timeline.rows; row += 1) {
        const x = area.x + (times[row] / xMaximum) * area.width;
        const probability = result.timeline.probabilities[row * 6 + classIndex];
        const y = area.y + (1 - probability) * area.height;
        if (row === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    }
    drawLegend(
      chart,
      ["background", "badminton", "jump_rope", "fly", "running", "table_tennis"].map(
        (key) => COPY.activities[key],
      ),
      COLORS,
      300,
    );

    const band = { x: area.x, y: 355, width: area.width, height: 34 };
    context.fillStyle = chart.text;
    context.font = "600 12px Inter, Segoe UI, sans-serif";
    context.textAlign = "left";
    context.fillText(COPY.decoded, band.x, band.y - 10);
    for (let row = 0; row < result.timeline.rows; row += 1) {
      const left = row === 0 ? 0 : (times[row - 1] + times[row]) / 2;
      const right =
        row === result.timeline.rows - 1 ? xMaximum : (times[row] + times[row + 1]) / 2;
      const x = band.x + (left / xMaximum) * band.width;
      const itemWidth = Math.max(1, ((right - left) / xMaximum) * band.width + 0.5);
      context.fillStyle = COLORS[result.timeline.decodedPath[row]];
      context.fillRect(x, band.y, itemWidth, band.height);
    }
    context.strokeStyle = chart.border;
    context.strokeRect(band.x, band.y, band.width, band.height);
    context.fillStyle = chart.muted;
    context.font = "12px Inter, Segoe UI, sans-serif";
    context.textAlign = "center";
    for (let tick = 0; tick <= 4; tick += 1) {
      const ratio = tick / 4;
      context.fillText((xMaximum * ratio).toFixed(0), band.x + band.width * ratio, band.y + 51);
    }
    context.fillText(COPY.time, width / 2, height - 7);
  }

  function activateTab(tabName) {
    for (const button of root.querySelectorAll("[data-demo-tab]")) {
      const selected = button.dataset.demoTab === tabName;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    }
    for (const panel of root.querySelectorAll("[data-demo-panel]")) {
      panel.hidden = panel.dataset.demoPanel !== tabName;
    }
    if (!latestResult) return;
    requestAnimationFrame(() => {
      if (tabName === "signal") drawRawSignals(latestResult);
      if (tabName === "timeline") drawTimeline(latestResult);
    });
  }

  for (const button of root.querySelectorAll("[data-demo-tab]")) {
    button.addEventListener("click", () => activateTab(button.dataset.demoTab));
  }
  elements.file.addEventListener("change", () => selectFile(elements.file.files?.[0]));
  elements.run.addEventListener("click", run);
  elements.reset.addEventListener("click", resetDemo);
  elements.duration.addEventListener("input", () => {
    elements.durationOutput.textContent = elements.duration.value;
  });
  elements.confidence.addEventListener("input", () => {
    elements.confidenceOutput.textContent = Number(elements.confidence.value).toFixed(2);
  });
  elements.topK.addEventListener("input", () => {
    elements.topKOutput.textContent = elements.topK.value;
  });

  for (const eventName of ["dragenter", "dragover"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!running) elements.dropZone.dataset.dragging = "true";
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      delete elements.dropZone.dataset.dragging;
    });
  }
  elements.dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer?.files?.[0]));

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!latestResult) return;
      const selected = root.querySelector('[data-demo-tab][aria-selected="true"]');
      activateTab(selected?.dataset.demoTab || "signal");
    }, 120);
  });
  new MutationObserver(() => {
    if (!latestResult) return;
    const selected = root.querySelector('[data-demo-tab][aria-selected="true"]');
    activateTab(selected?.dataset.demoTab || "signal");
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-md-color-scheme"] });

  resetDemo();
})();
