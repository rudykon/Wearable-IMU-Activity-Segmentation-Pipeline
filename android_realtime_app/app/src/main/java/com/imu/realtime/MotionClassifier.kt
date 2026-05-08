package com.imu.realtime

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Motion classifier using the paper inference pipeline over the complete
 * accumulated session history.  Each inference pass can revise all previously
 * displayed labels and segments.
 */
class MotionClassifier(context: Context) {

    companion object {
        val CLASS_NAMES  = arrayOf("无活动", "羽毛球", "跳绳", "飞鸟", "跑步", "乒乓球")
        val CLASS_EMOJIS = arrayOf("💤", "🏸", "🪢", "🦅", "🏃", "🏓")

        private const val TAG = "MotionClassifier"

        const val WINDOW_3S  = 300
        const val WINDOW_5S  = 500
        const val WINDOW_8S  = 800
        const val STEP_SIZE  = 100
        const val CHANNELS   = 6

        private const val SAMPLE_PERIOD_MS = 10L
        private const val REF_WINDOW_SEC = 3
        private const val SHORT_GAP_SEC = 60f
        const val OUTPUT_MIN_SEGMENT_SEC = 180
        private const val MIN_SEGMENT_SEC = 180f
        private const val TOP_K = 3
        private const val CONF_MIN = 0.45f

        private const val AVG_SMOOTH_SIZE = 7
        private const val MEDIAN_SIZE = 5

        private const val BASE_WEIGHT_3S = 0.20f
        private const val BASE_WEIGHT_5S = 0.35f
        private const val BASE_WEIGHT_8S = 0.45f
        private const val BOUNDARY_BOOST_3S = 0.30f
        private const val BOUNDARY_PENALTY_5S = 0.08f
        private const val BOUNDARY_PENALTY_8S = 0.22f
        private const val MIN_SCALE_WEIGHT = 0.05f
        private const val LOCAL_BOUNDARY_RADIUS = 3
        private const val LOCAL_BOUNDARY_SMOOTH = 3

        private val MODELS_3S = arrayOf("combined_model_3s_seed42.onnx")
        private val MODELS_5S = arrayOf("combined_model_5s_seed123.onnx")
        private val MODELS_8S = arrayOf("combined_model_8s_seed123.onnx")
        private const val LEGACY_MODEL = "hand_motion.onnx"
        private const val LEGACY_NORM  = "norm_params.json"

        private val FILTER_B = floatArrayOf(
            0.43284664499f, 1.73138657996f, 2.59707986994f,
            1.73138657996f, 0.43284664499f,
        )
        private val FILTER_A = floatArrayOf(
            1.0f, 2.36951300718f, 2.31398841442f,
            1.05466540588f, 0.187379492368f,
        )
        private val FILTER_ZI = floatArrayOf(
            0.56715335501f, 1.20527978223f,
            0.922188326705f, 0.245467152622f,
        )
        private const val FILTER_PAD = 15
        private const val ENERGY_SMOOTH_SIZE = 200
        private const val BOUNDARY_SEARCH_MS = 15_000L
    }

    private inner class OrtModel(
        val session: OrtSession,
        val windowSize: Int,
        val mean: FloatArray,
        val std: FloatArray,
    )

    private data class ScaleSeries(
        val suffix: String,
        val timestampsMs: LongArray,
        val probs: Array<FloatArray>,
    )

    private data class SensorSeries(
        val timestampsMs: LongArray,
        val samples: Array<FloatArray>,
        val hasOriginalTimestamps: Boolean,
    )

    private data class SegmentCandidate(
        var classIdx: Int,
        var startMs: Long,
        var endMs: Long,
        var confidence: Float,
        var startWindowIdx: Int,
        var endWindowIdx: Int,
    ) {
        val durationSec: Float get() = (endMs - startMs) / 1000f
    }

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val models3s = mutableListOf<OrtModel>()
    private val models5s = mutableListOf<OrtModel>()
    private val models8s = mutableListOf<OrtModel>()

    var loadedModelCount = 0
        private set
    var loaded3sModelCount = 0
        private set
    var loaded5sModelCount = 0
        private set
    var loaded8sModelCount = 0
        private set
    var usingLegacyModel = false
        private set
    var isAvailable = false
        private set
    var modelSummary = "未加载"
        private set

    private val rawHistory = ArrayList<FloatArray>(36_000)
    private val historyLock = Any()
    @Volatile private var totalSamples = 0
    @Volatile private var stepCount = 0

    private val inferScheduler = Executors.newSingleThreadExecutor()
    private val inferPool = Executors.newFixedThreadPool(3)
    @Volatile private var inferenceVersion = 0

    var onResult: ((RecognitionResult) -> Unit)? = null

    init {
        loadEnsemble(context)
    }

    private fun loadEnsemble(ctx: Context) {
        val norm3s = tryLoadNorm(ctx, "norm_params_3s.json")
        val norm5s = tryLoadNorm(ctx, "norm_params_5s.json")
        val norm8s = tryLoadNorm(ctx, "norm_params_8s.json")

        if (norm3s != null) {
            for (fileName in MODELS_3S) tryLoadModel(ctx, fileName, WINDOW_3S, norm3s)?.let { models3s += it }
        }
        if (norm5s != null) {
            for (fileName in MODELS_5S) tryLoadModel(ctx, fileName, WINDOW_5S, norm5s)?.let { models5s += it }
        }
        if (norm8s != null) {
            for (fileName in MODELS_8S) tryLoadModel(ctx, fileName, WINDOW_8S, norm8s)?.let { models8s += it }
        }

        loadedModelCount = models3s.size + models5s.size + models8s.size
        loaded3sModelCount = models3s.size
        loaded5sModelCount = models5s.size
        loaded8sModelCount = models8s.size

        if (loadedModelCount > 0) {
            isAvailable = true
            usingLegacyModel = false
            modelSummary = buildString {
                if (models3s.isNotEmpty()) append("3s×${models3s.size}")
                if (isNotEmpty() && models5s.isNotEmpty()) append(" + ")
                if (models5s.isNotEmpty()) append("5s×${models5s.size}")
                if (isNotEmpty() && models8s.isNotEmpty()) append(" + ")
                if (models8s.isNotEmpty()) append("8s×${models8s.size}")
            }
            Log.i(TAG, "Selected offline-style models loaded: $modelSummary")
            return
        }

        val normLegacy = tryLoadNorm(ctx, LEGACY_NORM) ?: return
        tryLoadModel(ctx, LEGACY_MODEL, WINDOW_3S, normLegacy)?.let {
            models3s += it
            loadedModelCount = 1
            loaded3sModelCount = 1
            loaded5sModelCount = 0
            loaded8sModelCount = 0
            usingLegacyModel = true
            isAvailable = true
            modelSummary = "单模型(fallback)"
            Log.w(TAG, "Ensemble assets missing - fell back to $LEGACY_MODEL")
        }
    }

    private fun tryLoadNorm(ctx: Context, fileName: String): Pair<FloatArray, FloatArray>? {
        return try {
            val json = JSONObject(BufferedReader(InputStreamReader(ctx.assets.open(fileName))).readText())
            val meanA = json.getJSONArray("mean")
            val stdA = json.getJSONArray("std")
            val mean = FloatArray(CHANNELS) { meanA.getDouble(it).toFloat() }
            val std = FloatArray(CHANNELS) { stdA.getDouble(it).toFloat().coerceAtLeast(1e-6f) }
            Pair(mean, std)
        } catch (e: Exception) {
            Log.d(TAG, "Cannot load norm $fileName: ${e.message}")
            null
        }
    }

    private fun tryLoadModel(
        ctx: Context,
        fileName: String,
        windowSize: Int,
        norm: Pair<FloatArray, FloatArray>,
    ): OrtModel? {
        return try {
            val bytes = ctx.assets.open(fileName).readBytes()
            val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) }
            val session = ortEnv.createSession(bytes, opts)
            OrtModel(session, windowSize, norm.first, norm.second).also {
                Log.i(TAG, "Loaded $fileName (${windowSize / 100}s window)")
            }
        } catch (e: Exception) {
            Log.d(TAG, "Cannot load model $fileName: ${e.message}")
            null
        }
    }

    fun addSample(accX: Float, accY: Float, accZ: Float, gyroX: Float, gyroY: Float, gyroZ: Float) {
        if (!isAvailable) return

        val snapshot: Array<FloatArray>?
        val version: Int
        synchronized(historyLock) {
            rawHistory.add(floatArrayOf(accX, accY, accZ, gyroX, gyroY, gyroZ))
            totalSamples = rawHistory.size
            stepCount++
            if (stepCount < STEP_SIZE || rawHistory.size < WINDOW_3S) return
            stepCount = 0
            snapshot = Array(rawHistory.size) { rawHistory[it].copyOf() }
            version = ++inferenceVersion
        }

        inferScheduler.submit {
            val samples = snapshot ?: return@submit
            val series = SensorSeries(
                timestampsMs = LongArray(samples.size) { it * SAMPLE_PERIOD_MS },
                samples = samples,
                hasOriginalTimestamps = false,
            )
            val result = runOfflinePipeline(series) ?: return@submit
            if (version == inferenceVersion) onResult?.invoke(result)
        }
    }

    fun reset() {
        synchronized(historyLock) {
            rawHistory.clear()
            totalSamples = 0
            stepCount = 0
            inferenceVersion++
        }
    }

    fun bufferFill(): Int = minOf(totalSamples, WINDOW_8S)

    fun classifyOffline(series: OfflineSensorSeries): RecognitionResult? {
        if (!isAvailable || series.samples.size < WINDOW_3S) return null
        val timestamps = series.timestampsMs.copyOf()
        val samples = Array(series.samples.size) { series.samples[it].copyOf() }
        return runOfflinePipeline(SensorSeries(timestamps, samples, hasOriginalTimestamps = true))
    }

    private fun runOfflinePipeline(series: SensorSeries): RecognitionResult? {
        if (series.samples.isEmpty() || series.samples.size != series.timestampsMs.size) return null
        val raw = series.samples
        val sampleTimestamps = series.timestampsMs
        val filtered = zeroPhaseButterworth(raw)
        val futures = mutableListOf<Future<ScaleSeries?>>()
        futures += inferPool.submit<ScaleSeries?> {
            predictScale("3s", models3s, filtered, sampleTimestamps, WINDOW_3S)
        }
        if (filtered.size >= WINDOW_5S) futures += inferPool.submit<ScaleSeries?> {
            predictScale("5s", models5s, filtered, sampleTimestamps, WINDOW_5S)
        }
        if (filtered.size >= WINDOW_8S) futures += inferPool.submit<ScaleSeries?> {
            predictScale("8s", models8s, filtered, sampleTimestamps, WINDOW_8S)
        }

        val scales = futures.mapNotNull { runCatching { it.get() }.getOrNull() }
        if (scales.isEmpty()) return null

        val ref = alignScaleProbabilities(scales)
        val fused = localBoundaryWindowFusion(ref.second)
        val smoothed = medianFilterZero(uniformFilterNearest(fused, AVG_SMOOTH_SIZE), MEDIAN_SIZE)
        val path = viterbiDecode(smoothed)
        if (path.isEmpty()) return null

        val stats = pathStats(path)
        val segments = postProcessSegments(
            path,
            ref.first,
            smoothed,
            filtered,
            sampleTimestamps,
            series.hasOriginalTimestamps,
        )
        val lastIdx = smoothed.lastIndex
        val cls = path[lastIdx]
        val warmup = minOf(raw.size.toFloat() / WINDOW_8S, 1f)

        return RecognitionResult(
            classIdx = cls,
            className = CLASS_NAMES.getOrElse(cls) { "未知" },
            confidence = smoothed[lastIdx][cls],
            probs = fused[lastIdx].copyOf(),
            smoothedProbs = smoothed[lastIdx].copyOf(),
            warmupFraction = warmup,
            decodedSeconds = path.size,
            segments = segments,
            stats = stats,
        )
    }

    private fun predictScale(
        suffix: String,
        models: List<OrtModel>,
        data: Array<FloatArray>,
        sampleTimestampsMs: LongArray,
        windowSize: Int,
    ): ScaleSeries? {
        if (models.isEmpty() || data.size < windowSize) return null
        val count = ((data.size - windowSize) / STEP_SIZE) + 1
        val timestamps = LongArray(count)
        val probs = Array(count) { FloatArray(CLASS_NAMES.size) }

        var outIdx = 0
        var start = 0
        while (start + windowSize <= data.size) {
            timestamps[outIdx] = sampleTimestampsMs[start + windowSize / 2]
            val scaleProbs = models.mapNotNull { runSingleWindow(it, data, start) }
            if (scaleProbs.isEmpty()) return null
            for (p in scaleProbs) for (c in p.indices) probs[outIdx][c] += p[c]
            val n = scaleProbs.size.toFloat()
            for (c in probs[outIdx].indices) probs[outIdx][c] /= n
            outIdx++
            start += STEP_SIZE
        }
        return ScaleSeries(suffix, timestamps, probs)
    }

    private fun runSingleWindow(model: OrtModel, data: Array<FloatArray>, start: Int): FloatArray? {
        val ws = model.windowSize
        return try {
            val floats = FloatArray(ws * CHANNELS)
            for (t in 0 until ws) {
                val src = data[start + t]
                for (c in 0 until CHANNELS) {
                    floats[t * CHANNELS + c] = (src[c] - model.mean[c]) / model.std[c]
                }
            }
            val tensor = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(floats),
                longArrayOf(1L, ws.toLong(), CHANNELS.toLong()),
            )
            val results = model.session.run(mapOf("input" to tensor))
            @Suppress("UNCHECKED_CAST")
            val logits = (results[0].value as Array<FloatArray>)[0]
            tensor.close()
            results.close()

            val maxL = logits.max()
            val expArr = FloatArray(logits.size) { exp((logits[it] - maxL).toDouble()).toFloat() }
            val expSum = expArr.sum().coerceAtLeast(1e-12f)
            FloatArray(logits.size) { expArr[it] / expSum }
        } catch (e: Exception) {
            Log.e(TAG, "Model inference error: ${e.message}")
            null
        }
    }

    private fun alignScaleProbabilities(scales: List<ScaleSeries>): Pair<LongArray, Map<String, Array<FloatArray>>> {
        val ref = scales.maxBy { it.timestampsMs.size }
        val aligned = linkedMapOf<String, Array<FloatArray>>()
        for (scale in scales.sortedBy { scaleOrder(it.suffix) }) {
            aligned[scale.suffix] = if (
                scale.timestampsMs.size == ref.timestampsMs.size &&
                scale.timestampsMs.contentEquals(ref.timestampsMs)
            ) {
                Array(scale.probs.size) { scale.probs[it].copyOf() }
            } else {
                interpolateProbs(scale, ref.timestampsMs)
            }
        }
        return Pair(ref.timestampsMs, aligned)
    }

    private fun scaleOrder(suffix: String): Int = when (suffix) {
        "3s" -> 0
        "5s" -> 1
        "8s" -> 2
        else -> 3
    }

    private fun interpolateProbs(scale: ScaleSeries, refTimes: LongArray): Array<FloatArray> {
        if (scale.timestampsMs.isEmpty()) return emptyArray()
        if (scale.timestampsMs.size == 1) return Array(refTimes.size) { scale.probs[0].copyOf() }

        val out = Array(refTimes.size) { FloatArray(CLASS_NAMES.size) }
        var j = 0
        for (i in refTimes.indices) {
            val t = refTimes[i]
            while (j < scale.timestampsMs.size - 2 && scale.timestampsMs[j + 1] < t) j++
            when {
                t <= scale.timestampsMs.first() -> System.arraycopy(scale.probs.first(), 0, out[i], 0, CLASS_NAMES.size)
                t >= scale.timestampsMs.last() -> System.arraycopy(scale.probs.last(), 0, out[i], 0, CLASS_NAMES.size)
                else -> {
                    val t0 = scale.timestampsMs[j]
                    val t1 = scale.timestampsMs[j + 1]
                    val alpha = (t - t0).toFloat() / (t1 - t0).toFloat().coerceAtLeast(1f)
                    for (c in out[i].indices) {
                        out[i][c] = scale.probs[j][c] * (1f - alpha) + scale.probs[j + 1][c] * alpha
                    }
                }
            }
        }
        return out
    }

    private fun localBoundaryWindowFusion(aligned: Map<String, Array<FloatArray>>): Array<FloatArray> {
        if (aligned.isEmpty()) return emptyArray()
        if (aligned.size == 1 || !aligned.containsKey("3s")) {
            val only = aligned.values.first()
            return Array(only.size) { only[it].copyOf() }
        }

        val suffixes = listOf("3s", "5s", "8s").filter { aligned.containsKey(it) }
        val nSteps = aligned[suffixes.first()]!!.size
        val baseWeights = suffixes.map {
            when (it) {
                "3s" -> BASE_WEIGHT_3S
                "5s" -> BASE_WEIGHT_5S
                "8s" -> BASE_WEIGHT_8S
                else -> 1f
            }
        }.toFloatArray()
        val baseSum = baseWeights.sum().coerceAtLeast(1e-6f)
        for (i in baseWeights.indices) baseWeights[i] /= baseSum

        val probs3s = aligned["3s"]!!
        val pred3s = IntArray(nSteps) { t -> argMax(probs3s[t]) }
        val mask = FloatArray(nSteps)
        for (t in 1 until nSteps) {
            if (pred3s[t] != pred3s[t - 1]) {
                val left = maxOf(0, t - LOCAL_BOUNDARY_RADIUS)
                val right = minOf(nSteps - 1, t + LOCAL_BOUNDARY_RADIUS)
                for (i in left..right) mask[i] = 1f
            }
        }
        val boundaryMask = uniformFilterNearest(mask, LOCAL_BOUNDARY_SMOOTH)

        return Array(nSteps) { t ->
            val weights = baseWeights.copyOf()
            val b = boundaryMask[t].coerceIn(0f, 1f)
            for (i in suffixes.indices) {
                weights[i] += when (suffixes[i]) {
                    "3s" -> BOUNDARY_BOOST_3S * b
                    "5s" -> -BOUNDARY_PENALTY_5S * b
                    "8s" -> -BOUNDARY_PENALTY_8S * b
                    else -> 0f
                }
                weights[i] = weights[i].coerceAtLeast(MIN_SCALE_WEIGHT)
            }
            val norm = weights.sum().coerceAtLeast(1e-6f)
            val fused = FloatArray(CLASS_NAMES.size)
            for (i in suffixes.indices) {
                val p = aligned[suffixes[i]]!![t]
                val w = weights[i] / norm
                for (c in fused.indices) fused[c] += p[c] * w
            }
            fused
        }
    }

    private fun zeroPhaseButterworth(data: Array<FloatArray>): Array<FloatArray> {
        if (data.size <= 100 || data.size <= FILTER_PAD) return Array(data.size) { data[it].copyOf() }
        val out = Array(data.size) { FloatArray(CHANNELS) }
        for (c in 0 until CHANNELS) {
            val x = FloatArray(data.size) { i -> data[i][c] }
            val y = filtfiltChannel(x)
            for (i in y.indices) out[i][c] = y[i]
        }
        return out
    }

    private fun filtfiltChannel(x: FloatArray): FloatArray {
        val pad = minOf(FILTER_PAD, x.size - 1)
        val extended = FloatArray(x.size + 2 * pad)
        for (i in 0 until pad) extended[i] = 2f * x[0] - x[pad - i]
        System.arraycopy(x, 0, extended, pad, x.size)
        for (i in 0 until pad) extended[pad + x.size + i] = 2f * x.last() - x[x.size - 2 - i]

        val forward = lfilter(extended, extended[0])
        forward.reverse()
        val backward = lfilter(forward, forward[0])
        backward.reverse()
        return backward.copyOfRange(pad, pad + x.size)
    }

    private fun lfilter(x: FloatArray, ziScale: Float): FloatArray {
        val z = FloatArray(FILTER_ZI.size) { FILTER_ZI[it] * ziScale }
        val y = FloatArray(x.size)
        for (i in x.indices) {
            val xi = x[i]
            val yi = FILTER_B[0] * xi + z[0]
            for (k in 1 until z.size) {
                z[k - 1] = FILTER_B[k] * xi + z[k] - FILTER_A[k] * yi
            }
            z[z.lastIndex] = FILTER_B.last() * xi - FILTER_A.last() * yi
            y[i] = yi
        }
        return y
    }

    private fun uniformFilterNearest(probs: Array<FloatArray>, size: Int): Array<FloatArray> {
        val half = size / 2
        return Array(probs.size) { t ->
            FloatArray(CLASS_NAMES.size) { c ->
                var sum = 0f
                for (k in -half..half) {
                    val idx = (t + k).coerceIn(0, probs.lastIndex)
                    sum += probs[idx][c]
                }
                sum / size.toFloat()
            }
        }
    }

    private fun uniformFilterNearest(values: FloatArray, size: Int): FloatArray {
        val half = size / 2
        return FloatArray(values.size) { t ->
            var sum = 0f
            for (k in -half..half) {
                val idx = (t + k).coerceIn(0, values.lastIndex)
                sum += values[idx]
            }
            sum / size.toFloat()
        }
    }

    private fun medianFilterZero(probs: Array<FloatArray>, size: Int): Array<FloatArray> {
        val half = size / 2
        return Array(probs.size) { t ->
            FloatArray(CLASS_NAMES.size) { c ->
                val vals = FloatArray(size) { k ->
                    val idx = t + k - half
                    if (idx in probs.indices) probs[idx][c] else 0f
                }
                vals.sort()
                vals[size / 2]
            }
        }
    }

    private fun viterbiDecode(probs: Array<FloatArray>): IntArray {
        val n = probs.size
        if (n == 0) return IntArray(0)
        val trans = Array(CLASS_NAMES.size) { FloatArray(CLASS_NAMES.size) { 0.001f } }
        for (i in trans.indices) trans[i][i] = 0.97f
        for (i in 1 until trans.size) {
            trans[0][i] = 0.01f
            trans[i][0] = 0.05f
        }
        val logTrans = Array(trans.size) { i ->
            val rowSum = trans[i].sum().coerceAtLeast(1e-6f)
            FloatArray(trans.size) { j -> ln((trans[i][j] / rowSum).toDouble() + 1e-10).toFloat() }
        }
        val logInit = ln(1.0 / CLASS_NAMES.size).toFloat()
        val v = Array(n) { FloatArray(CLASS_NAMES.size) }
        val bp = Array(n) { IntArray(CLASS_NAMES.size) }
        for (s in CLASS_NAMES.indices) v[0][s] = logInit + ln(probs[0][s].toDouble() + 1e-10).toFloat()
        for (t in 1 until n) {
            for (s in CLASS_NAMES.indices) {
                var best = Float.NEGATIVE_INFINITY
                var bestPrev = 0
                for (prev in CLASS_NAMES.indices) {
                    val score = v[t - 1][prev] + logTrans[prev][s]
                    if (score > best) {
                        best = score
                        bestPrev = prev
                    }
                }
                v[t][s] = best + ln(probs[t][s].toDouble() + 1e-10).toFloat()
                bp[t][s] = bestPrev
            }
        }
        val path = IntArray(n)
        path[n - 1] = argMax(v[n - 1])
        for (t in n - 2 downTo 0) path[t] = bp[t + 1][path[t + 1]]
        return path
    }

    private fun postProcessSegments(
        path: IntArray,
        timestampsMs: LongArray,
        probs: Array<FloatArray>,
        filteredData: Array<FloatArray>,
        sampleTimestampsMs: LongArray,
        includeAbsoluteTimestamps: Boolean,
    ): List<ActivitySegment> {
        var segments = extractSegments(path, timestampsMs, probs)
        segments = mergeSameClassSegments(segments)
        segments = refineBoundaries(segments, filteredData, sampleTimestampsMs)
        segments = resolveOverlaps(segments)
        segments = segments.filter { it.durationSec >= MIN_SEGMENT_SEC }
        if (segments.size > TOP_K) segments = selectTopK(segments, TOP_K)
        segments = segments.filter { it.confidence >= CONF_MIN }
        val sessionStartMs = sampleTimestampsMs.firstOrNull() ?: 0L
        return segments.sortedBy { it.startMs }.map {
            ActivitySegment(
                classIdx = it.classIdx,
                className = CLASS_NAMES.getOrElse(it.classIdx) { "未知" },
                startOffsetSec = ((it.startMs - sessionStartMs).coerceAtLeast(0L) / 1000L).toInt(),
                durationSec = it.durationSec.roundToInt(),
                isOngoing = false,
                confidence = it.confidence,
                absoluteStartMs = if (includeAbsoluteTimestamps) it.startMs else null,
                absoluteEndMs = if (includeAbsoluteTimestamps) it.endMs else null,
            )
        }
    }

    private fun extractSegments(
        path: IntArray,
        timestampsMs: LongArray,
        probs: Array<FloatArray>,
    ): List<SegmentCandidate> {
        if (path.isEmpty()) return emptyList()
        val segments = mutableListOf<SegmentCandidate>()
        var current = path[0]
        var startIdx = 0
        for (i in 1..path.size) {
            if (i == path.size || path[i] != current) {
                if (current > 0) {
                    val endIdx = i - 1
                    val startMs = timestampsMs[startIdx] - REF_WINDOW_SEC * 500L
                    val endMs = timestampsMs[endIdx] + REF_WINDOW_SEC * 500L
                    var conf = 0f
                    for (t in startIdx until i) conf += probs[t][current]
                    conf /= (i - startIdx).toFloat().coerceAtLeast(1f)
                    segments += SegmentCandidate(
                        classIdx = current,
                        startMs = startMs.coerceAtLeast(0L),
                        endMs = endMs.coerceAtLeast(0L),
                        confidence = conf,
                        startWindowIdx = startIdx,
                        endWindowIdx = endIdx,
                    )
                }
                if (i < path.size) {
                    current = path[i]
                    startIdx = i
                }
            }
        }
        return segments
    }

    private fun mergeSameClassSegments(input: List<SegmentCandidate>): List<SegmentCandidate> {
        if (input.size <= 1) return input
        val merged = mutableListOf<SegmentCandidate>()
        for (seg in input.sortedBy { it.startMs }) {
            val prev = merged.lastOrNull()
            val gapSec = if (prev == null) Float.POSITIVE_INFINITY else (seg.startMs - prev.endMs) / 1000f
            if (prev != null && seg.classIdx == prev.classIdx && gapSec < SHORT_GAP_SEC) {
                prev.endMs = seg.endMs
                prev.endWindowIdx = seg.endWindowIdx
                prev.confidence = (prev.confidence + seg.confidence) / 2f
            } else {
                merged += seg.copy()
            }
        }
        return merged
    }

    private fun refineBoundaries(
        input: List<SegmentCandidate>,
        filteredData: Array<FloatArray>,
        sampleTimestampsMs: LongArray,
    ): List<SegmentCandidate> {
        if (input.isEmpty() || filteredData.isEmpty() || sampleTimestampsMs.isEmpty()) return input
        val energy = FloatArray(filteredData.size) { i ->
            val acc = filteredData[i]
            sqrt((acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).toDouble()).toFloat()
        }
        val smoothEnergy = uniformFilterNearest(energy, ENERGY_SMOOTH_SIZE)
        val gradient = FloatArray(smoothEnergy.size) { i ->
            val prev = smoothEnergy[(i - 1).coerceAtLeast(0)]
            val next = smoothEnergy[(i + 1).coerceAtMost(smoothEnergy.lastIndex)]
            (next - prev) / 2f
        }
        val maxTimeMs = sampleTimestampsMs.last()

        return input.map { original ->
            val seg = original.copy()
            val oldStart = seg.startMs
            val oldEnd = seg.endMs
            findIndicesAround(seg.startMs, sampleTimestampsMs, maxTimeMs)
                .takeIf { !it.isEmpty() && (it.last - it.first + 1) > 100 }
                ?.let { range ->
                var bestIdx = range.first
                var bestScore = -1f
                for (idx in range) {
                    val score = abs(gradient[idx])
                    if (score > bestScore) {
                        bestScore = score
                        bestIdx = idx
                    }
                }
                val newStart = sampleTimestampsMs[bestIdx]
                if (abs(newStart - seg.startMs) < BOUNDARY_SEARCH_MS) seg.startMs = newStart
            }
            findIndicesAround(seg.endMs, sampleTimestampsMs, maxTimeMs)
                .takeIf { !it.isEmpty() && (it.last - it.first + 1) > 100 }
                ?.let { range ->
                var bestIdx = range.first
                var bestScore = Float.POSITIVE_INFINITY
                for (idx in range) {
                    val score = gradient[idx]
                    if (score < bestScore) {
                        bestScore = score
                        bestIdx = idx
                    }
                }
                val newEnd = sampleTimestampsMs[bestIdx]
                if (abs(newEnd - seg.endMs) < BOUNDARY_SEARCH_MS) seg.endMs = newEnd
            }
            if (seg.startMs >= seg.endMs) {
                seg.startMs = oldStart
                seg.endMs = oldEnd
            }
            seg
        }
    }

    private fun findIndicesAround(centerMs: Long, sampleTimestampsMs: LongArray, maxTimeMs: Long): IntRange {
        val startMs = (centerMs - BOUNDARY_SEARCH_MS).coerceAtLeast(0L)
        val endMs = (centerMs + BOUNDARY_SEARCH_MS).coerceAtMost(maxTimeMs)
        var start = 0
        while (start < sampleTimestampsMs.size && sampleTimestampsMs[start] < startMs) start++
        if (start >= sampleTimestampsMs.size) return IntRange.EMPTY

        var end = start
        while (end + 1 < sampleTimestampsMs.size && sampleTimestampsMs[end + 1] <= endMs) end++
        return start..end
    }

    private fun resolveOverlaps(input: List<SegmentCandidate>): List<SegmentCandidate> {
        if (input.size <= 1) return input
        val resolved = mutableListOf<SegmentCandidate>()
        for (segOriginal in input.sortedBy { it.startMs }) {
            val seg = segOriginal.copy()
            val prev = resolved.lastOrNull()
            if (prev != null && seg.startMs < prev.endMs) {
                if (seg.classIdx == prev.classIdx) {
                    prev.endMs = maxOf(prev.endMs, seg.endMs)
                    prev.confidence = maxOf(prev.confidence, seg.confidence)
                } else if (seg.confidence > prev.confidence) {
                    val mid = (seg.startMs + prev.endMs) / 2L
                    prev.endMs = mid
                    seg.startMs = mid
                    if (seg.endMs > seg.startMs) resolved += seg
                } else {
                    seg.startMs = prev.endMs
                    if (seg.endMs > seg.startMs) resolved += seg
                }
            } else {
                resolved += seg
            }
        }
        return resolved
    }

    private fun selectTopK(input: List<SegmentCandidate>, k: Int): List<SegmentCandidate> {
        if (input.size <= k) return input
        val byClass = linkedMapOf<Int, SegmentCandidate>()
        for (seg in input) {
            val existing = byClass[seg.classIdx]
            if (existing == null || seg.confidence > existing.confidence) byClass[seg.classIdx] = seg
        }
        val selected = byClass.values.toMutableList()
        if (selected.size >= k) {
            return selected.sortedByDescending { it.confidence }.take(k).sortedBy { it.startMs }
        }
        val remaining = input.filter { candidate -> selected.none { it === candidate } }
            .sortedByDescending { it.confidence }
        for (seg in remaining) {
            if (selected.size >= k) break
            selected += seg
        }
        return selected.sortedBy { it.startMs }
    }

    private fun pathStats(path: IntArray): Pair<Int, Map<Int, Int>> {
        val byClass = mutableMapOf<Int, Int>()
        for (c in path) byClass[c] = (byClass[c] ?: 0) + 1
        return Pair(path.size, byClass)
    }

    private fun argMax(values: FloatArray): Int {
        var bestIdx = 0
        var best = Float.NEGATIVE_INFINITY
        for (i in values.indices) {
            if (values[i] > best) {
                best = values[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    fun close() {
        inferScheduler.shutdown()
        inferPool.shutdown()
        models3s.forEach { runCatching { it.session.close() } }
        models5s.forEach { runCatching { it.session.close() } }
        models8s.forEach { runCatching { it.session.close() } }
    }
}
