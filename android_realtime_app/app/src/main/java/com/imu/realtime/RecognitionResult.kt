package com.imu.realtime

data class RecognitionResult(
    val classIdx: Int,
    val className: String,
    val confidence: Float,
    val probs: FloatArray,
    val smoothedProbs: FloatArray,
    val warmupFraction: Float,
    val decodedSeconds: Int = 0,
    val segments: List<ActivitySegment> = emptyList(),
    val stats: Pair<Int, Map<Int, Int>> = Pair(0, emptyMap()),
)
