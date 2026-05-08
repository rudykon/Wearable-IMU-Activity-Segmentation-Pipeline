package com.imu.realtime

/**
 * One recognized activity segment produced by [ActivityPostProcessor].
 *
 * @param classIdx        Model class index (0–5).
 * @param className       Human-readable name from [MotionClassifier.CLASS_NAMES].
 * @param startOffsetSec  Seconds from session start when this segment began.
 * @param durationSec     Length of this segment in seconds.
 * @param isOngoing       True for the last (still-growing) segment.
 * @param confidence      Mean smoothed class probability over the segment.
 * @param absoluteStartMs Original sensor timestamp for offline paper-style output.
 * @param absoluteEndMs   Original sensor timestamp for offline paper-style output.
 */
data class ActivitySegment(
    val classIdx: Int,
    val className: String,
    val startOffsetSec: Int,
    val durationSec: Int,
    val isOngoing: Boolean = false,
    val confidence: Float = 0f,
    val absoluteStartMs: Long? = null,
    val absoluteEndMs: Long? = null,
)
