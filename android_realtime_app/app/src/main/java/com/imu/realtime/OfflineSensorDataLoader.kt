package com.imu.realtime

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

data class OfflineSensorSeries(
    val sourceName: String,
    val timestampsMs: LongArray,
    val samples: Array<FloatArray>,
) {
    val sampleCount: Int get() = samples.size
    val durationSec: Int
        get() {
            if (timestampsMs.size < 2) return 0
            return ((timestampsMs.last() - timestampsMs.first()).coerceAtLeast(0L) / 1000L).toInt()
        }
}

object OfflineSensorDataLoader {
    const val BUILTIN_SAMPLE_ASSET = "offline_samples/Skipping_derived_sample.txt"
    const val BUILTIN_SAMPLE_NAME = "Skipping_derived_sample.txt"

    fun hasBuiltInSample(context: Context): Boolean {
        return try {
            context.assets.open(BUILTIN_SAMPLE_ASSET).close()
            true
        } catch (_: Exception) {
            false
        }
    }

    fun loadBuiltInSample(context: Context): OfflineSensorSeries {
        return context.assets.open(BUILTIN_SAMPLE_ASSET).use { input ->
            load(input, BUILTIN_SAMPLE_NAME)
        }
    }

    fun loadFromUri(context: Context, uri: Uri): OfflineSensorSeries {
        val name = displayName(context, uri)
        val input = context.contentResolver.openInputStream(uri)
            ?: throw IllegalArgumentException("Cannot open selected file")
        return input.use { load(it, name) }
    }

    fun load(input: InputStream, sourceName: String): OfflineSensorSeries {
        val reader = BufferedReader(InputStreamReader(input))
        val header = reader.readLine()?.trim()?.removePrefix("\uFEFF")
            ?: throw IllegalArgumentException("Empty sensor file")
        if (!header.contains("ACC_TIME")) {
            throw IllegalArgumentException("Unsupported file: missing ACC_TIME header")
        }

        val columns = header.split('\t').map { it.trim().removePrefix("\uFEFF") }
        val columnIndex = columns.withIndex().associate { it.value to it.index }
        fun column(name: String): Int = columnIndex[name]
            ?: throw IllegalArgumentException("Unsupported file: missing $name column")

        val accTimeIdx = column("ACC_TIME")
        val accXIdx = column("ACC_X")
        val accYIdx = column("ACC_Y")
        val accZIdx = column("ACC_Z")
        val gyroXIdx = column("GYRO_X")
        val gyroYIdx = column("GYRO_Y")
        val gyroZIdx = columnIndex["GYRO_Z"] ?: columnIndex["GYRO_"]
            ?: throw IllegalArgumentException("Unsupported file: missing GYRO_Z column")
        val maxIdx = listOf(accTimeIdx, accXIdx, accYIdx, accZIdx, gyroXIdx, gyroYIdx, gyroZIdx).maxOrNull() ?: 0

        val rows = ArrayList<Pair<Long, FloatArray>>(64_000)
        reader.lineSequence().forEach { line ->
            if (line.isBlank() || line.contains("ACC_TIME")) return@forEach
            val parts = line.split('\t')
            if (parts.size <= maxIdx) return@forEach

            val timestamp = parts[accTimeIdx].trim().toDoubleOrNull()?.toLong() ?: return@forEach
            if (timestamp <= 0L) return@forEach

            val sample = floatArrayOf(
                parts[accXIdx].trim().toFloatOrNull() ?: return@forEach,
                parts[accYIdx].trim().toFloatOrNull() ?: return@forEach,
                parts[accZIdx].trim().toFloatOrNull() ?: return@forEach,
                parts[gyroXIdx].trim().toFloatOrNull() ?: return@forEach,
                parts[gyroYIdx].trim().toFloatOrNull() ?: return@forEach,
                parts[gyroZIdx].trim().toFloatOrNull() ?: return@forEach,
            )
            rows += timestamp to sample
        }

        if (rows.isEmpty()) {
            throw IllegalArgumentException("No valid ACC/GYRO rows found")
        }

        val sorted = rows.sortedBy { it.first }
        return OfflineSensorSeries(
            sourceName = sourceName,
            timestampsMs = LongArray(sorted.size) { sorted[it].first },
            samples = Array(sorted.size) { sorted[it].second },
        )
    }

    private fun displayName(context: Context, uri: Uri): String {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0 && cursor.moveToFirst()) {
                    return cursor.getString(idx)
                }
            }
        return uri.lastPathSegment ?: "selected_sensor_data.txt"
    }
}
