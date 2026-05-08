package com.imu.realtime

import android.os.Environment
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * 单线程 Executor 写 CSV，写入 Downloads 目录。
 * API 29+ 无需权限，API < 29 需要 WRITE_EXTERNAL_STORAGE。
 */
class DataRecorder {

    private val executor = Executors.newSingleThreadExecutor()
    private var writer: BufferedWriter? = null
    var isRecording = false
        private set

    /** 返回新建文件的绝对路径，供 UI 提示 */
    fun start(): String {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
        val name = "imu_${sdf.format(Date())}.csv"
        val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        dir.mkdirs()
        val file = File(dir, name)
        executor.execute {
            writer = BufferedWriter(FileWriter(file, false))
            writer?.write("timestamp_ms,accX,accY,accZ,gyroX,gyroY,gyroZ,angleX,angleY,angleZ,magX,magY,magZ\n")
            writer?.flush()
        }
        isRecording = true
        return file.absolutePath
    }

    fun stop() {
        isRecording = false
        executor.execute {
            writer?.close()
            writer = null
        }
    }

    fun record(d: ImuData) {
        if (!isRecording) return
        executor.execute {
            writer?.write(
                "${d.timestamp},${d.accX},${d.accY},${d.accZ}," +
                "${d.gyroX},${d.gyroY},${d.gyroZ}," +
                "${d.angleX},${d.angleY},${d.angleZ}," +
                "${d.magX},${d.magY},${d.magZ}\n"
            )
        }
    }

    fun shutdown() {
        stop()
        executor.shutdown()
    }
}
