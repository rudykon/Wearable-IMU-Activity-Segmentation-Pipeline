package com.imu.realtime

data class ImuData(
    val timestamp: Long = System.currentTimeMillis(),
    val accX: Float = 0f, val accY: Float = 0f, val accZ: Float = 0f,
    val gyroX: Float = 0f, val gyroY: Float = 0f, val gyroZ: Float = 0f,
    val angleX: Float = 0f,   // Pitch °
    val angleY: Float = 0f,   // Roll  °
    val angleZ: Float = 0f,   // Yaw   °
    val magX: Float = Float.NaN,
    val magY: Float = Float.NaN,
    val magZ: Float = Float.NaN
)
