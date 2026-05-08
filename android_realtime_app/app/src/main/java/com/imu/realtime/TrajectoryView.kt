package com.imu.realtime

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.*

/**
 * 加速度积分轨迹（白色主题）
 * ZUPT 补偿，XY 平面俯视
 */
class TrajectoryView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val maxPoints = 500
    private val posX = FloatArray(maxPoints)
    private val posY = FloatArray(maxPoints)
    private var count = 0
    private var head  = 0

    private var velX = 0f; private var velY = 0f
    private var curX = 0f; private var curY = 0f

    private var filtAccX = 0f; private var filtAccY = 0f
    private val alpha = 0.1f

    private var lastTs = 0L
    private var zuptCount = 0
    private val ZUPT_THRESH = 0.02f
    private val ZUPT_FRAMES = 5

    private val paintBg = Paint().apply { color = Color.WHITE }
    private val paintGrid = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F1F5F9"); strokeWidth = 1f; style = Paint.Style.STROKE
    }
    private val paintAxis = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CBD5E1"); strokeWidth = 1.5f; style = Paint.Style.STROKE
    }
    private val paintTrack = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#3B82F6"); strokeWidth = 3f; style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
    }
    private val paintDot = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#10B981"); style = Paint.Style.FILL
    }
    private val paintOrigin = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#EF4444"); style = Paint.Style.FILL
    }
    private val paintText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#94A3B8"); textSize = 30f; textAlign = Paint.Align.CENTER
    }
    private val paintScale = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CBD5E1"); textSize = 24f; textAlign = Paint.Align.LEFT
    }

    fun addData(accX: Float, accY: Float, @Suppress("UNUSED_PARAMETER") accZ: Float, angleX: Float, angleY: Float) {
        val now = System.currentTimeMillis()
        val dt = if (lastTs == 0L) 0.01f else ((now - lastTs) / 1000f).coerceIn(0.005f, 0.1f)
        lastTs = now

        val gx = 9.8f * sin(Math.toRadians(angleY.toDouble())).toFloat()
        val gy = -9.8f * sin(Math.toRadians(angleX.toDouble())).toFloat()
        val linAccX = accX * 9.8f - gx
        val linAccY = accY * 9.8f - gy

        filtAccX = alpha * linAccX + (1 - alpha) * filtAccX
        filtAccY = alpha * linAccY + (1 - alpha) * filtAccY

        val mag = sqrt(filtAccX * filtAccX + filtAccY * filtAccY)
        if (mag < ZUPT_THRESH) {
            zuptCount++
            if (zuptCount >= ZUPT_FRAMES) { velX = 0f; velY = 0f }
        } else {
            zuptCount = 0
        }

        velX += filtAccX * dt; velY += filtAccY * dt
        curX += velX * dt;     curY += velY * dt

        posX[head] = curX; posY[head] = curY
        head = (head + 1) % maxPoints
        if (count < maxPoints) count++

        invalidate()
    }

    fun reset() {
        count = 0; head = 0
        velX = 0f; velY = 0f; curX = 0f; curY = 0f
        filtAccX = 0f; filtAccY = 0f; lastTs = 0L; zuptCount = 0
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        val cx = w / 2; val cy = h / 2

        canvas.drawRect(0f, 0f, w, h, paintBg)

        // 网格线
        val step = w / 6
        var x = cx % step; while (x < w) { canvas.drawLine(x, 0f, x, h, paintGrid); x += step }
        var y = cy % step; while (y < h) { canvas.drawLine(0f, y, w, y, paintGrid); y += step }

        // 坐标轴
        canvas.drawLine(0f, cy, w, cy, paintAxis)
        canvas.drawLine(cx, 0f, cx, h, paintAxis)

        if (count < 2) {
            canvas.drawText(context.getString(R.string.trajectory_waiting_data), cx, cy - 20f, paintText)
            return
        }

        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (i in 0 until count) {
            val idx = (head - count + i + maxPoints) % maxPoints
            minX = min(minX, posX[idx]); maxX = max(maxX, posX[idx])
            minY = min(minY, posY[idx]); maxY = max(maxY, posY[idx])
        }
        val rangeX = max(maxX - minX, 0.1f); val rangeY = max(maxY - minY, 0.1f)
        val scale = min((w - 80f) / rangeX, (h - 80f) / rangeY) * 0.75f
        val offX = cx - (minX + maxX) / 2 * scale
        val offY = cy + (minY + maxY) / 2 * scale

        val path = Path()
        var first = true
        for (i in 0 until count) {
            val idx = (head - count + i + maxPoints) % maxPoints
            val sx = posX[idx] * scale + offX
            val sy = offY - posY[idx] * scale
            if (first) { path.moveTo(sx, sy); first = false } else path.lineTo(sx, sy)
        }
        canvas.drawPath(path, paintTrack)

        val startIdx = (head - count + maxPoints) % maxPoints
        canvas.drawCircle(posX[startIdx] * scale + offX, offY - posY[startIdx] * scale, 9f, paintOrigin)

        val endIdx = (head - 1 + maxPoints) % maxPoints
        canvas.drawCircle(posX[endIdx] * scale + offX, offY - posY[endIdx] * scale, 11f, paintDot)

        // 当前坐标文字
        canvas.drawText("(%.2f, %.2f) m".format(curX, curY), cx, h - 12f, paintScale)
    }
}
