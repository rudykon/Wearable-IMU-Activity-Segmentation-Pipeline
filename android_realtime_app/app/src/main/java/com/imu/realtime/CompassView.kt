package com.imu.realtime

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * 罗盘视图 — 旋转盘面，红三角（北指针）固定朝上
 */
class CompassView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    var heading = 0f  // Yaw °，0=北，顺时针增大

    private val paintDial = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1E293B"); style = Paint.Style.FILL
    }
    private val paintRim = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#334155"); strokeWidth = 3f; style = Paint.Style.STROKE
    }
    private val paintTick = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#94A3B8"); strokeWidth = 2f; style = Paint.Style.STROKE
    }
    private val paintLabel = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 32f; textAlign = Paint.Align.CENTER
    }
    private val paintNeedle = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#EF4444"); style = Paint.Style.FILL
    }
    private val paintSmall = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#64748B"); textSize = 22f; textAlign = Paint.Align.CENTER
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        val cx = w / 2; val cy = h / 2
        val r = min(w, h) / 2f * 0.9f

        // 盘面背景
        canvas.drawCircle(cx, cy, r, paintDial)
        canvas.drawCircle(cx, cy, r, paintRim)

        // 旋转盘面
        canvas.save()
        canvas.rotate(-heading, cx, cy)

        // 刻度：每10°一短线，每30°一长线
        for (deg in 0 until 360 step 10) {
            val rad = Math.toRadians(deg.toDouble()).toFloat()
            val sinA = sin(rad); val cosA = cos(rad)
            val inner = if (deg % 30 == 0) r * 0.75f else r * 0.85f
            canvas.drawLine(
                cx + sinA * inner, cy - cosA * inner,
                cx + sinA * r,     cy - cosA * r, paintTick
            )
        }

        // 方位标
        val dirs = mapOf(0 to "N", 90 to "E", 180 to "S", 270 to "W")
        for ((deg, label) in dirs) {
            val rad = Math.toRadians(deg.toDouble()).toFloat()
            val tx = cx + sin(rad) * r * 0.62f
            val ty = cy - cos(rad) * r * 0.62f + paintLabel.textSize * 0.35f
            val p = if (deg == 0) paintNeedle.apply { style = Paint.Style.FILL }
                    .let { Paint(paintLabel).also { it.color = Color.parseColor("#EF4444") } }
                    else paintLabel
            canvas.drawText(label, tx, ty, p)
        }

        // 其他方位数字
        for (deg in 30 until 360 step 30) {
            if (deg % 90 == 0) continue
            val rad = Math.toRadians(deg.toDouble()).toFloat()
            val tx = cx + sin(rad) * r * 0.62f
            val ty = cy - cos(rad) * r * 0.62f + paintSmall.textSize * 0.35f
            canvas.drawText("$deg", tx, ty, paintSmall)
        }

        canvas.restore()

        // 固定北指针（红三角，始终朝上）
        val needlePath = Path()
        needlePath.moveTo(cx, cy - r * 0.5f)
        needlePath.lineTo(cx - 8f, cy)
        needlePath.lineTo(cx + 8f, cy)
        needlePath.close()
        canvas.drawPath(needlePath, paintNeedle)

        // 中心圆
        canvas.drawCircle(cx, cy, 10f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#64748B"); style = Paint.Style.FILL
        })

        // 当前航向文字
        val hdgText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE; textSize = 36f; textAlign = Paint.Align.CENTER
        }
        canvas.drawText(String.format("%.0f°", heading), cx, cy + r * 0.85f, hdgText)
    }

    fun update(heading: Float) {
        this.heading = heading; invalidate()
    }
}
