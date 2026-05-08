package com.imu.realtime

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * 270° 弧形仪表盘（白色主题）
 */
class GaugeView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    var minVal   = -20f
    var maxVal   =  20f
    var value    =   0f
    var label    = ""
    var unit     = ""
    var arcColor = Color.parseColor("#3B82F6")

    private val paintBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; style = Paint.Style.FILL
    }
    private val paintTrack = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E2E8F0"); style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val paintArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val paintValue = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#0F172A"); textAlign = Paint.Align.CENTER; typeface = Typeface.MONOSPACE
    }
    private val paintLabel = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#64748B"); textAlign = Paint.Align.CENTER
    }
    private val paintNeedle = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val paintRim = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E2E8F0"); style = Paint.Style.STROKE; strokeWidth = 1.5f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        val cx = w / 2; val cy = h / 2
        val r  = min(w, h) / 2f * 0.82f
        val sw = r * 0.13f

        // 白色圆形背景 + 细边框
        canvas.drawCircle(cx, cy, min(w, h) / 2f * 0.96f, paintBg)
        canvas.drawCircle(cx, cy, min(w, h) / 2f * 0.96f, paintRim)

        // 轨道
        paintTrack.strokeWidth = sw
        val oval = RectF(cx - r, cy - r, cx + r, cy + r)
        canvas.drawArc(oval, 135f, 270f, false, paintTrack)

        // 值弧
        val fraction = ((value - minVal) / (maxVal - minVal)).coerceIn(0f, 1f)
        val sweep = fraction * 270f
        paintArc.color = arcColor
        paintArc.strokeWidth = sw
        canvas.drawArc(oval, 135f, sweep, false, paintArc)

        // 针尖圆点
        val needleAngle = Math.toRadians((135.0 + fraction * 270.0))
        val nr = r * 0.68f
        val nx = (cx + cos(needleAngle) * nr).toFloat()
        val ny = (cy + sin(needleAngle) * nr).toFloat()
        paintNeedle.color = arcColor
        canvas.drawCircle(nx, ny, sw * 0.42f, paintNeedle)

        // 数值
        paintValue.textSize = r * 0.40f
        canvas.drawText(String.format("%.1f", value), cx, cy + r * 0.18f, paintValue)

        // 单位
        paintLabel.textSize = r * 0.22f
        paintLabel.color = Color.parseColor("#94A3B8")
        canvas.drawText(unit, cx, cy + r * 0.46f, paintLabel)

        // 标签（彩色）
        paintLabel.textSize = r * 0.27f
        paintLabel.color = arcColor
        canvas.drawText(label, cx, cy + r * 0.78f, paintLabel)

        // 最小/最大刻度文字
        val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#CBD5E1"); textSize = r * 0.18f; textAlign = Paint.Align.CENTER
        }
        // 起始刻度（左下 135°）
        val startA = Math.toRadians(135.0)
        canvas.drawText(String.format("%.0f", minVal),
            (cx + cos(startA) * r * 1.15f).toFloat(),
            (cy + sin(startA) * r * 1.15f + tickPaint.textSize / 3).toFloat(), tickPaint)
        // 结束刻度（右下 405° = 45°）
        val endA = Math.toRadians(45.0)
        canvas.drawText(String.format("%.0f", maxVal),
            (cx + cos(endA) * r * 1.15f).toFloat(),
            (cy + sin(endA) * r * 1.15f + tickPaint.textSize / 3).toFloat(), tickPaint)
    }

    fun update(v: Float) {
        value = v; invalidate()
    }
}
