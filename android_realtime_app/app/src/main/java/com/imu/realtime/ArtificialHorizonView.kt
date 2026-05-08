package com.imu.realtime

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * 人工地平线（Artificial Horizon / Attitude Indicator）
 * - 蓝色天空 / 棕色地面，随横滚旋转
 * - 俯仰平移，梯度线刻度
 * - 固定飞机符号（黄色）
 */
class ArtificialHorizonView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    var pitch = 0f  // angleX °（正值抬头）
    var roll  = 0f  // angleY °（正值右倾）

    private val paintSky  = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#1A6DC3") }
    private val paintGnd  = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#8B5E3C") }
    private val paintLine = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; strokeWidth = 2f; style = Paint.Style.STROKE
    }
    private val paintText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 28f; textAlign = Paint.Align.LEFT
    }
    private val paintPlane = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFD700"); strokeWidth = 6f; style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val paintRollArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; strokeWidth = 2f; style = Paint.Style.STROKE
    }
    private val clipPath = Path()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        val cx = w / 2; val cy = h / 2
        val r = min(w, h) / 2f * 0.95f

        // 圆形裁剪
        clipPath.reset()
        clipPath.addCircle(cx, cy, r, Path.Direction.CW)
        canvas.clipPath(clipPath)

        // 旋转画布实现横滚 + 俯仰平移
        val pitchPx = pitch * (h / 90f)

        canvas.save()
        canvas.rotate(-roll, cx, cy)
        canvas.translate(0f, -pitchPx)

        // 地面（下半部）
        canvas.drawRect(0f, cy, w, cy + h + r, paintGnd)
        // 天空（上半部）
        canvas.drawRect(0f, cy - h - r, w, cy, paintSky)
        // 水平线
        canvas.drawLine(0f, cy, w, cy, paintLine)

        // 俯仰刻度（每5°画一条）
        for (deg in -30..30 step 5) {
            if (deg == 0) continue
            val dy = -deg * (h / 90f)
            val lineHalf = if (deg % 10 == 0) w * 0.25f else w * 0.15f
            canvas.drawLine(cx - lineHalf, cy + dy, cx + lineHalf, cy + dy, paintLine)
            if (deg % 10 == 0) {
                canvas.drawText("${deg}°", cx + lineHalf + 4f, cy + dy + 8f, paintText)
            }
        }

        canvas.restore()

        // 横滚弧（不随俯仰移动，但随横滚旋转）
        val arcRect = RectF(cx - r * 0.85f, cy - r * 0.85f, cx + r * 0.85f, cy + r * 0.85f)
        canvas.save()
        canvas.rotate(-roll, cx, cy)
        canvas.drawArc(arcRect, 200f, 140f, false, paintRollArc)
        // 横滚指针小三角
        val path = Path()
        path.moveTo(cx, cy - r * 0.85f)
        path.lineTo(cx - 8f, cy - r * 0.85f + 16f)
        path.lineTo(cx + 8f, cy - r * 0.85f + 16f)
        path.close()
        val paintTri = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; style = Paint.Style.FILL }
        canvas.drawPath(path, paintTri)
        canvas.restore()

        // 固定飞机符号（始终居中）
        canvas.drawLine(cx - r * 0.4f, cy, cx - r * 0.1f, cy, paintPlane)
        canvas.drawLine(cx + r * 0.1f, cy, cx + r * 0.4f, cy, paintPlane)
        canvas.drawCircle(cx, cy, 5f, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#FFD700"); style = Paint.Style.FILL
        })
    }

    fun update(pitch: Float, roll: Float) {
        this.pitch = pitch; this.roll = roll; invalidate()
    }
}
