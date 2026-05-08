package com.imu.realtime

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.*

/**
 * 拟人化左手 + 手臂 3D 视图
 *
 * 传感器数据使用策略：
 *   · 陀螺仪 (gyroX/Y/Z °/s)：帧间积分，提供快速、低延迟的响应
 *   · 欧拉角 (angleX/Y/Z °)：传感器片内 acc+gyro 融合输出，用于纠正陀螺漂移
 *   · 加速度 (accX/Y/Z g)：检测动态加速度（偏离 1g），控制动作轨迹密度
 *     和视觉强度指示
 *
 * 互补滤波：filtered = α*(filtered + gyro*dt) + (1-α)*sensorAngle
 *   α=0.85：陀螺仪占 85%（快速响应），欧拉角纠偏 15%（防漂移）
 *
 * 默认下垂姿态：WT9011 左腕佩戴，手臂自然下垂时传感器 Pitch ≈ -90°
 *   calibPitch = -90，displayPitch = filtered - calibPitch，
 *   即手臂下垂时 displayPitch ≈ 0 → 模型呈下垂姿态
 *
 * 动作轨迹：保存最近 6 帧姿态，按透明度递减绘制"鬼影"
 *   高动态时（|accMag-1g| > 0.15）每帧记录；静止时每 5 帧记录一次
 */
class HandView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    // ── 互补滤波器状态 ──────────────────────────
    private var filtPitch = 0f   // 显示空间姿态角（校准后）
    private var filtRoll  = 0f
    private var filtYaw   = 0f
    private var lastTs    = 0L

    // ── 校准零点（传感器在手臂自然下垂时的读数） ──
    var calibPitch = -90f
    var calibRoll  =   0f
    var calibYaw   =   0f

    // ── 动作轨迹 ────────────────────────────────
    private val trail = ArrayDeque<FloatArray>()   // [pitch, roll, yaw]，最多 6 条
    private val MAX_TRAIL = 6
    private val trailAlpha = floatArrayOf(0.28f, 0.20f, 0.14f, 0.09f, 0.05f, 0.03f)
    private var trailSkip  = 0   // 帧计数器

    // ── 加速度动态强度（0=静止，1=剧烈运动） ─────
    private var motionIntensity = 0f

    // ── 皮肤配色 ────────────────────────────────
    private val skinBase   = Color.parseColor("#FADADB")
    private val skinMid    = Color.parseColor("#F2B9A0")
    private val skinDark   = Color.parseColor("#D08060")
    private val armColor   = Color.parseColor("#F5C4A8")
    private val outlineClr = Color.parseColor("#C07850")
    private val skinNail   = Color.parseColor("#FFE8E0")

    // ── 画笔 ────────────────────────────────────
    private val paintBg = Paint().apply { color = Color.parseColor("#F8FAFC") }

    private val paintBone = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
    }
    private val paintOutline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
        color = outlineClr
    }
    private val paintFill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val paintJoint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL; color = skinDark
    }
    private val paintNail = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL; color = skinNail
    }
    private val paintNailRim = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 1f; color = Color.parseColor("#C09080")
    }
    // 动态强度环
    private val paintRing = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 5f
    }

    // ── 透视参数 ────────────────────────────────
    private val focalLen  = 380f
    private val pixPerCm  = 24f

    // ── 互补滤波 α ─────────────────────────────
    private val ALPHA = 0.85f

    // ─────────────────────────────────────────────
    // 接收 ImuData（核心入口）
    // ─────────────────────────────────────────────
    fun update(data: ImuData) {
        val now = data.timestamp
        val dt  = if (lastTs > 0) ((now - lastTs) / 1000f).coerceIn(0.001f, 0.15f) else 0.01f
        lastTs  = now

        // 1. 目标角度（减去校准偏移）
        val tgtPitch = data.angleX - calibPitch
        val tgtRoll  = data.angleY - calibRoll
        val tgtYaw   = data.angleZ - calibYaw

        // 2. 互补滤波：陀螺仪积分 + 欧拉角纠偏
        filtPitch = ALPHA * (filtPitch + data.gyroX * dt) + (1f - ALPHA) * tgtPitch
        filtRoll  = ALPHA * (filtRoll  + data.gyroY * dt) + (1f - ALPHA) * tgtRoll
        filtYaw   = ALPHA * (filtYaw   + data.gyroZ * dt) + (1f - ALPHA) * tgtYaw

        // 3. 加速度动态强度（偏离 1g 的幅度）
        val accMag = sqrt(data.accX * data.accX + data.accY * data.accY + data.accZ * data.accZ)
        val dynAcc = (abs(accMag - 1f) / 1.5f).coerceIn(0f, 1f)
        motionIntensity = 0.75f * motionIntensity + 0.25f * dynAcc

        // 4. 动作轨迹采样：运动时高频，静止时低频
        trailSkip++
        val sampleInterval = if (motionIntensity > 0.12f) 1 else 5
        if (trailSkip >= sampleInterval) {
            trailSkip = 0
            trail.addFirst(floatArrayOf(filtPitch, filtRoll, filtYaw))
            while (trail.size > MAX_TRAIL) trail.removeLast()
        }

        invalidate()
    }

    /** 记录当前传感器读数为"零点"（手臂自然下垂姿态） */
    fun calibrate(data: ImuData) {
        calibPitch = data.angleX
        calibRoll  = data.angleY
        calibYaw   = data.angleZ
        filtPitch = 0f; filtRoll = 0f; filtYaw = 0f
        trail.clear()
        invalidate()
    }

    // ─────────────────────────────────────────────
    // 旋转矩阵（ZYX 内旋）
    // ─────────────────────────────────────────────
    private fun rotMatrix(p: Float, r: Float, y: Float): FloatArray {
        val pr = Math.toRadians(p.toDouble()); val rr = Math.toRadians(r.toDouble())
        val yr = Math.toRadians(y.toDouble())
        val cp = cos(pr); val sp = sin(pr)
        val cr = cos(rr); val sr = sin(rr)
        val cy = cos(yr); val sy = sin(yr)
        return floatArrayOf(
            (cy*cp).toFloat(),          (cy*sp*sr - sy*cr).toFloat(), (cy*sp*cr + sy*sr).toFloat(),
            (sy*cp).toFloat(),          (sy*sp*sr + cy*cr).toFloat(), (sy*sp*cr - cy*sr).toFloat(),
            (-sp).toFloat(),            (cp*sr).toFloat(),             (cp*cr).toFloat()
        )
    }

    private fun rotate(m: FloatArray, j: HandModel.Joint): Triple<Float, Float, Float> {
        return Triple(
            m[0]*j.x + m[1]*j.y + m[2]*j.z,
            m[3]*j.x + m[4]*j.y + m[5]*j.z,
            m[6]*j.x + m[7]*j.y + m[8]*j.z
        )
    }

    private fun project(cx: Float, cy: Float, rx: Float, ry: Float, rz: Float): Triple<Float, Float, Float> {
        val scale = focalLen / (focalLen + rz + 15f)
        return Triple(cx + rx * scale * pixPerCm, cy - ry * scale * pixPerCm, scale)
    }

    // ─────────────────────────────────────────────
    // 绘制单帧手部（被主绘制和轨迹鬼影共用）
    // ─────────────────────────────────────────────
    private data class PJ(val sx: Float, val sy: Float, val sc: Float, val rz: Float)

    private fun buildProjected(cx: Float, cy: Float, pitch: Float, roll: Float, yaw: Float): List<PJ> {
        val m = rotMatrix(pitch, roll, yaw)
        return HandModel.joints.map { j ->
            val (rx, ry, rz) = rotate(m, j)
            val (sx, sy, sc) = project(cx, cy, rx, ry, rz)
            PJ(sx, sy, sc, rz)
        }
    }

    private fun drawHand(canvas: Canvas, pj: List<PJ>) {
        // 手掌多边形
        drawPalm(canvas, pj)
        // 手臂（锥形）
        drawArm(canvas, pj[0].sx, pj[0].sy, 1.7f * pj[0].sc * pixPerCm,
                        pj[1].sx, pj[1].sy, 1.25f * pj[1].sc * pixPerCm)
        // 骨骼（从远到近）
        val sortedBones = HandModel.bones.sortedByDescending { b ->
            (pj[b.from].rz + pj[b.to].rz) / 2f
        }
        for (bone in sortedBones) {
            val a = pj[bone.from]; val b = pj[bone.to]
            val avgSc = (a.sc + b.sc) / 2f
            val halfW  = bone.widthCm / 2f * avgSc * pixPerCm
            val depthT = ((-a.rz + 5f) / 15f).coerceIn(0f, 1f)
            val fill   = lerp(skinMid, skinBase, depthT)

            paintOutline.strokeWidth = halfW * 2f + 2.5f
            canvas.drawLine(a.sx, a.sy, b.sx, b.sy, paintOutline)
            paintBone.color = fill
            paintBone.strokeWidth = halfW * 2f
            canvas.drawLine(a.sx, a.sy, b.sx, b.sy, paintBone)
        }
        // 关节圆点
        for (idx in listOf(1, 2, 3, 6, 7, 10, 11, 14, 15, 18, 19)) {
            val p = pj[idx]
            canvas.drawCircle(p.sx, p.sy, 4f * p.sc * pixPerCm / 20f, paintJoint)
        }
        // 指甲
        for (idx in listOf(5, 9, 13, 17, 21)) {
            val p = pj[idx]; val prev = pj[idx - 1]
            val ang = atan2(p.sy - prev.sy, p.sx - prev.sx) * 180f / PI.toFloat()
            val nw = 5f * p.sc * pixPerCm / 20f; val nh = nw * 0.62f
            canvas.save()
            canvas.translate(p.sx, p.sy); canvas.rotate(ang)
            val nr = RectF(-nw * 0.15f, -nh, nw * 1.1f, nh)
            canvas.drawRoundRect(nr, nh * 0.55f, nh * 0.55f, paintNail)
            canvas.drawRoundRect(nr, nh * 0.55f, nh * 0.55f, paintNailRim)
            canvas.restore()
        }
    }

    // ─────────────────────────────────────────────
    // onDraw
    // ─────────────────────────────────────────────
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width == 0 || height == 0) return
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paintBg)

        // 手腕对应屏幕中央偏上（手指向下更自然）
        val cx = width / 2f
        val cy = height * 0.42f

        // ── 轨迹鬼影（从旧到新，透明度递增） ──────
        for (i in trail.indices.reversed()) {
            val angles = trail[i]
            val alpha  = (trailAlpha.getOrElse(i) { 0.02f } * 255).toInt().coerceIn(0, 255)
            if (alpha < 4) continue
            val saved = canvas.saveLayerAlpha(0f, 0f, width.toFloat(), height.toFloat(), alpha)
            val pj = buildProjected(cx, cy, angles[0], angles[1], angles[2])
            drawHand(canvas, pj)
            canvas.restoreToCount(saved)
        }

        // ── 当前手（全不透明） ───────────────────
        val pj = buildProjected(cx, cy, filtPitch, filtRoll, filtYaw)
        drawHand(canvas, pj)

        // ── 动态强度环（腕部周围，蓝=静止，橙=运动） ─
        val wrist = pj[1]
        val ringR = 28f * wrist.sc * pixPerCm / 20f + 6f
        val blue  = Color.parseColor("#3B82F6")
        val orange = Color.parseColor("#F97316")
        paintRing.color = lerp(blue, orange, motionIntensity.coerceIn(0f, 1f))
        paintRing.alpha = (180 + (75 * motionIntensity).toInt()).coerceIn(0, 255)
        canvas.drawCircle(wrist.sx, wrist.sy, ringR, paintRing)
    }

    // ── 辅助：绘制手掌多边形 ─────────────────────
    private fun drawPalm(canvas: Canvas, pj: List<PJ>) {
        val pts = HandModel.palmFace.map { pj[it] }
        if (pts.size < 3) return
        val path = Path()
        path.moveTo(pts[0].sx, pts[0].sy)
        for (i in 1 until pts.size) path.lineTo(pts[i].sx, pts[i].sy)
        path.close()
        val shader = LinearGradient(
            pts[0].sx, pts[0].sy, pts[2].sx, pts[2].sy,
            skinBase, skinMid, Shader.TileMode.CLAMP
        )
        paintFill.shader = shader
        canvas.drawPath(path, paintFill)
        paintFill.shader = null
        paintOutline.strokeWidth = 2f
        canvas.drawPath(path, paintOutline)
    }

    // ── 辅助：绘制锥形手臂 ──────────────────────
    private fun drawArm(canvas: Canvas, x0: Float, y0: Float, w0: Float,
                         x1: Float, y1: Float, w1: Float) {
        val dx = x1 - x0; val dy = y1 - y0
        val len = sqrt(dx * dx + dy * dy)
        if (len < 1f) return
        val nx = -dy / len; val ny = dx / len
        val ang = atan2(ny, nx) * 180f / PI.toFloat()
        val path = Path().apply {
            moveTo(x0 + nx * w0, y0 + ny * w0)
            lineTo(x1 + nx * w1, y1 + ny * w1)
            arcTo(RectF(x1 - w1, y1 - w1, x1 + w1, y1 + w1), ang, 180f)
            lineTo(x0 - nx * w0, y0 - ny * w0)
            arcTo(RectF(x0 - w0, y0 - w0, x0 + w0, y0 + w0), ang + 180f, 180f)
            close()
        }
        paintFill.color = armColor
        canvas.drawPath(path, paintFill)
        paintOutline.strokeWidth = 2.5f
        canvas.drawPath(path, paintOutline)
    }

    // ── 辅助：线性插值颜色 ───────────────────────
    private fun lerp(c1: Int, c2: Int, t: Float): Int {
        val r = (Color.red(c1)   + (Color.red(c2)   - Color.red(c1))   * t).toInt()
        val g = (Color.green(c1) + (Color.green(c2) - Color.green(c1)) * t).toInt()
        val b = (Color.blue(c1)  + (Color.blue(c2)  - Color.blue(c1))  * t).toInt()
        return Color.rgb(r, g, b)
    }

    private val PI = Math.PI
}
