package com.imu.realtime

import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.imu.realtime.databinding.FragmentRecognitionBinding

class RecognitionFragment : Fragment() {

    private var _binding: FragmentRecognitionBinding? = null
    private val binding get() = _binding!!
    private val viewModel: ImuViewModel by activityViewModels()

    private lateinit var classBars:   Array<ProgressBar>
    private lateinit var classLabels: Array<TextView>

    private val offlineFileLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) viewModel.runOfflineFile(uri)
    }

    private val classColors = intArrayOf(
        Color.parseColor("#94A3B8"), // 0 无活动
        Color.parseColor("#EF4444"), // 1 羽毛球
        Color.parseColor("#10B981"), // 2 跳绳
        Color.parseColor("#3B82F6"), // 3 飞鸟
        Color.parseColor("#F97316"), // 4 跑步
        Color.parseColor("#8B5CF6"), // 5 乒乓球
    )

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = FragmentRecognitionBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        classBars   = arrayOf(binding.pbClass0, binding.pbClass1, binding.pbClass2,
                               binding.pbClass3, binding.pbClass4, binding.pbClass5)
        classLabels = arrayOf(binding.tvClass0, binding.tvClass1, binding.tvClass2,
                               binding.tvClass3, binding.tvClass4, binding.tvClass5)

        val classifier = viewModel.classifier
        binding.tvModelStatus.text = if (classifier.isAvailable)
            getString(R.string.recognition_model_ready, modelSummaryText(classifier))
        else
            getString(R.string.recognition_model_missing)
        binding.tvModelStatus.setTextColor(
            if (classifier.isAvailable) Color.parseColor("#10B981") else Color.parseColor("#F97316")
        )

        binding.pbBuffer.max = MotionClassifier.WINDOW_8S
        viewModel.bufferFill.observe(viewLifecycleOwner) { fill ->
            binding.pbBuffer.progress = fill
            binding.tvBufferCount.text = "$fill/${MotionClassifier.WINDOW_8S}"
        }

        viewModel.vitBuffer.observe(viewLifecycleOwner) { n ->
            binding.pbViterbi.max = MotionClassifier.OUTPUT_MIN_SEGMENT_SEC
            binding.pbViterbi.progress = n.coerceAtMost(MotionClassifier.OUTPUT_MIN_SEGMENT_SEC)
            binding.tvViterbiBuf.text  = "$n/${MotionClassifier.OUTPUT_MIN_SEGMENT_SEC}s"
        }

        viewModel.recognitionResult.observe(viewLifecycleOwner) { result ->
            if (result == null) return@observe
            updateResult(result)
        }

        // ── Activity history ──────────────────────────────────────────────────
        binding.btnResetSession.setOnClickListener {
            viewModel.resetSession()
        }

        viewModel.activitySegments.observe(viewLifecycleOwner) { segments ->
            renderSegments(segments)
        }

        viewModel.sessionStats.observe(viewLifecycleOwner) { (totalSec, byClass) ->
            updateStats(totalSec, byClass)
        }

        binding.btnOfflineSample.setOnClickListener {
            viewModel.runBuiltInOfflineSample()
        }

        binding.btnOfflineFile.setOnClickListener {
            offlineFileLauncher.launch(arrayOf("*/*"))
        }

        viewModel.offlineState.observe(viewLifecycleOwner) { state ->
            renderOfflineState(state)
        }
    }

    // ── Recognition result ────────────────────────────────────────────────────

    private fun updateResult(r: RecognitionResult) {
        val warmupPct = (r.warmupFraction * 100).toInt()
        if (warmupPct < 100) {
            binding.tvActivityEmoji.text = "⏳"
            binding.tvActivityName.text  = getString(R.string.recognition_warming_up)
            binding.tvActivityName.setTextColor(Color.parseColor("#94A3B8"))
        } else {
            binding.tvActivityEmoji.text = MotionClassifier.CLASS_EMOJIS.getOrElse(r.classIdx) { "❓" }
            binding.tvActivityName.text  = activityName(r.classIdx)
            binding.tvActivityName.setTextColor(classColors.getOrElse(r.classIdx) { Color.parseColor("#0F172A") })
        }

        val pct = (r.confidence * 100).toInt()
        binding.pbConfidence.progress = pct
        binding.tvConfidencePct.text  = "$pct%"
        val color = classColors.getOrElse(r.classIdx) { Color.parseColor("#3B82F6") }
        binding.pbConfidence.progressTintList =
            android.content.res.ColorStateList.valueOf(color)

        val display = r.smoothedProbs
        for (i in display.indices) {
            val p = (display[i] * 100).toInt()
            classBars[i].progress = p
            classLabels[i].text   = "$p%"
            classLabels[i].setTextColor(
                if (i == r.classIdx) classColors[i] else Color.parseColor("#64748B")
            )
        }
    }

    // ── Segment list rendering ────────────────────────────────────────────────

    private fun renderSegments(segments: List<ActivitySegment>) {
        val container = binding.llSegments
        container.removeAllViews()

        if (segments.isEmpty()) {
            binding.tvSegmentsEmpty.visibility = View.VISIBLE
            return
        }
        binding.tvSegmentsEmpty.visibility = View.GONE

        val ctx = requireContext()
        val dp = ctx.resources.displayMetrics.density

        // Render newest-first (reverse order)
        for (seg in segments.asReversed()) {
            val row = LinearLayout(ctx).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).also { it.bottomMargin = (8 * dp).toInt() }
            }

            val color = classColors.getOrElse(seg.classIdx) { Color.parseColor("#94A3B8") }
            val emoji = MotionClassifier.CLASS_EMOJIS.getOrElse(seg.classIdx) { "❓" }

            // Colored indicator stripe
            val stripe = View(ctx).apply {
                layoutParams = LinearLayout.LayoutParams((3 * dp).toInt(), (36 * dp).toInt())
                    .also { it.marginEnd = (10 * dp).toInt() }
                setBackgroundColor(color)
                background = android.graphics.drawable.GradientDrawable().apply {
                    setColor(color)
                    cornerRadius = 4 * dp
                }
            }
            row.addView(stripe)

            // Start time label (MM:SS)
            val tvStart = TextView(ctx).apply {
                text = formatSec(seg.startOffsetSec)
                textSize = 10f
                setTextColor(Color.parseColor("#94A3B8"))
                typeface = android.graphics.Typeface.MONOSPACE
                layoutParams = LinearLayout.LayoutParams(
                    (46 * dp).toInt(), LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }
            row.addView(tvStart)

            // Emoji + name
            val tvName = TextView(ctx).apply {
                text = "$emoji ${activityName(seg.classIdx)}"
                textSize = 13f
                setTextColor(color)
                layoutParams = LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f
                )
            }
            row.addView(tvName)

            // Duration (MM:SS or ongoing indicator)
            val tvDur = TextView(ctx).apply {
                text = if (seg.isOngoing)
                    "${formatSec(seg.durationSec)} ▶"
                else
                    formatSec(seg.durationSec)
                textSize = 11f
                setTextColor(if (seg.isOngoing) color else Color.parseColor("#94A3B8"))
                typeface = android.graphics.Typeface.MONOSPACE
                gravity = Gravity.END
                layoutParams = LinearLayout.LayoutParams(
                    (60 * dp).toInt(), LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }
            row.addView(tvDur)

            container.addView(row)
        }
    }

    // ── Session stats ─────────────────────────────────────────────────────────

    private fun updateStats(totalSec: Int, byClass: Map<Int, Int>) {
        if (totalSec == 0) {
            binding.tvSessionTotal.text  = getString(R.string.recognition_session_total_empty)
            binding.tvSessionActive.text = getString(R.string.recognition_session_active_empty)
            return
        }

        // "Active" = any class that is not 0 (无活动)
        val activeSec = byClass.entries
            .filter { it.key != 0 }
            .sumOf { it.value }

        binding.tvSessionTotal.text  = getString(R.string.recognition_session_total_fmt, formatSec(totalSec))
        binding.tvSessionActive.text = getString(R.string.recognition_session_active_fmt, formatSec(activeSec))
    }

    private fun renderOfflineState(state: OfflineRecognitionState) {
        binding.pbOffline.visibility = if (state.isRunning) View.VISIBLE else View.GONE
        binding.btnOfflineSample.isEnabled = !state.isRunning
        binding.btnOfflineFile.isEnabled = !state.isRunning

        val text = when {
            state.error != null -> getString(R.string.offline_status_error, state.error)
            state.isRunning && state.sourceName.isBlank() -> getString(R.string.offline_status_loading)
            state.isRunning -> getString(
                R.string.offline_status_running,
                state.sourceName,
                state.sampleCount,
                formatSec(state.durationSec),
            )
            state.sourceName.isNotBlank() -> getString(
                R.string.offline_status_done,
                state.sourceName,
                state.sampleCount,
                formatSec(state.durationSec),
                state.segmentCount,
            )
            else -> getString(R.string.offline_status_idle)
        }

        binding.tvOfflineStatus.text = text
        binding.tvOfflineStatus.setTextColor(
            when {
                state.error != null -> Color.parseColor("#EF4444")
                state.sourceName.isNotBlank() && !state.isRunning -> Color.parseColor("#10B981")
                else -> Color.parseColor("#64748B")
            }
        )
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun activityName(classIdx: Int): String {
        return resources.getStringArray(R.array.activity_names)
            .getOrElse(classIdx) { getString(R.string.unknown_activity) }
    }

    private fun modelSummaryText(classifier: MotionClassifier): String {
        if (classifier.usingLegacyModel) {
            return getString(R.string.model_summary_single_fallback)
        }

        val parts = mutableListOf<String>()
        if (classifier.loaded3sModelCount > 0) parts += "3s×${classifier.loaded3sModelCount}"
        if (classifier.loaded5sModelCount > 0) parts += "5s×${classifier.loaded5sModelCount}"
        if (classifier.loaded8sModelCount > 0) parts += "8s×${classifier.loaded8sModelCount}"
        if (parts.isEmpty()) return getString(R.string.model_summary_unloaded)

        return getString(
            R.string.model_summary_models,
            parts.joinToString(" + "),
            classifier.loadedModelCount
        )
    }

    private fun formatSec(sec: Int): String {
        val m = sec / 60
        val s = sec % 60
        return "%02d:%02d".format(m, s)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
