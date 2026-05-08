package com.imu.realtime

import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.imu.realtime.databinding.FragmentChartsBinding

class ChartsFragment : Fragment() {

    private var _binding: FragmentChartsBinding? = null
    private val binding get() = _binding!!
    private val viewModel: ImuViewModel by activityViewModels()

    private var xCounter = 0f
    private val WINDOW = 200

    private val RED   = Color.parseColor("#EF4444")
    private val GREEN = Color.parseColor("#10B981")
    private val BLUE  = Color.parseColor("#3B82F6")
    // 白色主题图表配色
    private val CHART_BG  = Color.parseColor("#FFFFFF")
    private val GRID      = Color.parseColor("#E2E8F0")
    private val ZERO_LINE = Color.parseColor("#94A3B8")
    private val AXIS_TEXT = Color.parseColor("#64748B")
    private val LEG_TEXT  = Color.parseColor("#475569")

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentChartsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupCharts()

        viewModel.imuData.observe(viewLifecycleOwner) { data ->
            updateCharts(data)
            updateValues(data)
        }
    }

    private fun setupCharts() {
        configChart(binding.chartAcc,   -20f,   20f,   listOf("AccX",  "AccY",  "AccZ"),  listOf(RED, GREEN, BLUE), autoScale = false)
        configChart(binding.chartGyro,    0f,    0f,   listOf("GyroX", "GyroY", "GyroZ"), listOf(RED, GREEN, BLUE), autoScale = true)
        configChart(binding.chartAngle, -185f,  185f,  listOf("Pitch", "Roll",  "Yaw"),   listOf(RED, GREEN, BLUE), autoScale = false)
    }

    private fun configChart(chart: LineChart, yMin: Float, yMax: Float, labels: List<String>, colors: List<Int>,
                             autoScale: Boolean = false) {
        chart.apply {
            description.isEnabled = false
            setNoDataText(getString(R.string.chart_waiting_connection))
            setNoDataTextColor(AXIS_TEXT)
            setBackgroundColor(CHART_BG)
            setDrawGridBackground(false)
            setTouchEnabled(false)
            // 伸缩尺度：随可见数据自动调整 Y 轴范围
            setAutoScaleMinMaxEnabled(autoScale)
            xAxis.isEnabled = false
            axisLeft.apply {
                if (autoScale) {
                    // 自动伸缩：重置固定范围，让 MPAndroidChart 根据数据计算
                    resetAxisMinimum()
                    resetAxisMaximum()
                    // 保留适当的上下边距（10%）
                    spaceTop    = 10f
                    spaceBottom = 10f
                } else {
                    axisMinimum = yMin
                    axisMaximum = yMax
                }
                gridColor   = GRID
                textColor   = AXIS_TEXT
                textSize    = 9f
                setDrawZeroLine(true)
                zeroLineColor = ZERO_LINE
                zeroLineWidth = 1f
            }
            axisRight.isEnabled = false
            legend.apply {
                textColor = LEG_TEXT
                textSize  = 10f
                form      = Legend.LegendForm.LINE
            }
            val sets = labels.mapIndexed { i, label ->
                LineDataSet(mutableListOf<Entry>(), label).apply {
                    color = colors[i]
                    lineWidth = 2f
                    setDrawCircles(false)
                    setDrawValues(false)
                    isHighlightEnabled = false
                    mode = LineDataSet.Mode.LINEAR
                }
            }
            data = LineData(sets)
        }
    }

    private fun updateCharts(d: ImuData) {
        val x = xCounter++
        pushChart(binding.chartAcc,   listOf(d.accX,   d.accY,   d.accZ),   x)
        pushChart(binding.chartGyro,  listOf(d.gyroX,  d.gyroY,  d.gyroZ),  x)
        pushChart(binding.chartAngle, listOf(d.angleX, d.angleY, d.angleZ), x)
    }

    private fun pushChart(chart: LineChart, values: List<Float>, x: Float) {
        val lineData = chart.data ?: return
        values.forEachIndexed { i, v ->
            val set = lineData.getDataSetByIndex(i) as LineDataSet
            set.addEntry(Entry(x, v))
            if (set.entryCount > WINDOW) set.removeEntry(set.getEntryForIndex(0))
        }
        lineData.notifyDataChanged()
        chart.notifyDataSetChanged()
        chart.setVisibleXRangeMaximum(WINDOW.toFloat())
        chart.moveViewToX(x)
    }

    private fun updateValues(d: ImuData) {
        binding.tvAccX.text  = String.format("%+.3f", d.accX)
        binding.tvAccY.text  = String.format("%+.3f", d.accY)
        binding.tvAccZ.text  = String.format("%+.3f", d.accZ)
        binding.tvGyroX.text = String.format("%+.1f", d.gyroX)
        binding.tvGyroY.text = String.format("%+.1f", d.gyroY)
        binding.tvGyroZ.text = String.format("%+.1f", d.gyroZ)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
