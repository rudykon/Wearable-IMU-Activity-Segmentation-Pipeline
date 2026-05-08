package com.imu.realtime

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.imu.realtime.databinding.FragmentHandBinding
import kotlin.math.roundToInt

class HandFragment : Fragment() {

    private var _binding: FragmentHandBinding? = null
    private val binding get() = _binding!!
    private val viewModel: ImuViewModel by activityViewModels()

    /** 保存最新一帧，供校准按钮使用 */
    private var latestData: ImuData? = null

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentHandBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // 校准按钮：将当前传感器读数设为"下垂零点"
        binding.btnCalibrate.setOnClickListener {
            val data = latestData
            if (data != null) {
                binding.handView.calibrate(data)
                Toast.makeText(requireContext(), getString(R.string.hand_calibrated), Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(requireContext(), getString(R.string.hand_no_sensor_data), Toast.LENGTH_SHORT).show()
            }
        }

        viewModel.imuData.observe(viewLifecycleOwner) { data ->
            latestData = data

            // 将完整 ImuData 传入 HandView（acc + gyro + angle 全部使用）
            binding.handView.update(data)

            // 更新数值显示
            binding.tvHandPitch.text = String.format("%+.1f°", data.angleX)
            binding.tvHandRoll.text  = String.format("%+.1f°", data.angleY)
            binding.tvHandYaw.text   = String.format("%+.1f°", data.angleZ)

            // 动作强度进度条（0~100）
            val accMag = kotlin.math.sqrt(
                data.accX * data.accX + data.accY * data.accY + data.accZ * data.accZ
            )
            val dynAccPct = (kotlin.math.abs(accMag - 1f) / 1.5f * 100).coerceIn(0f, 100f)
            binding.pbMotion.progress = dynAccPct.roundToInt()
            // 高动态时换橙色，静止时蓝色
            val tintColor = if (dynAccPct > 20f)
                android.graphics.Color.parseColor("#F97316")
            else
                android.graphics.Color.parseColor("#3B82F6")
            binding.pbMotion.progressTintList =
                android.content.res.ColorStateList.valueOf(tintColor)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
