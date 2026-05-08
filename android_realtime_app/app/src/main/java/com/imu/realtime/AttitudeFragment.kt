package com.imu.realtime

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.imu.realtime.databinding.FragmentAttitudeBinding

class AttitudeFragment : Fragment() {

    private var _binding: FragmentAttitudeBinding? = null
    private val binding get() = _binding!!
    private val viewModel: ImuViewModel by activityViewModels()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentAttitudeBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        viewModel.imuData.observe(viewLifecycleOwner) { data ->
            binding.horizonView.update(data.angleX, data.angleY)
            binding.compassView.update(data.angleZ)

            binding.tvPitch.text = String.format("Pitch: %+.1f°", data.angleX)
            binding.tvRoll.text  = String.format("Roll:  %+.1f°", data.angleY)
            binding.tvYaw.text   = String.format("Yaw:   %+.1f°", data.angleZ)

            if (!data.magX.isNaN()) {
                binding.tvMagX.text = String.format("MagX: %.0f", data.magX)
                binding.tvMagY.text = String.format("MagY: %.0f", data.magY)
                binding.tvMagZ.text = String.format("MagZ: %.0f", data.magZ)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
