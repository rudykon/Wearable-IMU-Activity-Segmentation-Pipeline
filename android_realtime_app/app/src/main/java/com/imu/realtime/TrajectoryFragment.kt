package com.imu.realtime

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.imu.realtime.databinding.FragmentTrajectoryBinding

class TrajectoryFragment : Fragment() {

    private var _binding: FragmentTrajectoryBinding? = null
    private val binding get() = _binding!!
    private val viewModel: ImuViewModel by activityViewModels()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentTrajectoryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.btnReset.setOnClickListener { binding.trajectoryView.reset() }

        viewModel.imuData.observe(viewLifecycleOwner) { data ->
            binding.trajectoryView.addData(data.accX, data.accY, data.accZ, data.angleX, data.angleY)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
