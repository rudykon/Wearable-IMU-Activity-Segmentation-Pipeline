package com.imu.realtime

import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.imu.realtime.databinding.FragmentDashboardBinding

class DashboardFragment : Fragment() {

    private var _binding: FragmentDashboardBinding? = null
    private val binding get() = _binding!!
    private val viewModel: ImuViewModel by activityViewModels()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentDashboardBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupGauges()

        viewModel.imuData.observe(viewLifecycleOwner) { data ->
            binding.gaugeAccX.update(data.accX)
            binding.gaugeAccY.update(data.accY)
            binding.gaugeAccZ.update(data.accZ)
            binding.gaugeGyroX.update(data.gyroX)
            binding.gaugeGyroY.update(data.gyroY)
            binding.gaugeGyroZ.update(data.gyroZ)
            binding.gaugeAngleX.update(data.angleX)
            binding.gaugeAngleY.update(data.angleY)
            binding.gaugeAngleZ.update(data.angleZ)
        }
    }

    private fun setupGauges() {
        val red   = Color.parseColor("#EF4444")
        val green = Color.parseColor("#10B981")
        val blue  = Color.parseColor("#3B82F6")

        binding.gaugeAccX.apply  { minVal=-20f; maxVal=20f;    label="AccX";  unit="g";   arcColor=red   }
        binding.gaugeAccY.apply  { minVal=-20f; maxVal=20f;    label="AccY";  unit="g";   arcColor=green }
        binding.gaugeAccZ.apply  { minVal=-20f; maxVal=20f;    label="AccZ";  unit="g";   arcColor=blue  }
        binding.gaugeGyroX.apply { minVal=-2000f; maxVal=2000f; label="GyroX"; unit="°/s"; arcColor=red   }
        binding.gaugeGyroY.apply { minVal=-2000f; maxVal=2000f; label="GyroY"; unit="°/s"; arcColor=green }
        binding.gaugeGyroZ.apply { minVal=-2000f; maxVal=2000f; label="GyroZ"; unit="°/s"; arcColor=blue  }
        binding.gaugeAngleX.apply{ minVal=-180f; maxVal=180f;  label="Pitch"; unit="°";   arcColor=red   }
        binding.gaugeAngleY.apply{ minVal=-180f; maxVal=180f;  label="Roll";  unit="°";   arcColor=green }
        binding.gaugeAngleZ.apply{ minVal=-180f; maxVal=180f;  label="Yaw";   unit="°";   arcColor=blue  }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
