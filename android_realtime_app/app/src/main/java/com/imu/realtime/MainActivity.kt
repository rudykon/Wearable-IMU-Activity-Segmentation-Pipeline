package com.imu.realtime

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.imu.realtime.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var ble: BleManager
    private val viewModel: ImuViewModel by viewModels()

    private val foundDevices = mutableListOf<BluetoothDevice>()
    private val mainHandler = Handler(Looper.getMainLooper())

    // 录制计时
    private var recordStartMs = 0L
    private val timerRunnable = object : Runnable {
        override fun run() {
            val elapsed = (System.currentTimeMillis() - recordStartMs) / 1000L
            val m = elapsed / 60
            val s = elapsed % 60
            binding.btnRecord.text = String.format("⏹ %d:%02d", m, s)
            mainHandler.postDelayed(this, 1000)
        }
    }

    // ──────────────────────────────────────────
    // 权限
    // ──────────────────────────────────────────
    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        if (grants.values.all { it }) startScan()
        else Toast.makeText(this, getString(R.string.permission_bluetooth_required), Toast.LENGTH_LONG).show()
    }

    private fun neededPerms() =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S)
            arrayOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        else
            arrayOf(
                Manifest.permission.BLUETOOTH,
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.ACCESS_FINE_LOCATION
            )

    // ──────────────────────────────────────────
    // 生命周期
    // ──────────────────────────────────────────
    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(LocaleHelper.wrap(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        LocaleHelper.applyToResources(this)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ble = BleManager(this) { data -> viewModel.onData(data) }
        ble.onDeviceFound = { device -> addFoundDevice(device) }
        ble.onConnectionChanged = { connected, name ->
            viewModel.onConnectionChanged(connected, name)
        }

        // 显示默认 Fragment
        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, ChartsFragment())
                .commit()
        }

        setupBottomNav()
        observeViewModel()
        refreshLocalizedUi()

        binding.btnScan.setOnClickListener { checkAndScan() }
        binding.btnDisconnect.setOnClickListener { ble.disconnect() }
        binding.btnRecord.setOnClickListener { toggleRecording() }
        binding.btnLanguage.setOnClickListener { switchLanguage() }
    }

    override fun onDestroy() {
        super.onDestroy()
        ble.disconnect()
        mainHandler.removeCallbacks(timerRunnable)
    }

    private fun setupBottomNav() {
        binding.bottomNav.setOnItemSelectedListener { item ->
            showFragmentForNavItem(item.itemId)
        }
    }

    private fun showFragmentForNavItem(itemId: Int): Boolean {
        val fragment = fragmentForNavItem(itemId) ?: return false
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .commit()
        return true
    }

    private fun fragmentForNavItem(itemId: Int): Fragment? = when (itemId) {
        R.id.nav_charts      -> ChartsFragment()
        R.id.nav_attitude    -> AttitudeFragment()
        R.id.nav_hand        -> HandFragment()
        R.id.nav_trajectory  -> TrajectoryFragment()
        R.id.nav_dashboard   -> DashboardFragment()
        R.id.nav_recognition -> RecognitionFragment()
        else                 -> null
    }

    private fun observeViewModel() {
        viewModel.bleState.observe(this) { state ->
            renderBleState(state)
        }

        viewModel.deviceName.observe(this) { name ->
            renderDeviceName(name)
        }

        viewModel.hz.observe(this) { hz ->
            renderSamplingRate(hz)
        }

        viewModel.isRecording.observe(this) { recording ->
            if (recording) {
                recordStartMs = System.currentTimeMillis()
                mainHandler.post(timerRunnable)
            } else {
                mainHandler.removeCallbacks(timerRunnable)
                binding.btnRecord.text = getString(R.string.action_record)
            }
        }

        viewModel.recordPath.observe(this) { path ->
            if (path.isNotEmpty()) {
                Toast.makeText(this, getString(R.string.recording_path, path), Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun renderBleState(state: BleState) {
        when (state) {
            BleState.CONNECTED -> {
                binding.dotStatus.setBackgroundResource(R.drawable.dot_green)
                binding.btnDisconnect.visibility = View.VISIBLE
                binding.btnScan.isEnabled = false
            }
            BleState.SCANNING -> {
                binding.dotStatus.setBackgroundResource(R.drawable.dot_red)
                binding.btnDisconnect.visibility = View.GONE
                binding.btnScan.isEnabled = false
                setMsg(getString(R.string.status_scanning))
            }
            BleState.CONNECTING -> {
                binding.dotStatus.setBackgroundResource(R.drawable.dot_red)
                binding.btnDisconnect.visibility = View.GONE
                binding.btnScan.isEnabled = false
                setMsg(getString(R.string.status_connecting))
            }
            else -> {
                binding.dotStatus.setBackgroundResource(R.drawable.dot_red)
                binding.btnDisconnect.visibility = View.GONE
                binding.btnScan.isEnabled = true
                setMsg("")
            }
        }
    }

    private fun renderDeviceName(name: String) {
        if (name.isNotEmpty()) {
            binding.tvStatus.text = name
            binding.tvStatus.setTextColor(Color.parseColor("#10B981"))
            setMsg("")
        } else {
            binding.tvStatus.text = getString(R.string.status_disconnected)
            binding.tvStatus.setTextColor(Color.parseColor("#64748B"))
        }
    }

    private fun refreshLocalizedUi() {
        binding.btnScan.text = getString(R.string.action_scan)
        binding.btnDisconnect.text = getString(R.string.action_disconnect)
        if (viewModel.isRecording.value != true) {
            binding.btnRecord.text = getString(R.string.action_record)
        }
        updateLanguageButton()
        updateBottomNavTitles()
        renderBleState(viewModel.bleState.value ?: BleState.DISCONNECTED)
        renderDeviceName(viewModel.deviceName.value.orEmpty())
        renderSamplingRate(viewModel.hz.value ?: 0)
    }

    private fun updateLanguageButton() {
        binding.btnLanguage.text = if (LocaleHelper.isEnglish(this)) {
            getString(R.string.action_switch_to_chinese)
        } else {
            getString(R.string.action_switch_to_english)
        }
        binding.btnLanguage.contentDescription = getString(R.string.language_switch_content_description)
    }

    private fun updateBottomNavTitles() {
        binding.bottomNav.menu.findItem(R.id.nav_charts)?.setTitle(R.string.nav_charts)
        binding.bottomNav.menu.findItem(R.id.nav_attitude)?.setTitle(R.string.nav_attitude)
        binding.bottomNav.menu.findItem(R.id.nav_hand)?.setTitle(R.string.nav_hand)
        binding.bottomNav.menu.findItem(R.id.nav_trajectory)?.setTitle(R.string.nav_trajectory)
        binding.bottomNav.menu.findItem(R.id.nav_dashboard)?.setTitle(R.string.nav_dashboard)
        binding.bottomNav.menu.findItem(R.id.nav_recognition)?.setTitle(R.string.nav_recognition)
    }

    private fun renderSamplingRate(hz: Int) {
        binding.tvHz.text = if (hz > 0) {
            getString(R.string.status_sampling_rate_fmt, hz)
        } else {
            getString(R.string.status_sampling_rate_placeholder)
        }
    }

    private fun switchLanguage() {
        val selectedItemId = binding.bottomNav.selectedItemId.takeIf { it != 0 } ?: R.id.nav_charts
        LocaleHelper.setLanguageTag(this, LocaleHelper.nextLanguageTag(this))
        refreshLocalizedUi()
        showFragmentForNavItem(selectedItemId)
    }

    // ──────────────────────────────────────────
    // 录制
    // ──────────────────────────────────────────
    private fun toggleRecording() {
        if (viewModel.isRecording.value == true) {
            viewModel.stopRecording()
            Toast.makeText(this, getString(R.string.recording_stopped), Toast.LENGTH_SHORT).show()
        } else {
            viewModel.startRecording()
        }
    }

    // ──────────────────────────────────────────
    // 扫描 & 连接
    // ──────────────────────────────────────────
    private fun checkAndScan() {
        val perms = neededPerms()
        if (perms.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED })
            startScan()
        else
            permLauncher.launch(perms)
    }

    private fun startScan() {
        foundDevices.clear()
        viewModel.onScanning()
        ble.startScan()

        mainHandler.postDelayed({
            ble.stopScan()
            binding.btnScan.isEnabled = true
            if (foundDevices.isEmpty()) setMsg(getString(R.string.scan_no_device))
            else showDeviceDialog()
        }, 10_000)
    }

    @SuppressLint("MissingPermission")
    private fun addFoundDevice(device: BluetoothDevice) {
        if (foundDevices.none { it.address == device.address }) {
            foundDevices.add(device)
            setMsg(getString(R.string.scan_found_devices, foundDevices.size))
        }
    }

    @SuppressLint("MissingPermission")
    private fun showDeviceDialog() {
        val items = foundDevices
            .map { "${it.name ?: getString(R.string.device_unknown)}\n${it.address}" }
            .toTypedArray()

        AlertDialog.Builder(this)
            .setTitle(R.string.dialog_select_device)
            .setItems(items) { _, idx ->
                ble.stopScan()
                viewModel.onConnecting()
                ble.connect(foundDevices[idx])
            }
            .setNegativeButton(R.string.action_cancel) { _, _ ->
                binding.btnScan.isEnabled = true
                setMsg("")
            }
            .show()
    }

    private fun setMsg(msg: String) {
        binding.tvMsg.text = msg
        binding.tvMsg.visibility = if (msg.isEmpty()) View.GONE else View.VISIBLE
    }
}
