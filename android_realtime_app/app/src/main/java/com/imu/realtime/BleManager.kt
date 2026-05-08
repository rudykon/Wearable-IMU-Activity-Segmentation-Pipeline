package com.imu.realtime

import android.annotation.SuppressLint
import android.bluetooth.*
import android.bluetooth.le.*
import android.content.Context
import android.os.Handler
import android.os.Looper
import java.util.UUID

/**
 * BLE 管理器
 *   - 扫描含 "WT" 的设备
 *   - 连接并订阅 FFE4 通知
 *   - 解析 55 61 帧（Acc/Gyro/Angle）和 55 54 帧（Mag），回调 ImuData
 */
@SuppressLint("MissingPermission")
class BleManager(
    private val context: Context,
    private val onData: (ImuData) -> Unit
) {
    companion object {
        private val SERVICE_UUID = UUID.fromString("0000ffe5-0000-1000-8000-00805f9a34fb")
        private val NOTIFY_UUID  = UUID.fromString("0000ffe4-0000-1000-8000-00805f9a34fb")
        private val WRITE_UUID   = UUID.fromString("0000ffe9-0000-1000-8000-00805f9a34fb")
        private val CCC_UUID     = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
    }

    /** 扫描到含 WT 的设备时触发（主线程） */
    var onDeviceFound: ((BluetoothDevice) -> Unit)? = null

    /** 连接状态变化时触发（主线程）：connected, deviceName */
    var onConnectionChanged: ((Boolean, String) -> Unit)? = null

    private val btAdapter: BluetoothAdapter =
        (context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter

    private var gatt: BluetoothGatt? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    // 帧缓冲（仅在 BLE 回调线程访问，无并发问题）
    private val tmp = ArrayList<Int>(24)

    // 保存最近一次已知的完整数据，用于 mag 帧合并
    @Volatile private var lastKnown = ImuData()

    // ─────────────────────────────────────────
    // 扫描
    // ─────────────────────────────────────────
    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val name = result.device.name ?: return
            if (name.contains("WT")) {
                mainHandler.post { onDeviceFound?.invoke(result.device) }
            }
        }
    }

    fun startScan() {
        tmp.clear()
        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()
        btAdapter.bluetoothLeScanner?.startScan(null, settings, scanCallback)
    }

    fun stopScan() {
        btAdapter.bluetoothLeScanner?.stopScan(scanCallback)
    }

    // ─────────────────────────────────────────
    // 连接
    // ─────────────────────────────────────────
    fun connect(device: BluetoothDevice) {
        gatt?.close()
        gatt = device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
    }

    fun disconnect() {
        gatt?.disconnect()
        gatt?.close()
        gatt = null
        mainHandler.post { onConnectionChanged?.invoke(false, "") }
    }

    private val gattCallback = object : BluetoothGattCallback() {

        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED    -> gatt.discoverServices()
                BluetoothProfile.STATE_DISCONNECTED ->
                    mainHandler.post { onConnectionChanged?.invoke(false, "") }
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (status != BluetoothGatt.GATT_SUCCESS) return
            val service = gatt.getService(SERVICE_UUID) ?: return

            // 使能 FFE4 通知
            val notifyChar = service.getCharacteristic(NOTIFY_UUID) ?: return
            gatt.setCharacteristicNotification(notifyChar, true)
            notifyChar.getDescriptor(CCC_UUID)?.let { desc ->
                desc.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                gatt.writeDescriptor(desc)
            }

            val name = gatt.device.name ?: gatt.device.address
            mainHandler.post { onConnectionChanged?.invoke(true, name) }
        }

        // Android < 13
        @Suppress("DEPRECATION")
        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            if (characteristic.uuid == NOTIFY_UUID) parseBytes(characteristic.value)
        }

        // Android 13+
        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            if (characteristic.uuid == NOTIFY_UUID) parseBytes(value)
        }
    }

    // ─────────────────────────────────────────
    // 帧解析（55 61 / 55 54 / 55 71）
    // ─────────────────────────────────────────
    private fun parseBytes(data: ByteArray) {
        for (b in data) {
            val v = b.toInt() and 0xFF
            tmp.add(v)
            if (tmp.size == 1 && tmp[0] != 0x55) { tmp.clear(); continue }
            if (tmp.size == 2 && tmp[1] != 0x61 && tmp[1] != 0x71 && tmp[1] != 0x54) {
                tmp.clear(); continue
            }
            if (tmp.size == 20) {
                val frame = ArrayList(tmp)
                when (tmp[1]) {
                    0x61 -> {
                        val d = parseFrame61(frame)
                        lastKnown = d
                        mainHandler.post { onData(d) }
                    }
                    0x54 -> {
                        val d = parseFrame54(frame)
                        mainHandler.post { onData(d) }
                    }
                    // 0x71 reserved / ignored
                }
                tmp.clear()
            }
        }
    }

    /** 小端 int16 有符号转换：byte[lo] + byte[hi] */
    private fun s16(lo: Int, hi: Int): Float {
        var v = (hi shl 8) or lo
        if (v >= 32768) v -= 65536
        return v.toFloat()
    }

    /** 55 61 帧：Acc / Gyro / Angle */
    private fun parseFrame61(b: List<Int>) = ImuData(
        accX   = s16(b[2],  b[3])  / 32768f * 16f,
        accY   = s16(b[4],  b[5])  / 32768f * 16f,
        accZ   = s16(b[6],  b[7])  / 32768f * 16f,
        gyroX  = s16(b[8],  b[9])  / 32768f * 2000f,
        gyroY  = s16(b[10], b[11]) / 32768f * 2000f,
        gyroZ  = s16(b[12], b[13]) / 32768f * 2000f,
        angleX = s16(b[14], b[15]) / 32768f * 180f,
        angleY = s16(b[16], b[17]) / 32768f * 180f,
        angleZ = s16(b[18], b[19]) / 32768f * 180f,
        magX   = lastKnown.magX,
        magY   = lastKnown.magY,
        magZ   = lastKnown.magZ
    )

    /** 55 54 帧：磁场（合并上次已知的 acc/gyro/angle） */
    private fun parseFrame54(b: List<Int>) = ImuData(
        accX   = lastKnown.accX,
        accY   = lastKnown.accY,
        accZ   = lastKnown.accZ,
        gyroX  = lastKnown.gyroX,
        gyroY  = lastKnown.gyroY,
        gyroZ  = lastKnown.gyroZ,
        angleX = lastKnown.angleX,
        angleY = lastKnown.angleY,
        angleZ = lastKnown.angleZ,
        magX   = s16(b[2], b[3]),
        magY   = s16(b[4], b[5]),
        magZ   = s16(b[6], b[7])
    )
}
