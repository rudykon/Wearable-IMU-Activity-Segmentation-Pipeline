package com.imu.realtime

import android.app.Application
import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import java.util.concurrent.Executors

data class OfflineRecognitionState(
    val isRunning: Boolean = false,
    val sourceName: String = "",
    val sampleCount: Int = 0,
    val durationSec: Int = 0,
    val segmentCount: Int = 0,
    val error: String? = null,
)

class ImuViewModel(app: Application) : AndroidViewModel(app) {

    private val _imuData = MutableLiveData<ImuData>()
    val imuData: LiveData<ImuData> = _imuData

    private val _bleState = MutableLiveData(BleState.DISCONNECTED)
    val bleState: LiveData<BleState> = _bleState

    private val _deviceName = MutableLiveData("")
    val deviceName: LiveData<String> = _deviceName

    private val _hz = MutableLiveData(0)
    val hz: LiveData<Int> = _hz

    private val _isRecording = MutableLiveData(false)
    val isRecording: LiveData<Boolean> = _isRecording

    private val _recordPath = MutableLiveData("")
    val recordPath: LiveData<String> = _recordPath

    // ── 运动识别 ──────────────────────────────────────────────────────────────
    val classifier = MotionClassifier(app)

    private val _recognitionResult = MutableLiveData<RecognitionResult?>(null)
    val recognitionResult: LiveData<RecognitionResult?> = _recognitionResult

    private val _bufferFill = MutableLiveData(0)
    val bufferFill: LiveData<Int> = _bufferFill

    // Offline-style pipeline depth: the classifier recomputes the complete
    // probability, Viterbi and segment history on every inference pass.
    private val _vitBuffer = MutableLiveData(0)
    val vitBuffer: LiveData<Int> = _vitBuffer

    private val _activitySegments = MutableLiveData<List<ActivitySegment>>(emptyList())
    val activitySegments: LiveData<List<ActivitySegment>> = _activitySegments

    private val _sessionStats = MutableLiveData<Pair<Int, Map<Int, Int>>>(Pair(0, emptyMap()))
    val sessionStats: LiveData<Pair<Int, Map<Int, Int>>> = _sessionStats

    private val _offlineState = MutableLiveData(OfflineRecognitionState())
    val offlineState: LiveData<OfflineRecognitionState> = _offlineState

    init {
        classifier.onResult = { result ->
            _recognitionResult.postValue(result)
            _vitBuffer.postValue(result.decodedSeconds)
            _activitySegments.postValue(result.segments)
            _sessionStats.postValue(result.stats)
        }
    }

    private val offlineExecutor = Executors.newSingleThreadExecutor()

    fun resetSession() {
        classifier.reset()
        _recognitionResult.postValue(null)
        _activitySegments.postValue(emptyList())
        _sessionStats.postValue(Pair(0, emptyMap()))
        _bufferFill.postValue(0)
        _vitBuffer.postValue(0)
    }

    fun runBuiltInOfflineSample() {
        if (!OfflineSensorDataLoader.hasBuiltInSample(getApplication())) {
            _offlineState.postValue(
                OfflineRecognitionState(error = getApplication<Application>().getString(R.string.offline_builtin_unavailable))
            )
            return
        }
        runOfflineLoadAndClassify {
            OfflineSensorDataLoader.loadBuiltInSample(getApplication())
        }
    }

    fun runOfflineFile(uri: Uri) {
        runOfflineLoadAndClassify {
            OfflineSensorDataLoader.loadFromUri(getApplication(), uri)
        }
    }

    private fun runOfflineLoadAndClassify(loader: () -> OfflineSensorSeries) {
        if (!classifier.isAvailable) {
            _offlineState.postValue(
                OfflineRecognitionState(error = getApplication<Application>().getString(R.string.recognition_model_missing))
            )
            return
        }

        _offlineState.postValue(OfflineRecognitionState(isRunning = true))
        offlineExecutor.execute {
            try {
                val series = loader()
                if (series.sampleCount < MotionClassifier.WINDOW_3S) {
                    throw IllegalArgumentException("Need at least ${MotionClassifier.WINDOW_3S} rows of ACC/GYRO data")
                }

                _offlineState.postValue(
                    OfflineRecognitionState(
                        isRunning = true,
                        sourceName = series.sourceName,
                        sampleCount = series.sampleCount,
                        durationSec = series.durationSec,
                    )
                )

                val result = classifier.classifyOffline(series)
                    ?: throw IllegalStateException("No prediction windows were produced")

                _recognitionResult.postValue(result)
                _vitBuffer.postValue(result.decodedSeconds)
                _activitySegments.postValue(result.segments)
                _sessionStats.postValue(result.stats)
                _bufferFill.postValue(minOf(series.sampleCount, MotionClassifier.WINDOW_8S))
                _offlineState.postValue(
                    OfflineRecognitionState(
                        sourceName = series.sourceName,
                        sampleCount = series.sampleCount,
                        durationSec = series.durationSec,
                        segmentCount = result.segments.size,
                    )
                )
            } catch (e: Exception) {
                _offlineState.postValue(
                    OfflineRecognitionState(error = e.message ?: e.javaClass.simpleName)
                )
            }
        }
    }

    // ── BLE data ──────────────────────────────────────────────────────────────

    private val recorder    = DataRecorder()
    private val mainHandler = Handler(Looper.getMainLooper())
    private var hzFrames    = 0
    private var hzTs        = System.currentTimeMillis()

    fun onData(data: ImuData) {
        _imuData.postValue(data)
        recorder.record(data)

        classifier.addSample(data.accX, data.accY, data.accZ,
                             data.gyroX, data.gyroY, data.gyroZ)
        _bufferFill.postValue(classifier.bufferFill())

        hzFrames++
        val now = System.currentTimeMillis()
        if (now - hzTs >= 1000L) {
            val frames = hzFrames
            mainHandler.post { _hz.value = frames }
            hzFrames = 0
            hzTs = now
        }
    }

    fun onConnectionChanged(connected: Boolean, name: String) {
        _bleState.postValue(if (connected) BleState.CONNECTED else BleState.DISCONNECTED)
        _deviceName.postValue(if (connected) name else "")
        if (connected) {
            classifier.reset()
            _recognitionResult.postValue(null)
            _bufferFill.postValue(0)
            _vitBuffer.postValue(0)
        }
    }

    fun onScanning()   { _bleState.postValue(BleState.SCANNING) }
    fun onConnecting() { _bleState.postValue(BleState.CONNECTING) }

    fun startRecording() {
        val path = recorder.start()
        _isRecording.postValue(true)
        _recordPath.postValue(path)
    }

    fun stopRecording() {
        recorder.stop()
        _isRecording.postValue(false)
    }

    override fun onCleared() {
        super.onCleared()
        recorder.shutdown()
        offlineExecutor.shutdown()
        classifier.close()
    }
}
