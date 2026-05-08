"""
WT9011DCL-BT50  实时 acc / gyro 采集与展示
==========================================
用法：
    python collect.py                 # 自动扫描并列出含 "WT" 的设备，手动选择
    python collect.py AA:BB:CC:DD:EE:FF   # 直接指定 MAC，跳过扫描

窗口说明：
  上半部分  —— 加速度（AccX/Y/Z，单位 g）
  下半部分  —— 角速度（AsX/Y/Z，单位 °/s）
  标题行实时更新最新数值。
  关闭窗口即断开 BLE 并退出。
"""

import asyncio
import sys
import threading
from collections import deque

import bleak
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────
# 中文字体
# ──────────────────────────────────────────────
matplotlib.rcParams["font.family"] = ["SimHei", "Microsoft JhengHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# BLE UUID（WT9011DCL-BT50 固定值）
# ──────────────────────────────────────────────
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

# ──────────────────────────────────────────────
# 滚动缓冲区配置
# ──────────────────────────────────────────────
WINDOW   = 200    # 显示最近 N 帧
INTERVAL = 40     # 图形刷新间隔（ms），约 25 fps

buf: dict[str, deque] = {k: deque([0.0] * WINDOW, maxlen=WINDOW)
                         for k in ("AccX", "AccY", "AccZ", "AsX", "AsY", "AsZ")}
buf_lock  = threading.Lock()
latest: dict = {}          # 最新一帧数值，用于标题显示
frame_count = 0            # 已接收帧计数

# ──────────────────────────────────────────────
# 55 61 帧解析
# ──────────────────────────────────────────────
_tmp: list[int] = []

def _s16(hi: int, lo: int) -> int:
    """把两字节（大端高位先）拼成 int16 有符号整数"""
    v = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v

def _on_notify(sender, data: bytearray) -> None:
    global _tmp, frame_count
    for b in data:
        _tmp.append(b)
        # 帧头校验
        if len(_tmp) == 1 and _tmp[0] != 0x55:
            _tmp.clear(); continue
        if len(_tmp) == 2 and _tmp[1] not in (0x61, 0x71):
            _tmp.clear(); continue
        if len(_tmp) == 20:
            if _tmp[1] == 0x61:
                _parse_61(_tmp)
                frame_count += 1
            _tmp.clear()

def _parse_61(b: list[int]) -> None:
    """解析 55 61 姿态帧，字节布局（小端 int16）：
    [2:4]=AccX  [4:6]=AccY  [6:8]=AccZ
    [8:10]=AsX  [10:12]=AsY [12:14]=AsZ
    [14:16]=AngX [16:18]=AngY [18:20]=AngZ
    """
    ax = _s16(b[3], b[2]) / 32768.0 * 16
    ay = _s16(b[5], b[4]) / 32768.0 * 16
    az = _s16(b[7], b[6]) / 32768.0 * 16
    gx = _s16(b[9],  b[8])  / 32768.0 * 2000
    gy = _s16(b[11], b[10]) / 32768.0 * 2000
    gz = _s16(b[13], b[12]) / 32768.0 * 2000
    with buf_lock:
        buf["AccX"].append(ax); buf["AccY"].append(ay); buf["AccZ"].append(az)
        buf["AsX"].append(gx);  buf["AsY"].append(gy);  buf["AsZ"].append(gz)
    latest.update(AccX=ax, AccY=ay, AccZ=az, AsX=gx, AsY=gy, AsZ=gz)

# ──────────────────────────────────────────────
# BLE 扫描 & 连接
# ──────────────────────────────────────────────
connected  = threading.Event()
stop_event = threading.Event()
_ble_loop  = asyncio.new_event_loop()

async def _ble_main(target_addr: str | None) -> None:
    if target_addr:
        device = await bleak.BleakScanner.find_device_by_address(target_addr, timeout=20)
        if device is None:
            print(f"[错误] 未找到 MAC={target_addr}，请检查地址或蓝牙是否开启。")
            return
    else:
        print("正在扫描 BLE 设备（最长 15 秒）……")
        devices = await bleak.BleakScanner.discover(timeout=15)
        found = [d for d in devices if d.name and "WT" in d.name]
        if not found:
            print("[错误] 未发现含 'WT' 的设备，请确认设备已开机且蓝牙可用。")
            return
        print(f"\n发现 {len(found)} 台设备：")
        for i, d in enumerate(found):
            print(f"  [{i}] {d.address}  {d.name}")
        if len(found) == 1:
            idx = 0
            print(f"自动选择唯一设备 [{idx}]")
        else:
            idx = int(input(f"请输入序号 [0-{len(found)-1}]: ").strip())
        device = found[idx]

    print(f"\n正在连接 {device.address}（{device.name}）……")
    try:
        async with bleak.BleakClient(device, timeout=15) as client:
            await client.start_notify(NOTIFY_UUID, _on_notify)
            print(f"已连接 ✓  关闭图形窗口即退出。\n")
            connected.set()
            while not stop_event.is_set():
                await asyncio.sleep(0.3)
            await client.stop_notify(NOTIFY_UUID)
    except Exception as e:
        print(f"[错误] BLE 连接失败：{e}")
        connected.set()   # 避免主线程永久等待

def _ble_thread(target_addr: str | None) -> None:
    asyncio.set_event_loop(_ble_loop)
    _ble_loop.run_until_complete(_ble_main(target_addr))

# ──────────────────────────────────────────────
# matplotlib 实时绘图
# ──────────────────────────────────────────────
def _start_plot() -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    fig.suptitle("WT9011DCL-BT50  实时 IMU 数据", fontsize=13, fontweight="bold")

    x = np.arange(WINDOW)

    # ── 加速度子图 ──
    ax1.set_xlim(0, WINDOW - 1)
    ax1.set_ylim(-20, 20)
    ax1.set_ylabel("g", fontsize=10)
    ax1.set_xticklabels([])
    ax1.grid(True, alpha=0.25)
    ln_ax, = ax1.plot(x, list(buf["AccX"]), lw=1.2, label="AccX", color="#e74c3c")
    ln_ay, = ax1.plot(x, list(buf["AccY"]), lw=1.2, label="AccY", color="#2ecc71")
    ln_az, = ax1.plot(x, list(buf["AccZ"]), lw=1.2, label="AccZ", color="#3498db")
    ax1.legend(loc="upper right", fontsize=9, ncol=3)
    ax1.axhline(0, color="gray", lw=0.5, ls="--")

    # ── 角速度子图 ──
    ax2.set_xlim(0, WINDOW - 1)
    ax2.set_ylim(-2200, 2200)
    ax2.set_ylabel("°/s", fontsize=10)
    ax2.set_xlabel(f"最近 {WINDOW} 帧", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ln_gx, = ax2.plot(x, list(buf["AsX"]), lw=1.2, label="GyroX", color="#e74c3c")
    ln_gy, = ax2.plot(x, list(buf["AsY"]), lw=1.2, label="GyroY", color="#2ecc71")
    ln_gz, = ax2.plot(x, list(buf["AsZ"]), lw=1.2, label="GyroZ", color="#3498db")
    ax2.legend(loc="upper right", fontsize=9, ncol=3)
    ax2.axhline(0, color="gray", lw=0.5, ls="--")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def _update(_frame):
        with buf_lock:
            a = [list(buf["AccX"]), list(buf["AccY"]), list(buf["AccZ"])]
            g = [list(buf["AsX"]),  list(buf["AsY"]),  list(buf["AsZ"])]
        ln_ax.set_ydata(a[0]); ln_ay.set_ydata(a[1]); ln_az.set_ydata(a[2])
        ln_gx.set_ydata(g[0]); ln_gy.set_ydata(g[1]); ln_gz.set_ydata(g[2])
        if latest:
            ax1.set_title(
                f"加速度 (g)     "
                f"X = {latest['AccX']:+.3f}    "
                f"Y = {latest['AccY']:+.3f}    "
                f"Z = {latest['AccZ']:+.3f}",
                fontsize=10,
            )
            ax2.set_title(
                f"角速度 (°/s)   "
                f"X = {latest['AsX']:+7.1f}    "
                f"Y = {latest['AsY']:+7.1f}    "
                f"Z = {latest['AsZ']:+7.1f}",
                fontsize=10,
            )
        return ln_ax, ln_ay, ln_az, ln_gx, ln_gy, ln_gz

    ani = animation.FuncAnimation(   # noqa: F841（必须持有引用，防止 GC）
        fig, _update, interval=INTERVAL, blit=True, cache_frame_data=False
    )

    fig.canvas.mpl_connect("close_event", lambda _e: stop_event.set())
    plt.show()

# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None

    # 启动 BLE 线程
    t = threading.Thread(target=_ble_thread, args=(target,), daemon=True)
    t.start()

    print("等待 BLE 连接……")
    if not connected.wait(timeout=45):
        print("[超时] 未能在 45 秒内完成连接，退出。")
        sys.exit(1)

    # 检查是否真正连入（_ble_main 出错时也会 set connected）
    if not latest and stop_event.is_set():
        sys.exit(1)

    try:
        _start_plot()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("已退出。")
