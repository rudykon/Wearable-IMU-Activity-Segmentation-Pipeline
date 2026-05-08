"""
WT9011DCL-BT50  实时 IMU 数据  Web 服务
========================================
启动：  python server.py
访问：  http://127.0.0.1:8765

可选参数：  python server.py AA:BB:CC:DD:EE:FF
  指定 MAC 地址时，启动后直接连接，无需在页面上扫描。
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Set

import bleak
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

PORT = 8765
NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
HERE = Path(__file__).parent
AUTO_CONNECT_ADDR = sys.argv[1] if len(sys.argv) > 1 else None

# ──────────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────────
ws_clients: Set[WebSocket] = set()

device_info: dict = {"connected": False, "name": "", "address": "", "error": ""}

_ble_task: Optional[asyncio.Task] = None
_connect_done: Optional[asyncio.Event] = None
_loop: Optional[asyncio.AbstractEventLoop] = None   # 由 _ble_run 在事件循环内赋值

_tmp: list[int] = []   # 帧字节缓冲（单线程 asyncio，无需加锁）

# ──────────────────────────────────────────────
# 55 61 帧解析
# ──────────────────────────────────────────────
def _s16(hi: int, lo: int) -> int:
    v = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v


def _on_notify(sender, data: bytearray) -> None:
    """BLE 通知回调（同步）。完整 20 字节帧调度为 asyncio 任务广播。"""
    global _tmp
    for b in data:
        _tmp.append(b)
        if len(_tmp) == 1 and _tmp[0] != 0x55:
            _tmp.clear(); continue
        if len(_tmp) == 2 and _tmp[1] not in (0x61, 0x71):
            _tmp.clear(); continue
        if len(_tmp) == 20:
            frame = _tmp[:]
            _tmp.clear()
            if frame[1] == 0x61 and _loop is not None:
                _loop.create_task(_send_frame(frame))


async def _send_frame(b: list[int]) -> None:
    msg = json.dumps({
        "t":    "data",
        "AccX": round(_s16(b[3],  b[2])  / 32768 * 16,   3),
        "AccY": round(_s16(b[5],  b[4])  / 32768 * 16,   3),
        "AccZ": round(_s16(b[7],  b[6])  / 32768 * 16,   3),
        "AsX":  round(_s16(b[9],  b[8])  / 32768 * 2000, 2),
        "AsY":  round(_s16(b[11], b[10]) / 32768 * 2000, 2),
        "AsZ":  round(_s16(b[13], b[12]) / 32768 * 2000, 2),
    })
    await _broadcast(msg)


async def _broadcast(msg: str) -> None:
    dead: Set[WebSocket] = set()
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_clients -= dead

# ──────────────────────────────────────────────
# BLE 连接任务
# ──────────────────────────────────────────────
async def _ble_run(addr: str) -> None:
    global device_info, _connect_done, _tmp, _loop
    _tmp = []
    _loop = asyncio.get_running_loop()
    try:
        device = await bleak.BleakScanner.find_device_by_address(addr, timeout=10)
        if device is None:
            device_info["error"] = f"未找到设备 {addr}"
            if _connect_done:
                _connect_done.set()
            return

        async with bleak.BleakClient(device) as client:
            await client.start_notify(NOTIFY_UUID, _on_notify)
            device_info.update(
                connected=True, name=device.name or addr,
                address=addr, error=""
            )
            if _connect_done:
                _connect_done.set()
            await _broadcast(json.dumps({"t": "status", **device_info}))
            # 持续保持连接，等待被取消
            await asyncio.Event().wait()

    except asyncio.CancelledError:
        pass
    except Exception as e:
        device_info.update(connected=False, error=str(e))
        if _connect_done:
            _connect_done.set()
        await _broadcast(json.dumps({"t": "status", **device_info}))
    finally:
        device_info["connected"] = False

# ──────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────
app = FastAPI()


@app.on_event("startup")
async def startup_auto_connect():
    """Connect automatically when a MAC address is supplied on the command line."""
    if not AUTO_CONNECT_ADDR:
        return
    global _ble_task, _connect_done
    _connect_done = asyncio.Event()
    _ble_task = asyncio.create_task(_ble_run(AUTO_CONNECT_ADDR))
    print(f"正在连接 {AUTO_CONNECT_ADDR} …")


@app.get("/")
async def root():
    return HTMLResponse((HERE / "index.html").read_text(encoding="utf-8"))


@app.get("/api/scan")
async def api_scan():
    """扫描约 10 秒，返回名称含 'WT' 的设备列表。"""
    raw = await bleak.BleakScanner.discover(timeout=10)
    found = [
        {"address": d.address, "name": d.name or "(无名)"}
        for d in raw if d.name and "WT" in d.name
    ]
    return JSONResponse({"devices": found})


class ConnectReq(BaseModel):
    addr: str


@app.post("/api/connect")
async def api_connect(req: ConnectReq):
    """连接指定 MAC，最长等待 15 秒。"""
    global _ble_task, _connect_done
    if _ble_task and not _ble_task.done():
        _ble_task.cancel()
        await asyncio.sleep(0.3)
    device_info.update(connected=False, name="", address="", error="")
    _connect_done = asyncio.Event()
    _ble_task = asyncio.create_task(_ble_run(req.addr))
    try:
        await asyncio.wait_for(_connect_done.wait(), timeout=15)
    except asyncio.TimeoutError:
        device_info["error"] = "连接超时"
    return JSONResponse({"ok": device_info["connected"], "device": device_info})


@app.post("/api/disconnect")
async def api_disconnect():
    """主动断开当前 BLE 连接。"""
    global _ble_task
    if _ble_task and not _ble_task.done():
        _ble_task.cancel()
        await asyncio.sleep(0.3)
    device_info.update(connected=False, name="", address="", error="")
    await _broadcast(json.dumps({"t": "status", **device_info}))
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    # 推送当前连接状态
    await ws.send_text(json.dumps({"t": "status", **device_info}))
    try:
        while True:
            await ws.receive_text()   # 保持心跳
    except WebSocketDisconnect:
        ws_clients.discard(ws)


# ──────────────────────────────────────────────
# 启动入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  WT9011DCL-BT50 实时 IMU 数据服务")
    print(f"  打开浏览器访问 → http://127.0.0.1:{PORT}\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
