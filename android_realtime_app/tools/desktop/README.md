# Desktop Debug Tools

These tools are optional and are not required for the Android app build.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Matplotlib collector

Scan for devices whose name contains `WT`:

```bash
python collect.py
```

Connect directly by MAC address:

```bash
python collect.py AA:BB:CC:DD:EE:FF
```

## Web dashboard

Start the local server:

```bash
python server.py
```

Open:

```text
http://127.0.0.1:8765
```

You can also auto-connect on startup:

```bash
python server.py AA:BB:CC:DD:EE:FF
```
