[Русский](README.md) | [English](README.en.md)

# Vegas — Voice Assistant on ESP32-S3

A DIY voice assistant that sits on your shelf: press a button, say something, get a spoken reply. No cloud smart speakers, no subscriptions, no eavesdropping by default — STT, LLM, and TTS run on your own hardware (or with your own DeepSeek API key).

```
┌──────────────┐   Wi-Fi / WebSocket    ┌─────────────────────────────┐
│  ESP32-S3    │ ─────────────────────> │  voice_server.py (FastAPI)  │
│  INMP441     │   PCM 16 kHz, int16    │  Sherpa / Vosk  →  text     │
│  MAX98357A   │ <───────────────────── │  DeepSeek / Ollama → reply  │
│  BOOT button │   JSON + PCM 22 kHz    │  Piper TTS → speech         │
└──────────────┘                        └─────────────────────────────┘
```

## Features

- Activated by a short press of **BOOT** (GPIO0), no wake word required.
- **Adaptive VAD**: measures room noise and adjusts speech/silence thresholds automatically.
- Recording stops after **500 ms of silence** (300 ms in very quiet conditions), but not earlier than 500 ms after speech starts — natural pauses are not cut off.
- ~320 ms pre-roll so the first words are not lost.
- Replies are played back through the MAX98357A speaker; the server also returns text.
- **Deep sleep** after 2 minutes of inactivity, wake-up via the BOOT button.
- **OTA** updates over Wi-Fi (ArduinoOTA, UDP port 3232).
- Wi-Fi diagnostics: `GET /status` and `GET /log` (last ~4000 characters of the VAD log).
- Weather via met.no + ipwho.is, no API keys required; supports granular questions (e.g. "tomorrow?", "in the evening?").
- Small talk (non-weather): quick rules (time, date, jokes) + LLM (DeepSeek / Ollama) for everything else.
- Voice server reboot: saying "restart the server" (RU: «перезагрузи сервер») makes the server confirm and restart itself.
- Greetings ("hi", "hello", etc.) go to the **LLM**; weather requests go to **fast rules**.

## Hardware

| Component | Purpose | Connection |
|---|---|---|
| ESP32-S3-DevKitC-1 **N16R8** | MCU, 16 MB flash, 8 MB PSRAM | — |
| INMP441 | I2S microphone | SCK → GPIO4, WS → GPIO5, SD → GPIO6, L/R → GND, VDD → 3.3V |
| MAX98357A | Amplifier + speaker | BCLK → GPIO15, LRC → GPIO16, DIN → GPIO17, VIN → 5V |
| BOOT button | Activation / wake-up | GPIO0 → GND (on-board) |
| RGB LED | Status indication | GPIO48 (on-board) |
| 5 V ≥ 1 A power supply | Power | 5V / GND |

> Power the board via **5V/GND**. Do not power it through `3.3V`. If the enclosure blocks USB access, keep a service access option or rely on OTA.

## Firmware

```bash
cd firmware
cp include/secrets.example.h include/secrets.h
# fill in Wi-Fi, SERVER_HOST, OTA_PASSWORD

~/.platformio/penv/bin/pio run -e esp32-s3-devkitc-1
~/.platformio/penv/bin/pio run -e esp32-s3-devkitc-1 -t upload --upload-port /dev/ttyACM0
```

`secrets.h` is in `.gitignore` and will not be committed.

### OTA

```bash
export OTA_HOST=192.168.1.100
export OTA_PASSWORD=your-password
~/.platformio/penv/bin/pio run -e esp32-s3-ota -t upload
```

The device must be awake (press BOOT if it has gone to sleep). Port **3232 is UDP**, so TCP checks (`nc -z`) are meaningless. The first attempt after boot may fail authentication — retrying usually succeeds.

### Important

- **Never** run `pio run -t uploadfs` — it erases LittleFS.
- Partition layout `partitions.csv`: two 3 MB OTA slots (`app0` / `app1`).
- Serial output goes to UART0 (GPIO43/44) and can be read via CH343 as `/dev/ttyACM0` at 115200 baud.

## Server

Requires Python 3.10–3.12 and ~2 GB RAM (models are loaded into memory). Python 3.13 removed the `audioop` module used by `tools/test_wake_server.py`.

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set DEEPSEEK_API_KEY or configure Ollama
python voice_server.py
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### Models

Place models in the following paths (or override `MODEL_DIR` / `PIPER_MODEL` in `.env`):

```
server/
├── models/
│   ├── vosk-model-small-ru-0.22/     # wake-word / Vosk fallback
│   ├── vosk-model-ru-0.54-onnx/      # Sherpa Big Russian
│   └── sherpa-en-zipformer/          # Sherpa Zipformer EN
└── piper/
    ├── ru_RU-denis-medium.onnx
    └── ru_RU-denis-medium.onnx.json
```

- Vosk models: <https://alphacephei.com/vosk/models>
- Sherpa-onnx: <https://github.com/k2-fsa/sherpa-onnx/releases>
- Piper voices: <https://huggingface.co/rhasspy/piper-voices>

Models and voices are **not committed** — they are large and have their own licenses.

### systemd

```bash
sudo cp server/systemd/voice-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voice-server
```

Adjust paths and the user in the unit file for your setup.

## Protocol

- ESP → server: binary PCM chunks (16 kHz, mono, int16), then the text `END_OF_SPEECH`.
- Server → ESP: JSON with the reply text, then binary PCM (22,050 Hz, int16), then `END_OF_AUDIO`.
- `GET /health` — server status, LLM mode, model availability.
- `GET /status` on ESP — firmware version, state, RSSI, heap, time until sleep.
- `GET /log` on ESP — VAD log in plain text.

## VAD

Parameters in `firmware/include/config.h`:

| Parameter | Value | Meaning |
|---|---|---|
| `MIC_GAIN` | 8 | microphone gain (final value after VAD/mic tuning, was 4) |
| `VAD_START_RMS` | 500 | lower bound for speech threshold |
| `VAD_SILENCE_RMS` | 300 | lower bound for silence threshold |
| `VAD_SILENCE_MS` | 500 | how long to wait for silence after speech |
| `VAD_SILENCE_FAST_MS` | 300 | fast stop in near-total silence |
| `VAD_MIN_SPEECH_MS` | 500 | minimum speech duration before stop is allowed |
| `MAX_RECORD_MS` | 12000 | safety limit |
| `VAD_START_TIMEOUT` | 6000 | timeout if speech never starts |

Noise floor is measured before the beep; the first 100 ms after I2S start are skipped, and DC offset is taken from a real audio block. Speech threshold = `max(500, noise × 1.6)`, silence threshold = `max(300, noise × 1.25)`.

## 3D Models

`hardware/3d/` contains `vrgas.stl` and `vrgas1.stl` (case/holder). Print as you like, but keep USB accessible or rely on OTA for updates.

## License

MIT. Third-party components (ESP32 Arduino, WebSockets, ArduinoJson, NeoPixel, FastAPI, Vosk, Sherpa-onnx, Piper, Ollama) are distributed under their respective licenses.
