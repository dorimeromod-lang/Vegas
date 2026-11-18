[Русский](README.md) | [English](README.en.md)

# Вегас — голосовой ассистент на ESP32-S3

DIY-ассистент, который живёт на полке: нажал кнопку, сказал фразу — получил голосовой ответ. Без облачных «умных колонок», без подписки и без прослушки: STT, LLM и TTS крутятся на твоём железе (или на твоём ключе DeepSeek).

```
┌──────────────┐   Wi-Fi / WebSocket    ┌─────────────────────────────┐
│  ESP32-S3    │ ─────────────────────> │  voice_server.py (FastAPI)  │
│  INMP441     │   PCM 16 kHz, int16    │  Sherpa / Vosk  →  текст    │
│  MAX98357A   │ <───────────────────── │  DeepSeek / Ollama → ответ  │
│  кнопка BOOT │   JSON + PCM 22 kHz    │  Piper TTS → голос          │
└──────────────┘                        └─────────────────────────────┘
```

## Что умеет

- Активация коротким нажатием **BOOT** (GPIO0), без wake-word.
- **Адаптивный VAD**: измеряет фон комнаты, сам подбирает порог речи и тишины.
- Запись завершается через **500 мс тишины** (300 мс при почти полной тишине), но не раньше чем через 500 мс после начала речи — естественные паузы не режутся.
- Пре-ролл ~320 мс: первые слова не теряются.
- Ответ приходит звуком через MAX98357A, параллельно сервер отдаёт текст.
- **Deep sleep** через 2 минуты простоя, пробуждение кнопкой BOOT.
- **OTA** по Wi-Fi (ArduinoOTA, порт 3232/UDP).
- Диагностика по Wi-Fi: `GET /status` и `GET /log` (последние ~4000 символов VAD-лога).
- Погода: met.no + ipwho.is, без API-ключей; поддерживаются гранулярные вопросы («а завтра?», «а вечером?»).
- Болтовня (кроме погоды): быстрые правила (время, дата, анекдот) + LLM DeepSeek / Ollama для всего остального.
- Голосовая перезагрузка сервера: фраза «перезагрузи сервер» → сервер отвечает и перезапускает сам себя.
- Привет («привет», «здравствуй» и т.п.) уходит в **LLM**; погода — в **быстрые правила**.

## Железо

| Компонент | Назначение | Подключение |
|---|---|---|
| ESP32-S3-DevKitC-1 **N16R8** | мозг, 16 МБ flash, 8 МБ PSRAM | — |
| INMP441 | микрофон I2S | SCK → GPIO4, WS → GPIO5, SD → GPIO6, L/R → GND, VDD → 3.3V |
| MAX98357A | усилитель + динамик | BCLK → GPIO15, LRC → GPIO16, DIN → GPIO17, VIN → 5V |
| Кнопка BOOT | активация / пробуждение | GPIO0 → GND (встроена в плату) |
| RGB LED | индикация состояния | GPIO48 (встроен) |
| БП 5 В ≥ 1 А | питание | 5V / GND |

> Питание подавать на **5V/GND**. На `3.3V` питание не подавать. Если корпус закрывает USB — оставь сервисный доступ или используй OTA.

## Прошивка

```bash
cd firmware
cp include/secrets.example.h include/secrets.h
# заполни Wi-Fi, SERVER_HOST, OTA_PASSWORD

~/.platformio/penv/bin/pio run -e esp32-s3-devkitc-1
~/.platformio/penv/bin/pio run -e esp32-s3-devkitc-1 -t upload --upload-port /dev/ttyACM0
```

`secrets.h` в `.gitignore` — в репозиторий он не попадёт.

### OTA

```bash
export OTA_HOST=192.168.1.100
export OTA_PASSWORD=твой-пароль
~/.platformio/penv/bin/pio run -e esp32-s3-ota -t upload
```

Устройство должно бодрствовать (нажми BOOT, если оно уже ушло в сон). Порт **3232 — UDP**, поэтому TCP-проверки (`nc -z`) бессмысленны. Первая попытка после загрузки иногда падает на аутентификации — повторная обычно проходит.

### Важно

- **Никогда** не запускай `pio run -t uploadfs` — это стирает LittleFS.
- Разметка `partitions.csv`: два OTA-слота по 3 МБ (`app0` / `app1`).
- Serial идёт в UART0 (GPIO43/44), читается через CH343 как `/dev/ttyACM0`, 115200.

## Сервер

Нужен Python 3.10–3.12 и ~2 ГБ RAM (модели грузятся в память). В Python 3.13 убрали модуль `audioop`, который использует `tools/test_wake_server.py`.

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# впиши DEEPSEEK_API_KEY или настрой Ollama
python voice_server.py
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
```

### Модели

Разложи по путям (или переопредели `MODEL_DIR` / `PIPER_MODEL` в `.env`):

```
server/
├── models/
│   ├── vosk-model-small-ru-0.22/     # wake-word / fallback Vosk
│   ├── vosk-model-ru-0.54-onnx/      # Sherpa Big Russian
│   └── sherpa-en-zipformer/          # Sherpa Zipformer EN
└── piper/
    ├── ru_RU-denis-medium.onnx
    └── ru_RU-denis-medium.onnx.json
```

- Vosk-модели: <https://alphacephei.com/vosk/models>
- Sherpa-onnx: <https://github.com/k2-fsa/sherpa-onnx/releases>
- Голоса Piper: <https://huggingface.co/rhasspy/piper-voices>

Модели и голоса **не коммитятся** — они большие и у них свои лицензии.

### systemd

```bash
sudo cp server/systemd/voice-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voice-server
```

Поправь пути и пользователя в юните под свою установку.

## Протокол

- ESP → сервер: бинарные блоки PCM (16 кГц, моно, int16), затем текст `END_OF_SPEECH`.
- Сервер → ESP: JSON с текстом ответа, затем бинарный PCM (22 050 Гц, int16), затем `END_OF_AUDIO`.
- `GET /health` — состояние сервера, режим LLM, наличие моделей.
- `GET /status` на ESP — версия, состояние, RSSI, heap, время до сна.
- `GET /log` на ESP — VAD-лог в текстовом виде.

## VAD

Параметры в `firmware/include/config.h`:

| Параметр | Значение | Смысл |
|---|---|---|
| `MIC_GAIN` | 8 | усиление микрофона (итог после серии VAD/микрофонных правок, было 4) |
| `VAD_START_RMS` | 500 | нижняя граница порога речи |
| `VAD_SILENCE_RMS` | 300 | нижняя граница порога тишины |
| `VAD_SILENCE_MS` | 500 | сколько тишины ждать после речи |
| `VAD_SILENCE_FAST_MS` | 300 | быстрый стоп при почти полной тишине |
| `VAD_MIN_SPEECH_MS` | 500 | минимум речи до разрешённого стопа |
| `MAX_RECORD_MS` | 12000 | предохранитель |
| `VAD_START_TIMEOUT` | 6000 | если речь так и не началась |

Фон измеряется до писка, первые 100 мс после старта I2S пропускаются, DC-смещение берётся из реального блока. Порог речи = `max(500, фон × 1.6)`, порог тишины = `max(300, фон × 1.25)`.

## 3D-модели

В `hardware/3d/` лежат `vrgas.stl` и `vrgas1.stl` — корпус/держатель. Печатай как хочешь, но оставь доступ к USB или полагайся на OTA.

## Лицензия

MIT. Сторонние компоненты (ESP32 Arduino, WebSockets, ArduinoJson, NeoPixel, FastAPI, Vosk, Sherpa-onnx, Piper, Ollama) — под своими лицензиями.
