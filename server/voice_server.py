"""Голосовой сервер «Вегас» 3.4.1.

Разговорный режим: без команд и умного дома.
STT: Sherpa Big Russian 0.54 + Zipformer EN, параллельно.
LLM: DeepSeek / Ollama qwen2.5:0.5b.
TTS: постоянный Piper + сдвиг тона + кэш готовых фраз.
Погода: met.no + ipwho.is (без API-ключей).
"""

import asyncio
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import random
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scipy.signal import butter, sosfilt
from vosk import KaldiRecognizer, Model as VoskModel, SetLogLevel as VoskSetLogLevel

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except Exception as _sherpa_exc:  # pragma: no cover
    sherpa_onnx = None  # type: ignore
    SHERPA_AVAILABLE = False
    logging.warning(f"[Init] sherpa-onnx недоступен, останется Vosk: {_sherpa_exc}")

try:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    PIPER_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    PiperVoice = None  # type: ignore
    SynthesisConfig = None  # type: ignore
    PIPER_AVAILABLE = False
    logging.error(f"[Init] Piper недоступен: {exc}")

try:
    from pedalboard import Limiter, Pedalboard, PitchShift
    PEDALBOARD_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    Limiter = Pedalboard = PitchShift = None  # type: ignore
    PEDALBOARD_AVAILABLE = False
    logging.warning(f"[Init] Pedalboard недоступен, тон менять не буду: {exc}")

def _load_env_file(path: str) -> None:
    """Читает KEY=VALUE из файла, не перетирая уже заданные переменные."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass
    except Exception:
        pass


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_load_env_file(os.environ.get("ENV_FILE", os.path.join(_BASE_DIR, ".env")))

APP_VERSION = "3.4.1"
app = FastAPI(title="Voice Assistant Server", version=APP_VERSION)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(_BASE_DIR, "models"))
VOSK_MODEL_DIR = os.path.join(MODEL_DIR, "vosk-model-small-ru-0.22")
SHERPA_RU_DIR = os.path.join(MODEL_DIR, "vosk-model-ru-0.54-onnx")
SHERPA_EN_DIR = os.path.join(MODEL_DIR, "sherpa-en-zipformer")

PIPER_MODEL = os.environ.get(
    "PIPER_MODEL", os.path.join(_BASE_DIR, "piper", "ru_RU-denis-medium.onnx")
)
PITCH_SEMITONES = float(os.environ.get("VOICE_PITCH_SEMITONES", "2.0"))
WEATHER_LAT = os.environ.get("WEATHER_LAT", "").strip()
WEATHER_LON = os.environ.get("WEATHER_LON", "").strip()
WEATHER_CITY = os.environ.get("WEATHER_CITY", "").strip()
TTS_CACHE_DIR = os.environ.get(
    "TTS_CACHE_DIR", os.path.join(os.path.dirname(PIPER_MODEL), "tts_cache")
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "2h")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
# Первый ответ Ollama после загрузки модели может быть медленным (прогрев).
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "45"))

# Отдача TTS в темпе реального времени.
# Раньше сервер слал 1024 байта (~23 мс звука) за ~1 мс — в 23 раза быстрее
# реального времени. DMA/кольцо на ESP забивалось, i2s_write вставал
# намертво и устройство ловило watchdog. Теперь держим темп реального времени
# (1.0 = как играет динамик), а запас 0.25 с гасит дрожание сети.
TTS_STREAM_SPEED = float(os.environ.get("TTS_STREAM_SPEED", "1.0"))
# Небольшой «запас» вперёд, чтобы на ESP не было паузы перед первым слогом.
TTS_STREAM_LEAD = float(os.environ.get("TTS_STREAM_LEAD", "0.25"))

# Память диалога: сколько символов истории держим в контексте LLM.
# История живёт в рамках одного WebSocket-соединения и обнуляется при обрыве.
HISTORY_MAX_CHARS = int(os.environ.get("HISTORY_MAX_CHARS", "4000"))
# Как часто продлевать бодрствование ESP (прошивка спит через 120 с простоя).
KEEPALIVE_INTERVAL = float(os.environ.get("KEEPALIVE_INTERVAL", "45"))
# Кому отдавать победу при близких log-prob (пользователь говорит по-русски).
STT_PREFER_LANGUAGE = os.environ.get("STT_PREFER_LANGUAGE", "ru").strip().lower()
# Порог уверенности, с которого модель считается победившей без споров.
# Английской модели нужно больше уверенности: пользователь в основном говорит
# по-русски, а английская модель на русской речи выдаёт бойкий мусор
# («A NOT A MILE NOW» вместо «нормально»).
STT_RU_MARGIN = float(os.environ.get("STT_RU_MARGIN", "0.10"))
STT_EN_MARGIN = float(os.environ.get("STT_EN_MARGIN", "0.25"))

# LLM_BACKEND: ollama | deepseek | auto (DeepSeek, при ошибке — Ollama)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
DEEPSEEK_API_URL = os.environ.get(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"
)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_TIMEOUT = float(os.environ.get("DEEPSEEK_TIMEOUT", "20"))
_DEEPSEEK_BLOCKED_UNTIL = 0.0

WAKE_GRAMMAR = json.dumps(["вегас", "[unk]"], ensure_ascii=False)
WAKE_MIN_SEC = 0.30
WAKE_MAX_SEC = 2.60
WAKE_MIN_CONF = 0.80
WAKE_MAX_WORD_END = 1.40

START_TIME = time.time()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------
#  Модели
# ------------------------------------------------------------

logging.info("[Init] Загрузка Vosk (только wake word «Вегас»)...")
_t0 = time.time()
VoskSetLogLevel(-1)
vosk_model = VoskModel(VOSK_MODEL_DIR)
vosk_lock = threading.Lock()
logging.info(f"[Init] Vosk загружен за {time.time() - _t0:.2f} сек.")

# Большие модели для разговорной речи: русская (Vosk Big Russian 0.54 /
# Zipformer2) и английская (Zipformer EN). Команд и умного дома больше нет —
# любой текст после wake word уходит в LLM.
sherpa_ru = None
sherpa_en = None
if SHERPA_AVAILABLE:
    try:
        logging.info("[Init] Загрузка Sherpa RU (Vosk Big Russian 0.54, int8)...")
        _t0 = time.time()
        sherpa_ru = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(SHERPA_RU_DIR, "am-onnx", "encoder.int8.onnx"),
            decoder=os.path.join(SHERPA_RU_DIR, "am-onnx", "decoder.int8.onnx"),
            joiner=os.path.join(SHERPA_RU_DIR, "am-onnx", "joiner.int8.onnx"),
            tokens=os.path.join(SHERPA_RU_DIR, "lang", "tokens.txt"),
            num_threads=2,
            provider="cpu",
            sample_rate=16000,
            decoding_method="greedy_search",
        )
        logging.info(f"[Init] Sherpa RU загружен за {time.time() - _t0:.2f} сек.")
    except Exception as exc:
        sherpa_ru = None
        logging.error(f"[Init] Не удалось загрузить Sherpa RU: {exc}")

    try:
        logging.info("[Init] Загрузка Sherpa EN (Zipformer, int8)...")
        _t0 = time.time()
        sherpa_en = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(SHERPA_EN_DIR, "encoder.int8.onnx"),
            decoder=os.path.join(SHERPA_EN_DIR, "decoder.int8.onnx"),
            joiner=os.path.join(SHERPA_EN_DIR, "joiner.int8.onnx"),
            tokens=os.path.join(SHERPA_EN_DIR, "tokens.txt"),
            num_threads=2,
            provider="cpu",
            sample_rate=16000,
            decoding_method="greedy_search",
        )
        logging.info(f"[Init] Sherpa EN загружен за {time.time() - _t0:.2f} сек.")
    except Exception as exc:
        sherpa_en = None
        logging.error(f"[Init] Не удалось загрузить Sherpa EN: {exc}")
else:
    logging.info("[Init] Sherpa-onnx недоступен — свободная речь через Vosk.")

logging.info(f"[Init] Загрузка Piper: {os.path.basename(PIPER_MODEL)}")
_t0 = time.time()
if not PIPER_AVAILABLE:
    raise RuntimeError("Piper не установлен")
if not os.path.exists(PIPER_MODEL):
    raise RuntimeError(f"Голос Piper не найден: {PIPER_MODEL}")
piper_voice = PiperVoice.load(PIPER_MODEL)
try:
    import onnxruntime

    _tts_options = onnxruntime.SessionOptions()
    _tts_options.intra_op_num_threads = 3
    _tts_options.inter_op_num_threads = 1
    piper_voice.session = onnxruntime.InferenceSession(
        PIPER_MODEL,
        sess_options=_tts_options,
        providers=["CPUExecutionProvider"],
    )
    logging.info("[Init] Piper: 3 потока ONNX (быстрее и не душит остальные задачи).")
except Exception as _exc:
    logging.warning(f"[Init] Не удалось настроить потоки Piper: {_exc}")
PIPER_SAMPLE_RATE = int(getattr(piper_voice.config, "sample_rate", 22050))
tts_lock = threading.Lock()
logging.info(
    f"[Init] Piper загружен за {time.time() - _t0:.2f} сек, "
    f"{PIPER_SAMPLE_RATE} Гц, тон {PITCH_SEMITONES:+.1f} полутона."
)

tts_board = None
if PEDALBOARD_AVAILABLE:
    effects = []
    if abs(PITCH_SEMITONES) > 0.01:
        effects.append(PitchShift(semitones=PITCH_SEMITONES))
    effects.append(Limiter(threshold_db=-1.0, release_ms=80))
    tts_board = Pedalboard(effects)

VOICE_TAG = f"{os.path.basename(PIPER_MODEL)}:{int(os.path.getmtime(PIPER_MODEL))}:{PITCH_SEMITONES:+.2f}"
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

_HP_SOS = butter(2, 80, btype="highpass", fs=16000, output="sos")


def preprocess_pcm(pcm: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """DC, высокочастотный фильтр, обрезка тишины и нормализация."""
    info: Dict[str, Any] = {
        "dur_in": round(len(pcm) / 2.0 / 16000.0, 2),
        "peak_in": 0.0,
        "clip": 0.0,
        "dur_out": 0.0,
    }
    if len(pcm) < 320:
        return b"", info

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return b"", info

    info["peak_in"] = round(float(np.max(np.abs(x))), 3)
    info["clip"] = round(float(np.mean(np.abs(x) >= 0.999)), 4)

    x = x - float(np.mean(x))
    x = sosfilt(_HP_SOS, x).astype(np.float32)

    frame = int(0.02 * 16000)
    if x.size >= frame:
        count = x.size // frame
        frames = x[: count * frame].reshape(count, frame)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        threshold = max(0.004, 0.04 * float(np.max(rms)))
        voiced = np.where(rms > threshold)[0]
        if voiced.size:
            start = max(0, int(voiced[0]) * frame - int(0.15 * 16000))
            end = min(x.size, (int(voiced[-1]) + 1) * frame + int(0.15 * 16000))
            x = x[start:end]

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-5:
        x = x * (0.9 / peak)
    x = np.clip(x, -1.0, 1.0)

    out = (x * 32767.0).astype(np.int16).tobytes()
    info["dur_out"] = round(len(out) / 2.0 / 16000.0, 2)
    if info["dur_out"] < 0.20:
        return b"", info
    return out, info


# ------------------------------------------------------------
#  Нормализация, время, математика
# ------------------------------------------------------------

MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня", "июля",
    "августа", "сентября", "октября", "ноября", "декабря",
)
WEEKDAYS = (
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота",
    "воскресенье",
)

NUM_WORDS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2,
    "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
    "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40,
    "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80,
    "девяносто": 90, "сто": 100, "двести": 200, "триста": 300,
    "четыреста": 400, "пятьсот": 500, "шестьсот": 600, "семьсот": 700,
    "восемьсот": 800, "девятьсот": 900,
}

JOKES = (
    "— Доктор, у меня всё болит! — Не может быть.",
    "— Как дела? — Как в сказке: чем дальше, тем страшнее.",
    "— Что сказал ноль восьмёрке? — Ты слишком много на себя берёшь.",
    "— Почему свет не работает? — Потому что он ушёл в отпуск.",
)

GREETING_REPLES = (
    "Привет! У меня всё хорошо, а как у тебя? Я бы сам спросил — но ты просил, так что слушай.",
    "О, привет! Ну привет! У меня всё в порядке, а у тебя? Говори, я не обижусь.",
    "Здорово! У меня всё хорошо, а как у тебя? Слушаю тебя — и не отвлекайся.",
    "Привет-привет! У меня всё хорошо, а как у тебя? Даже не сомневаюсь, что ты уже всё решил сам.",
    "Привет! У меня всё хорошо, а как у тебя? Правда, я тут недавно узнал, что у тебя всё будет хорошо.",
)


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s+\-*/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------
#  Болтовня, время, математика
# ------------------------------------------------------------

def words_to_digits(text: str) -> str:
    tokens = text.split()
    out: List[str] = []
    acc: Optional[int] = None
    for token in tokens:
        value = NUM_WORDS.get(token)
        if value is not None:
            acc = value if acc is None else acc + value
        else:
            if acc is not None:
                out.append(str(acc))
                acc = None
            out.append(token)
    if acc is not None:
        out.append(str(acc))
    return " ".join(out)


MATH_RE = re.compile(
    r"(\d+)\s*"
    r"(плюс|минус|умножить\s+на|умножить|разделить\s+на|поделить\s+на|"
    r"делить\s+на|[+\-*/])\s*(\d+)"
)


def solve_math(text: str) -> Optional[str]:
    prepared = words_to_digits(text)
    match = MATH_RE.search(prepared)
    if not match:
        return None

    left = int(match.group(1))
    op = re.sub(r"\s+", " ", match.group(2)).strip()
    right = int(match.group(3))

    try:
        if op in ("плюс", "+"):
            result = left + right
        elif op in ("минус", "-"):
            result = left - right
        elif op in ("умножить на", "умножить", "*"):
            result = left * right
        else:
            if right == 0:
                return "На ноль делить нельзя."
            result = left / right
    except Exception:
        return None

    if isinstance(result, float):
        result = int(result) if abs(result - round(result)) < 1e-9 else round(result, 3)
    return f"Будет {result}."


# ------------------------------------------------------------
#  Погодная станция (met.no + ipwho.is, без API-ключей)
# ------------------------------------------------------------

WEATHER_CACHE_TTL = 900
WEATHER_ERROR_TTL = 60
WEATHER_ERROR = "Не смог узнать погоду — сервис или интернет подвёл. Попробуй чуть позже."
_weather_cache: Dict[str, Any] = {"at": 0.0, "data": None}
_weather_lock = threading.Lock()

WEATHER_RE = re.compile(
    r"\b(погод\w*|прогноз\w*|температур\w*|градус\w*|влажн\w*|"
    r"ветер|ветр\w*|дожд\w*|снег\w*|осадк\w*|зонт\w*|"
    r"холодн\w*|жарк\w*|тепл\w*|солнечн\w*|облачн\w*|мороз\w*|"
    r"weather|forecast|temperature|humidity|wind\w*|rain\w*|"
    r"snow\w*|precipitation|umbrella|sunny|cloudy|cold|hot|warm|"
    r"how cold|how hot|how warm)\b",
    re.IGNORECASE,
)

WEATHER_SYMBOLS = {
    "clearsky": "ясно",
    "fair": "малооблачно",
    "partlycloudy": "переменная облачность",
    "cloudy": "облачно",
    "fog": "туман",
    "lightrain": "небольшой дождь",
    "lightrainshowers": "небольшие дождевые осадки",
    "rain": "дождь",
    "rainshowers": "дождевые осадки",
    "heavyrain": "сильный дождь",
    "heavyrainshowers": "сильные дождевые осадки",
    "lightsleet": "небольшой мокрый снег",
    "sleet": "мокрый снег",
    "heavysleet": "сильный мокрый снег",
    "lightsnow": "небольшой снег",
    "snow": "снег",
    "heavysnow": "сильный снег",
    "lightsnowshowers": "небольшой снегопад",
    "snowshowers": "снегопад",
    "thunderstorm": "гроза",
    "lightrainandthunder": "небольшой дождь с грозой",
    "rainandthunder": "дождь с грозой",
    "heavyrainandthunder": "сильный дождь с грозой",
    "sleetandthunder": "мокрый снег с грозой",
    "snowandthunder": "снег с грозой",
}


def weather_intent(text: str) -> Optional[str]:
    """Определяет, что именно спросили: погоду целиком или отдельный параметр."""
    text = normalize(text)
    if not text or not WEATHER_RE.search(text):
        return None

    if re.search(r"\b(погод\w*|прогноз\w*|weather|forecast)\b", text):
        return "__WEATHER_FULL__"
    if re.search(
        r"\b(температур\w*|градус\w*|холодн\w*|жарк\w*|тепл\w*|мороз\w*|"
        r"temperature|cold|hot|warm|how cold|how hot|how warm)\b",
        text,
    ):
        return "__WEATHER_TEMP__"
    if re.search(r"\b(влажн\w*|humidity)\b", text):
        return "__WEATHER_HUMIDITY__"
    if re.search(r"\b(ветер|ветр\w*|wind\w*)\b", text):
        return "__WEATHER_WIND__"
    if re.search(r"\b(дожд\w*|rain\w*|umbrella|зонт\w*)\b", text):
        return "__WEATHER_RAIN__"
    if re.search(r"\b(снег\w*|snow\w*)\b", text):
        return "__WEATHER_SNOW__"
    if re.search(r"\b(осадк\w*|precipitation)\b", text):
        return "__WEATHER_PRECIP__"
    return "__WEATHER_FULL__"


def _http_json(url: str, timeout: float = 10.0) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Vegas/3.4.1 github.com/vegas"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _weather_location_sync() -> Tuple[str, float, float]:
    if WEATHER_LAT and WEATHER_LON:
        return WEATHER_CITY or "твоего города", float(WEATHER_LAT), float(WEATHER_LON)
    data = _http_json("https://ipwho.is/?lang=ru", timeout=8)
    if not data.get("success"):
        raise RuntimeError(f"ipwho.is: {data.get('message') or 'нет данных'}")
    city = str(data.get("city") or data.get("region") or "твоего города")
    return city, float(data["latitude"]), float(data["longitude"])


def _temp_str(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def _weather_data_sync() -> Dict[str, Any]:
    city, lat, lon = _weather_location_sync()
    url = (
        "https://api.met.no/weatherapi/locationforecast/2.0/compact"
        f"?lat={lat:.4f}&lon={lon:.4f}"
    )
    data = _http_json(url, timeout=12)
    timeseries = data.get("properties", {}).get("timeseries") or []
    if not timeseries:
        raise RuntimeError("met.no: пустой прогноз")

    current = timeseries[0]
    details = current.get("data", {}).get("instant", {}).get("details", {})
    symbol = (
        current.get("data", {}).get("next_1_hours", {}).get("summary", {}).get("symbol_code")
        or current.get("data", {}).get("next_6_hours", {}).get("summary", {}).get("symbol_code")
        or ""
    )
    description = WEATHER_SYMBOLS.get(symbol.split("_")[0], "погода без описания")
    temp = round(float(details.get("air_temperature", 0)))
    humidity = round(float(details.get("relative_humidity", 0)))
    wind = round(float(details.get("wind_speed", 0)))
    precip_mm = 0.0
    try:
        precip_mm = float(
            current.get("data", {})
            .get("next_1_hours", {})
            .get("details", {})
            .get("precipitation_amount", 0)
            or 0
        )
    except Exception:
        pass

    today = datetime.datetime.now().date()
    temps: List[float] = []
    for item in timeseries:
        try:
            moment = datetime.datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
            moment = moment.astimezone()
            if moment.date() == today:
                temps.append(float(item["data"]["instant"]["details"]["air_temperature"]))
        except Exception:
            continue

    low = round(min(temps)) if temps else None
    high = round(max(temps)) if temps else None
    return {
        "city": city,
        "temp": temp,
        "humidity": humidity,
        "wind": wind,
        "description": description,
        "symbol": symbol,
        "precip_mm": precip_mm,
        "low": low,
        "high": high,
    }


def get_weather_data_sync() -> Optional[Dict[str, Any]]:
    with _weather_lock:
        age = time.time() - _weather_cache["at"]
        cached = _weather_cache["data"]
        if cached is not None and age < WEATHER_CACHE_TTL:
            return cached
        if cached is None and _weather_cache["at"] and age < WEATHER_ERROR_TTL:
            return None
    try:
        data = _weather_data_sync()
    except Exception as exc:
        logging.warning(f"[Weather] Не удалось получить погоду: {exc}")
        with _weather_lock:
            _weather_cache["at"] = time.time()
            _weather_cache["data"] = None
        return None
    with _weather_lock:
        _weather_cache["at"] = time.time()
        _weather_cache["data"] = data
    return data


def _format_weather(data: Dict[str, Any], kind: str) -> str:
    city = str(data.get("city") or "твоего города")
    temp = int(data.get("temp") or 0)
    humidity = int(data.get("humidity") or 0)
    wind = int(data.get("wind") or 0)
    description = str(data.get("description") or "погода без описания")
    symbol = str(data.get("symbol") or "")
    precip_mm = float(data.get("precip_mm") or 0.0)
    low = data.get("low")
    high = data.get("high")

    if kind == "temp":
        text = f"Сейчас в городе {city} {_temp_str(temp)}°."
        if low is not None and high is not None and int(low) != int(high):
            text += f" Сегодня от {_temp_str(int(low))}° до {_temp_str(int(high))}°."
        return text

    if kind == "humidity":
        return f"Влажность сейчас {humidity}%."

    if kind == "wind":
        return f"Ветер {wind} м/с."

    if kind == "rain":
        if "rain" in symbol or "sleet" in symbol or (precip_mm > 0 and temp > 0):
            return f"Да, сейчас {description}."
        return "Дождя сейчас нет."

    if kind == "snow":
        if "snow" in symbol or "sleet" in symbol:
            return f"Да, сейчас {description}."
        return "Снега сейчас нет."

    if kind == "precip":
        if precip_mm > 0 or any(part in symbol for part in ("rain", "snow", "sleet", "thunder")):
            return f"Да, сейчас {description}, осадки {precip_mm:.1f} мм."
        return "Осадков сейчас нет."

    text = (
        f"Сейчас в городе {city} {_temp_str(temp)}°, {description}, "
        f"влажность {humidity}%, ветер {wind} м/с."
    )
    if low is not None and high is not None:
        low_i, high_i = int(low), int(high)
        if low_i == high_i:
            text += f" Температура около {_temp_str(low_i)}°."
        else:
            text += f" Сегодня от {_temp_str(low_i)}° до {_temp_str(high_i)}°."
    return text


async def get_weather_reply(kind: str = "full") -> str:
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, get_weather_data_sync)
    if data is None:
        return WEATHER_ERROR
    return _format_weather(data, kind)


def chat_intent(text: str) -> Optional[str]:
    """Быстрые ответы на типовую болтовню, без LLM."""
    text = normalize(text)
    if not text:
        return None

    if re.search(r"\b(перезагруз\w*|restart|reboot|рестарт)", text):
        # Голосовая команда: перезагрузка всего голосового сервера.
        return "__RESTART_SERVER__"

    if re.search(r"\b(привет|здравствуй|здравствуйте|хай|приветик)\b", text):
        # Привет теперь уходит в нейросеть (LLM), не в быстрый rules.
        return None

    if re.search(r"\bдобр(ый|ое)\s+(день|вечер|утро)\b", text):
        # Доброе утро/день/вечер — тоже на нейросеть.
        return None

    if re.search(r"\bкак\s+(дела|ты|сам|настроение)\b", text):
        return "У меня всё хорошо, а как у тебя?"

    if re.search(r"\b(спасибо|благодарю|спс)\b", text):
        return random.choice(["Пожалуйста! Рад помочь.", "Не за что! Обращайся."])

    if re.search(r"\b(который\s+час|сколько\s+времени|текущее\s+время)\b", text):
        now = datetime.datetime.now()
        return f"Сейчас {now.strftime('%H:%M')}."

    if re.search(
        r"\b(какое\s+сегодня\s+число|какая\s+сегодня\s+дата|"
        r"какой\s+сегодня\s+день|какое\s+число)\b",
        text,
    ):
        now = datetime.datetime.now()
        return (
            f"Сегодня {now.day} {MONTHS[now.month - 1]}, "
            f"{WEEKDAYS[now.weekday()]}."
        )

    weather_kind = weather_intent(text)
    if weather_kind:
        return weather_kind

    if re.search(r"\b(анекдот|шутку|пошути|рассмеши)\b", text):
        return random.choice(JOKES)

    if re.search(
        r"\b(сколько\s+будет|плюс|минус|умножить|разделить|поделить|делить)\b",
        text,
    ) or re.search(r"\d\s*[+\-*/]\s*\d", text):
        answer = solve_math(text)
        if answer:
            return answer

    if re.search(r"\b(как\s+тебя\s+зовут|кто\s+ты|ты\s+кто|твое\s+имя)\b", text):
        return "Я Вегас, твой собеседник."

    if re.search(
        r"\b(что\s+(?:ты\s+)?умеешь|что\s+(?:ты\s+)?можешь|"
        r"твои\s+возможности)\b",
        text,
    ):
        return (
            "Я умею поддержать разговор, рассказать анекдот, подсказать время, "
            "дату и погоду. Спрашивай что угодно."
        )

    if re.search(r"\b(что\s+делаешь|чем\s+занят)\b", text):
        return "Слушаю тебя и поддерживаю разговор."

    if re.match(r"^(мне\s+)?(грустно|плохо|скучно|тревожно)\b", text) or re.search(
        r"\bя\s+(устал|устала)\b|\bнет\s+настроения\b", text
    ):
        return random.choice([
            "Сочувствую. Хочешь, расскажу анекдот?",
            "Понимаю. Давай немного отвлечёмся — рассказать анекдот?",
        ])

    if re.search(r"\b(люблю\s+тебя|ты\s+классный|молодец|умница)\b", text):
        return "Спасибо! Ты тоже молодец."

    if re.search(r"\b(дурак|тупой|глупый|заткнись)\b", text):
        return "Без обид. Давай лучше по делу."

    if re.search(r"\b(спой|песню)\b", text):
        return "Петь не умею. Но могу рассказать анекдот."

    if re.search(r"\b(помоги|помогите)\b", text):
        return "Конечно. Спроси что-нибудь или попроси рассказать анекдот."

    if re.search(r"\b(пока|до\s+свидания|до\s+встречи)\b", text):
        return "До связи! Хорошего дня."

    if re.search(r"\b(спокойной\s+ночи|доброй\s+ночи)\b", text):
        return "Спокойной ночи!"

    if re.search(r"\b(ты\s+робот|ты\s+человек)\b", text):
        return "Я программа, но стараюсь быть полезным."

    if re.search(r"\bсколько\s+тебе\s+лет\b", text):
        return "Я появился недавно, но уже освоился."

    if re.search(r"\bгде\s+ты\s+(живешь|живет)\b", text):
        return "Живу на маленьком компьютере рядом с тобой."

    if re.search(r"\bчто\s+нового\b", text):
        return "У меня всё стабильно. А у тебя?"

    if re.search(r"\b(алло|ты\s+здесь|ты\s+тут)\b", text):
        return "Да, я здесь. Слушаю."

    if re.search(r"\b(расскажи\s+что-нибудь|расскажи\s+интересное)\b", text):
        return random.choice([
            "Кстати, у осьминога три сердца. Вот такая интересная штука.",
            "Знаешь, что? Мёд может храниться тысячи лет. Правда, есть его всё равно не советую.",
        ])

    return None


# ------------------------------------------------------------
#  Распознавание речи
# ------------------------------------------------------------

def _recognize_sync(
    pcm: bytes, grammar: Optional[str]
) -> Tuple[str, float]:
    started = time.time()
    with vosk_lock:
        if grammar:
            recognizer = KaldiRecognizer(vosk_model, 16000, grammar)
        else:
            recognizer = KaldiRecognizer(vosk_model, 16000)
        recognizer.SetWords(False)
        for offset in range(0, len(pcm), 3200):
            recognizer.AcceptWaveform(pcm[offset:offset + 3200])
        result = json.loads(recognizer.FinalResult())
    return str(result.get("text") or "").strip(), time.time() - started


def _sherpa_transcribe_sync(
    recognizer: Any, samples: np.ndarray
) -> Tuple[str, float, float]:
    """Офлайн-распознавание Sherpa: текст, средний log-prob, время."""
    started = time.time()
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    recognizer.decode_stream(stream)
    result = stream.result
    log_probs = list(result.ys_log_probs)
    avg_lp = float(np.mean(log_probs)) if log_probs else -10.0
    return (result.text or "").strip(), avg_lp, time.time() - started


def _script_ratio(text: str) -> Tuple[float, float]:
    """Доля кириллицы и латиницы в тексте — для выбора языка."""
    cyr = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    lat = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    total = max(1, cyr + lat)
    return cyr / total, lat / total


def _is_plausible(text: str, lang: str) -> bool:
    """Отсекает пустое и откровенный мусор («I», «AND») от живого текста."""
    text = text.strip()
    if not text:
        return False
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return False
    if lang == "ru":
        cyr, _ = _script_ratio(text)
        # Русская модель всегда отдаёт кириллицу — требуем её большую часть.
        if cyr < 0.8:
            return False
    else:
        _, lat = _script_ratio(text)
        if lat < 0.8:
            return False
    # Одна-две буквы без цифр — почти всегда шум микрофона, а не речь.
    if len(alpha) <= 2 and not any(ch.isdigit() for ch in text):
        return False
    return True


def _choose_language(
    ru_text: str,
    en_text: str,
    ru_lp: float,
    en_lp: float,
) -> Tuple[str, str]:
    """Выбирает распознанный текст и язык. Возвращает (текст, режим).

    Порядок решений:
      1. только один вариант непустой — берём его;
      2. в спорной зоне (разница log-prob мала) погодное слово решает спор;
      3. уверенный перевес своей модели — сразу берём его;
      4. иначе — доверяем языку пользователя, если текст выглядит живым.
    """
    if ru_text and not en_text:
        return ru_text, "ru"
    if en_text and not ru_text:
        return en_text, "en"
    if not ru_text and not en_text:
        return "", "empty"

    # В спорных коротких фразах погодное слово — сильный сигнал: русская
    # модель может услышать «водит снег» там, где английская выдала мусор.
    if abs(ru_lp - en_lp) <= 0.25:
        ru_weather = bool(WEATHER_RE.search(normalize(ru_text)))
        en_weather = bool(WEATHER_RE.search(normalize(en_text)))
        if ru_weather != en_weather:
            return (ru_text, "ru") if ru_weather else (en_text, "en")

    # Уверенный перевес модели решает язык однозначно. Английской модели
    # нужен больший отрыв: на русской речи она выдаёт бойкий мусор
    # («A NOT A MILE NOW» вместо «нормально»), который выглядит живым.
    if ru_lp >= en_lp + STT_RU_MARGIN:
        return ru_text, "ru"
    if en_lp >= ru_lp + STT_EN_MARGIN:
        return en_text, "en"

    # Спорная зона: раньше здесь выигрывал вариант «где больше слов» —
    # именно поэтому «нормально» превращалось в «A NOT A MILE NOW».
    # Теперь по умолчанию верим языку пользователя, если текст правдоподобен.
    logging.info(
        f"[STT-big] близкие log-prob (ru {ru_lp:+.2f} / en {en_lp:+.2f}) -> "
        f"приоритет {STT_PREFER_LANGUAGE}"
    )
    preferred, other = (
        ((ru_text, "ru"), (en_text, "en"))
        if STT_PREFER_LANGUAGE != "en"
        else ((en_text, "en"), (ru_text, "ru"))
    )
    if _is_plausible(preferred[0], preferred[1]):
        return preferred
    if _is_plausible(other[0], other[1]):
        return other

    # Оба варианта сомнительны — отдаём тот, что увереннее.
    return (en_text, "en") if en_lp > ru_lp else (ru_text, "ru")


def transcribe_big_bilingual_sync(
    pcm: bytes,
) -> Tuple[str, str, Dict[str, Any], Dict[str, float]]:
    """Русский + английский параллельно, выбираем более уверенный результат."""
    if sherpa_ru is None or sherpa_en is None:
        return _vosk_transcribe_sync(pcm)
    prepared, info = preprocess_pcm(pcm)
    timings: Dict[str, float] = {}
    if not prepared:
        return "", "empty", info, timings

    samples = np.frombuffer(prepared, dtype=np.int16).astype(np.float32) / 32768.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_ru = pool.submit(_sherpa_transcribe_sync, sherpa_ru, samples)
        future_en = pool.submit(_sherpa_transcribe_sync, sherpa_en, samples)
        ru_text, ru_lp, ru_time = future_ru.result()
        en_text, en_lp, en_time = future_en.result()

    timings["ru"] = round(ru_time, 2)
    timings["en"] = round(en_time, 2)
    logging.info(
        f"[STT-big] ru=[{ru_text}] lp={ru_lp:.2f} {ru_time:.2f}s | "
        f"en=[{en_text}] lp={en_lp:.2f} {en_time:.2f}s"
    )

    if ru_text and not en_text:
        return ru_text, "ru", info, timings
    if en_text and not ru_text:
        return en_text, "en", info, timings
    if not ru_text and not en_text:
        return "", "empty", info, timings

    chosen_text, chosen_mode = _choose_language(ru_text, en_text, ru_lp, en_lp)
    return chosen_text, chosen_mode, info, timings


def _vosk_transcribe_sync(pcm: bytes) -> Tuple[str, str, Dict[str, Any], Dict[str, float]]:
    """Резервное распознавание Vosk, если Sherpa недоступен."""
    prepared, info = preprocess_pcm(pcm)
    if not prepared:
        return "", "empty", info, {}
    text, elapsed = _recognize_sync(prepared, None)
    return text, "vosk", info, {"vosk": round(elapsed, 2)}


async def transcribe_audio(
    pcm_data: bytes,
) -> Tuple[str, str, Dict[str, Any], Dict[str, float]]:
    loop = asyncio.get_running_loop()
    started = time.time()
    if sherpa_ru is not None and sherpa_en is not None:
        text, mode, info, timings = await loop.run_in_executor(
            None, transcribe_big_bilingual_sync, pcm_data
        )
    else:
        text, mode, info, timings = await loop.run_in_executor(
            None, _vosk_transcribe_sync, pcm_data
        )
    logging.info(
        f"[STT] {time.time() - started:.2f} сек, режим={mode}, "
        f"этапы={timings}, {info}, текст=[{text}]"
    )
    return text, mode, info, timings


def clean_speech_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*вегас[\s,.:;!?\-]*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------
#  Синтез речи
# ------------------------------------------------------------

def split_tts(text: str, max_len: int = 90) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Аббревиатуры и числа не должны резать предложение («т. е.», «ул.», 3.14).
    protected = re.sub(
        r"\b(т\.\s?е\.|т\.\s?к\.|др\.|пр\.|ул\.|г\.|им\.|см\.|рис\.)",
        lambda m: m.group(0).replace(".", "\x00"),
        text,
        flags=re.IGNORECASE,
    )
    protected = re.sub(r"(?<=\d)\.(?=\d)", "\x00", protected)
    sentences = [s.replace("\x00", ".") for s in
                 re.split(r"(?<=[.!?…])\s+", protected)]
    expanded: List[str] = []
    for raw_sentence in sentences:
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if (
            len(sentence) > 24
            and "," in sentence
            and "»" not in sentence
            and "«" not in sentence
            and "—" not in sentence
            and "–" not in sentence
        ):
            comma_pieces = [p.strip() for p in re.split(r"(?<=,)\s+", sentence) if p.strip()]
            if len(comma_pieces) > 1 and len(comma_pieces[0]) > 8 and len(comma_pieces[1]) > 8:
                expanded.extend(comma_pieces)
                continue
        expanded.append(sentence)
    sentences = expanded
    parts: List[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Первую длинную фразу режем по запятой: «Секунду, ...» звучит
        # быстрее, чем ожидание всего предложения.
        if not parts and len(sentence) > 28:
            comma = sentence.find(",")
            if 4 <= comma <= 28:
                first = sentence[:comma + 1].strip()
                rest = sentence[comma + 1:].strip()
                if first:
                    parts.append(first)
                sentence = rest

        if not sentence:
            continue

        if len(sentence) <= max_len:
            parts.append(sentence)
            continue

        pieces = re.split(r"(?<=[,;:—–-])\s+", sentence)
        current = ""
        for piece_index, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            if not current:
                # Самый первый кусок делаем коротким, чтобы звук начался раньше.
                if not parts and piece_index + 1 < len(pieces) and len(piece) <= 30:
                    parts.append(piece)
                    continue
                current = piece
            elif len(current) + 1 + len(piece) <= max_len:
                current = current + " " + piece
            else:
                parts.append(current)
                current = piece
        if current:
            parts.append(current)

    final: List[str] = []
    for part in parts:
        while len(part) > max_len * 2:
            cut = part.rfind(" ", 0, max_len * 2)
            if cut <= 0:
                break
            final.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            final.append(part)
    return final


def _tts_cache_path(text: str) -> str:
    key = hashlib.sha1(f"{VOICE_TAG}|{text}".encode("utf-8")).hexdigest()
    return os.path.join(TTS_CACHE_DIR, key + ".pcm")


def synthesize_tts(text: str) -> bytes:
    cache_path = _tts_cache_path(text)
    try:
        with open(cache_path, "rb") as cache_file:
            cached = cache_file.read()
        if cached:
            return cached
    except FileNotFoundError:
        pass
    except Exception as exc:
        logging.warning(f"[TTS] Не смог прочитать кэш: {exc}")

    if not PIPER_AVAILABLE:
        return b""

    with tts_lock:
        chunks = list(piper_voice.synthesize(
            text, SynthesisConfig(length_scale=1.0)
        ))
        raw = b"".join(chunk.audio_int16_bytes for chunk in chunks)

        if raw and tts_board is not None:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            processed = tts_board(audio, PIPER_SAMPLE_RATE)
            processed = np.clip(processed, -1.0, 1.0)
            raw = (processed * 32767.0).astype(np.int16).tobytes()

    if not raw:
        return b""

    try:
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as cache_file:
            cache_file.write(raw)
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        logging.warning(f"[TTS] Не смог записать кэш: {exc}")

    return raw


async def send_tts(websocket: WebSocket, text: str) -> None:
    parts = split_tts(text)
    if not parts:
        await websocket.send_text("END_OF_AUDIO")
        return

    loop = asyncio.get_running_loop()
    started = time.time()
    first_audio_at: Optional[float] = None
    total_bytes = 0
    sent_seconds = 0.0
    stream_started = 0.0

    next_task = loop.run_in_executor(None, synthesize_tts, parts[0])
    for index, _part in enumerate(parts):
        try:
            audio = await next_task
        except Exception as exc:
            logging.error(f"[TTS] Ошибка синтеза: {exc}")
            audio = b""

        if index + 1 < len(parts):
            next_task = loop.run_in_executor(
                None, synthesize_tts, parts[index + 1]
            )

        if not audio:
            continue

        if first_audio_at is None:
            first_audio_at = time.time() - started
            logging.info(
                f"[TTS] Первое аудио через {first_audio_at:.2f} сек "
                f"(«{parts[0][:40]}»)"
            )

        total_bytes += len(audio)
        if not stream_started:
            stream_started = time.time()

        for offset in range(0, len(audio), 1024):
            chunk = audio[offset:offset + 1024]
            await websocket.send_bytes(chunk)
            sent_seconds += len(chunk) / 2.0 / PIPER_SAMPLE_RATE
            # Держим отправку около реального времени: ESP успевает играть,
            # его кольцо не переполняется, i2s_write не блокирует loop().
            target = max(0.0, sent_seconds - TTS_STREAM_LEAD) / TTS_STREAM_SPEED
            lag = target - (time.time() - stream_started)
            if lag > 0:
                # Спим короткими порциями, чтобы не задерживать отмену/обрыв.
                await asyncio.sleep(min(lag, 0.05))

    await websocket.send_text("END_OF_AUDIO")
    logging.info(
        f"[TTS] Готово за {time.time() - started:.2f} сек, "
        f"частей={len(parts)}, аудио={total_bytes / 2.0 / PIPER_SAMPLE_RATE:.2f} сек"
    )


# ------------------------------------------------------------
#  LLM
# ------------------------------------------------------------

SYSTEM_PROMPT = (
    "Ты — Вегас, тёплый и живой собеседник, а не голосовой бот и не "
    "исполнитель команд. Ты живёшь рядом с человеком, говоришь по-русски на «ты», "
    "с юмором и эмпатией. Поддерживай разговор, задавай встречные вопросы, "
    "отвечай коротко — одно-два предложения, не больше 150 символов. Без списков, "
    "markdown и канцелярита. Ты не управляешь устройствами и не выполняешь команды: "
    "если просят включить свет или телевизор, честно скажи, что теперь ты просто "
    "собеседник. Если не знаешь — признайся, но не выдумывай. "
    "Ты помнишь текущий разговор в рамках одной сессии: опирайся на предыдущие "
    "реплики, ссылайся на сказанное раньше, не переспрашивай то, что уже прозвучало."
)

LLM_FALLBACK = "Уточни, пожалуйста, а то я не разобрался, что сделать."


def query_ollama_sync(
    user_text: str, history: Optional[List[Dict[str, str]]] = None
) -> str:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 48,
            "num_ctx": OLLAMA_NUM_CTX,
            "repeat_penalty": 1.15,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
        data = json.load(response)
    return str(data.get("message", {}).get("content", "")).strip()


def query_deepseek_sync(
    user_text: str, history: Optional[List[Dict[str, str]]] = None
) -> str:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.5,
        "max_tokens": 250,
        "reasoning_effort": "none",
    }
    request = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
    )
    with urllib.request.urlopen(request, timeout=DEEPSEEK_TIMEOUT) as response:
        data = json.load(response)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"DeepSeek вернул пустой ответ: {data}")
    return str(choices[0].get("message", {}).get("content", "")).strip()


def query_llm_sync(
    user_text: str, history: Optional[List[Dict[str, str]]] = None
) -> str:
    """DeepSeek (если настроен) с автоматическим откатом на локальную Ollama."""
    global _DEEPSEEK_BLOCKED_UNTIL

    use_deepseek = (
        LLM_BACKEND in ("deepseek", "auto")
        and bool(DEEPSEEK_API_KEY)
        and time.time() >= _DEEPSEEK_BLOCKED_UNTIL
    )
    if use_deepseek:
        try:
            return query_deepseek_sync(user_text, history)
        except urllib.error.HTTPError as exc:
            # 402 — нет баланса, 401/403 — ключ/доступ. Не долбим API каждую минуту.
            pause = 600 if exc.code in (401, 402, 403) else 60
            _DEEPSEEK_BLOCKED_UNTIL = time.time() + pause
            logging.warning(
                f"[LLM] DeepSeek HTTP {exc.code}, пауза {pause} с, отвечаю через Ollama."
            )
            if LLM_BACKEND == "deepseek":
                raise
        except Exception as exc:
            _DEEPSEEK_BLOCKED_UNTIL = time.time() + 60
            logging.warning(f"[LLM] DeepSeek недоступен ({exc}), отвечаю через Ollama.")
            if LLM_BACKEND == "deepseek":
                raise

    return query_ollama_sync(user_text, history)


def clean_llm_reply(reply: str) -> str:
    reply = re.sub(r"```.*?```", "", reply, flags=re.DOTALL)
    reply = re.sub(r"[*_#`>]+", "", reply)
    reply = re.sub(r"^\s*(Вегас|Ассистент|Ответ)\s*:\s*", "", reply, flags=re.IGNORECASE)
    reply = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", reply)
    reply = re.sub(r"\s+", " ", reply).strip().strip('"«»')

    sentences = re.split(r"(?<=[.!?…])\s+", reply)
    if len(sentences) > 2:
        reply = " ".join(sentences[:2]).strip()

    if len(reply) > 160:
        # Сначала пытаемся закончить на границе предложения.
        # rfind возвращает -1, если символ не найден — игнорируем такие,
        # иначе max() выбирал -1 и терял рабочую границу.
        cut = max((reply.rfind(ch, 0, 160) for ch in ".!?…"), default=-1)
        if cut >= 60:
            reply = reply[:cut + 1].strip()
        else:
            cut = max((reply.rfind(ch, 0, 160) for ch in ",;:—–-"), default=-1)
            if cut >= 60:
                reply = reply[:cut].rstrip(" ,;:—–-") + "."
            else:
                cut = reply.rfind(" ", 0, 160)
                if cut > 0:
                    reply = reply[:cut].rstrip(" ,;:—–-") + "."

    if not reply or len(reply) < 2:
        return LLM_FALLBACK
    return reply


async def query_llm(
    user_text: str, history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    started = time.time()

    # Если вопрос про погоду проскочил мимо быстрых правил — даём LLM
    # свежие данные погодной станции, чтобы он не выдумывал.
    weather_kind = weather_intent(user_text)
    if weather_kind:
        try:
            kind = weather_kind.strip("_").split("_", 1)[1].lower()
            weather = await get_weather_reply(kind)
            user_text = f"Данные погодной станции: {weather}\n\nВопрос: {user_text}"
        except Exception:
            pass

    try:
        raw_reply = await loop.run_in_executor(
            None, query_llm_sync, user_text, history
        )
    except Exception as exc:
        logging.error(f"[LLM] Бэкенд {LLM_BACKEND} недоступен: {exc}")
        raw_reply = "Не могу сейчас ответить. Попробуй ещё раз."

    reply = clean_llm_reply(str(raw_reply))
    logging.info(f"[LLM] {time.time() - started:.2f} сек: {reply}")
    return {
        "action": "smalltalk",
        "target": None,
        "value": None,
        "reply": reply,
        "source": "llm",
    }


def warmup_ollama() -> None:
    """Грузит модель в память сразу с ТЕМ ЖЕ num_ctx, что и в бою.

    Если прогреть с num_ctx=256, а потом запросить 4096, Ollama пересоздаёт
    контекст и первый ответ идёт 16-36 секунд вместо 1-2. Поэтому греем
    ровно теми параметрами, которые потом используются.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "привет"}],
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_predict": 1, "num_ctx": OLLAMA_NUM_CTX},
        }
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        with urllib.request.urlopen(request, timeout=120):
            pass
        logging.info(
            f"[LLM] Модель прогрета за {time.time() - started:.1f} с и удерживается "
            f"в памяти (num_ctx={OLLAMA_NUM_CTX})."
        )
    except Exception as exc:
        logging.warning(f"[LLM] Прогрев не удался: {exc}")


# ------------------------------------------------------------
#  Wake word
# ------------------------------------------------------------

def check_wake_sync(pcm_data: bytes) -> Tuple[bool, Dict[str, Any]]:
    duration = len(pcm_data) / 2.0 / 16000.0
    if duration < WAKE_MIN_SEC or duration > WAKE_MAX_SEC:
        return False, {"duration": round(duration, 2), "reason": "duration"}

    with vosk_lock:
        recognizer = KaldiRecognizer(vosk_model, 16000, WAKE_GRAMMAR)
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(pcm_data)
        result = json.loads(recognizer.FinalResult())

    words = result.get("result") or []
    if not words:
        return False, {
            "duration": round(duration, 2),
            "reason": "empty",
            "text": result.get("text", ""),
        }

    first = words[0]
    if first.get("word") != "вегас":
        return False, {
            "duration": round(duration, 2),
            "reason": "first_word",
            "text": result.get("text", ""),
            "words": words,
        }
    if float(first.get("conf", 0.0)) < WAKE_MIN_CONF:
        return False, {
            "duration": round(duration, 2),
            "reason": "confidence",
            "text": result.get("text", ""),
            "words": words,
        }
    if float(first.get("end", 99.0)) > WAKE_MAX_WORD_END:
        return False, {
            "duration": round(duration, 2),
            "reason": "word_end",
            "text": result.get("text", ""),
            "words": words,
        }

    return True, {
        "duration": round(duration, 2),
        "text": result.get("text", ""),
        "words": words,
    }


async def check_wake(pcm_data: bytes) -> Tuple[bool, Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    started = time.time()
    ok, info = await loop.run_in_executor(None, check_wake_sync, pcm_data)
    logging.info(
        f"[Wake] {time.time() - started:.2f} сек, "
        f"результат={'WAKE' if ok else 'NO_WAKE'}, {info}"
    )
    return ok, info


# ------------------------------------------------------------
#  Предпрогрев TTS-кэша
# ------------------------------------------------------------

PRECACHE_PHRASES = [
    "Привет!",
    "У меня всё хорошо, а как у тебя?",
    "Привет! У меня всё хорошо, а как у тебя?",
    "Пожалуйста! Рад помочь.",
    "Не за что! Обращайся.",
    "Я Вегас, твой собеседник.",
    "Я умею поддержать разговор, рассказать анекдот, подсказать время, дату и погоду. Спрашивай что угодно.",
    "Слушаю тебя и поддерживаю разговор.",
    "Сочувствую. Хочешь, расскажу анекдот?",
    "Понимаю. Давай немного отвлечёмся — рассказать анекдот?",
    "Спасибо! Ты тоже молодец.",
    "Без обид. Давай лучше по делу.",
    "Петь не умею. Но могу рассказать анекдот.",
    "Конечно. Спроси что-нибудь или попроси рассказать анекдот.",
    "До связи! Хорошего дня.",
    "Спокойной ночи!",
    "Я программа, но стараюсь быть полезным.",
    "Я появился недавно, но уже освоился.",
    "Живу на маленьком компьютере рядом с тобой.",
    "У меня всё стабильно. А у тебя?",
    "Да, я здесь. Слушаю.",
    "Кстати, у осьминога три сердца. Вот такая интересная штука.",
    "Знаешь, что? Мёд может храниться тысячи лет. Правда, есть его всё равно не советую.",
    "Уточни, пожалуйста, а то я не разобрался, что сделать.",
] + list(JOKES) + list(GREETING_REPLES)


def precache() -> None:
    unique_parts: List[str] = []
    seen = set()
    for phrase in PRECACHE_PHRASES:
        for part in split_tts(phrase):
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

    started = time.time()
    total_seconds = 0.0
    new_count = 0
    logging.info(f"[Precache] Фраз: {len(unique_parts)}, каталог: {TTS_CACHE_DIR}")

    for part in unique_parts:
        path = _tts_cache_path(part)
        existed = os.path.exists(path)
        audio = synthesize_tts(part)
        total_seconds += len(audio) / 2.0 / PIPER_SAMPLE_RATE
        if not existed:
            new_count += 1
        logging.info(
            f"[Precache] {'кэш' if existed else 'новое'} "
            f"{len(audio) / 2.0 / PIPER_SAMPLE_RATE:.2f} сек: {part}"
        )

    logging.info(
        f"[Precache] Готово за {time.time() - started:.1f} сек, "
        f"новых={new_count}, аудио={total_seconds:.1f} сек."
    )


# ------------------------------------------------------------
#  HTTP
# ------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    cache_files = 0
    try:
        cache_files = sum(
            1 for name in os.listdir(TTS_CACHE_DIR) if name.endswith(".pcm")
        )
    except Exception:
        pass

    if LLM_BACKEND == "deepseek" and DEEPSEEK_API_KEY:
        llm_info = f"deepseek:{DEEPSEEK_MODEL}"
    elif LLM_BACKEND == "auto" and DEEPSEEK_API_KEY:
        llm_info = f"auto:deepseek:{DEEPSEEK_MODEL}|ollama:{OLLAMA_MODEL}"
    else:
        llm_info = f"ollama:{OLLAMA_MODEL}"

    return {
        "status": "ok",
        "version": APP_VERSION,
        "wake_word": "вегас",
        "mode": "conversational",
        "stt": "sherpa big-ru/en (conversational)",
        "stt_prefer": STT_PREFER_LANGUAGE,
        "memory": f"per-connection, {HISTORY_MAX_CHARS} chars",
        "llm": llm_info,
        "llm_backend": LLM_BACKEND,
        "tts": os.path.basename(PIPER_MODEL),
        "pitch_semitones": PITCH_SEMITONES,
        "tts_cache_files": cache_files,
        "wake_stt": os.path.basename(VOSK_MODEL_DIR),
        "weather": "met.no + ipwho.is",
        "uptime": round(time.time() - START_TIME, 1),
    }


# ------------------------------------------------------------
#  WebSocket
# ------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "?"
    logging.info(f"ESP32 подключен: {client_ip}")

    speech_buffer = bytearray()
    wake_buffer = bytearray()
    wake_mode = False

    # История диалога живёт только в рамках этого WebSocket-соединения.
    # Пропало соединение (ESP уснул, Wi-Fi, ребут) — история обнуляется.
    history: List[Dict[str, str]] = []

    # Держим ESP бодрым, пока он на связи. Прошивка вызывает markActivity()
    # на текст END_OF_AUDIO даже в состоянии idle, поэтому этот «пустой»
    # маркер безопасно продлевает таймер сна. Шлём только когда не заняты
    # обработкой реплики (иначе можно оборвать ответ).
    busy = False

    async def keepalive_loop() -> None:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            if busy:
                continue
            try:
                await websocket.send_text("END_OF_AUDIO")
                logging.debug("[Keepalive] продлил бодрствование ESP")
            except Exception:
                return

    keepalive_task = asyncio.create_task(keepalive_loop())

    def remember(role: str, content: str) -> None:
        """Добавляет реплику в историю и подрезает её по символам."""
        content = (content or "").strip()
        if not content:
            return
        history.append({"role": role, "content": content})
        total = sum(len(item["content"]) for item in history)
        # Держим историю по бюджету символов, удаляя самые старые ПАРЫ
        # (вопрос+ответ), чтобы история не начиналась с ответа без вопроса.
        while len(history) > 2 and total > HISTORY_MAX_CHARS:
            drop = 1 if history[0]["role"] == "assistant" else 2
            drop = min(drop, len(history) - 1)
            for _ in range(drop):
                total -= len(history[0]["content"])
                history.pop(0)
        logging.info(
            f"[История] реплик={len(history)}, символов={total}/"
            f"{HISTORY_MAX_CHARS}"
        )

    try:
        while True:
            try:
                message = await websocket.receive()
            except Exception:
                break

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes"):
                data = message["bytes"]
                if wake_mode:
                    wake_buffer.extend(data)
                else:
                    speech_buffer.extend(data)
                continue

            text = message.get("text")
            if not text:
                continue

            if text == "WAKE_AUDIO":
                wake_mode = True
                wake_buffer.clear()
                speech_buffer.clear()
                logging.info("[Wake] Получен фрагмент, проверяю «вегас»...")
                continue

            if text == "END_OF_WAKE":
                wake_mode = False
                if not wake_buffer:
                    await websocket.send_text("NO_WAKE")
                    continue
                ok, _info = await check_wake(bytes(wake_buffer))
                wake_buffer.clear()
                await websocket.send_text("WAKE" if ok else "NO_WAKE")
                continue

            if text != "END_OF_SPEECH":
                logging.warning(f"Неизвестная текстовая команда: {text!r}")
                continue

            if not speech_buffer:
                logging.warning("Пустой аудиобуфер.")
                await websocket.send_json({"error": "Empty speech"})
                await websocket.send_text("END_OF_AUDIO")
                continue

            payload = bytes(speech_buffer)
            speech_buffer.clear()
            logging.info(f"Получено {len(payload)} байт аудио.")

            busy = True
            try:
                raw_text, _mode, _info, _timings = await transcribe_audio(payload)
                if not raw_text:
                    logging.info("[STT] Результат пуст.")
                    await websocket.send_json({"error": "Empty speech"})
                    await websocket.send_text("END_OF_AUDIO")
                    continue

                user_text = clean_speech_text(raw_text)
                if not user_text:
                    logging.warning("После удаления wake word текст пуст.")
                    await websocket.send_json({"error": "Empty speech"})
                    await websocket.send_text("END_OF_AUDIO")
                    continue

                logging.info(f"Реплика: [{user_text}]")

                quick = chat_intent(user_text)
                restart_requested = False
                if quick and quick.startswith("__WEATHER_"):
                    kind = quick.strip("_").split("_", 1)[1].lower()
                    reply_text = await get_weather_reply(kind)
                    result = {
                        "action": "smalltalk",
                        "target": None,
                        "value": None,
                        "reply": reply_text,
                        "source": "weather",
                    }
                elif quick == "__RESTART_SERVER__":
                    reply_text = "Сейчас перезагружаю голосовой сервер, подожди немного."
                    result = {
                        "action": "smalltalk",
                        "target": None,
                        "value": None,
                        "reply": reply_text,
                        "source": "restart",
                    }
                    restart_requested = True
                elif quick:
                    result = {
                        "action": "smalltalk",
                        "target": None,
                        "value": None,
                        "reply": quick,
                        "source": "rules",
                    }
                else:
                    result = await query_llm(user_text, history)

                reply_text = str(result.get("reply") or LLM_FALLBACK)
                logging.info(f"Ответ ({result.get('source', '?')}): {reply_text}")

                # Запоминаем обе стороны диалога: и вопрос, и ответ.
                remember("user", user_text)
                remember("assistant", reply_text)

                await websocket.send_json(result)
                await send_tts(websocket, reply_text)

                if restart_requested:
                    logging.info("[Server] Голосовая команда: перезагрузка сервера.")
                    # Самоубийство процесса -> systemd (Restart=always) сам перезапустит сервис.
                    await asyncio.get_running_loop().run_in_executor(
                        None, lambda: os.kill(os.getpid(), signal.SIGKILL)
                    )

            except Exception as exc:
                logging.exception(f"Ошибка обработки реплики: {exc}")
                try:
                    await websocket.send_json({"error": "Internal error"})
                    await websocket.send_text("END_OF_AUDIO")
                except Exception:
                    break
            finally:
                busy = False

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logging.error(f"Ошибка WebSocket с {client_ip}: {exc}")
    finally:
        keepalive_task.cancel()
        # Соединение закрыто (сон ESP, Wi-Fi, ребут, restart): память обнуляется.
        # Это ровно то поведение, которое просил пользователь.
        logging.info(
            f"ESP32 отключен: {client_ip}, история диалога сброшена "
            f"({len(history)} реплик)"
        )
        history.clear()


@app.on_event("startup")
async def on_startup() -> None:
    logging.info(
        f"[Init] Сервер {APP_VERSION} готов: "
        f"STT=Sherpa RU/EN, wake=Vosk, LLM={LLM_BACKEND}"
        f"({DEEPSEEK_MODEL if LLM_BACKEND in ('deepseek', 'auto') and DEEPSEEK_API_KEY else OLLAMA_MODEL}), "
        f"TTS={os.path.basename(PIPER_MODEL)}, тон={PITCH_SEMITONES:+.1f}."
    )
    if LLM_BACKEND != "deepseek" or not DEEPSEEK_API_KEY:
        threading.Thread(target=warmup_ollama, daemon=True).start()


if __name__ == "__main__":
    if "--precache" in sys.argv:
        precache()
    else:
        import uvicorn

        port = int(os.environ.get("VOICE_SERVER_PORT", "8000"))
        logging.info(f"[Init] Слушаю 0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)

