#!/usr/bin/env python3
"""Проверка wake word «вегас» без ESP32.

Синтезирует фразы через Piper, приводит к 16 кГц моно и отправляет
на сервер по протоколу WAKE_AUDIO -> PCM -> END_OF_WAKE.
"""

import argparse
import asyncio
import audioop
import os
import subprocess
import sys

# Пути можно переопределить переменными окружения.
PIPER = os.environ.get("PIPER_BIN", "piper")
MODEL = os.environ.get(
    "PIPER_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "..", "server", "piper",
                 "ru_RU-denis-medium.onnx"),
)

DEFAULT_PHRASES = [
    "Вегас",
    "Вегас, включи свет",
    "Привет, как дела?",
    "Выключить свет гостиной и включить увлажнитель воздуха",
]


def tts_16k(text: str) -> bytes:
    raw = subprocess.run(
        [PIPER, "--model", MODEL, "--output_raw"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout
    pcm, _ = audioop.ratecv(raw, 2, 1, 22050, 16000, None)
    return pcm


async def check(ws, pcm: bytes) -> str:
    await ws.send("WAKE_AUDIO")
    for i in range(0, len(pcm), 1024):
        await ws.send(pcm[i:i + 1024])
        await asyncio.sleep(0.001)
    await ws.send("END_OF_WAKE")
    return await asyncio.wait_for(ws.recv(), timeout=15)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phrases", nargs="*", help="фразы для проверки")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import websockets

    phrases = args.phrases or DEFAULT_PHRASES
    failed = False

    async with websockets.connect(
            f"ws://{args.host}:{args.port}/ws", max_size=None) as ws:
        for text in phrases:
            pcm = tts_16k(text)
            answer = await check(ws, pcm)
            print(f"{text!r:60} {len(pcm) / 32000:5.2f} с -> {answer}")
            expected = "WAKE" if text.lower().startswith("вегас") and len(pcm) / 32000 <= 2.6 else "NO_WAKE"
            if answer != expected:
                failed = True
                print(f"  [!] ожидался {expected}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
