/*
 * ESP32-S3 — голосовой ассистент «Вегас» (без музыки, активация по кнопке).
 *
 *   - активация: кнопка BOOT (GPIO0);
 *   - микрофон: INMP441 (I2S);
 *   - динамик:  MAX98357A (I2S);
 *   - сервер:   voice_server.py (Sherpa/Vosk + DeepSeek/Ollama + Piper);
 *   - обновление: OTA по Wi-Fi (espota, порт 3232, пароль из secrets.h);
 *
 *   Сценарий: короткое/долгое нажатие BOOT -> писк -> запись -> отправка на сервер
 *   -> ответ Piper звучит через MAX98357A.
 *
 *   3.7.0: воспроизведение стало НЕБЛОКИРУЮЩИМ (кольцевой буфер + i2s_write
 *   с таймаутом 0). Раньше i2s_write(portMAX_DELAY) вызывался прямо из
 *   callback WebSocket и блокировал loop() на всю длину ответа — отсюда
 *   перезагрузки во время/после длинных ответов. Плюс добавлен crash-лог
 *   в RTC-память, который переживает перезагрузку.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <ArduinoOTA.h>
#include <WebServer.h>
#include <math.h>
#include <Adafruit_NeoPixel.h>
#include "driver/i2s.h"
#include "esp_sleep.h"
#include "config.h"

#define FW_VERSION "3.7.3"

// ------------------------------------------------------------
//  Константы и состояние
// ------------------------------------------------------------

// 512 сэмплов при 16 кГц = 32 мс
static const int FRAMES_PER_BLOCK = 512;

// Пре-ролл команды: ~320 мс звука до момента, когда VAD заметил речь.
static const int PREROLL_BLOCKS = 10;

static const i2s_port_t I2S_MIC_PORT = I2S_NUM_0;
static const i2s_port_t I2S_SPK_PORT = I2S_NUM_1;

enum State {
    ST_IDLE,        // слушаем кнопку
    ST_LISTENING,   // пишем команду
    ST_THINKING,    // ждём ответ сервера
    ST_SPEAKING     // играем ответ
};

static State state = ST_IDLE;
static uint32_t stateSince = 0;

// Состояния
static uint32_t lastActivityAt = 0;
static bool bootJustWoke = false;
static uint32_t bootWakeAt = 0;
static bool otaStarted = false;
static bool otaInProgress = false;
static bool statusServerStarted = false;
static uint32_t bootAt = 0;
static WebServer statusServer(80);

// Встроенный RGB-светодиод ESP32-S3 DevKitC-1 — GPIO 48 (WS2812)
#define PIN_LED 48
static Adafruit_NeoPixel led(1, PIN_LED, NEO_GRB + NEO_KHZ800);

static void ledSet(uint8_t r, uint8_t g, uint8_t b) {
    // Adafruit_NeoPixel сам переставляет байты под NEO_GRB — передаём обычный RGB.
    led.setPixelColor(0, led.Color(r, g, b));
    led.show();
}

// ------------------------------------------------------------
//  Сетевой лог: GET http://<ip>/log (читается по Wi-Fi)
// ------------------------------------------------------------
static String netLogBuf;

static void netLog(const char *fmt, ...) {
    char tmp[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, args);
    va_end(args);
    Serial.print(tmp);
    netLogBuf += tmp;
    if (netLogBuf.length() > 4000) {
        netLogBuf.remove(0, netLogBuf.length() - 4000);
    }
}

// Тестовая анимация при старте: фиолетовый вспыхивает 3 раза.
static void ledShowTest() {
    for (int i = 0; i < 3; ++i) {
        ledSet(120, 0, 120);
        delay(120);
        ledSet(0, 0, 0);
        delay(120);
    }
}

// Треугольная волна 0..255 с заданным периодом (не зависит от скорости loop).
static uint8_t triWave(uint32_t now, uint32_t period) {
    uint32_t p = now % period;
    uint32_t half = period / 2;
    uint32_t v = (p < half) ? (p * 255 / half)
                            : ((period - p) * 255 / half);
    if (v > 255) v = 255;
    return (uint8_t)v;
}

// Разное фиолетовое переливание для каждого действия.
static void ledUpdate() {
    static uint32_t lastLedAt = 0;
    uint32_t now = millis();
    if (now - lastLedAt < 20) return; // ~50 кадров/с, не грузим loop
    lastLedAt = now;

    uint8_t r = 0, g = 0, b = 0;

    switch (state) {
        case ST_IDLE: {
            // Медленное глубокое «дыхание», приглушённый фиолетовый.
            uint8_t lvl = (uint8_t)((uint16_t)triWave(now, 3200) * 45 / 255);
            r = (uint8_t)((uint16_t)lvl * 160 / 255);
            b = lvl;
            break;
        }
        case ST_LISTENING: {
            // Пульс при записи: ярче и чаще, с небольшим свечением в минимуме.
            uint8_t lvl = (uint8_t)(10 + (uint16_t)triWave(now, 900) * 110 / 255);
            r = (uint8_t)((uint16_t)lvl * 200 / 255);
            b = lvl;
            break;
        }
        case ST_THINKING: {
            // Быстрое мерцание «думает», сдвиг в пурпурно-фиолетовый.
            uint8_t lvl = (uint8_t)((uint16_t)triWave(now, 240) * 190 / 255);
            r = lvl;
            b = (uint8_t)((uint16_t)lvl * 210 / 255);
            break;
        }
        case ST_SPEAKING: {
            // Плавная волна во время ответа, сине-фиолетовый оттенок.
            uint8_t lvl = (uint8_t)(15 + (uint16_t)triWave(now, 1600) * 145 / 255);
            r = (uint8_t)((uint16_t)lvl * 120 / 255);
            b = lvl;
            break;
        }
        default:
            break;
    }

    ledSet(r, g, b);
}

static bool micRunning = false;
static bool speechStarted = false;
static bool followUpMode = false;
static uint32_t recordStart = 0;
static uint32_t speechStartAt = 0;
static uint32_t lastVoiceAt = 0;
static uint32_t quietSince = 0;
static bool quietActive = false;
static float smoothRms = 0;
static float noiseFloor = 0;
static uint32_t lastRmsLog = 0;

static bool audioDone = false;
static uint32_t lastAudioAt = 0;
static uint32_t thinkingSince = 0;

// ------------------------------------------------------------
//  Crash-лог в RTC-памяти (переживает перезагрузку)
// ------------------------------------------------------------

#define CRASH_MAGIC 0x56454741u // "VEGA"

RTC_NOINIT_ATTR struct {
    uint32_t magic;       // CRASH_MAGIC, если снимок валиден
    uint32_t uptimeMs;    // сколько прожили к моменту снимка
    uint32_t stateCode;   // в каком состоянии были
    uint32_t heap;        // сколько heap оставалось
    uint32_t slept;       // 1 = ушли в deep sleep штатно
    char log[384];        // последние строки netLog
} crashRec;

static uint32_t prevResetReason = 0;
static uint32_t prevUptimeMs = 0;
static uint32_t prevStateCode = 0;
static uint32_t prevHeap = 0;
static String prevLog;

static const char *resetReasonName(uint32_t reason) {
    switch (reason) {
        case 1: return "POWERON";
        case 2: return "EXT_PIN";
        case 3: return "SW";
        case 4: return "PANIC";
        case 5: return "INT_WDT";
        case 6: return "TASK_WDT";
        case 7: return "WDT";
        case 8: return "DEEPSLEEP";
        case 9: return "BROWNOUT";
        case 10: return "SDIO";
        case 11: return "SW_CPU0";
        case 12: return "SW_CPU1";
        default: return "?";
    }
}

// Регулярно (раз в 5 с) сохраняем «снимок» состояния в RTC-память.
// Если случится WDT/panic/brownout, после ребута в /status будет видно,
// в каком состоянии и с каким heap устройство умерло.
static void crashSaveSnapshot() {
    crashRec.magic = CRASH_MAGIC;
    crashRec.uptimeMs = millis();
    crashRec.stateCode = (uint32_t)state;
    crashRec.heap = (uint32_t)ESP.getFreeHeap();
    int len = netLogBuf.length();
    int take = len < (int)sizeof(crashRec.log) - 1 ? len : (int)sizeof(crashRec.log) - 1;
    if (take < 0) take = 0;
    memcpy(crashRec.log, netLogBuf.c_str() + (len - take), take);
    crashRec.log[take] = 0;
}

// ------------------------------------------------------------
//  Буферы
// ------------------------------------------------------------

static int32_t rawBuf[FRAMES_PER_BLOCK * 2]; // RX: 2 слота (L+R) по 32 бита
static int16_t pcmBuf[FRAMES_PER_BLOCK];

static int16_t cmdPreroll[PREROLL_BLOCKS][FRAMES_PER_BLOCK];
static int cmdPrerollFrames[PREROLL_BLOCKS];
static int cmdPrerollWrite = 0;
static int cmdPrerollCount = 0;

static int16_t stereoBuf[512]; // 256 стерео-кадров для TTS/писка
static int32_t dcOffsetL = 0;
static int32_t dcOffsetR = 0;
static int lastRmsL = 0;
static int lastRmsR = 0;

// ------------------------------------------------------------
//  Кольцевой буфер динамика (неблокирующее воспроизведение)
//  16384 сэмплов ≈ 0.74 с при 22.05 кГц. Сервер шлёт аудио в темпе
//  реального времени, поэтому кольцо гасит дрожание сети, а не копит
//  «на будущее» — callback WebSocket не блокирует loop().
// ------------------------------------------------------------

static const int SPK_RING_SAMPLES = 16384;
static int16_t spkRing[SPK_RING_SAMPLES];
static int spkHead = 0;   // индекс записи
static int spkTail = 0;   // индекс чтения
static int spkCount = 0;  // сколько сэмплов ждёт отправки
static uint32_t spkDrainAt = 0;     // когда кольцо опустело (для хвоста DMA)
static uint32_t spkProgressAt = 0;  // последний успешный прогресс записи

static int spkFree() { return SPK_RING_SAMPLES - spkCount; }

static bool spkAppend(const int16_t *data, size_t frames) {
    if ((int)frames > spkFree()) return false;
    for (size_t i = 0; i < frames; ++i) {
        spkRing[spkHead] = data[i];
        spkHead = (spkHead + 1) % SPK_RING_SAMPLES;
    }
    spkCount += (int)frames;
    return true;
}

static void spkReset() {
    spkHead = 0;
    spkTail = 0;
    spkCount = 0;
    spkDrainAt = 0;
}

// Butterworth high-pass 100 Гц, 2-й порядок, fs=16 кГц.
// Убирает постоянку и наводку 50 Гц, речь выше 200 Гц не трогает.
struct BiquadState { float x1, x2, y1, y2; };
static BiquadState hpL = {0, 0, 0, 0};
static BiquadState hpR = {0, 0, 0, 0};
static const float HP_B0 = 0.9726138985f;
static const float HP_B1 = -1.9452277970f;
static const float HP_B2 = 0.9726138985f;
static const float HP_A1 = -1.9444776578f;
static const float HP_A2 = 0.9459779362f;

static inline float hpStep(BiquadState &s, float x) {
    float y = HP_B0 * x + HP_B1 * s.x1 + HP_B2 * s.x2
              - HP_A1 * s.y1 - HP_A2 * s.y2;
    s.x2 = s.x1; s.x1 = x;
    s.y2 = s.y1; s.y1 = y;
    return y;
}

static float filtL[FRAMES_PER_BLOCK];
static float filtR[FRAMES_PER_BLOCK];

// ------------------------------------------------------------
//  Объекты
// ------------------------------------------------------------

static WebSocketsClient wsClient;

// ------------------------------------------------------------
//  I2S / звук
// ------------------------------------------------------------

static void micStop() {
    if (!micRunning) return;
    i2s_stop(I2S_MIC_PORT);
    micRunning = false;
}

static void micStart() {
    if (micRunning) return;
    i2s_start(I2S_MIC_PORT);
    micRunning = true;

    // Слить старые данные DMA и заодно оценить DC-смещение по последнему
    // блоку. Иначе фильтр стартует с нуля, первые блоки уходят в клип,
    // и «фон комнаты» получается 7000+ RMS.
    delay(30);
    size_t bytesRead = 0;
    uint32_t t0 = millis();
    int64_t sumL = 0, sumR = 0;
    int count = 0;
    int minL = 32767, maxL = -32768;
    while (millis() - t0 < 80) {
        if (i2s_read(I2S_MIC_PORT, rawBuf, sizeof(rawBuf), &bytesRead,
                     pdMS_TO_TICKS(10)) != ESP_OK) {
            break;
        }
        if (bytesRead == 0) continue;
        int n = bytesRead / (sizeof(int32_t) * 2);
        sumL = 0;
        sumR = 0;
        for (int i = 0; i < n; ++i) {
            int32_t lv = rawBuf[2 * i] >> 16;
            if (lv < minL) minL = lv;
            if (lv > maxL) maxL = lv;
            sumL += lv;
            sumR += (rawBuf[2 * i + 1] >> 16);
        }
        count = n;
    }
    if (count > 0) {
        dcOffsetL = (int32_t)(sumL / count);
        dcOffsetR = (int32_t)(sumR / count);
    } else {
        dcOffsetL = 0;
        dcOffsetR = 0;
    }
    hpL = {0, 0, 0, 0};
    hpR = {0, 0, 0, 0};
    netLog("[Mic] DC L=%ld R=%ld min=%d max=%d\n",
           (long)dcOffsetL, (long)dcOffsetR, minL, maxL);
}

static void initMic() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = MIC_SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT, // читаем оба канала, выбираем активный
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 8,
        .dma_buf_len          = 256,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0,
    };

    ESP_ERROR_CHECK(i2s_driver_install(I2S_MIC_PORT, &cfg, 0, nullptr));

    i2s_pin_config_t pins = {
        .mck_io_num   = I2S_PIN_NO_CHANGE,
        .bck_io_num   = PIN_MIC_SCK,
        .ws_io_num    = PIN_MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = PIN_MIC_SD,
    };
    ESP_ERROR_CHECK(i2s_set_pin(I2S_MIC_PORT, &pins));

    // Драйвер стартует сам — останавливаем, пока микрофон не нужен.
    i2s_stop(I2S_MIC_PORT);
}

static void initSpeaker() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate          = SPK_SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 16,
        .dma_buf_len          = 512,
        .use_apll             = false,
        .tx_desc_auto_clear   = true,
        .fixed_mclk           = 0,
    };
    ESP_ERROR_CHECK(i2s_driver_install(I2S_SPK_PORT, &cfg, 0, nullptr));
    i2s_pin_config_t pins = {
        .mck_io_num   = I2S_PIN_NO_CHANGE,
        .bck_io_num   = PIN_SPK_BCLK,
        .ws_io_num    = PIN_SPK_LRC,
        .data_out_num = PIN_SPK_DIN,
        .data_in_num  = I2S_PIN_NO_CHANGE,
    };
    ESP_ERROR_CHECK(i2s_set_pin(I2S_SPK_PORT, &pins));
    i2s_zero_dma_buffer(I2S_SPK_PORT);
    i2s_set_sample_rates(I2S_SPK_PORT, SPK_SAMPLE_RATE);
}

// Моно PCM -> стерео (драйвер ESP32-I2S работает в RIGHT_LEFT).
// ВАЖНО: не блокируется. Если DMA занят — возвращает false, вызывающий
// код подождёт в loop() и вызовет снова. Так loop() никогда не встаёт
// намертво и watchdog не срабатывает.
// Пишет максимум n кадров. Возвращает число РЕАЛЬНО принятых кадров
// (может быть меньше n: i2s_write с таймаутом 0 отдаёт столько, сколько
// влезло в DMA). Никогда не блокирует — именно поэтому loop() жив.
static int writeMonoPcmCount(const int16_t *data, int frames, TickType_t wait) {
    if (frames <= 0) return 0;
    int n = frames > 256 ? 256 : frames;
    for (int i = 0; i < n; ++i) {
        stereoBuf[2 * i]     = data[i];
        stereoBuf[2 * i + 1] = data[i];
    }
    size_t written = 0;
    esp_err_t err = i2s_write(I2S_SPK_PORT, stereoBuf, (size_t)n * 2 * sizeof(int16_t),
                              &written, wait);
    if (err != ESP_OK) return 0;
    return (int)(written / (2 * sizeof(int16_t)));
}

// Отправляет из кольца столько, сколько принимает DMA. Не блокирует.
static void spkPump() {
    while (spkCount > 0) {
        int want = spkCount > 256 ? 256 : spkCount;
        // Читаем, НЕ сдвигая хвост: сдвиг только на реально записанное.
        int16_t chunk[256];
        for (int i = 0; i < want; ++i) {
            chunk[i] = spkRing[(spkTail + i) % SPK_RING_SAMPLES];
        }
        int sent = writeMonoPcmCount(chunk, want, 0);
        if (sent <= 0) {
            // DMA полон — выходим, loop() вернётся сюда в следующей итерации.
            return;
        }
        spkTail = (spkTail + sent) % SPK_RING_SAMPLES;
        spkCount -= sent;
        spkProgressAt = millis();
        if (sent < want) {
            // Частичная запись: DMA почти полон, дальше ждать смысла нет.
            return;
        }
    }
    if (spkDrainAt == 0) spkDrainAt = millis();
}

static void playTone(uint16_t freq, uint16_t ms) {
    const int total = (SPK_SAMPLE_RATE * ms) / 1000;
    int16_t mono[256];

    // Тон короткий (до нескольких сотен мс), поэтому здесь допустимо
    // подождать: пишем порциями с таймаутом и не забываем кормить loop.
    for (int done = 0; done < total; ) {
        int n = total - done;
        if (n > 256) n = 256;
        for (int i = 0; i < n; ++i) {
            float t = (float)(done + i) / (float)SPK_SAMPLE_RATE;
            mono[i] = (int16_t)(sinf(2.0f * PI * freq * t) * 6000.0f);
        }
        int left = n;
        int off = 0;
        uint32_t t0 = millis();
        while (left > 0 && millis() - t0 < 2000) {
            int sent = writeMonoPcmCount(mono + off, left, pdMS_TO_TICKS(50));
            if (sent <= 0) {
                delay(2);
                continue;
            }
            off += sent;
            left -= sent;
        }
        done += n;
    }
}

// ------------------------------------------------------------
//  Микрофон -> PCM (авто-детект канала L/R)
// ------------------------------------------------------------

static int readMicBlock(int16_t *out, int &framesOut) {
    size_t bytesRead = 0;
    esp_err_t err = i2s_read(I2S_MIC_PORT, rawBuf, sizeof(rawBuf), &bytesRead,
                             pdMS_TO_TICKS(50));
    if (err != ESP_OK || bytesRead == 0) {
        framesOut = 0;
        return 0;
    }

    const int n = bytesRead / (sizeof(int32_t) * 2); // n stereo-кадров
    framesOut = n;
    if (n == 0) return 0;

    int64_t sumSqL = 0, sumSqR = 0;
    for (int i = 0; i < n; ++i) {
        int32_t l = rawBuf[2 * i] >> 16;
        int32_t r = rawBuf[2 * i + 1] >> 16;
        dcOffsetL += (l - dcOffsetL) >> 10;
        dcOffsetR += (r - dcOffsetR) >> 10;
        filtL[i] = hpStep(hpL, (float)(l - dcOffsetL));
        filtR[i] = hpStep(hpR, (float)(r - dcOffsetR));
        int32_t vl = (int32_t)(filtL[i] * MIC_GAIN);
        int32_t vr = (int32_t)(filtR[i] * MIC_GAIN);
        if (vl > 32767) vl = 32767; if (vl < -32768) vl = -32768;
        if (vr > 32767) vr = 32767; if (vr < -32768) vr = -32768;
        sumSqL += (int64_t)vl * vl;
        sumSqR += (int64_t)vr * vr;
    }

    lastRmsL = (int)sqrtf((float)sumSqL / (float)n);
    lastRmsR = (int)sqrtf((float)sumSqR / (float)n);

    // Выбираем канал с сигналом (активный канал INMP441: L или R).
    bool useRight = lastRmsR > lastRmsL;
    int64_t sumSq = useRight ? sumSqR : sumSqL;
    for (int i = 0; i < n; ++i) {
        int32_t v = (int32_t)((useRight ? filtR[i] : filtL[i]) * MIC_GAIN);
        if (v > 32767)  v = 32767;
        if (v < -32768) v = -32768;
        out[i] = (int16_t)v;
    }

    return (int)sqrtf((float)sumSq / (float)n);
}

// Измеряет фон комнаты до писка. Читает блоки и усредняет без самого
// громкого — защита от щелчка/всплеска при старте I2S.
static float measureNoiseFloor(uint32_t durationMs) {
    // Первые 200 мс после старта I2S — переходный процесс, не учитываем.
    uint32_t skipStart = millis();
    while (millis() - skipStart < 200) {
        int frames = 0;
        readMicBlock(pcmBuf, frames);
    }

    // Собираем RMS блоков и берём 25-й процентиль: одиночные всплески,
    // щелчки и раннее начало речи не задирают фон.
    const int MAX_BLOCKS = 32;
    int vals[MAX_BLOCKS];
    int count = 0;
    uint32_t start = millis();
    while (millis() - start < durationMs && count < MAX_BLOCKS) {
        int frames = 0;
        int rms = readMicBlock(pcmBuf, frames);
        if (frames <= 0) continue;
        vals[count++] = rms;
    }
    if (count == 0) return 0.0f;

    for (int i = 1; i < count; ++i) {
        int v = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > v) {
            vals[j + 1] = vals[j];
            --j;
        }
        vals[j + 1] = v;
    }
    int p25 = vals[count / 4];
    netLog("[Фон] min=%d p25=%d med=%d max=%d (блоков %d)\n",
           vals[0], p25, vals[count / 2], vals[count - 1], count);
    return (float)p25;
}

// ------------------------------------------------------------
//  Состояния / помощники
// ------------------------------------------------------------

// Активность пользователя/OTA: только она сдвигает отсчёт сна.
// Переподключения WebSocket и смена состояния сами по себе сон не откладывают.
static void markActivity() {
    lastActivityAt = millis();
}

static void setState(State s) {
    state = s;
    stateSince = millis();
}

static void cleanupAfterConversation() {
    // Освобождаем память/состояние после каждого разговора.
    netLogBuf = "";
    netLogBuf.reserve(0);
    cmdPrerollCount = 0;
    cmdPrerollWrite = 0;
    speechStarted = false;
    quietActive = false;
    quietSince = 0;
    smoothRms = 0.0f;
    noiseFloor = 0.0f;
    speechStartAt = 0;
}

static void enterIdle(const char *reason) {
    micStop();
    if (state == ST_SPEAKING || state == ST_THINKING) {
        i2s_zero_dma_buffer(I2S_SPK_PORT);
        audioDone = false;
        spkReset();
    }
    followUpMode = false;
    setState(ST_IDLE);
    if (reason && *reason) Serial.printf("[Состояние] idle (%s)\n", reason);
    cleanupAfterConversation();
}

static void stopSpeakingNow() {
    markActivity();
    i2s_zero_dma_buffer(I2S_SPK_PORT);
    spkReset();
    enterIdle("tts stop");
}

static void goToSleep() {
    if (otaInProgress) {
        Serial.println("[Сон] OTA идёт — сон отложен.");
        markActivity();
        return;
    }
    Serial.printf("[Сон] Deep sleep: idle=%lu с, uptime=%lu с, wake: BOOT/GPIO0.\n",
                  (unsigned long)((millis() - lastActivityAt) / 1000),
                  (unsigned long)(millis() / 1000));
    // Помечаем штатный сон, чтобы следующий запуск не считал его падением.
    crashRec.slept = 1;
    ledSet(0, 0, 0); // выключаем RGB перед сном
    if (statusServerStarted) {
        statusServer.stop();
        statusServerStarted = false;
    }
    wsClient.disconnect();
    WiFi.disconnect(true);
    Serial.flush();
    esp_err_t err = esp_sleep_enable_ext0_wakeup((gpio_num_t)PIN_BUTTON, 0);
    if (err != ESP_OK) {
        Serial.printf("[Сон] Не смог включить wake по GPIO0: %d\n", (int)err);
    }
    delay(100);
    esp_deep_sleep_start();
}

// ------------------------------------------------------------
//  Приемка команды на сервер
// ------------------------------------------------------------

static void sendPcmBlock(const int16_t *data, int frames) {
    if (wsClient.isConnected()) {
        wsClient.sendBIN((const uint8_t *)data, frames * sizeof(int16_t));
    }
}

static void pushCmdPreroll(const int16_t *data, int frames) {
    memcpy(cmdPreroll[cmdPrerollWrite], data, frames * sizeof(int16_t));
    cmdPrerollFrames[cmdPrerollWrite] = frames;
    cmdPrerollWrite = (cmdPrerollWrite + 1) % PREROLL_BLOCKS;
    if (cmdPrerollCount < PREROLL_BLOCKS) cmdPrerollCount++;
}

static void flushCmdPreroll() {
    int start = (cmdPrerollWrite - cmdPrerollCount + PREROLL_BLOCKS) % PREROLL_BLOCKS;
    for (int i = 0; i < cmdPrerollCount; ++i) {
        int idx = (start + i) % PREROLL_BLOCKS;
        sendPcmBlock(cmdPreroll[idx], cmdPrerollFrames[idx]);
    }
    cmdPrerollCount = 0;
    cmdPrerollWrite = 0;
}

// ------------------------------------------------------------
//  Запись команды
// ------------------------------------------------------------

static void startListening() {
    markActivity();
    followUpMode = false;

    if (WiFi.status() != WL_CONNECTED || !wsClient.isConnected()) {
        netLog("[Запись] Нет соединения с сервером.\n");
        playTone(300, 120);
        enterIdle("no server");
        return;
    }

    micStop();
    micStart();                    // микрофон уже работает — меряем фон ДО писка
    delay(50);
    noiseFloor = measureNoiseFloor(300);
    if (noiseFloor < 50.0f) noiseFloor = 50.0f;
    netLog("[Запись] Фон комнаты: %d RMS\n", (int)noiseFloor);

    playTone(1200, 70);            // «пик» — можно говорить
    delay(100);                    // дать пику уйти

    cmdPrerollCount = 0;
    cmdPrerollWrite = 0;
    speechStarted = false;
    quietSince = 0;
    quietActive = false;
    smoothRms = 0;

    recordStart = millis();
    speechStartAt = 0;
    lastVoiceAt = recordStart;
    lastRmsLog = recordStart;
    setState(ST_LISTENING);
    netLog("[Запись] Старт. Говорите...\n");
}

static void startFollowUpListening() {
    if (WiFi.status() != WL_CONNECTED || !wsClient.isConnected()) {
        netLog("[Диалог] Нет соединения для продолжения.\n");
        followUpMode = false;
        enterIdle("no server");
        return;
    }
    markActivity();

    micStop();
    micStart();
    delay(30);
    noiseFloor = measureNoiseFloor(150);
    if (noiseFloor < 50.0f) noiseFloor = 50.0f;

    cmdPrerollCount = 0;
    cmdPrerollWrite = 0;
    speechStarted = false;
    quietSince = 0;
    quietActive = false;
    smoothRms = 0.0f;
    speechStartAt = 0;

    recordStart = millis();
    lastVoiceAt = recordStart;
    lastRmsLog = recordStart;
    followUpMode = true;
    setState(ST_LISTENING);
    netLog("[Диалог] Слушаю продолжение (%u мс, шум=%d)...\n",
           (unsigned)FOLLOWUP_LISTEN_TIMEOUT_MS, (int)noiseFloor);
}

static void finishListening(bool sendToServer) {
    micStop();
    markActivity();

    bool wasFollowUp = followUpMode;
    followUpMode = false;

    if (sendToServer && wsClient.isConnected()) {
        wsClient.sendTXT("END_OF_SPEECH");
        if (!wasFollowUp) {
            playTone(800, 60);         // «конец записи» в обычном режиме
        }
        thinkingSince = millis();
        setState(ST_THINKING);
        netLog("[Запись] Отправлено, жду ответ сервера...\n");
    } else {
        if (wasFollowUp) {
            netLog("[Диалог] Голоса нет, завершаю диалог.\n");
            enterIdle("followup timeout");
        } else {
            playTone(300, 100);
            enterIdle("no speech");
        }
    }
}

static void recordStep() {
    int frames = 0;
    int rms = readMicBlock(pcmBuf, frames);
    uint32_t startTimeout = followUpMode ? FOLLOWUP_LISTEN_TIMEOUT_MS : VAD_START_TIMEOUT;
    if (frames <= 0) {
        uint32_t now = millis();
        if (!speechStarted && now - recordStart >= startTimeout) {
            finishListening(false);
        } else if (now - recordStart >= MAX_RECORD_MS) {
            finishListening(speechStarted);
        }
        return;
    }

    uint32_t now = millis();

    // Сглаженный RMS: одиночные всплески/провалы не должны дёргать VAD.
    smoothRms = (smoothRms <= 0.0f) ? (float)rms
                                    : (0.7f * smoothRms + 0.3f * (float)rms);

    if (!speechStarted) {
        if (now - recordStart < 400) {
            // Первые 400 мс — писк и его хвост. В VAD не лезем, но в пре-ролл пишем.
            pushCmdPreroll(pcmBuf, frames);
        } else {
            uint32_t startThr = (uint32_t)fmaxf((float)VAD_START_RMS,
                                                noiseFloor * 1.6f);
            if (smoothRms >= (float)startThr) {
                speechStarted = true;
                speechStartAt = now;
                lastVoiceAt = now;
                netLog("[Запись] Речь обнаружена (RMS=%d sm=%d шум=%d порог=%u L=%d R=%d), пре-ролл %d блоков\n",
                       rms, (int)smoothRms, (int)noiseFloor, startThr,
                       lastRmsL, lastRmsR, cmdPrerollCount);
                flushCmdPreroll();
                sendPcmBlock(pcmBuf, frames);
            } else {
                // Медленно подстраиваем фон под комнату (шум/вентилятор/ТВ).
                noiseFloor = 0.995f * noiseFloor + 0.005f * (float)rms;
                pushCmdPreroll(pcmBuf, frames);
            }
        }

        if (!speechStarted && now - recordStart >= startTimeout) {
            finishListening(false);
            return;
        }
    } else {
        sendPcmBlock(pcmBuf, frames);

        // Порог тишины — от измеренного фона, а не жёсткие 300 RMS.
        uint32_t silenceThr = (uint32_t)fmaxf((float)VAD_SILENCE_RMS,
                                              noiseFloor * 1.25f);
        float fastThr = fmaxf((float)VAD_SILENCE_FAST_RMS, noiseFloor * 1.05f);

        if (smoothRms >= (float)silenceThr) {
            lastVoiceAt = now;
            quietActive = false;
        } else {
            if (!quietActive) {
                quietActive = true;
                quietSince = now;
            }
            uint32_t quietFor = now - quietSince;
            bool fastStop = (smoothRms < fastThr &&
                             quietFor >= VAD_SILENCE_FAST_MS);
            // Не обрываем запись в первые VAD_MIN_SPEECH_MS после старта речи:
            // короткая пауза в начале фразы не должна завершать запись.
            bool minSpeechElapsed = (now - speechStartAt >= VAD_MIN_SPEECH_MS);
            if ((fastStop || quietFor >= VAD_SILENCE_MS) && minSpeechElapsed) {
                netLog("[Запись] Конец речи: тишина %lu мс (RMS=%d sm=%d шум=%d порог=%u)%s\n",
                       (unsigned long)quietFor, rms, (int)smoothRms,
                       (int)noiseFloor, silenceThr,
                       fastStop ? ", быстрый стоп" : "");
                finishListening(true);
                return;
            }
        }
    }

    if (now - lastRmsLog >= 1000) {
        lastRmsLog = now;
        netLog("[Запись] RMS=%d sm=%d шум=%d (L=%d R=%d)%s\n",
               rms, (int)smoothRms, (int)noiseFloor,
               lastRmsL, lastRmsR,
               speechStarted ? " (речь)" : "");
    }

    if (now - recordStart >= MAX_RECORD_MS) {
        netLog("[Запись] Достигнут лимит времени.\n");
        finishListening(speechStarted);
    }
}

// ------------------------------------------------------------
//  WebSocket
// ------------------------------------------------------------

static void onWsEvent(WStype_t type, uint8_t *payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            Serial.printf("[WS] Подключено к ws://%s:%d%s\n",
                          SERVER_HOST, SERVER_PORT, SERVER_PATH);
            if (state != ST_IDLE) {
                enterIdle("ws connected");
            }
            break;

        case WStype_DISCONNECTED:
            Serial.println("[WS] Отключено. Переподключение...");
            if (state == ST_SPEAKING) {
                i2s_zero_dma_buffer(I2S_SPK_PORT);
            }
            if (state != ST_IDLE) {
                enterIdle("ws disconnected");
            }
            break;

        case WStype_TEXT: {
            char text[512];
            size_t n = length < sizeof(text) - 1 ? length : sizeof(text) - 1;
            memcpy(text, payload, n);
            text[n] = '\0';

            if (strcmp(text, "END_OF_AUDIO") == 0) {
                // Сервер периодически присылает END_OF_AUDIO как keepalive.
                // В IDLE не считаем это активностью пользователя, иначе сон
                // никогда не наступит при активном WS-соединении.
                if (state != ST_IDLE && !otaInProgress) markActivity();
                if (state == ST_SPEAKING) {
                    // Аудио принято, но в кольце может быть ещё не проиграно.
                    // Не ставим audioDone сразу: loop() доиграет остаток.
                    audioDone = true;
                    lastAudioAt = millis();
                    netLog("[Ответ] Конец аудио, в кольце %d сэмплов.\n",
                           spkCount);
                } else if (state == ST_THINKING) {
                    Serial.println("[Ответ] Аудио нет, ответ только текстом.");
                    startFollowUpListening();
                }
                break;
            }

            Serial.printf("[Ответ] JSON: %s\n", text);

            JsonDocument doc;
            DeserializationError err = deserializeJson(doc, text);
            if (err) {
                Serial.printf("[Ответ] Не смог разобрать JSON: %s\n", err.c_str());
                break;
            }

            if (doc["error"].is<const char *>()) {
                Serial.printf("[Ответ] Ошибка сервера: %s\n",
                              doc["error"].as<const char *>());
                enterIdle("server error");
                break;
            }

            const char *reply = doc["reply"].as<const char *>();
            const char *action = doc["action"].as<const char *>();
            if (reply) Serial.printf("[Ассистент] %s\n", reply);
            if (action) Serial.printf("[Команда] action=%s\n", action);
            break;
        }

        case WStype_BIN:
            if (state == ST_THINKING) {
                markActivity();
                i2s_set_sample_rates(I2S_SPK_PORT, SPK_SAMPLE_RATE);
                audioDone = false;
                spkReset();
                spkProgressAt = millis();
                setState(ST_SPEAKING);
                Serial.printf("[Ответ] Начало аудио (%u байт)...\n",
                              (unsigned)length);
            }
            if (state == ST_SPEAKING && length > 0) {
                if (length & 1) length--;
                size_t samples = length / 2;
                const int16_t *src = (const int16_t *)payload;
                // Кольцо может заполниться: сервер шлёт быстрее реального
                // времени. Тогда ждём в коротком цикле, но ОБЯЗАТЕЛЬНО
                // возвращаем управление в loop() (через delay(1)), иначе
                // watchdog снова прибьёт устройство во время длинного ответа.
                uint32_t stallSince = millis();
                while (samples > 0) {
                    size_t chunk = samples > 256 ? 256 : samples;
                    if (spkAppend(src, chunk)) {
                        src += chunk;
                        samples -= chunk;
                        continue;
                    }
                    spkPump();
                    if (millis() - stallSince > 2000) {
                        // 2 секунды без прогресса — DMA встал. Не висим.
                        netLog("[Динамик] DMA не принимает 2 с, сброс.\n");
                        i2s_zero_dma_buffer(I2S_SPK_PORT);
                        spkReset();
                        stallSince = millis();
                    }
                    delay(1);
                }
                lastAudioAt = millis();
                markActivity();
            }
            break;

        case WStype_ERROR:
            Serial.println("[WS] Ошибка сокета.");
            break;

        default:
            break;
    }
}

// ------------------------------------------------------------
//  Кнопка BOOT (GPIO0)
// ------------------------------------------------------------

static void handleButton() {
    static bool lastRaw = false;
    static bool stable = false;
    static uint32_t changedAt = 0;

    bool raw = (digitalRead(PIN_BUTTON) == LOW);
    if (raw != lastRaw) {
        lastRaw = raw;
        changedAt = millis();
    }

    if (millis() - changedAt <= 30 || stable == raw) return;
    stable = raw;

    if (!stable) return; // отпущено

    markActivity();
    // нажатие
    if (bootJustWoke) {
        // кнопка удерживается после пробуждения из deep sleep — автостарт уже решил,
        // здесь просто игнорируем это же «нажатие».
        return;
    }

    switch (state) {
        case ST_IDLE:
            startListening();
            break;
        case ST_LISTENING:
            finishListening(speechStarted);
            break;
        case ST_SPEAKING:
            stopSpeakingNow();
            break;
        default:
            break;
    }
}

// ------------------------------------------------------------
//  Serial-команды
// ------------------------------------------------------------

static void handleSerial() {
    static String cmd;

    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            cmd.trim();
            if (cmd.length() == 0) {
                cmd = "";
                continue;
            }

            markActivity();
            if (cmd.equalsIgnoreCase("n")) {
                Serial.println("[Команда] n — без музыки, не нужно.");
            } else if (cmd.equalsIgnoreCase("s")) {
                enterIdle("serial stop");
            } else if (cmd.equalsIgnoreCase("m")) {
                Serial.println("[Микрофон] Тест отключён: сигнал L-канала уже проверен (~31k).");
            } else if (cmd.equalsIgnoreCase("a")) {
                startListening();
            } else if (cmd.equalsIgnoreCase("t")) {
                playTone(1000, 500); // тест громкоговорителя: тон 1 кГц на 0.5 с
                Serial.println("[Команда] t — тест динамика: тон 1 кГц на 0.5 с");
            } else if (cmd.startsWith("v")) {
                Serial.println("[Команда] v — без музыки, громкость фиксирована.");
            } else {
                Serial.println("[Команда] a — старт записи, s — стоп, m — микрофон тест");
            }
            cmd = "";
        } else if (cmd.length() < 32) {
            cmd += c;
        }
    }
}

// ------------------------------------------------------------
//  HTTP-диагностика /status
// ------------------------------------------------------------

static const char *stateName(State s) {
    switch (s) {
        case ST_IDLE:      return "idle";
        case ST_LISTENING: return "listening";
        case ST_THINKING:  return "thinking";
        case ST_SPEAKING:  return "speaking";
        default:           return "?";
    }
}

static void handleStatus() {
    uint32_t now = millis();
    uint32_t age = now - lastActivityAt;
    int32_t sleepIn = -1;
    if (state == ST_IDLE && !micRunning) {
        sleepIn = (age >= IDLE_SLEEP_MS) ? 0 : (int32_t)(IDLE_SLEEP_MS - age);
    }

    JsonDocument doc;
    doc["fw"] = FW_VERSION;
    doc["uptime_ms"] = now;
    doc["boot_at_ms"] = bootAt;
    doc["state"] = stateName(state);
    doc["state_ms"] = now - stateSince;
    doc["last_activity_age_ms"] = age;
    doc["sleep_in_ms"] = sleepIn;
    doc["mic_running"] = micRunning;
    doc["ws_connected"] = wsClient.isConnected();
    doc["rssi"] = WiFi.RSSI();
    doc["heap"] = ESP.getFreeHeap();
    doc["boot_just_woke"] = bootJustWoke;
    doc["ota_started"] = otaStarted;
    doc["ota_in_progress"] = otaInProgress;
    // reset_reason — почему закончилась ПРОШЛАЯ жизнь.
    doc["reset_reason"] = (int)prevResetReason;
    doc["reset_reason_name"] = resetReasonName(prevResetReason);
    doc["prev_uptime_ms"] = prevUptimeMs;
    doc["prev_state"] = stateName((State)prevStateCode);
    doc["prev_heap"] = prevHeap;
    doc["prev_log"] = prevLog;
    // Кольцо динамика: видно, если звук «залип».
    doc["spk_ring"] = spkCount;
    doc["spk_free"] = spkFree();
    doc["wake_cause"] = (int)esp_sleep_get_wakeup_cause();
    doc["ip"] = WiFi.localIP().toString();

    String out;
    serializeJson(doc, out);
    statusServer.send(200, "application/json", out);
}

static void handleLog() {
    String out = netLogBuf;
    if (out.length() == 0) out = "(лог пуст)\n";
    statusServer.send(200, "text/plain; charset=utf-8", out);
}

static void startStatusServer() {
    if (statusServerStarted) return;
    statusServer.on("/status", handleStatus);
    statusServer.on("/log", handleLog);
    statusServer.begin();
    statusServerStarted = true;
    Serial.printf("[HTTP] /status и /log: http://%s/\n",
                  WiFi.localIP().toString().c_str());
}

// ------------------------------------------------------------
//  Wi-Fi
// ------------------------------------------------------------

static void connectWiFi() {
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.setHostname(HOSTNAME);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.printf("[WiFi] Подключение к \"%s\"", WIFI_SSID);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
        delay(300);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        String ip = WiFi.localIP().toString();
        Serial.printf("[WiFi] OK, IP: %s\n", ip.c_str());
        Serial.printf("[WiFi] Сайт: http://%s/\n", ip.c_str());
    } else {
        Serial.println("[WiFi] Не подключился, перепробую в loop.");
    }
}

// ------------------------------------------------------------
//  setup / loop
// ------------------------------------------------------------

void setup() {
    Serial.begin(115200);
    delay(300);

    Serial.println("\n=== ESP32-S3: голосовой ассистент «Вегас» ===");

    // Причина текущего сброса = то, КАК закончилась прошлая жизнь.
    prevResetReason = esp_reset_reason();
    // Снимок состояния, сделанный за мгновение до падения (раз в 5 с).
    if (crashRec.magic == CRASH_MAGIC) {
        prevUptimeMs = crashRec.uptimeMs;
        prevStateCode = crashRec.stateCode;
        prevHeap = crashRec.heap;
        prevLog = String(crashRec.log);
        if (prevResetReason != 8 /* DEEPSLEEP */) {
            Serial.printf("[CRASH] сброс %s (%lu) после uptime=%lu мс, "
                          "state=%lu, heap=%lu\n",
                          resetReasonName(prevResetReason),
                          (unsigned long)prevResetReason,
                          (unsigned long)prevUptimeMs,
                          (unsigned long)prevStateCode,
                          (unsigned long)prevHeap);
            if (prevLog.length()) {
                Serial.printf("[CRASH] последний лог:\n%s\n", prevLog.c_str());
            }
        }
        if (crashRec.slept) {
            Serial.println("[CRASH] предыдущий выход был штатным deep sleep.");
        }
    }
    crashRec.magic = 0;
    crashRec.uptimeMs = 0;
    crashRec.stateCode = 0;
    crashRec.heap = 0;
    crashRec.slept = 0;
    crashRec.log[0] = 0;

    Serial.printf("[BOOT] fw=%s, reset_reason=%d (%s), wake_cause=%d\n",
                  FW_VERSION, (int)prevResetReason,
                  resetReasonName(prevResetReason),
                  (int)esp_sleep_get_wakeup_cause());
    Serial.printf("Сервер: %s:%d%s\n", SERVER_HOST, SERVER_PORT, SERVER_PATH);
    Serial.println("Динамик: BCLK=15, LRC=16, DIN=17; микрофон: SCK=4, WS=5, SD=6");
    Serial.println("Активация: короткое нажатие BOOT -> запись -> ответ.");

    pinMode(PIN_BUTTON, INPUT_PULLUP);

    initSpeaker();
    initMic();

    connectWiFi();

    // OTA-обновление по Wi-Fi (ArduinoOTA / espota, порт 3232).
    ArduinoOTA.setHostname(HOSTNAME);
    ArduinoOTA.setPassword(OTA_PASSWORD);
    ArduinoOTA.onStart([]() {
        otaInProgress = true;
        markActivity();
        Serial.println("[OTA] Обновление началось...");
    });
    ArduinoOTA.onEnd([]() {
        otaInProgress = false;
        Serial.println("\n[OTA] Готово, перезагрузка.");
    });
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        markActivity();
        static unsigned int lastPct = 0;
        unsigned int pct = (total > 0) ? (progress * 100 / total) : 0;
        if (pct >= lastPct + 10) {
            lastPct = pct;
            Serial.printf("[OTA] %u%%\n", pct);
        }
    });
    ArduinoOTA.onError([](ota_error_t error) {
        otaInProgress = false;
        Serial.printf("[OTA] Ошибка %u\n", error);
    });

    wsClient.begin(SERVER_HOST, SERVER_PORT, SERVER_PATH);
    wsClient.onEvent(onWsEvent);
    wsClient.setReconnectInterval(3000);
    // Терпимее к кратким потерям Wi-Fi: пинг раз в 20 с, ждём 5 с, 3 промаха до обрыва.
    wsClient.enableHeartbeat(20000, 5000, 3);

    playTone(900, 80);             // проверка динамика при старте

    led.begin();
    led.setBrightness(255); // яркость задаём в ledUpdate(), чтобы idle был приглушённым
    ledShowTest();

    enterIdle("boot");

    bootAt = millis();
    markActivity();
    // Пробуждение из deep sleep по BOOT определяется по причине сброса,
    // а не по удержанию кнопки (её можно отпустить раньше конца загрузки).
    bootJustWoke = (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT0);
    bootWakeAt = millis();

    Serial.println("[Готов] Нажмите BOOT — начнётся запись команды.");
}

void loop() {
    // OTA и HTTP-диагностика стартуют, как только Wi-Fi реально поднялся.
    if (WiFi.status() == WL_CONNECTED) {
        if (!otaStarted) {
            ArduinoOTA.begin();
            otaStarted = true;
            Serial.printf("[OTA] espota %s:3232, пароль из secrets.h\n",
                          WiFi.localIP().toString().c_str());
        }
        if (!statusServerStarted) {
            startStatusServer();
        }
    }
    if (otaStarted) ArduinoOTA.handle();
    if (statusServerStarted) statusServer.handleClient();

    // Снимок состояния в RTC раз в 5 с: если сейчас watchdog/panic/броунаут,
    // после перезагрузки /status покажет, где именно мы были и сколько heap.
    static uint32_t lastSnapshotAt = 0;
    if (millis() - lastSnapshotAt >= 5000) {
        lastSnapshotAt = millis();
        crashSaveSnapshot();
    }

    // Диагностика: RSSI, WebSocket, heap, uptime — раз в 10 секунд.
    static uint32_t lastDiagAt = 0;
    if (millis() - lastDiagAt >= 10000) {
        lastDiagAt = millis();
        Serial.printf("[DIAG] RSSI=%d dBm, WS=%d, heap=%u, uptime=%lus\n",
                      WiFi.RSSI(), wsClient.isConnected() ? 1 : 0,
                      (unsigned)ESP.getFreeHeap(),
                      (unsigned long)(millis() / 1000));
    }

    // Диагностика сна: раз в 30 секунд, пока стоим в idle.
    static uint32_t lastSleepLogAt = 0;
    if (state == ST_IDLE && !micRunning && millis() - lastSleepLogAt >= 30000) {
        lastSleepLogAt = millis();
        uint32_t age = millis() - lastActivityAt;
        long left = (long)((age >= IDLE_SLEEP_MS) ? 0 : (IDLE_SLEEP_MS - age) / 1000);
        Serial.printf("[Сон] idle=%lus, до сна ~%lds, WS=%d\n",
                      (unsigned long)(age / 1000), left,
                      wsClient.isConnected() ? 1 : 0);
    }

    wsClient.loop();
    handleButton();
    handleSerial();

    // Wi-Fi: переподключаемся, если сеть пропала.
    static uint32_t lastWifiCheck = 0;
    if (millis() - lastWifiCheck > 10000) {
        lastWifiCheck = millis();
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[WiFi] Переподключение...");
            WiFi.reconnect();
        }
    }

    switch (state) {
        case ST_IDLE:
            // Активация — только кнопкой BOOT (или serial 'a'). Автостарт убран.
            if (micRunning) {
                recordStep();
            } else if (!otaInProgress && millis() - lastActivityAt >= IDLE_SLEEP_MS) {
                goToSleep();
            } else if (bootJustWoke) {
                if (millis() - bootWakeAt > 30000) {
                    // Сеть так и не поднялась — не начинаем запись неожиданно позже.
                    bootJustWoke = false;
                } else if (WiFi.status() == WL_CONNECTED && wsClient.isConnected()) {
                    bootJustWoke = false;
                    startListening();
                }
            }
            break;

        case ST_LISTENING:
            recordStep();
            break;

        case ST_THINKING:
            if (millis() - thinkingSince > RESPONSE_TIMEOUT_MS) {
                Serial.println("[Таймаут] Сервер не ответил за 90 с.");
                playTone(300, 150);
                markActivity();
                enterIdle("response timeout");
            }
            break;

        case ST_SPEAKING:
            // Сначала досылаем кольцо в DMA, потом уже решаем, что ответ кончился.
            spkPump();
            if (audioDone && spkCount == 0 && millis() - lastAudioAt > 500) {
                audioDone = false;
                startFollowUpListening();
            }
            break;

        default:
            break;
    }

    ledUpdate();
    yield();

    delay(1);
}
