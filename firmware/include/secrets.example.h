#pragma once

// Скопируй этот файл в secrets.h и заполни своими значениями.
// secrets.h добавлен в .gitignore и не должен попадать в репозиторий.

// Wi-Fi (ESP32-S3 умеет только 2.4 ГГц)
#define WIFI_SSID       "YourWiFi"
#define WIFI_PASSWORD   "your-wifi-password"

// Адрес машины, на которой запущен voice_server.py
#define SERVER_HOST     "192.168.1.100"

// Пароль ArduinoOTA / espota (порт 3232/UDP)
#define OTA_PASSWORD    "change-me-please"
