# Развёртывание LTX-воркера

Воркер — сервис, который живёт на GPU-машине рядом с ComfyUI и выполняет
генерацию видео для Content Factory. Бот общается с ним по HTTPS.

## Архитектура

```
Telegram-бот (VPS)
      │  HTTPS + Bearer-токен
      ▼
TLS-терминатор (см. «Публикация наружу»)
      │  http://127.0.0.1:8080
      ▼
ltx_worker (этот сервис, только loopback)
      │  http://127.0.0.1:8188
      ▼
ComfyUI (только loopback) → GPU
```

Оба сервиса на машине слушают **только 127.0.0.1**. Наружу смотрит только
TLS-терминатор.

## Требования

- GPU с 48+ ГБ VRAM (L40S, A100 80, H100). Конфигурации с 40 ГБ требуют отдельно подтверждённого low-memory профиля и не входят в этот deployment contract.
- Ubuntu 22.04+, Python 3.11+, ~120 ГБ свободного диска.
- Развёрнутый runtime ComfyUI с моделями (см. [`assets/runtime-manifest.json`](assets/runtime-manifest.json);
  файлы проверяются по SHA-256 при каждом запуске задачи).

## Установка рантайма

```bash
# 1. Распаковать pinned-архив ComfyUI (хеши сверяются с receipts)
python3 -m ltx_worker.bootstrap prepare --archive /path/to/comfyui-runtime.tar.zst

# 2. Установить pinned-зависимости в venv рантайма
python3 -m ltx_worker.bootstrap install-dependencies --python /path/to/venv/bin/python

# 3. Модели НЕ скачиваются воркером — их размещает оператор в
#    <comfy_root>/models/... строго по runtime-manifest.json
```

## Конфигурация воркера (env)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LTX_API_TOKEN` | — (обязательно) | Bearer-токен API. Сгенерировать: `openssl rand -hex 24` |
| `LTX_WORKER_HOST` | `127.0.0.1` | Адрес прослушивания. Не менять без TLS-терминатора |
| `LTX_WORKER_PORT` | `8080` | Порт воркера |
| `LTX_STUDIO_ROOT` | корень Studio | Корень расположения рантайма |
| `LTX_COMFY_ROOT` | `<studio>/runtime/ComfyUI` | Корень ComfyUI |
| `LTX_WORKER_ROOT` | `<studio>/worker-data` | Рабочие данные (SQLite, задачи, результаты) |
| `LTX_COMFY_BASE_URL` | `http://127.0.0.1:8188` | URL ComfyUI. Принимается только loopback HTTP |
| `LTX_INFERENCE_ENABLED` | выкл. | Гейт инференса. Включать только после preflight |
| `LTX_INFERENCE_TIMEOUT_SECONDS` | `1800` | Таймаут одной генерации |
| `LTX_MAX_REQUEST_BYTES` | `26214400` | Лимит тела запроса (входное изображение ≤ 20 МБ) |

## Запуск ComfyUI

ComfyUI поднимается оператором отдельным процессом до старта воркера:

```bash
cd <comfy_root>
# 48 ГБ VRAM:
<venv>/bin/python main.py --listen 127.0.0.1 --port 8188 --lowvram --disable-all-custom-nodes
# 80 ГБ VRAM (A100): --lowvram можно опустить
```

## Запуск воркера

```bash
python3 -m ltx_worker   # читает env, поднимает HTTP на 127.0.0.1:8080
```

systemd-юнит (`/etc/systemd/system/ltx-worker.service`):

```ini
[Unit]
Description=LTX video worker
After=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/ltx-worker/worker.env
WorkingDirectory=/opt/ltx-worker
ExecStart=/opt/ltx-worker/venv/bin/python -m ltx_worker
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Перезапуск безопасен: незавершённые задачи переводятся в
`reconciliation_required` и **никогда не перезапускаются автоматически**.

## Публикация наружу (TLS)

Воркер говорит plain HTTP и не терминирует TLS сам. Варианты:

1. **Lightning Studio**: открыть порт 8080 через встроенный механизм портов
   Studio — он выдаёт `https://<порт>-<studio>.cloud.lightning.ai` с TLS.
   Этот URL идёт в `LTX_BASE_URL` бота.
2. **Reverse proxy на той же машине** (caddy/nginx + Let's Encrypt):
   проксировать 443 → `127.0.0.1:8080`.
3. **SSH-туннель с VPS** для разового теста:
   `ssh -R 8080:127.0.0.1:8080 ...` + локальный прокси с TLS на VPS.

`LTX_BASE_URL` на стороне бота обязан быть `https://` — адаптер отклонит
незашифрованную схему.

## Настройка бота (env бота, не воркера)

```
LTX_VIDEO_ENABLED=true          # feature flag, по умолчанию выкл.
LTX_BASE_URL=https://...        # публичный TLS-адрес воркера
LTX_API_TOKEN=<тот же токен>    # хранить только в env-файле, не в git/чате
```

Флаг проверяется в меню бота, в callback-обработчиках и на границе
создания задания (`VideoJobManager.prepare`) — выключенный провайдер
не создаст даже черновик подтверждения.

## Preflight-чеклист

```bash
curl -s http://127.0.0.1:8080/health                       # {"ok": true}, без авторизации
curl -s -H "Authorization: Bearer $LTX_API_TOKEN" \
     http://127.0.0.1:8080/ready                           # готовность рантайма и ComfyUI
```

`LTX_INFERENCE_ENABLED=true` включать только когда `/ready` зелёный.

## Эксплуатационные правила

- Одновременно выполняется **одна** задача (GPU-память и контракт).
- Одна задача = одна физическая отправка в ComfyUI. Повторной отправки нет
  ни при каких ошибках; неоднозначный исход → `reconciliation_required` →
  ручная сверка.
- `job_id` детерминированный (`cf-<idempotency-key>`): повторный POST с тем
  же телом возвращает существующую задачу, с другим телом — 409.
- Результат проверяется (MP4-сигнатура, ffprobe-контракт, лимит размера)
  до выдачи наружу.
