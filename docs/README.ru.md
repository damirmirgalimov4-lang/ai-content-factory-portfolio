# AI Content Factory — кратко по-русски

Это Telegram-first платформа для управляемого производства коротких AI-видео. Она проводит идею через роли продюсера, сценариста, раскадровщика и prompt engineer, проверяет структурные контракты, создаёт визуальные материалы и только после подтверждения отправляет платную видеогенерацию.

## Что демонстрирует проект

- управляемый agentic workflow вместо одного prompt-вызова;
- машинные контракты для сцен, референсов и изображений;
- human-in-the-loop подтверждение каждого смыслового этапа;
- защита от повторных платных запросов;
- несколько video provider adapters;
- отдельный authenticated LTX/ComfyUI worker;
- SQLite/WAL, восстановление jobs и управление удалённой GPU VM;
- большой provider-free regression suite;
- security/privacy gate для публичного репозитория.

## Проверка без ключей и платных API

```bash
python3 scripts/security_scan.py
python3 -m compileall -q agent_platform ltx_worker tests scripts
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Важные границы

Проект показывает Applied AI / LLM Systems Engineering, а не обучение моделей. LTX и платные провайдеры выключены по умолчанию. Автоматический LLM-failover и горизонтальное масштабирование остаются в roadmap.

Основная техническая документация написана на английском:

- [Architecture](ARCHITECTURE.md)
- [Demo](DEMO.md)
- [Evaluation](EVALUATION.md)
- [Security](SECURITY.md)
- [Limitations](LIMITATIONS.md)
