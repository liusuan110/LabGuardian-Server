"""RAG / 检索契约的评测套件。

WP-4 (2026-05-24): 60 题 gold set 是检索契约的"事实标准"。任何后续对
``app/agent/**`` / ``app/services/scene_resolver.py`` /
``app/services/datasheet_kb_service.py`` 的改动都应该跑这里的评测，
而不是只看单元测试通过。schema 与正确性校验见 ``schema.py``。
"""
