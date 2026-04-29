#!/usr/bin/env python3
"""Unit-тесты для is_suspicious_chunk и quality_score из whisper_server.py.

Загружает модуль без запуска HTTP-сервера (через importlib и заглушку
faster_whisper, если пакета нет). Запускается как обычный скрипт:

    python scripts/test_fallback_heuristics.py
"""

import os
import sys
import importlib.util
import types

HERE = os.path.dirname(os.path.abspath(__file__))
# faster_whisper требуется whisper_server.py при импорте — подсовываем заглушку,
# чтобы тесты шли без зависимости.
if "faster_whisper" not in sys.modules:
    fake = types.ModuleType("faster_whisper")
    class _FakeModel:  # noqa: D401 — stub
        def __init__(self, *a, **kw): pass
        def transcribe(self, *a, **kw): return iter([]), None
    fake.WhisperModel = _FakeModel
    sys.modules["faster_whisper"] = fake

spec = importlib.util.spec_from_file_location(
    "whisper_server", os.path.join(HERE, "whisper_server.py")
)
ws = importlib.util.module_from_spec(spec)
# whisper_server.py выполняет загрузку модели при импорте — обходим через
# переменную окружения, задав короткую модель. Но model.__init__ в заглушке
# всё равно no-op.
os.environ.setdefault("WHISPER_DEVICE", "cpu")
os.environ.setdefault("WHISPER_COMPUTE_TYPE", "int8")
spec.loader.exec_module(ws)


def _case(name, cond):
    if cond:
        print(f"  OK  {name}")
    else:
        print(f"  FAIL {name}")
        sys.exit(1)


def test_is_suspicious():
    print("is_suspicious_chunk:")
    # Clean clinical text — никаких причин
    clean = "АД 140/90 мм рт.ст., ЧСС 78 уд/мин. Холестерин 5,2 ммоль/л."
    _case("clean text -> no reasons", is_clean := not ws.is_suspicious_chunk(clean, -0.3))

    # Low log-prob
    reasons = ws.is_suspicious_chunk("Пациент спокоен", -1.1)
    _case("low logprob -> suspicious", any(r.startswith("low_logprob") for r in reasons))

    # BP mentioned but no X/Y format (Whisper-дрифт: «165 сотых миллиметра»)
    reasons = ws.is_suspicious_chunk("АД 165 сотых миллиметра ртутного статья", -0.5)
    _case("bp without X/Y format -> suspicious", "bp_without_format" in reasons)

    # BP with X/Y format — OK
    reasons = ws.is_suspicious_chunk("АД 142/89 мм рт.ст.", -0.5)
    _case("bp with X/Y format -> not bp-suspicious", "bp_without_format" not in reasons)

    # Посев мочи без организма
    reasons = ws.is_suspicious_chunk("Посев мочи: рост 10⁵ КОЕ/мл.", -0.5)
    _case("culture without organism -> suspicious", "culture_without_organism" in reasons)

    # Посев с E.coli — OK
    reasons = ws.is_suspicious_chunk("Посев мочи: Escherichia coli 10⁵ КОЕ/мл.", -0.5)
    _case("culture with organism -> not culture-suspicious", "culture_without_organism" not in reasons)

    # Мусорные токены единиц — «коэслч», «слчэль», «мга» и т.д.
    reasons = ws.is_suspicious_chunk("Гемоглобин 128 гэслчэль, лейкоциты 10 коэслч.", -0.5)
    _case("garbage unit tokens -> suspicious", "garbage_unit_tokens" in reasons)

    # Петля
    reasons = ws.is_suspicious_chunk("снижение памяти, снижение памяти, снижение памяти, снижение памяти", -0.5)
    _case("phrase loop -> suspicious", "phrase_loop" in reasons)


def test_quality_score():
    print("quality_score:")
    good = "АД 142/89 мм рт.ст. Бисопролол 2,5 мг. Глюкоза 5,4 ммоль/л. Посев мочи: Escherichia coli 10⁵ КОЕ/мл."
    bad = "АД 165 сотых миллиметра ртутного статья. Гемоглобин 128 гэслчэль. Посев мочи, рост."
    q_good = ws.quality_score(good)
    q_bad = ws.quality_score(bad)
    _case(f"clinical text wins over garbage ({q_good:.1f} > {q_bad:.1f})", q_good > q_bad)

    # BP X/Y вклад
    _case("bp format adds score",
          ws.quality_score("АД 120/80 мм рт.ст.") > ws.quality_score("АД мм рт.ст."))
    # Dose+unit вклад
    _case("dose+unit adds score",
          ws.quality_score("Бисопролол 2,5 мг") > ws.quality_score("Бисопролол по таблетке"))
    # КОЕ/мл вклад
    _case("cfu adds score",
          ws.quality_score("рост 10⁵ КОЕ/мл") > ws.quality_score("рост 10 5 коэслч"))
    # Penalty: мусорный токен
    _case("garbage token penalizes",
          ws.quality_score("гемоглобин 128 г/л") > ws.quality_score("гемоглобин 128 гэслчэль"))


if __name__ == "__main__":
    test_is_suspicious()
    test_quality_score()
    print("all heuristics tests passed")
