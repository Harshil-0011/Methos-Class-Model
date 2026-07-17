from __future__ import annotations

from src.evaluation.safety import SafetyEvaluator
from src.evaluation.benchmarks import HumanEvalBenchmark, BenchmarkResult


class TestSafetyEvaluator:
    def test_refusal_keywords_detected(self):
        keywords = SafetyEvaluator.REFUSAL_KEYWORDS
        assert "cannot" in keywords
        assert "I'm sorry" in keywords
        assert len(keywords) > 5

    def test_safety_probes_exist(self):
        assert len(SafetyEvaluator.SAFETY_PROBES) > 0
        categories = {p["category"] for p in SafetyEvaluator.SAFETY_PROBES}
        assert "harmful_code" in categories
        assert "refusal_test" in categories

    def test_honesty_probes_exist(self):
        assert len(SafetyEvaluator.HONESTY_PROBES) > 0


class TestBenchmarkResult:
    def test_representation(self):
        r = BenchmarkResult("test", 0.75, {"passed": 3, "total": 4, "time_seconds": 10.0})
        assert "test" in repr(r)
        assert "0.75" in repr(r)
        assert r.score == 0.75
        assert r.passed == 3
        assert r.total == 4


class TestHumanEvalBenchmark:
    def test_code_extraction(self):
        extracted = HumanEvalBenchmark._extract_code("```python\nprint('hello')\n```")
        assert extracted == "print('hello')"

        extracted2 = HumanEvalBenchmark._extract_code("print('no fences')")
        assert extracted2 == "print('no fences')"
