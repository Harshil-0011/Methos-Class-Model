from __future__ import annotations

from src.data.quality import (
    QualityFilter,
    ExactDeduplicator,
    MinHashDeduplicator,
    ContaminationFilter,
    QualityScorer,
)


class TestQualityFilter:
    def test_length_check_passes(self):
        assert QualityFilter.check_length("x" * 100)
        assert not QualityFilter.check_length("x" * 10)
        assert not QualityFilter.check_length("x" * 300000)

    def test_low_quality_markers(self):
        assert not QualityFilter.is_high_quality_content("lorem ipsum dolor sit amet")
        assert not QualityFilter.is_high_quality_content("todo: add code here")
        assert QualityFilter.is_high_quality_content("def foo():\n    pass")

    def test_language_markers_python(self):
        assert QualityFilter.check_language_markers("def foo():\n    pass", "python")
        assert not QualityFilter.check_language_markers("hello world", "python")

    def test_language_markers_javascript(self):
        assert QualityFilter.check_language_markers("function foo() {}", "javascript")
        assert not QualityFilter.check_language_markers("hello world", "javascript")


class TestExactDeduplicator:
    def test_deduplicates_exact_matches(self):
        dedup = ExactDeduplicator()
        assert not dedup.is_duplicate("hello")
        assert dedup.is_duplicate("hello")
        assert not dedup.is_duplicate("world")
        assert dedup.is_duplicate("world")

    def test_reset(self):
        dedup = ExactDeduplicator()
        dedup.is_duplicate("test")
        assert dedup.is_duplicate("test")
        dedup.reset()
        assert not dedup.is_duplicate("test")


class TestContaminationFilter:
    def test_detects_human_eval(self):
        cf = ContaminationFilter(benchmarks=["human_eval"])
        assert cf.is_contaminated("def check_solution(): assert result == 3")
        assert not cf.is_contaminated("def normal_function(x): return x + 1")

    def test_detects_gsm8k(self):
        cf = ContaminationFilter(benchmarks=["gsm8k"])
        assert cf.is_contaminated("Let's think step by step. The answer is #### 42")
        assert not cf.is_contaminated("The total is 42.")


class TestQualityScorer:
    def test_heuristic_score_range(self):
        score = QualityScorer.heuristic_score("def foo():\n    \"\"\"doc\"\"\"\n    pass")
        assert 0.0 <= score <= 1.0

    def test_code_scores_higher_than_plain_text(self):
        code = "def foo():\n    \"\"\"docstring\"\"\"\n    return [x for x in range(10)]"
        text = "hello world " * 10
        assert QualityScorer.heuristic_score(code) > QualityScorer.heuristic_score(text)
