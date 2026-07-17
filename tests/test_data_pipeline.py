from __future__ import annotations

from typing import List

import pytest

from src.config.schema import Config
from src.data.streaming import MassiveDataCollector
from src.data.quality import QualityFilter, ExactDeduplicator


SAMPLE_ENTRIES = [
    {"instruction": "def foo", "output": "def foo():\n    pass", "language": "python"},
    {"instruction": "add", "output": "def add(a, b): return a + b", "language": "python"},
]


class TestStreamingCollector:
    def test_clean_text(self):
        assert MassiveDataCollector._clean_text(" hello ") == "hello"
        assert MassiveDataCollector._clean_text(None) == ""
        assert MassiveDataCollector._clean_text(["a", "b"]) == "a\nb"

    def test_normalize_language(self):
        assert MassiveDataCollector._normalize_language("py") == "python"
        assert MassiveDataCollector._normalize_language("js") == "javascript"
        assert MassiveDataCollector._normalize_language("c++") == "cpp"
        assert MassiveDataCollector._normalize_language("python") == "python"

    def test_is_valid_sample(self):
        long_code = "def foo():" + " x = 1\n" * 20
        assert MassiveDataCollector._is_valid_sample(long_code, "python")
        assert not MassiveDataCollector._is_valid_sample("hi", "python")
        assert not MassiveDataCollector._is_valid_sample("lorem ipsum dolor sit amet, consectetur adipiscing elit", "python")

    def test_detect_language(self):
        assert MassiveDataCollector._detect_language("def foo(): import os") == "python"
        assert MassiveDataCollector._detect_language("fn main() { let x = 1; }") == "rust"
        assert MassiveDataCollector._detect_language("const hello = () => { return 1; }") == "javascript"
        assert MassiveDataCollector._detect_language("The quick brown fox is jumping") == "text"
        assert MassiveDataCollector._detect_language("public class Hello { void main() {} }") == "java"

    def test_extract_standard_fields(self):
        collector = MassiveDataCollector([])
        entry = {"instruction": "write a function", "input": "add two numbers", "output": "def add(a,b): return a+b"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "write a function"
        assert inp == "add two numbers"
        assert out == "def add(a,b): return a+b"

    def test_extract_problem_solution(self):
        collector = MassiveDataCollector([])
        entry = {"problem": "solve x+2=5", "solution": "x=3"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "solve x+2=5"
        assert out == "x=3"

    def test_extract_question_answer(self):
        collector = MassiveDataCollector([])
        entry = {"question": "what is 2+2?", "answer": "4"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "what is 2+2?"
        assert out == "4"

    def test_extract_dolly_format(self):
        collector = MassiveDataCollector([])
        entry = {"context": "math context", "instruction": "compute 2+2", "response": "4"}
        inst, inp, out = collector._extract_fields(entry)
        assert "math context" in inst
        assert "compute 2+2" in inst
        assert out == "4"

    def test_extract_flan_format(self):
        collector = MassiveDataCollector([])
        entry = {"inputs": "translate to french: hello", "targets": "bonjour"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "translate to french: hello"
        assert out == "bonjour"

    def test_extract_sharegpt_fields(self):
        collector = MassiveDataCollector([])
        human = {"from": "human", "value": "hello there"}
        gpt = {"from": "gpt", "value": "hi how can I help"}
        inst1, _, _ = collector._extract_sharegpt_fields(human)
        assert inst1 == "hello there"
        _, _, out2 = collector._extract_sharegpt_fields(gpt)
        assert out2 == "hi how can I help"

    def test_extract_chat_messages(self):
        collector = MassiveDataCollector([])
        entry = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "hello"
        assert out == "hi there"

    def test_extract_orca_style(self):
        collector = MassiveDataCollector([])
        entry = {"system_prompt": "you are a math tutor", "question": "what is 2+2?", "response": "4"}
        inst, inp, out = collector._extract_fields(entry)
        assert "you are a math tutor" in inst
        assert "what is 2+2?" in inst
        assert out == "4"

    def test_extract_tool_use(self):
        collector = MassiveDataCollector([])
        entry = {"tool_definition": "fn add(a,b) -> int", "instruction": "call add(1,2)", "response": "3"}
        inst, inp, out = collector._extract_fields(entry)
        assert "fn add(a,b) -> int" in inst
        assert "call add(1,2)" in inst
        assert out == "3"

    def test_extract_code_explanation(self):
        collector = MassiveDataCollector([])
        entry = {"code": "print('hello')", "explanation": "prints hello"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "prints hello"
        assert out == "print('hello')"

    def test_extract_text_fallback(self):
        collector = MassiveDataCollector([])
        entry = {"text": "some long article text about programming in python"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == ""
        assert out == "some long article text about programming in python"

    def test_extract_natural_question(self):
        collector = MassiveDataCollector([])
        entry = {"sentence1": "the cat sat", "sentence2": "the cat sat on mat", "label": "entailment"}
        inst, inp, out = collector._extract_fields(entry)
        assert inst == "the cat sat"
        assert inp == "the cat sat on mat"
        assert out == "entailment"

    def test_valid_sample_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5
        assert MassiveDataCollector._is_valid_sample(text, "text")

    def test_valid_sample_rejects_too_short(self):
        assert not MassiveDataCollector._is_valid_sample("short", "text")


class TestQualityFilter:
    def test_valid_content(self):
        assert QualityFilter.check_length("hello world " * 10, min_len=50, max_len=250000)
        assert not QualityFilter.check_length("short", min_len=50)
        assert QualityFilter.is_high_quality_content("def valid_function(): pass")

    def test_contamination_detection(self):
        from src.data.quality import ContaminationFilter
        cf = ContaminationFilter(benchmarks=["human_eval"])
        assert cf.is_contaminated("def check_solution(): assert result == 42")
        assert not cf.is_contaminated("def normal(x): return x")


class TestDeduplicator:
    def test_exact_dedup(self):
        dedup = ExactDeduplicator()
        assert not dedup.is_duplicate("unique text")
        assert dedup.is_duplicate("unique text")
        assert not dedup.is_duplicate("other text")

    def test_exact_dedup_reset(self):
        dedup = ExactDeduplicator()
        dedup.is_duplicate("test")
        dedup.reset()
        assert not dedup.is_duplicate("test")
