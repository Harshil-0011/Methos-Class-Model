from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


_CONTAMINATION_BENCHMARK_PATTERNS: Dict[str, List[str]] = {
    "human_eval": [
        r"def\s+(check\w*|candidate|human_eval)", r"def test_", r"assert.*==",
    ],
    "mbpp": [
        r"\"\"\".*>>>.*", r"def\s+check\w+",
    ],
    "mmlu": [
        r"(?:Answer|answer):\s*[A-D]", r"Question \d+:", r"\([A-D]\)\s",
    ],
    "gsm8k": [
        r"####\s+\-?\d+", r"Let's think step by step", r"<<.*=.*>>",
    ],
    "arc": [
        r"grid\s*=\s*\[\[", r"A\.\s*\[", r"output_grid",
    ],
}


class QualityFilter:
    LOW_QUALITY_MARKERS = [
        "lorem ipsum", "todo: add code", "your code here",
        "coming soon", "placeholder", "fixme", "not implemented",
    ]

    @staticmethod
    def check_length(content: str, min_len: int = 50, max_len: int = 250000) -> bool:
        return min_len <= len(content) <= max_len

    @staticmethod
    def is_high_quality_content(content: str) -> bool:
        """Returns True if content has NO low-quality markers (i.e., is high quality)."""
        lowered = content.lower()
        return not any(m in lowered for m in QualityFilter.LOW_QUALITY_MARKERS)

    @staticmethod
    def check_language_markers(content: str, language: str) -> bool:
        signals = {
            "python": [r"\bdef\s+", r"\bimport\s+", r"\bclass\s+"],
            "rust": [r"\bfn\s+", r"\blet\s+", r"\buse\s+"],
            "golang": [r"\bfunc\s+", r"\bpackage\s+"],
            "cpp": [r"#include", r"\bint\s+\w+\("],
            "java": [r"\bclass\s+", r"\bpublic\s+"],
            "typescript": [r"\binterface\s+", r"\bexport\s+"],
            "javascript": [r"\bfunction\b", r"\bconst\s+", r"\blet\s+"],
            "text": [r".{80}"],
        }
        sigs = signals.get(language, [r"\s+"])
        return any(re.search(s, content) for s in sigs)


class ExactDeduplicator:
    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def is_duplicate(self, content: str) -> bool:
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def reset(self) -> None:
        self._seen.clear()


class MinHashDeduplicator:
    def __init__(self, threshold: float = 0.85, num_hashes: int = 128) -> None:
        self.threshold = threshold
        self.num_hashes = num_hashes
        self._signatures: List[Tuple[int, ...]] = []

    def _shingles(self, text: str, k: int = 5) -> List[str]:
        return [text[i:i + k] for i in range(max(1, len(text) - k + 1))]

    def _signature(self, text: str) -> Tuple[int, ...]:
        shingles = self._shingles(text)
        sig = []
        for seed in range(1, self.num_hashes + 1):
            min_hash = min(hashlib.sha256((str(seed) + s).encode()).hexdigest() for s in shingles)
            sig.append(hash(min_hash))
        return tuple(sig)

    def is_duplicate(self, content: str) -> bool:
        sig = self._signature(content)
        for other in self._signatures:
            matches = sum(1 for a, b in zip(sig, other) if a == b)
            similarity = matches / self.num_hashes
            if similarity >= self.threshold:
                return True
        self._signatures.append(sig)
        return False


class ContaminationFilter:
    def __init__(self, benchmarks: Optional[List[str]] = None) -> None:
        self.benchmarks = benchmarks or list(_CONTAMINATION_BENCHMARK_PATTERNS.keys())
        self._patterns: List[re.Pattern] = []
        for bm in self.benchmarks:
            for pat in _CONTAMINATION_BENCHMARK_PATTERNS.get(bm, []):
                self._patterns.append(re.compile(pat, re.IGNORECASE))

    def is_contaminated(self, content: str) -> bool:
        return any(p.search(content) for p in self._patterns)

    def filter_items(self, items: List[Dict]) -> List[Dict]:
        clean = []
        for item in items:
            text = " ".join(str(v) for v in item.values())
            if not self.is_contaminated(text):
                clean.append(item)
        return clean


class QualityScorer:
    @staticmethod
    def heuristic_score(content: str) -> float:
        score = 0.0
        length = len(content)
        if 200 <= length <= 50000:
            score += 0.3
        elif length > 50000:
            score += 0.2
        else:
            score += 0.1
        code_indicators = sum(content.count(c) for c in ["def ", "class ", "import ", "func ", "function "])
        score += min(0.3, code_indicators * 0.02)
        has_docstring = '"""' in content or "'''" in content or "/**" in content
        if has_docstring:
            score += 0.2
        has_types = bool(re.search(r":[ \t]*\w+[\[\(]?", content))
        if has_types:
            score += 0.1
        has_tests = bool(re.search(r"(?:def test|assert |unittest|pytest)", content))
        if has_tests:
            score += 0.1
        return min(1.0, score)
