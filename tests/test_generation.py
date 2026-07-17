"""Tests for code generation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.generator import CodeGenerator


class TestCodeGenerator(unittest.TestCase):
    """Unit tests for the CodeGenerator class."""

    def _mock_model(self, return_code: str = "def foo(): pass") -> MagicMock:
        model = MagicMock()
        model.is_ready = True
        model.generate_code.return_value = return_code
        return model

    # ------------------------------------------------------------------
    # Language normalisation
    # ------------------------------------------------------------------

    def test_normalise_python_aliases(self):
        for alias in ("python", "py", "Python", "PYTHON", "python3"):
            self.assertEqual(CodeGenerator._normalise_language(alias), "python")

    def test_normalise_javascript_aliases(self):
        for alias in ("javascript", "js", "JavaScript", "node", "nodejs"):
            self.assertEqual(CodeGenerator._normalise_language(alias), "javascript")

    def test_normalise_unsupported_raises(self):
        with self.assertRaises(ValueError):
            CodeGenerator._normalise_language("brainfuck")

    # ------------------------------------------------------------------
    # Code extraction
    # ------------------------------------------------------------------

    def test_extract_fenced_code(self):
        raw = "Here is the code:\n```python\ndef hello():\n    pass\n```\nDone."
        result = CodeGenerator._extract_code(raw, "python")
        self.assertEqual(result, "def hello():\n    pass")

    def test_extract_plain_code(self):
        raw = "def hello():\n    pass"
        result = CodeGenerator._extract_code(raw, "python")
        self.assertEqual(result, "def hello():\n    pass")

    def test_extract_strips_trailing_markers(self):
        raw = "def foo(): pass\n### Instruction\nMore stuff"
        result = CodeGenerator._extract_code(raw, "python")
        self.assertEqual(result, "def foo(): pass")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def test_generate_calls_model(self):
        model = self._mock_model("def sort(arr): return sorted(arr)")
        gen = CodeGenerator(model)
        code = gen.generate("Sort a list", "python")
        model.generate_code.assert_called_once()
        self.assertIn("sort", code.lower())

    def test_generate_batch(self):
        model = self._mock_model("code")
        gen = CodeGenerator(model)
        results = gen.generate_batch(["p1", "p2", "p3"], "python")
        self.assertEqual(len(results), 3)
        self.assertEqual(model.generate_code.call_count, 3)

    def test_generate_all_languages(self):
        model = self._mock_model("code")
        gen = CodeGenerator(model)
        results = gen.generate_all_languages("Reverse a string")
        self.assertIn("python", results)
        self.assertIn("javascript", results)
        self.assertEqual(model.generate_code.call_count, len(CodeGenerator.SUPPORTED_LANGUAGES))

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_model_not_ready_raises(self):
        model = MagicMock()
        model.is_ready = False
        with self.assertRaises(RuntimeError):
            CodeGenerator(model)


if __name__ == "__main__":
    unittest.main()
