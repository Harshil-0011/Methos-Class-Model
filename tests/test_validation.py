"""Tests for code validation."""

from __future__ import annotations

import unittest

from src.validator import CodeValidator


class TestCodeValidator(unittest.TestCase):
    """Unit tests for the CodeValidator class."""

    def setUp(self):
        self.validator = CodeValidator()

    # ------------------------------------------------------------------
    # Python syntax
    # ------------------------------------------------------------------

    def test_python_valid_syntax(self):
        code = "def add(a, b):\n    return a + b\n"
        result = self.validator.check_syntax(code, "python")
        self.assertTrue(result.success)

    def test_python_invalid_syntax(self):
        code = "def add(a, b)\n    return a + b\n"  # missing colon
        result = self.validator.check_syntax(code, "python")
        self.assertFalse(result.success)
        self.assertIn("SyntaxError", result.error)

    # ------------------------------------------------------------------
    # Python execution
    # ------------------------------------------------------------------

    def test_python_execution_success(self):
        code = "print(1 + 2)"
        result = self.validator.execute(code, "python")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "3")

    def test_python_execution_error(self):
        code = "raise ValueError('boom')"
        result = self.validator.execute(code, "python")
        self.assertFalse(result.success)
        self.assertIn("ValueError", result.error)

    def test_python_execution_timeout(self):
        code = "while True: pass"
        result = self.validator.execute(code, "python", timeout=1)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def test_python_validate_output_match(self):
        code = "print('hello world')"
        result = self.validator.validate_output(code, "hello world", "python")
        self.assertTrue(result.success)

    def test_python_validate_output_mismatch(self):
        code = "print('foo')"
        result = self.validator.validate_output(code, "bar", "python")
        self.assertFalse(result.success)
        self.assertIn("mismatch", result.error.lower())

    # ------------------------------------------------------------------
    # Batch validation
    # ------------------------------------------------------------------

    def test_batch_validation(self):
        codes = [
            "print('ok')",
            "def f(): return 1",
            "invalid((",
        ]
        results = self.validator.validate_batch(codes, "python")
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].success)   # runs fine
        self.assertTrue(results[1].success)   # runs fine (no output, no error)
        self.assertFalse(results[2].success)  # syntax error

    # ------------------------------------------------------------------
    # Unsupported language
    # ------------------------------------------------------------------

    def test_unsupported_language_syntax(self):
        result = self.validator.check_syntax("++++++++", "brainfuck")
        self.assertFalse(result.success)

    def test_unsupported_language_execute(self):
        result = self.validator.execute("++++++++", "brainfuck")
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
