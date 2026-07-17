#!/usr/bin/env python3
"""
Professional-grade coding benchmark for the Specialized Coding Model.

Inspired by HumanEval, MBPP, LiveCodeBench, and SWE-bench.
Each problem has:
  - A natural-language description with constraints
  - A required function signature
  - Hidden test cases with assertions (execution-based scoring)
  - Difficulty tier (easy / medium / hard / expert)
  - Category tags (algorithms, data-structures, math, strings, etc.)

Scoring uses pass@1: the model must produce code that passes ALL test
cases on the first attempt to earn credit.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.validator import CodeValidator  # type: ignore[import]

logger = logging.getLogger(__name__)


# ===================================================================
# Data structures
# ===================================================================


@dataclass
class BenchmarkProblem:
    """A single benchmark problem with hidden test cases."""

    id: str
    title: str
    description: str
    signature: str  # Required function signature
    test_code: str  # Python assertions that call the function
    difficulty: str  # easy | medium | hard | expert
    category: str  # algorithms, data-structures, math, strings, etc.
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of evaluating one problem."""

    problem_id: str
    title: str
    difficulty: str
    category: str
    passed: bool
    syntax_valid: bool
    error: str = ""
    generated_code: str = ""
    elapsed_seconds: float = 0.0


# ===================================================================
# Problem bank — 30 problems across 5 difficulty tiers
# ===================================================================

BENCHMARK_PROBLEMS: List[BenchmarkProblem] = [
    # ---------------------------------------------------------------
    #  EASY (6 problems)
    # ---------------------------------------------------------------
    BenchmarkProblem(
        id="E01",
        title="Two Sum",
        description=(
            "Given a list of integers `nums` and an integer `target`, return the "
            "indices of the two numbers that add up to `target`. You may assume "
            "each input has exactly one solution, and you may not use the same "
            "element twice. Return the answer as a list of two indices."
        ),
        signature="def two_sum(nums: list[int], target: int) -> list[int]:",
        test_code="""\
assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]
assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]
assert sorted(two_sum([3, 3], 6)) == [0, 1]
assert sorted(two_sum([1, 5, 3, 7, 2], 9)) == [1, 3]
assert sorted(two_sum([-1, -2, -3, -4, -5], -8)) == [2, 4]
""",
        difficulty="easy",
        category="hash-table",
        tags=["arrays", "hash-map"],
    ),
    BenchmarkProblem(
        id="E02",
        title="Valid Parentheses",
        description=(
            "Given a string `s` containing just the characters '(', ')', '{', '}', "
            "'[' and ']', determine if the input string is valid. An input string "
            "is valid if: open brackets are closed by the same type, and in the "
            "correct order. Every close bracket has a corresponding open bracket."
        ),
        signature="def is_valid(s: str) -> bool:",
        test_code="""\
assert is_valid("()") == True
assert is_valid("()[]{}") == True
assert is_valid("(]") == False
assert is_valid("([)]") == False
assert is_valid("{[]}") == True
assert is_valid("") == True
assert is_valid("((((") == False
assert is_valid("({[)") == False
""",
        difficulty="easy",
        category="stack",
        tags=["strings", "stack"],
    ),
    BenchmarkProblem(
        id="E03",
        title="Palindrome Check",
        description=(
            "Given a string `s`, return `True` if it is a palindrome considering "
            "only alphanumeric characters and ignoring cases. Return `False` otherwise."
        ),
        signature="def is_palindrome(s: str) -> bool:",
        test_code="""\
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome(" ") == True
assert is_palindrome("0P") == False
assert is_palindrome("Was it a car or a cat I saw?") == True
assert is_palindrome("No 'x' in Nixon") == True
""",
        difficulty="easy",
        category="strings",
        tags=["two-pointers", "strings"],
    ),
    BenchmarkProblem(
        id="E04",
        title="Roman to Integer",
        description=(
            "Convert a Roman numeral string to an integer. Roman numerals: "
            "I=1, V=5, X=10, L=50, C=100, D=500, M=1000. "
            "Subtraction rules: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900."
        ),
        signature="def roman_to_int(s: str) -> int:",
        test_code="""\
assert roman_to_int("III") == 3
assert roman_to_int("LVIII") == 58
assert roman_to_int("MCMXCIV") == 1994
assert roman_to_int("IV") == 4
assert roman_to_int("IX") == 9
assert roman_to_int("XLII") == 42
assert roman_to_int("CDXLIV") == 444
assert roman_to_int("MMMCMXCIX") == 3999
""",
        difficulty="easy",
        category="math",
        tags=["strings", "math"],
    ),
    BenchmarkProblem(
        id="E05",
        title="Merge Two Sorted Lists",
        description=(
            "Given two sorted lists of integers, merge them into one sorted list "
            "and return it. The result should also be sorted in ascending order."
        ),
        signature="def merge_sorted(list1: list[int], list2: list[int]) -> list[int]:",
        test_code="""\
assert merge_sorted([1, 2, 4], [1, 3, 4]) == [1, 1, 2, 3, 4, 4]
assert merge_sorted([], []) == []
assert merge_sorted([], [0]) == [0]
assert merge_sorted([1], []) == [1]
assert merge_sorted([1, 3, 5, 7], [2, 4, 6, 8]) == [1, 2, 3, 4, 5, 6, 7, 8]
assert merge_sorted([-3, -1, 0], [-2, 2, 4]) == [-3, -2, -1, 0, 2, 4]
""",
        difficulty="easy",
        category="sorting",
        tags=["two-pointers", "merge"],
    ),
    BenchmarkProblem(
        id="E06",
        title="FizzBuzz",
        description=(
            "Return a list of strings from 1 to n. For multiples of 3, use "
            "'Fizz'; for multiples of 5, use 'Buzz'; for multiples of both, "
            "use 'FizzBuzz'; otherwise, use the string representation of the number."
        ),
        signature="def fizzbuzz(n: int) -> list[str]:",
        test_code="""\
assert fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"]
assert fizzbuzz(15)[-1] == "FizzBuzz"
assert fizzbuzz(1) == ["1"]
assert fizzbuzz(3) == ["1", "2", "Fizz"]
assert len(fizzbuzz(100)) == 100
assert fizzbuzz(15).count("FizzBuzz") == 1
""",
        difficulty="easy",
        category="math",
        tags=["loops", "conditionals"],
    ),

    # ---------------------------------------------------------------
    #  MEDIUM (8 problems)
    # ---------------------------------------------------------------
    BenchmarkProblem(
        id="M01",
        title="Longest Substring Without Repeating Characters",
        description=(
            "Given a string `s`, find the length of the longest substring "
            "without repeating characters."
        ),
        signature="def length_of_longest_substring(s: str) -> int:",
        test_code="""\
assert length_of_longest_substring("abcabcbb") == 3
assert length_of_longest_substring("bbbbb") == 1
assert length_of_longest_substring("pwwkew") == 3
assert length_of_longest_substring("") == 0
assert length_of_longest_substring(" ") == 1
assert length_of_longest_substring("dvdf") == 3
assert length_of_longest_substring("abcdefghijklmnop") == 16
assert length_of_longest_substring("aab") == 2
""",
        difficulty="medium",
        category="sliding-window",
        tags=["hash-map", "strings"],
    ),
    BenchmarkProblem(
        id="M02",
        title="Group Anagrams",
        description=(
            "Given a list of strings `strs`, group the anagrams together. "
            "An anagram is a word formed by rearranging the letters of another. "
            "Return a list of groups (each group is a sorted list of strings). "
            "The groups themselves can be in any order."
        ),
        signature="def group_anagrams(strs: list[str]) -> list[list[str]]:",
        test_code="""\
result = group_anagrams(["eat","tea","tan","ate","nat","bat"])
result = [sorted(g) for g in result]
result.sort()
assert result == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]

result2 = group_anagrams([""])
assert result2 == [[""]]

result3 = group_anagrams(["a"])
assert result3 == [["a"]]
""",
        difficulty="medium",
        category="hash-table",
        tags=["strings", "sorting"],
    ),
    BenchmarkProblem(
        id="M03",
        title="Product of Array Except Self",
        description=(
            "Given an integer array `nums`, return an array `answer` where "
            "`answer[i]` is the product of all elements of `nums` except "
            "`nums[i]`. You must solve it in O(n) time WITHOUT using division."
        ),
        signature="def product_except_self(nums: list[int]) -> list[int]:",
        test_code="""\
assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
assert product_except_self([2, 3]) == [3, 2]
assert product_except_self([1, 1, 1, 1]) == [1, 1, 1, 1]
assert product_except_self([0, 0]) == [0, 0]
assert product_except_self([5]) == [1]
""",
        difficulty="medium",
        category="arrays",
        tags=["prefix-sum"],
    ),
    BenchmarkProblem(
        id="M04",
        title="3Sum",
        description=(
            "Given an integer array `nums`, return all the triplets "
            "`[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, "
            "`j != k`, and `nums[i] + nums[j] + nums[k] == 0`. "
            "The solution set must not contain duplicate triplets. "
            "Return each triplet sorted, and the overall list sorted."
        ),
        signature="def three_sum(nums: list[int]) -> list[list[int]]:",
        test_code="""\
result = three_sum([-1, 0, 1, 2, -1, -4])
result = sorted([sorted(t) for t in result])
assert result == [[-1, -1, 2], [-1, 0, 1]]

assert three_sum([0, 1, 1]) == []
assert three_sum([0, 0, 0]) == [[0, 0, 0]]
assert three_sum([]) == []
assert three_sum([1, -1]) == []

result2 = three_sum([-2, 0, 1, 1, 2])
result2 = sorted([sorted(t) for t in result2])
assert result2 == [[-2, 0, 2], [-2, 1, 1]]
""",
        difficulty="medium",
        category="two-pointers",
        tags=["sorting", "arrays"],
    ),
    BenchmarkProblem(
        id="M05",
        title="LRU Cache",
        description=(
            "Design a Least Recently Used (LRU) cache class with `get(key)` "
            "and `put(key, value)` methods. `get` returns the value or -1 if "
            "not found. `put` inserts or updates; if capacity is exceeded, "
            "evict the least recently used item. Both must run in O(1) average time."
        ),
        signature="class LRUCache:\n    def __init__(self, capacity: int):\n        pass\n    def get(self, key: int) -> int:\n        pass\n    def put(self, key: int, value: int) -> None:\n        pass",
        test_code="""\
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(4, 4)
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4

cache2 = LRUCache(1)
cache2.put(2, 1)
assert cache2.get(2) == 1
cache2.put(3, 2)
assert cache2.get(2) == -1
assert cache2.get(3) == 2
""",
        difficulty="medium",
        category="design",
        tags=["hash-map", "linked-list"],
    ),
    BenchmarkProblem(
        id="M06",
        title="Binary Tree Level Order Traversal",
        description=(
            "Given a binary tree represented as a list (level-order with None "
            "for missing nodes), return its level order traversal as a list of "
            "lists. First, reconstruct the tree from the list, then traverse it."
        ),
        signature="def level_order(tree_list: list) -> list[list[int]]:",
        test_code="""\
assert level_order([3, 9, 20, None, None, 15, 7]) == [[3], [9, 20], [15, 7]]
assert level_order([1]) == [[1]]
assert level_order([]) == []
assert level_order([1, 2, 3, 4, 5]) == [[1], [2, 3], [4, 5]]
assert level_order([1, None, 2]) == [[1], [2]]
""",
        difficulty="medium",
        category="trees",
        tags=["bfs", "binary-tree"],
    ),
    BenchmarkProblem(
        id="M07",
        title="Coin Change",
        description=(
            "Given an array of coin denominations `coins` and a total `amount`, "
            "return the fewest number of coins needed to make up that amount. "
            "If it cannot be made, return -1. You have infinite supply of each coin."
        ),
        signature="def coin_change(coins: list[int], amount: int) -> int:",
        test_code="""\
assert coin_change([1, 5, 10, 25], 30) == 2
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
assert coin_change([1, 2, 5], 11) == 3
assert coin_change([186, 419, 83, 408], 6249) == 20
assert coin_change([1], 1) == 1
assert coin_change([1], 2) == 2
assert coin_change([3, 7], 11) == -1
""",
        difficulty="medium",
        category="dynamic-programming",
        tags=["dp", "greedy"],
    ),
    BenchmarkProblem(
        id="M08",
        title="Top K Frequent Elements",
        description=(
            "Given an integer array `nums` and an integer `k`, return the `k` most "
            "frequent elements. You may return the answer in any order but it must "
            "be sorted in descending order of frequency."
        ),
        signature="def top_k_frequent(nums: list[int], k: int) -> list[int]:",
        test_code="""\
assert set(top_k_frequent([1,1,1,2,2,3], 2)) == {1, 2}
assert top_k_frequent([1], 1) == [1]
assert set(top_k_frequent([4,4,4,4,3,3,3,2,2,1], 2)) == {4, 3}
assert len(top_k_frequent([1,2,3,4,5], 3)) == 3
assert set(top_k_frequent([5,5,5,5,1,1,1,2,2,3], 3)) == {5, 1, 2}
""",
        difficulty="medium",
        category="heap",
        tags=["hash-map", "sorting", "bucket-sort"],
    ),

    # ---------------------------------------------------------------
    #  HARD (8 problems)
    # ---------------------------------------------------------------
    BenchmarkProblem(
        id="H01",
        title="Longest Increasing Subsequence",
        description=(
            "Given an integer array `nums`, return the length of the longest "
            "strictly increasing subsequence. An O(n log n) solution is expected."
        ),
        signature="def length_of_lis(nums: list[int]) -> int:",
        test_code="""\
assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4
assert length_of_lis([7, 7, 7, 7, 7, 7, 7]) == 1
assert length_of_lis([]) == 0
assert length_of_lis([1]) == 1
assert length_of_lis([1, 3, 6, 7, 9, 4, 10, 5, 6]) == 6
assert length_of_lis(list(range(100, 0, -1))) == 1
assert length_of_lis(list(range(1, 101))) == 100
""",
        difficulty="hard",
        category="dynamic-programming",
        tags=["binary-search", "dp"],
    ),
    BenchmarkProblem(
        id="H02",
        title="Word Break",
        description=(
            "Given a string `s` and a dictionary of strings `word_dict`, return "
            "`True` if `s` can be segmented into a space-separated sequence of "
            "one or more dictionary words."
        ),
        signature="def word_break(s: str, word_dict: list[str]) -> bool:",
        test_code="""\
assert word_break("leetcode", ["leet", "code"]) == True
assert word_break("applepenapple", ["apple", "pen"]) == True
assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) == False
assert word_break("a", ["a"]) == True
assert word_break("ab", ["a", "b"]) == True
assert word_break("aaaaaaa", ["aaaa", "aaa"]) == True
assert word_break("goalspecial", ["go", "goal", "goals", "special"]) == True
assert word_break("abcd", ["a","abc","b","cd"]) == True
""",
        difficulty="hard",
        category="dynamic-programming",
        tags=["dp", "strings", "trie"],
    ),
    BenchmarkProblem(
        id="H03",
        title="Merge K Sorted Lists",
        description=(
            "Given `k` sorted lists of integers, merge them into one sorted list "
            "and return it. Aim for O(N log k) where N is total elements."
        ),
        signature="def merge_k_sorted(lists: list[list[int]]) -> list[int]:",
        test_code="""\
assert merge_k_sorted([[1,4,5],[1,3,4],[2,6]]) == [1,1,2,3,4,4,5,6]
assert merge_k_sorted([]) == []
assert merge_k_sorted([[]]) == []
assert merge_k_sorted([[1],[0]]) == [0,1]
assert merge_k_sorted([[-1,0,1],[-2,2],[-3,3]]) == [-3,-2,-1,0,1,2,3]
assert merge_k_sorted([[i] for i in range(100, 0, -1)]) == list(range(1, 101))
""",
        difficulty="hard",
        category="heap",
        tags=["divide-and-conquer", "merge"],
    ),
    BenchmarkProblem(
        id="H04",
        title="Serialize and Deserialize Binary Tree",
        description=(
            "Implement `serialize(root_list)` and `deserialize(data)`. "
            "`serialize` takes a tree in level-order list form (with None for "
            "missing nodes) and converts it to a string. `deserialize` takes that "
            "string and reconstructs the same level-order list. The round-trip must "
            "be lossless."
        ),
        signature="def serialize(tree_list: list) -> str:\n    pass\ndef deserialize(data: str) -> list:",
        test_code="""\
cases = [
    [1, 2, 3, None, None, 4, 5],
    [],
    [1],
    [1, None, 2],
    [1, 2, 3, 4, 5, None, None, 6, 7],
]
for tree in cases:
    assert deserialize(serialize(tree)) == tree, f"Failed for {tree}"
""",
        difficulty="hard",
        category="trees",
        tags=["design", "bfs"],
    ),
    BenchmarkProblem(
        id="H05",
        title="Minimum Window Substring",
        description=(
            "Given two strings `s` and `t`, return the minimum window substring "
            "of `s` that contains every character (including duplicates) in `t`. "
            "If no such window exists, return the empty string."
        ),
        signature="def min_window(s: str, t: str) -> str:",
        test_code="""\
assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "a") == "a"
assert min_window("a", "aa") == ""
assert min_window("aa", "aa") == "aa"
assert min_window("bba", "ab") == "ba"
assert min_window("aaflslflsldkalskaaa", "aaa") == "aaa"
""",
        difficulty="hard",
        category="sliding-window",
        tags=["hash-map", "two-pointers"],
    ),
    BenchmarkProblem(
        id="H06",
        title="Course Schedule (Topological Sort)",
        description=(
            "There are `num_courses` courses (0 to n-1). You are given a list of "
            "prerequisite pairs `[a, b]` meaning you must take course `b` before `a`. "
            "Return `True` if you can finish all courses (no cycle), else `False`."
        ),
        signature="def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:",
        test_code="""\
assert can_finish(2, [[1, 0]]) == True
assert can_finish(2, [[1, 0], [0, 1]]) == False
assert can_finish(5, [[1,0],[2,1],[3,2],[4,3]]) == True
assert can_finish(3, [[0,1],[1,2],[2,0]]) == False
assert can_finish(1, []) == True
assert can_finish(4, [[1,0],[2,0],[3,1],[3,2]]) == True
assert can_finish(7, [[1,0],[2,0],[3,1],[4,2],[5,3],[6,4],[3,6]]) == False
""",
        difficulty="hard",
        category="graph",
        tags=["topological-sort", "dfs", "bfs"],
    ),
    BenchmarkProblem(
        id="H07",
        title="Knapsack 0/1",
        description=(
            "Given `n` items with `weights` and `values`, and a knapsack of "
            "capacity `W`, return the maximum value that can be carried. "
            "Each item can be used at most once."
        ),
        signature="def knapsack(weights: list[int], values: list[int], W: int) -> int:",
        test_code="""\
assert knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
assert knapsack([2, 3, 4, 5], [3, 4, 5, 6], 5) == 7
assert knapsack([], [], 10) == 0
assert knapsack([10], [100], 5) == 0
assert knapsack([10], [100], 10) == 100
assert knapsack([1,1,1,1,1], [1,1,1,1,1], 3) == 3
assert knapsack([1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1], 15) == 30
""",
        difficulty="hard",
        category="dynamic-programming",
        tags=["dp", "optimization"],
    ),
    BenchmarkProblem(
        id="H08",
        title="Trie (Prefix Tree)",
        description=(
            "Implement a Trie class with `insert(word)`, `search(word)` (exact), "
            "and `starts_with(prefix)` methods."
        ),
        signature="class Trie:\n    def __init__(self):\n        pass\n    def insert(self, word: str) -> None:\n        pass\n    def search(self, word: str) -> bool:\n        pass\n    def starts_with(self, prefix: str) -> bool:\n        pass",
        test_code="""\
trie = Trie()
trie.insert("apple")
assert trie.search("apple") == True
assert trie.search("app") == False
assert trie.starts_with("app") == True
trie.insert("app")
assert trie.search("app") == True
assert trie.starts_with("appl") == True
assert trie.search("banana") == False
trie.insert("banana")
assert trie.search("banana") == True
assert trie.starts_with("ban") == True
assert trie.starts_with("bana") == True
assert trie.starts_with("xyz") == False
""",
        difficulty="hard",
        category="trie",
        tags=["design", "strings"],
    ),

    # ---------------------------------------------------------------
    #  EXPERT (8 problems) — frontier-model caliber
    # ---------------------------------------------------------------
    BenchmarkProblem(
        id="X01",
        title="Regular Expression Matching",
        description=(
            "Implement regular expression matching with '.' (matches any single "
            "character) and '*' (matches zero or more of the preceding element). "
            "The matching should cover the ENTIRE input string."
        ),
        signature="def is_match(s: str, p: str) -> bool:",
        test_code="""\
assert is_match("aa", "a") == False
assert is_match("aa", "a*") == True
assert is_match("ab", ".*") == True
assert is_match("aab", "c*a*b") == True
assert is_match("mississippi", "mis*is*ip*.") == True
assert is_match("", ".*") == True
assert is_match("", "") == True
assert is_match("abc", "") == False
assert is_match("aaa", "a*a") == True
assert is_match("aaa", "ab*a*c*a") == True
assert is_match("a", "ab*") == True
assert is_match("bbbba", ".*a*a") == True
""",
        difficulty="expert",
        category="dynamic-programming",
        tags=["recursion", "strings"],
    ),
    BenchmarkProblem(
        id="X02",
        title="N-Queens",
        description=(
            "Return the total number of distinct solutions to the N-Queens puzzle. "
            "Given an integer `n`, place `n` queens on an `n x n` board such that "
            "no two queens threaten each other."
        ),
        signature="def total_n_queens(n: int) -> int:",
        test_code="""\
assert total_n_queens(1) == 1
assert total_n_queens(4) == 2
assert total_n_queens(5) == 10
assert total_n_queens(6) == 4
assert total_n_queens(7) == 40
assert total_n_queens(8) == 92
assert total_n_queens(9) == 352
""",
        difficulty="expert",
        category="backtracking",
        tags=["recursion", "constraint-satisfaction"],
    ),
    BenchmarkProblem(
        id="X03",
        title="Longest Common Subsequence (3 strings)",
        description=(
            "Given three strings, find the length of their longest common subsequence."
        ),
        signature="def lcs3(a: str, b: str, c: str) -> int:",
        test_code="""\
assert lcs3("abcde", "ace", "aue") == 2
assert lcs3("abc", "abc", "abc") == 3
assert lcs3("abc", "def", "ghi") == 0
assert lcs3("abcxyz", "xyzabc", "abcxyz") == 3
assert lcs3("", "abc", "def") == 0
assert lcs3("aaa", "aa", "a") == 1
assert lcs3("aggtab", "gxtxayb", "aggtxyb") == 4
""",
        difficulty="expert",
        category="dynamic-programming",
        tags=["dp", "strings"],
    ),
    BenchmarkProblem(
        id="X04",
        title="Skyline Problem",
        description=(
            "Given a list of buildings `[left, right, height]`, return the skyline "
            "formed by these buildings as a list of `[x, height]` key points. "
            "Key points are the left endpoints of horizontal segments. "
            "The last point always has height 0. Points should be sorted by x."
        ),
        signature="def get_skyline(buildings: list[list[int]]) -> list[list[int]]:",
        test_code="""\
assert get_skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]) == [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
assert get_skyline([[0,2,3],[2,5,3]]) == [[0,3],[5,0]]
assert get_skyline([]) == []
assert get_skyline([[1,2,1],[1,2,2],[1,2,3]]) == [[1,3],[2,0]]
""",
        difficulty="expert",
        category="divide-and-conquer",
        tags=["heap", "sweep-line"],
    ),
    BenchmarkProblem(
        id="X05",
        title="Word Ladder (Shortest Transformation)",
        description=(
            "Given `begin_word`, `end_word`, and a `word_list`, return the number "
            "of words in the shortest transformation sequence from begin to end. "
            "Each transformation changes exactly one letter. Every transformed word "
            "must exist in the word list. Return 0 if no sequence exists."
        ),
        signature="def ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int:",
        test_code="""\
assert ladder_length("hit", "cog", ["hot","dot","dog","lot","log","cog"]) == 5
assert ladder_length("hit", "cog", ["hot","dot","dog","lot","log"]) == 0
assert ladder_length("a", "c", ["a","b","c"]) == 2
assert ladder_length("hot", "dog", ["hot","dog","dot"]) == 3
assert ladder_length("hot", "dog", ["hot","dog"]) == 0
""",
        difficulty="expert",
        category="graph",
        tags=["bfs", "strings"],
    ),
    BenchmarkProblem(
        id="X06",
        title="Max Rectangle in Histogram",
        description=(
            "Given a list of integers representing bar heights in a histogram "
            "where each bar has width 1, find the area of the largest rectangle "
            "that can be formed in the histogram."
        ),
        signature="def largest_rectangle_area(heights: list[int]) -> int:",
        test_code="""\
assert largest_rectangle_area([2, 1, 5, 6, 2, 3]) == 10
assert largest_rectangle_area([2, 4]) == 4
assert largest_rectangle_area([1]) == 1
assert largest_rectangle_area([0]) == 0
assert largest_rectangle_area([1, 1, 1, 1, 1]) == 5
assert largest_rectangle_area([6, 2, 5, 4, 5, 1, 6]) == 12
assert largest_rectangle_area([2, 1, 2]) == 3
assert largest_rectangle_area([]) == 0
""",
        difficulty="expert",
        category="stack",
        tags=["monotonic-stack"],
    ),
    BenchmarkProblem(
        id="X07",
        title="Median of Two Sorted Arrays",
        description=(
            "Given two sorted arrays `nums1` and `nums2` of size m and n, "
            "return the median of the two sorted arrays. "
            "The solution should run in O(log(m+n)) time."
        ),
        signature="def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:",
        test_code="""\
assert find_median_sorted_arrays([1, 3], [2]) == 2.0
assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
assert find_median_sorted_arrays([0, 0], [0, 0]) == 0.0
assert find_median_sorted_arrays([], [1]) == 1.0
assert find_median_sorted_arrays([2], []) == 2.0
assert find_median_sorted_arrays([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == 5.5
assert abs(find_median_sorted_arrays([1, 3, 5, 7], [2, 4, 6, 8]) - 4.5) < 1e-9
""",
        difficulty="expert",
        category="binary-search",
        tags=["divide-and-conquer", "arrays"],
    ),
    BenchmarkProblem(
        id="X08",
        title="Edit Distance (Levenshtein)",
        description=(
            "Given two strings `word1` and `word2`, return the minimum number of "
            "operations required to convert `word1` to `word2`. You may: insert a "
            "character, delete a character, or replace a character."
        ),
        signature="def min_distance(word1: str, word2: str) -> int:",
        test_code="""\
assert min_distance("horse", "ros") == 3
assert min_distance("intention", "execution") == 5
assert min_distance("", "") == 0
assert min_distance("", "abc") == 3
assert min_distance("abc", "") == 3
assert min_distance("abc", "abc") == 0
assert min_distance("kitten", "sitting") == 3
assert min_distance("saturday", "sunday") == 3
assert min_distance("abcdefghij", "jihgfedcba") == 8
""",
        difficulty="expert",
        category="dynamic-programming",
        tags=["dp", "strings"],
    ),
]


# ===================================================================
# Benchmark runner
# ===================================================================


class CodingBenchmark:
    """Runs the full benchmark suite against a CodeGenerator."""

    DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]

    def __init__(
        self,
        problems: Optional[List[BenchmarkProblem]] = None,
        timeout: int = 10,
    ) -> None:
        self.problems = problems or BENCHMARK_PROBLEMS
        self.timeout = timeout
        self.validator = CodeValidator()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_code(self, problem: BenchmarkProblem, code: str) -> BenchmarkResult:
        """Run generated code against hidden test cases."""
        t0 = time.time()

        # 1. Syntax check
        syntax = self.validator.check_syntax(code, "python")
        if not syntax.success:
            return BenchmarkResult(
                problem_id=problem.id,
                title=problem.title,
                difficulty=problem.difficulty,
                category=problem.category,
                passed=False,
                syntax_valid=False,
                error=f"Syntax error: {syntax.error}",
                generated_code=code,
                elapsed_seconds=time.time() - t0,
            )

        # 2. Execution: combine generated code + test assertions
        full_code = code + "\n\n" + problem.test_code
        exec_result = self.validator.execute(full_code, "python", timeout=self.timeout)

        return BenchmarkResult(
            problem_id=problem.id,
            title=problem.title,
            difficulty=problem.difficulty,
            category=problem.category,
            passed=exec_result.success,
            syntax_valid=True,
            error=exec_result.error if not exec_result.success else "",
            generated_code=code,
            elapsed_seconds=time.time() - t0,
        )

    def run(
        self,
        generator: Any,
        difficulty: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full benchmark.

        Args:
            generator: A ``CodeGenerator`` instance.
            difficulty: Optional filter — only run problems of this difficulty.
            category: Optional filter — only run problems in this category.

        Returns:
            A dictionary with results, scores, and breakdown.
        """
        problems = self._filter_problems(difficulty, category)
        results: List[BenchmarkResult] = []

        total = len(problems)
        for i, problem in enumerate(problems):
            logger.info(
                "[%d/%d] %s — %s (%s)",
                i + 1, total, problem.id, problem.title, problem.difficulty.upper(),
            )

            # Build a detailed prompt including the function signature
            prompt = (
                f"{problem.description}\n\n"
                f"Implement the following:\n"
                f"```python\n{problem.signature}\n```\n\n"
                f"Return ONLY the Python implementation. Do not include test cases."
            )

            try:
                code = generator.generate(
                    problem=prompt,
                    language="python",
                    temperature=0.2,  # Low temp for deterministic output
                    max_new_tokens=768,
                )
            except Exception as exc:
                logger.error("Generation failed for %s: %s", problem.id, exc)
                results.append(BenchmarkResult(
                    problem_id=problem.id,
                    title=problem.title,
                    difficulty=problem.difficulty,
                    category=problem.category,
                    passed=False,
                    syntax_valid=False,
                    error=f"Generation error: {exc}",
                    generated_code="",
                ))
                continue

            result = self.evaluate_code(problem, code)
            results.append(result)

            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info("  %s (%.1fs)", status, result.elapsed_seconds)
            if result.error:
                error_str = str(result.error)
                logger.info("  Error: %s", error_str[0:200])  # type: ignore[index]

        return self._compile_report(results)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _compile_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Produce a structured report from raw results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        syntax_ok = sum(1 for r in results if r.syntax_valid)

        # Breakdown by difficulty
        by_difficulty: Dict[str, Dict[str, int]] = {}
        for diff in self.DIFFICULTY_ORDER:
            subset = [r for r in results if r.difficulty == diff]
            if subset:
                by_difficulty[diff] = {
                    "total": len(subset),
                    "passed": sum(1 for r in subset if r.passed),
                }

        # Breakdown by category
        categories: Dict[str, Dict[str, int]] = {}
        for r in results:
            if r.category not in categories:
                categories[r.category] = {"total": 0, "passed": 0}
            categories[r.category]["total"] += 1
            if r.passed:
                categories[r.category]["passed"] += 1

        report = {
            "summary": {
                "total_problems": total,
                "passed": passed,
                "failed": total - passed,
                "syntax_valid": syntax_ok,
                "pass_at_1": (float(passed) / float(total) * 100.0) if total else 0.0,
            },
            "by_difficulty": by_difficulty,
            "by_category": categories,
            "results": [r.__dict__ for r in results],
        }

        return report

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Pretty-print the benchmark report to stdout."""
        s = report["summary"]

        print(f"\n{'━' * 70}")
        print(f"  🏆  CODING BENCHMARK RESULTS")
        print(f"{'━' * 70}")
        print(f"  Total Problems :  {s['total_problems']}")
        print(f"  Passed         :  {s['passed']}")
        print(f"  Failed         :  {s['failed']}")
        print(f"  Syntax Valid   :  {s['syntax_valid']}")
        print(f"  Pass@1 Score   :  {s['pass_at_1']}%")
        print(f"{'━' * 70}")

        # By difficulty
        print(f"\n  📊  BREAKDOWN BY DIFFICULTY")
        print(f"  {'Difficulty':<12} {'Passed':>8} {'Total':>8} {'Rate':>8}")
        print(f"  {'─' * 40}")
        for diff in CodingBenchmark.DIFFICULTY_ORDER:
            if diff in report["by_difficulty"]:
                d = report["by_difficulty"][diff]
                _p, _t = int(d["passed"]), int(d["total"])
                rate = (float(_p) / float(_t) * 100.0) if _t else 0.0
                rate = float(int(rate * 10) / 10)
                emoji = "🟢" if rate >= 80 else "🟡" if rate >= 50 else "🔴"
                print(f"  {emoji} {diff:<10} {d['passed']:>8} {d['total']:>8} {rate:>7}%")

        # By category
        print(f"\n  📂  BREAKDOWN BY CATEGORY")
        print(f"  {'Category':<22} {'Passed':>8} {'Total':>8} {'Rate':>8}")
        print(f"  {'─' * 50}")
        for cat, d in sorted(report["by_category"].items()):
            _p2, _t2 = int(d["passed"]), int(d["total"])
            rate = float(int((float(_p2) / float(_t2) * 100.0) * 10) / 10) if _t2 else 0.0
            print(f"  {cat:<22} {d['passed']:>8} {d['total']:>8} {rate:>7}%")

        # Individual results
        print(f"\n  📝  INDIVIDUAL RESULTS")
        print(f"  {'ID':<6} {'Title':<42} {'Diff':<8} {'Result':<8}")
        print(f"  {'─' * 66}")
        for r in report["results"]:
            status = "✅" if r["passed"] else "❌"
            print(f"  {r['problem_id']:<6} {r['title']:<42} {r['difficulty']:<8} {status}")

        print(f"\n{'━' * 70}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_problems(
        self,
        difficulty: Optional[str],
        category: Optional[str],
    ) -> List[BenchmarkProblem]:
        problems = self.problems
        if difficulty:
            problems = [p for p in problems if p.difficulty == difficulty.lower()]
        if category:
            problems = [p for p in problems if p.category == category.lower()]
        return problems

    def save_report(self, report: Dict[str, Any], path: str | Path) -> None:
        """Save the benchmark report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Benchmark report saved to %s", path)
