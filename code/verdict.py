"""Verdict classifier for pytest output — 10-verdict pure-regex classifier.

No LLM, no filesystem access. Classifies a pytest stdout string into one of
10 Verdict values. Priority order (highest to lowest):

  COMPILE_ERROR > TIMEOUT > INFRA_ERROR > DESELECTED > XPASS > XFAIL >
  GOOD_FAIL > WRONG_FAIL > UNEXPECTED_PASS > NO_TEST

Design mirrors jina_test/verifier.py Verdict enum but is self-contained.
"""

from __future__ import annotations

import re
from enum import Enum


class Verdict(str, Enum):
    # Original 5 (from jina_test/verifier.py)
    GOOD_FAIL = "GOOD_FAIL"              # test fails with issue-related assertion error
    WRONG_FAIL = "WRONG_FAIL"            # test fails but failure is unrelated/unknown
    UNEXPECTED_PASS = "UNEXPECTED_PASS"  # test passes on buggy code — no bug caught
    NO_TEST = "NO_TEST"                  # no test files created/modified
    INFRA_ERROR = "INFRA_ERROR"          # import/fixture/environment failure
    # Extended 5 (new)
    DESELECTED = "DESELECTED"            # pytest deselected the test(s)
    XFAIL = "XFAIL"                      # expected failure marker
    XPASS = "XPASS"                      # xfail test unexpectedly passed
    TIMEOUT = "TIMEOUT"                  # test execution timed out
    COMPILE_ERROR = "COMPILE_ERROR"      # syntax/compile error in test file


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# COMPILE_ERROR — syntax/indentation errors that prevent import
_COMPILE_ERROR_PATTERNS = [
    re.compile(r"\bSyntaxError\b"),
    re.compile(r"\bIndentationError\b"),
    re.compile(r"\bCOMPILE_ERROR\b"),
]

# TIMEOUT
_TIMEOUT_PATTERNS = [
    re.compile(r"\bTIMEOUT\b"),
    re.compile(r"timed out", re.IGNORECASE),
    re.compile(r"\bTimeoutExpired\b"),
]

# INFRA_ERROR — environment / import / fixture failures
_INFRA_ERROR_PATTERNS = [
    re.compile(r"\bImportError\b"),
    re.compile(r"\bModuleNotFoundError\b"),
    re.compile(r"\bNameError\b"),
    re.compile(r"\bFileNotFoundError\b"),
    re.compile(r"E\s+fixture\s+\S+\s+not found"),
    re.compile(r"collected 0 items"),
    re.compile(r"ERRORS\s*="),
    re.compile(r"INTERNALERROR>"),
]

# DESELECTED
_DESELECTED_RE = re.compile(r"\d+\s+deselected")

# XPASS
_XPASS_PATTERNS = [
    re.compile(r"\bxpassed\b", re.IGNORECASE),
    re.compile(r"\bXPASS\b"),
]

# XFAIL
_XFAIL_PATTERNS = [
    re.compile(r"\bxfailed\b", re.IGNORECASE),
    re.compile(r"\bXFAIL\b"),
]

# GOOD_FAIL — real test failures
_GOOD_FAIL_PATTERNS = [
    # pytest format
    re.compile(r"\d+\s+failed"),
    re.compile(r"FAILED\s+\S+::\S+"),
    # Bug 3 fix: Django unittest-style (./tests/runtests.py)
    # Examples: "FAILED (failures=1)", "FAILED (errors=2)", "FAILED (failures=1, errors=2)"
    re.compile(r"^FAILED\s*\(", re.MULTILINE),
    re.compile(r"FAILED\s*\(\s*(?:failures|errors)\s*="),
    # sympy bin/test format
    re.compile(r"^\s*\d+\s+tests\s+failed", re.MULTILINE),
    # Generic unittest: "... FAIL" or "... ERROR" at end of test lines
    re.compile(r"^\S+\s+\.\.\.\s+(FAIL|ERROR)\s*$", re.MULTILINE),
]

# UNEXPECTED_PASS — all tests passed
_UNEXPECTED_PASS_PATTERNS = [
    # pytest format
    re.compile(r"\d+\s+passed"),
    # Bug 3 fix: Django unittest-style "Ran N tests in X.Xs\n\nOK"
    # The OK line appears ONLY when no failures/errors occurred.
    re.compile(r"Ran\s+\d+\s+tests?\s+in\s+[\d.]+s\s*\n+OK\b"),
    re.compile(r"^OK$", re.MULTILINE),
    # sympy bin/test: "tests finished: ..."
    re.compile(r"tests\s+finished:\s+\d+\s+passed"),
]


def classify_pytest_output(
    output: str,
    test_patch: str | None = None,  # reserved for future use, unused in pure-regex mode
) -> Verdict:
    """Classify pytest stdout into one of 10 Verdict values.

    Priority order (first match wins):
      COMPILE_ERROR > TIMEOUT > INFRA_ERROR > DESELECTED > XPASS > XFAIL >
      GOOD_FAIL > WRONG_FAIL > UNEXPECTED_PASS > NO_TEST

    Args:
        output: Raw pytest stdout/stderr string.
        test_patch: Optional diff of the test patch (unused, for API compatibility).

    Returns:
        The classified Verdict.
    """
    if not output or not output.strip():
        return Verdict.NO_TEST

    # 1. COMPILE_ERROR — syntax errors prevent collection entirely
    if any(p.search(output) for p in _COMPILE_ERROR_PATTERNS):
        return Verdict.COMPILE_ERROR

    # 2. TIMEOUT — subprocess or harness timeout
    if any(p.search(output) for p in _TIMEOUT_PATTERNS):
        return Verdict.TIMEOUT

    # 3. DESELECTED — check before INFRA_ERROR because "collected 0 items / N deselected"
    #    contains "collected 0 items" which is also an INFRA_ERROR pattern.
    #    Deselected takes priority when the output explicitly mentions deselection.
    if _DESELECTED_RE.search(output):
        return Verdict.DESELECTED

    # 4. INFRA_ERROR — import/fixture/env failures
    if any(p.search(output) for p in _INFRA_ERROR_PATTERNS):
        return Verdict.INFRA_ERROR

    # 5. XPASS — unexpected pass on an xfail-marked test
    if any(p.search(output) for p in _XPASS_PATTERNS):
        return Verdict.XPASS

    # 6. XFAIL — expected failure (still counts as "test ran")
    if any(p.search(output) for p in _XFAIL_PATTERNS):
        return Verdict.XFAIL

    has_fail = any(p.search(output) for p in _GOOD_FAIL_PATTERNS)
    has_pass = any(p.search(output) for p in _UNEXPECTED_PASS_PATTERNS)

    # 7. GOOD_FAIL — real failure (assume it's a content-level failure at this point)
    if has_fail:
        return Verdict.GOOD_FAIL

    # 8. UNEXPECTED_PASS — all tests passed on buggy code
    if has_pass:
        return Verdict.UNEXPECTED_PASS

    # 9. WRONG_FAIL — something went wrong but doesn't match known patterns
    #    (exit code non-zero implied by caller, or unknown output)
    return Verdict.WRONG_FAIL


def verdict_kills(verdict: Verdict) -> bool:
    """Return True if this verdict means the test 'killed' the fix candidate.

    A test kills a fix when it produces GOOD_FAIL when both are applied —
    i.e., the test fails on the (supposedly fixed) code, meaning the fix
    didn't actually address what the test exercises.

    In the CXV matrix context this is called from cxv.py after running
    test_k against fix_j. GOOD_FAIL = the test catches a regression in
    the fix (the fix is wrong from the test's perspective).
    """
    return verdict == Verdict.GOOD_FAIL
