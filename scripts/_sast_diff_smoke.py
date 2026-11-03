# TEMP: intentional Semgrep finding to verify SAST_MODE=diff gates on NEW issues.
# Delete this file after the smoke check.
from __future__ import annotations


def run_untrusted(expr: str) -> object:
    # Semgrep python.lang.security.audit.eval-detected (ERROR)
    return eval(expr)
