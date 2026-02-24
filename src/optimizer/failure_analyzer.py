"""Failure pattern analysis for Loop 2 optimization."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.schemas.evaluation_result import BenchmarkCaseResult, BenchmarkReport
from src.tools.cad import ErrorCategory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FailurePattern:
    """A categorized failure pattern."""

    def __init__(self, category: str, description: str, count: int = 0, severity: float = 1.0):
        self.category = category
        self.description = description
        self.count = count
        self.severity = severity
        self.examples: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "count": self.count,
            "severity": self.severity,
            "examples": self.examples[:3],
        }


class FailureAnalyzer:
    """Analyzes failure patterns across benchmark results."""

    # Known error patterns and their categories
    ERROR_PATTERNS = [
        (r"SyntaxError|IndentationError", "syntax", "Code syntax errors", 0.8),
        (r"NameError.*not defined", "undefined_variable", "Undefined variables", 0.7),
        (r"TypeError.*argument", "type_error", "Type/argument errors in build123d API", 0.9),
        (r"Hole.*radius|radius.*too", "hole_sizing", "Hole sizing issues", 0.6),
        (r"Boolean|Fuse|Cut|standard_boolean", "boolean_op", "Boolean operation failures", 1.0),
        (r"Extrude.*amount|extrusion", "extrusion", "Extrusion errors", 0.8),
        (r"Sketch.*closed|wire.*closed", "open_sketch", "Non-closed sketch profiles", 0.9),
        (r"export|step|STEP", "export", "STEP export failures", 0.5),
        (r"timed out|timeout|Timeout", "timeout", "Execution timeouts", 0.7),
        (r"import.*build123d|ModuleNotFoundError", "import", "Import/environment errors", 0.3),
    ]

    def analyze(self, report: BenchmarkReport) -> list[FailurePattern]:
        """Analyze failure patterns in benchmark results.
        
        Args:
            report: BenchmarkReport with case results.
        
        Returns:
            List of FailurePattern objects sorted by frequency * severity.
        """
        logger.info("failure_analysis_start", cases=len(report.case_results))

        pattern_map: dict[str, FailurePattern] = {}

        for case in report.case_results:
            if case.metrics.execution_success and case.metrics.composite_score > 0.8:
                continue  # Skip successes

            self._classify_failure(case, pattern_map)

        # Sort by impact (count * severity)
        patterns = sorted(
            pattern_map.values(),
            key=lambda p: p.count * p.severity,
            reverse=True,
        )

        logger.info(
            "failure_analysis_complete",
            patterns_found=len(patterns),
            top_pattern=patterns[0].category if patterns else "none",
        )
        return patterns

    def summarize(self, patterns: list[FailurePattern]) -> str:
        """Generate a text summary of failure patterns for the prompt optimizer."""
        if not patterns:
            return "No significant failure patterns detected."

        lines = ["## Failure Pattern Analysis", ""]
        for i, p in enumerate(patterns[:5], 1):
            lines.append(f"{i}. **{p.category}** (count: {p.count}, severity: {p.severity:.1f})")
            lines.append(f"   {p.description}")
            for ex in p.examples[:2]:
                lines.append(f"   - Example: {ex[:200]}")
            lines.append("")

        return "\n".join(lines)

    def _classify_failure(
        self,
        case: BenchmarkCaseResult,
        pattern_map: dict[str, FailurePattern],
    ) -> None:
        """Classify a single case failure into patterns."""
        error_text = case.error or ""

        matched = False
        for regex, category, description, severity in self.ERROR_PATTERNS:
            if re.search(regex, error_text, re.IGNORECASE):
                if category not in pattern_map:
                    pattern_map[category] = FailurePattern(
                        category=category,
                        description=description,
                        severity=severity,
                    )
                pattern_map[category].count += 1
                pattern_map[category].examples.append(
                    f"[{case.case_id}] {error_text[:300]}"
                )
                matched = True
                break

        if not matched and error_text:
            cat = "unknown"
            if cat not in pattern_map:
                pattern_map[cat] = FailurePattern(
                    category=cat,
                    description="Unclassified errors",
                    severity=0.5,
                )
            pattern_map[cat].count += 1
            pattern_map[cat].examples.append(f"[{case.case_id}] {error_text[:300]}")

        # Also check for quality failures (execution succeeded but low score)
        if case.metrics.execution_success and case.metrics.composite_score < 0.5:
            cat = "low_quality"
            if cat not in pattern_map:
                pattern_map[cat] = FailurePattern(
                    category=cat,
                    description="Execution succeeded but low quality score",
                    severity=0.6,
                )
            pattern_map[cat].count += 1
            pattern_map[cat].examples.append(
                f"[{case.case_id}] score={case.metrics.composite_score:.3f}"
            )
