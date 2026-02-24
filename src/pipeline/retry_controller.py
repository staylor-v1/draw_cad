"""Retry controller: manages error-feedback retry loop for code generation."""
from __future__ import annotations

from src.schemas.pipeline_config import PipelineConfig
from src.tools.cad import ErrorCategory, ExecutionResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RetryController:
    """Controls the retry loop for failed code generation attempts."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.max_retries = config.pipeline.max_retries
        self.attempt = 0
        self.history: list[dict] = []

    def should_retry(self, result: ExecutionResult) -> bool:
        """Determine if we should retry after a failed execution."""
        if result.success:
            return False
        if self.attempt >= self.max_retries:
            logger.info("retry_limit_reached", attempt=self.attempt)
            return False

        # Don't retry import errors (environment issue, not code issue)
        if result.error_category == ErrorCategory.IMPORT_ERROR:
            logger.info("no_retry_import_error")
            return False

        return True

    def record_attempt(self, code: str, result: ExecutionResult) -> None:
        """Record an attempt for context building."""
        self.history.append({
            "attempt": self.attempt,
            "code": code,
            "success": result.success,
            "error_category": result.error_category.value,
            "stderr": result.stderr,
        })
        self.attempt += 1

    def build_error_context(self) -> str:
        """Build enriched error context from attempt history."""
        if not self.history:
            return ""

        latest = self.history[-1]
        lines = [
            f"## Attempt {latest['attempt'] + 1} of {self.max_retries + 1} failed",
            f"Error category: {latest['error_category']}",
            "",
            "### Error Output:",
            f"```\n{latest['stderr']}\n```",
        ]

        # Add progressive context for later retries
        if len(self.history) > 1:
            lines.append("")
            lines.append("### Previous attempts also failed:")
            for h in self.history[:-1]:
                lines.append(f"- Attempt {h['attempt'] + 1}: {h['error_category']}")

        # Add specific guidance based on error category
        guidance = self._get_error_guidance(latest["error_category"])
        if guidance:
            lines.append("")
            lines.append("### Guidance:")
            lines.append(guidance)

        return "\n".join(lines)

    def get_previous_code(self) -> str:
        """Get the most recent code attempt."""
        if self.history:
            return self.history[-1]["code"]
        return ""

    def reset(self) -> None:
        """Reset the retry controller for a new pipeline run."""
        self.attempt = 0
        self.history.clear()

    @staticmethod
    def _get_error_guidance(category: str) -> str:
        """Provide category-specific guidance for the LLM."""
        guidance_map = {
            "syntax_error": (
                "Check for proper Python syntax. Common issues: "
                "missing colons, incorrect indentation, unclosed brackets."
            ),
            "geometry_error": (
                "The geometry operation failed. Common issues:\n"
                "- Sketch not closed before Extrude\n"
                "- Boolean operation on non-intersecting bodies\n"
                "- Hole radius larger than the face\n"
                "- Zero-thickness extrusion\n"
                "Try simplifying the geometry or using a different construction approach."
            ),
            "export_error": (
                "The STEP export failed. Ensure the part variable is accessible "
                "and the geometry is a valid solid."
            ),
            "runtime_error": (
                "A runtime error occurred. Check variable names, API usage, "
                "and ensure all build123d operations are used correctly."
            ),
            "timeout": (
                "The script timed out. Simplify the geometry or reduce "
                "the number of operations."
            ),
        }
        return guidance_map.get(category, "")
