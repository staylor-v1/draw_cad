"""LLM-driven prompt optimization for Loop 2."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.inference.base import BaseLLMClient, ChatMessage
from src.optimizer.failure_analyzer import FailurePattern
from src.schemas.pipeline_config import PipelineConfig
from src.utils.file_utils import load_prompt, save_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

PROMPT_OPTIMIZER_SYSTEM = """You are a prompt engineering expert. Your task is to improve a system prompt for a CAD code generation AI.

Given:
1. The current system prompt
2. A summary of failure patterns from recent runs
3. A strategy to apply

Generate a PATCH to add to the system prompt. Output ONLY the text to append.
Rules:
- Keep patches concise (under 500 characters)
- Be specific and actionable
- Include concrete examples when helpful
- Do not repeat existing content
"""


class PromptOptimizer:
    """Generates prompt patches to improve pipeline performance."""

    def __init__(self, config: PipelineConfig, llm_client: BaseLLMClient | None = None):
        self.config = config
        self.llm_client = llm_client
        self._patch_history: list[dict] = []

    def generate_patch(
        self,
        failure_summary: str,
        strategy: str = "add_error_prevention",
    ) -> str:
        """Generate a prompt patch based on failure analysis.
        
        Args:
            failure_summary: Text summary of failure patterns.
            strategy: Patch strategy to apply.
        
        Returns:
            Text to append to the system prompt.
        """
        if self.llm_client is None:
            return self._generate_mock_patch(failure_summary, strategy)

        try:
            current_prompt = load_prompt(self.config.prompts.system_prompt)
        except FileNotFoundError:
            current_prompt = "[System prompt not found]"

        user_message = (
            f"## Current System Prompt (abbreviated)\n"
            f"{current_prompt[:2000]}\n\n"
            f"## Failure Patterns\n{failure_summary}\n\n"
            f"## Strategy: {strategy}\n"
            f"Generate a patch to append to the system prompt that addresses "
            f"the top failure patterns using the '{strategy}' strategy."
        )

        messages = [
            ChatMessage(role="system", content=PROMPT_OPTIMIZER_SYSTEM),
            ChatMessage(role="user", content=user_message),
        ]

        try:
            response = self.llm_client.chat(
                messages=messages,
                model=self.config.models.reasoning.name,
                temperature=0.3,
                max_tokens=1024,
            )
            patch = response.content.strip()
            self._patch_history.append({
                "strategy": strategy,
                "patch": patch,
                "failure_summary": failure_summary[:500],
            })
            logger.info("prompt_patch_generated", strategy=strategy, length=len(patch))
            return patch
        except Exception as e:
            logger.error("prompt_patch_generation_failed", error=str(e))
            return ""

    def apply_patch(self, prompt_path: str | Path, patch: str) -> str:
        """Apply a patch to a prompt file by appending.
        
        Returns:
            The new prompt content.
        """
        try:
            current = load_prompt(prompt_path)
        except FileNotFoundError:
            current = ""

        patched = current + "\n\n" + patch
        Path(prompt_path).write_text(patched)
        logger.info("prompt_patch_applied", path=str(prompt_path), new_length=len(patched))
        return patched

    def rollback_patch(self, prompt_path: str | Path, original_content: str) -> None:
        """Rollback a prompt to its original content."""
        Path(prompt_path).write_text(original_content)
        logger.info("prompt_patch_rolled_back", path=str(prompt_path))

    def _generate_mock_patch(self, failure_summary: str, strategy: str) -> str:
        """Generate a mock patch for testing."""
        patches = {
            "add_error_prevention": (
                "\n## Common Pitfalls to Avoid\n"
                "- Always close sketches before extruding\n"
                "- Ensure hole radius is smaller than the face dimension\n"
                "- Use absolute coordinates for positioning when possible\n"
            ),
            "add_constraint": (
                "\n## Additional Constraints\n"
                "- Always verify that Boolean operations have overlapping volumes\n"
                "- Check that extrusion amounts are positive\n"
            ),
            "add_example_pattern": (
                "\n## Recommended Code Pattern\n"
                "```python\n"
                "# Always structure code as: Parameters -> Base -> Features -> Export\n"
                "```\n"
            ),
            "clarify_ambiguity": (
                "\n## Handling Ambiguity\n"
                "- When hole positions are not specified, place them at the center\n"
                "- Default fillet radius: 1mm unless specified\n"
            ),
        }
        return patches.get(strategy, patches["add_error_prevention"])
