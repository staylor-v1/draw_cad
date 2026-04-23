"""Reasoning stage: generate build123d code from reconciled geometry."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from src.inference.base import BaseLLMClient, ChatMessage
from src.schemas.geometry import ReconciledGeometry
from src.schemas.pipeline_config import PipelineConfig
from src.utils.file_utils import load_prompt, load_yaml
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks in LLM response."""
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try generic code block
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "from build123d" in code or "import build123d" in code:
            return code
    stripped = text.strip()
    if "from build123d" in stripped or "import build123d" in stripped:
        return stripped
    return ""


class ReasoningStage:
    """Generates build123d code from reconciled geometry."""

    def __init__(
        self,
        config: PipelineConfig,
        llm_client: BaseLLMClient | None = None,
    ):
        self.config = config
        self.llm_client = llm_client

    def run(
        self,
        reconciled: ReconciledGeometry,
        error_context: str = "",
        previous_code: str = "",
        fewshot_examples: list[str] | None = None,
    ) -> str:
        """Generate build123d code from reconciled geometry.
        
        Args:
            reconciled: Reconciled 3D geometry description.
            error_context: Error from a previous attempt (for retries).
            previous_code: Code from a previous attempt (for retries).
            fewshot_examples: Optional few-shot examples to include.
        
        Returns:
            Generated Python code string.
        """
        logger.info("reasoning_stage_start", has_error_context=bool(error_context))

        if self.llm_client is None:
            logger.info("reasoning_stage_using_mock")
            return self._mock_generate(reconciled)

        # Build messages
        messages = self._build_messages(
            reconciled, error_context, previous_code, fewshot_examples
        )

        model_cfg = self.config.models.reasoning
        response = self.llm_client.chat(
            messages=messages,
            model=model_cfg.name,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )

        code = extract_code(response.content)
        if not code:
            logger.warning("reasoning_no_code_extracted", response_length=len(response.content))
        else:
            logger.info("reasoning_code_generated", code_length=len(code))

        return code

    def _build_messages(
        self,
        reconciled: ReconciledGeometry,
        error_context: str,
        previous_code: str,
        fewshot_examples: list[str] | None,
    ) -> list[ChatMessage]:
        """Build the message sequence for the reasoning LLM."""
        messages = []

        # System prompt
        try:
            system_prompt = load_prompt(self.config.prompts.system_prompt)
        except FileNotFoundError:
            system_prompt = "You are an expert CAD engineer. Generate build123d Python code."
        messages.append(ChatMessage(role="system", content=system_prompt))

        # Few-shot examples
        if fewshot_examples:
            for example in fewshot_examples:
                messages.append(ChatMessage(role="user", content="Here is an example:"))
                messages.append(ChatMessage(role="assistant", content=example))

        # Main geometry prompt
        geometry_text = reconciled.to_prompt_text()
        user_prompt = (
            "Here is the analysis of the engineering drawing:\n\n"
            f"{geometry_text}\n\n"
            "Please generate the build123d code to create this part. "
            "Export the result as 'output.step'."
        )

        # If retry, include error context
        if error_context and previous_code:
            try:
                retry_prompt_template = load_prompt(self.config.prompts.retry_prompt)
                user_prompt += "\n\n" + retry_prompt_template.format(
                    error=error_context,
                    previous_code=previous_code,
                )
            except (FileNotFoundError, KeyError):
                user_prompt += (
                    f"\n\n## Previous Attempt Failed\n"
                    f"Error:\n```\n{error_context}\n```\n"
                    f"Previous code:\n```python\n{previous_code}\n```\n"
                    f"Please fix the code and try again."
                )

        messages.append(ChatMessage(role="user", content=user_prompt))
        return messages

    def _mock_generate(self, reconciled: ReconciledGeometry) -> str:
        """Generate mock code based on reconciled geometry."""
        dims = reconciled.overall_dimensions
        length = dims.get("length", 100)
        width = dims.get("width", 50)
        thickness = dims.get("thickness", dims.get("height", 10))

        code = f'''from build123d import *

# Parameters
length = {length}
width = {width}
thickness = {thickness}

# Construction
with BuildPart() as part:
    with BuildSketch(Plane.XY):
        Rectangle(length, width)
    Extrude(amount=thickness)
'''
        # Add features
        for feat in reconciled.features:
            feat_lower = feat.type.lower()
            if "hole" in feat_lower:
                for dim in feat.dimensions:
                    if "diameter" in dim.label.lower():
                        radius = dim.value / 2
                        code += f'''
    # {feat.description}
    with Locations(part.faces().sort_by(Axis.Z)[-1]):
        Hole(radius={radius})
'''
                        break
                else:
                    code += f'''
    # {feat.description}
    with Locations(part.faces().sort_by(Axis.Z)[-1]):
        Hole(radius=2.5)
'''

        code += '''
# Export
part.part.export_step("output.step")
'''
        return code

    def load_fewshot_examples(self, count: int = 3) -> list[str]:
        """Load few-shot examples from the configured index."""
        try:
            index = load_yaml(self.config.prompts.fewshot_index)
        except FileNotFoundError:
            logger.debug("fewshot_index_not_found")
            return []

        examples = []
        for entry in index.get("examples", [])[:count]:
            try:
                example_text = load_prompt(entry["path"])
                examples.append(example_text)
            except (FileNotFoundError, KeyError):
                continue
        return examples
