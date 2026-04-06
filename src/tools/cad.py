"""Enhanced CAD execution tool with error categorization."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ErrorCategory(str, Enum):
    SUCCESS = "success"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    GEOMETRY_ERROR = "geometry_error"
    EXPORT_ERROR = "export_error"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


@dataclass
class ExecutionResult:
    """Result of executing a build123d script."""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    output_file: Optional[str] = None
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    return_code: int = -1


def categorize_error(stderr: str) -> ErrorCategory:
    """Categorize the error from stderr output."""
    stderr_lower = stderr.lower()
    if "importerror" in stderr_lower or "modulenotfounderror" in stderr_lower:
        return ErrorCategory.IMPORT_ERROR
    if "syntaxerror" in stderr_lower or "indentationerror" in stderr_lower:
        return ErrorCategory.SYNTAX_ERROR
    if any(kw in stderr_lower for kw in [
        "standard_boolean", "gp_pnt", "topods", "brep", "occ",
        "geometry", "shape", "solid", "wire", "edge", "face",
        "build123d", "buildpart", "buildsketch",
    ]):
        return ErrorCategory.GEOMETRY_ERROR
    if "export" in stderr_lower or "step" in stderr_lower:
        return ErrorCategory.EXPORT_ERROR
    if "timed out" in stderr_lower:
        return ErrorCategory.TIMEOUT
    return ErrorCategory.RUNTIME_ERROR


def execute_build123d(
    script_content: str,
    output_path: str = "output.step",
    timeout: int = 60,
) -> ExecutionResult:
    """Execute a build123d script in a subprocess.
    
    Args:
        script_content: Python code using build123d.
        output_path: Path where the STEP file should be written.
        timeout: Maximum execution time in seconds.
    
    Returns:
        ExecutionResult with success status and details.
    """
    logger.info("executing_build123d", output_path=output_path, timeout=timeout)

    # Build the wrapper script
    wrapper_lines = [
        "import sys",
        "import os",
        "try:",
        "    from build123d import *",
        "except ImportError:",
        '    print("Error: build123d not installed.", file=sys.stderr)',
        "    sys.exit(1)",
        "",
        "# --- User script start ---",
        script_content,
        "# --- User script end ---",
        "",
        "# Auto-export if not already done",
        f'_output_path = r"{output_path}"',
        "if 'part' in dir() and not os.path.exists(_output_path):",
        "    try:",
        "        if hasattr(part, 'part'):",
        "            part.part.export_step(_output_path)",
        "        elif hasattr(part, 'export_step'):",
        "            part.export_step(_output_path)",
        "        else:",
        "            export_step(part, _output_path)",
        '        print(f"Auto-exported to {_output_path}")',
        "    except Exception as e:",
        '        print(f"Auto-export error: {e}", file=sys.stderr)',
        "if not os.path.exists(_output_path):",
        "    # Try to find any BuildPart context",
        "    for _name, _val in list(locals().items()):",
        "        if hasattr(_val, 'part') and hasattr(_val.part, 'export_step'):",
        "            try:",
        "                _val.part.export_step(_output_path)",
        '                print(f"Auto-exported {_name} to {_output_path}")',
        "                break",
        "            except Exception:",
        "                pass",
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write("\n".join(wrapper_lines))
        tmp_path = tmp.name

    try:
        python_executable = _resolve_python_executable()
        result = subprocess.run(
            [python_executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output_exists = os.path.exists(output_path)
        success = result.returncode == 0 and output_exists

        if success:
            error_cat = ErrorCategory.SUCCESS
        elif result.returncode != 0:
            error_cat = categorize_error(result.stderr)
        else:
            error_cat = ErrorCategory.EXPORT_ERROR

        exec_result = ExecutionResult(
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            output_file=output_path if output_exists else None,
            error_category=error_cat,
            return_code=result.returncode,
        )
        logger.info(
            "build123d_execution_complete",
            success=success,
            error_category=error_cat.value,
        )
        return exec_result

    except subprocess.TimeoutExpired:
        logger.warning("build123d_timeout", timeout=timeout)
        return ExecutionResult(
            success=False,
            stderr=f"Execution timed out after {timeout} seconds.",
            error_category=ErrorCategory.TIMEOUT,
        )
    finally:
        os.remove(tmp_path)


def _resolve_python_executable() -> str:
    """Prefer an explicit or local project interpreter for CAD execution."""
    configured = os.environ.get("DRAWING_TO_CAD_PYTHON")
    if configured and Path(configured).exists():
        return configured

    project_venv = Path.cwd() / ".venv" / "bin" / "python"
    if project_venv.exists():
        return str(project_venv)

    return sys.executable
