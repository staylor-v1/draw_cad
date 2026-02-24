"""Report generation for benchmark results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.schemas.evaluation_result import BenchmarkReport
from src.utils.file_utils import save_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_report(report: BenchmarkReport, output_path: str | Path) -> str:
    """Generate a human-readable report from benchmark results.
    
    Args:
        report: BenchmarkReport with all results.
        output_path: Path to write the report JSON.
    
    Returns:
        Formatted text summary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    save_json(report.model_dump(), output_path)

    # Generate text summary
    lines = [
        f"# Benchmark Report: {report.suite_name}",
        f"",
        f"## Summary",
        f"- Total cases: {report.total_cases}",
        f"- Successful: {report.successful_cases}",
        f"- Failed: {report.failed_cases}",
        f"- Composite score: {report.aggregate_metrics.composite_score:.4f}",
        f"",
        f"## Aggregate Metrics",
        f"- Dimension fidelity: {report.aggregate_metrics.dimension_fidelity:.4f}",
        f"- Feature recall: {report.aggregate_metrics.feature_recall:.4f}",
        f"- Feature precision: {report.aggregate_metrics.feature_precision:.4f}",
        f"- Bounding box IoU: {report.aggregate_metrics.bounding_box_iou:.4f}",
        f"- Volume ratio: {report.aggregate_metrics.volume_ratio:.4f}",
        f"- Geometry valid: {report.aggregate_metrics.geometry_valid:.4f}",
        f"- Retry efficiency: {report.aggregate_metrics.retry_efficiency:.4f}",
        f"",
        f"## Case Results",
    ]

    for case in report.case_results:
        status = "PASS" if case.metrics.execution_success else "FAIL"
        lines.append(f"### {case.case_id} [{status}]")
        lines.append(f"  Score: {case.metrics.composite_score:.4f}")
        if case.error:
            lines.append(f"  Error: {case.error[:200]}")
        lines.append("")

    summary = "\n".join(lines)

    # Also save text summary
    text_path = output_path.with_suffix(".txt")
    text_path.write_text(summary)

    logger.info("report_generated", json_path=str(output_path), text_path=str(text_path))
    return summary
