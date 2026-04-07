"""Gemma 4 experiments on Total_view_data orthographic triplets."""
from __future__ import annotations

import base64
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageOps

from src.pipeline.reasoning_stage import extract_code
from src.reconstruction import (
    OrthographicTriplet,
    OrthographicTripletReconstructor,
    ReconstructionCandidate,
    TotalViewArchive,
    TotalViewPngArchive,
    evaluate_step_against_triplet,
)
from src.schemas.pipeline_config import PipelineConfig
from src.tools.cad import execute_build123d
from src.utils.file_utils import ensure_dir, load_prompt, load_yaml, save_json, save_yaml
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BUCKET_ORDER = ("axisymmetric", "prismatic_hidden", "prismatic")


@dataclass(frozen=True)
class PromptCandidateConfig:
    """One prompt candidate used during calibration."""

    id: str
    path: str


@dataclass(frozen=True)
class SelectedCase:
    """Selected evaluation case metadata."""

    case_id: str
    dataset: str
    bucket: str
    split: str
    baseline_score: float
    baseline_visible_f1: float
    baseline_hidden_f1: float
    selected_candidate: str
    hidden_feature_count: int
    total_polylines: int


def _strategy_description(name: str) -> str:
    mapping = {
        "visual_hull_hidden": "Intersect the three silhouettes and carve supported hidden cylinders.",
        "visual_hull_base": "Intersect the three silhouettes without hidden-feature carving.",
        "axisymmetric_hidden": "Fit a revolve profile and carve supported hidden cylinders.",
        "axisymmetric_base": "Fit a revolve profile without hidden-feature carving.",
    }
    return mapping.get(name, name.replace("_", " "))


class Gemma4TotalViewExperiment:
    """Run prompt search and evaluation for Gemma 4 on Total_view_data."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.settings = load_yaml(self.config_path)
        self.pipeline_config = PipelineConfig.from_yaml(
            self.settings.get("pipeline_config", "config/total_view_data.yaml")
        )

        self.dataset = self.settings.get("dataset", "ABC")
        self.model_name = self.settings["model"]["name"]
        self.temperature = float(self.settings["model"].get("temperature", 0.0))
        self.max_tokens = int(self.settings["model"].get("max_tokens", 2048))
        self.max_tool_rounds = int(self.settings["model"].get("max_tool_rounds", 4))
        self.api_url = self.settings["model"].get(
            "api_url",
            f"{self.pipeline_config.inference.ollama.base_url}/api/chat",
        )
        self.report_dir = ensure_dir(
            self.settings.get("report_dir", "reports/gemma4_total_view")
        )
        self.scratch_dir = ensure_dir(
            self.settings.get("scratch_dir", "experiments/gemma4_total_view_runtime")
        )

        self.svg_archive = TotalViewArchive(
            self.pipeline_config.total_view_data.get_svg_archive(self.dataset)
        )
        self.png_archive = TotalViewPngArchive(
            self.pipeline_config.total_view_data.get_png_archive(self.dataset)
        )
        self.reconstructor = OrthographicTripletReconstructor.from_pipeline_config(
            self.pipeline_config
        )

        self._triplets: dict[str, OrthographicTriplet] = {}
        self._candidates: dict[str, list[ReconstructionCandidate]] = {}
        self._http = httpx.Client(timeout=600)

    def close(self) -> None:
        self._http.close()

    def run(self) -> dict[str, Any]:
        """Run selection, prompt search, evaluation, and report generation."""
        pool = self._load_or_build_selection_pool()
        selected_cases = self._select_cases(pool)
        self._write_selected_cases(selected_cases)
        self._render_contact_sheet(selected_cases)

        calibration_cases = [case for case in selected_cases if case.split == "calibration"]
        evaluation_cases = [case for case in selected_cases if case.split == "evaluation"]

        prompt_search = {
            "raw": self._run_prompt_search(
                mode="raw",
                cases=calibration_cases,
                prompt_settings=self.settings["prompt_search"]["raw"],
            ),
            "tool": self._run_prompt_search(
                mode="tool",
                cases=calibration_cases,
                prompt_settings=self.settings["prompt_search"]["tool"],
            ),
        }

        evaluation = {
            "gemma4_raw": self._evaluate_mode(
                mode="raw",
                cases=evaluation_cases,
                prompt_id=prompt_search["raw"]["best_prompt_id"],
                prompt_path=prompt_search["raw"]["best_prompt_path"],
            ),
            "gemma4_with_tools": self._evaluate_mode(
                mode="tool",
                cases=evaluation_cases,
                prompt_id=prompt_search["tool"]["best_prompt_id"],
                prompt_path=prompt_search["tool"]["best_prompt_path"],
            ),
            "tools_only": self._evaluate_tools_only_subset(evaluation_cases, pool),
        }

        result = {
            "config_path": str(self.config_path),
            "dataset": self.dataset,
            "model": self.model_name,
            "selected_cases": [asdict(case) for case in selected_cases],
            "prompt_search": prompt_search,
            "evaluation": evaluation,
        }
        save_json(result, self.report_dir / "results.json")
        report_markdown = self._write_report(result)
        result["report_path"] = str(self.report_dir / "report.md")
        result["report_markdown"] = report_markdown
        return result

    def _load_or_build_selection_pool(self) -> list[dict[str, Any]]:
        pool_path = self.report_dir / "selection_pool.json"
        if pool_path.exists():
            return load_yaml(pool_path) if pool_path.suffix in {".yaml", ".yml"} else json.loads(pool_path.read_text(encoding="utf-8"))

        selection_cfg = self.settings["selection"]
        case_ids = self.svg_archive.case_ids()[: int(selection_cfg.get("scan_limit", 50))]
        records: list[dict[str, Any]] = []
        run_dir = ensure_dir(self.scratch_dir / "selection_pool")
        for index, case_id in enumerate(case_ids, start=1):
            case_record = self._run_tools_only_case(
                case_id=case_id,
                output_dir=run_dir / case_id,
            )
            logger.info(
                "gemma4_selection_case",
                index=index,
                total=len(case_ids),
                case_id=case_id,
                score=case_record["score"],
                bucket=case_record["bucket"],
            )
            records.append(case_record)

        save_json(records, pool_path)
        return records

    def _select_cases(self, pool: list[dict[str, Any]]) -> list[SelectedCase]:
        selection_cfg = self.settings["selection"]
        min_score = float(selection_cfg.get("min_baseline_score", 0.25))
        calibration_per_bucket = int(selection_cfg.get("calibration_per_bucket", 1))
        evaluation_per_bucket = int(selection_cfg.get("evaluation_per_bucket", 2))
        window_multiplier = int(selection_cfg.get("window_multiplier", 3))

        selected: list[SelectedCase] = []
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in pool:
            if float(record["score"]) >= min_score:
                by_bucket[record["bucket"]].append(record)

        for bucket in BUCKET_ORDER:
            bucket_records = sorted(
                by_bucket[bucket],
                key=lambda item: (-float(item["score"]), item["case_id"]),
            )
            total_needed = calibration_per_bucket + evaluation_per_bucket
            if len(bucket_records) < total_needed:
                raise RuntimeError(
                    f"Not enough eligible cases for bucket {bucket}: "
                    f"need {total_needed}, found {len(bucket_records)}"
                )

            window = bucket_records[: max(total_needed * window_multiplier, total_needed)]
            picks = self._evenly_spaced(window, total_needed)
            calibration_index = len(picks) // 2

            for index, record in enumerate(picks):
                split = "calibration" if index == calibration_index else "evaluation"
                selected.append(
                    SelectedCase(
                        case_id=record["case_id"],
                        dataset=self.dataset,
                        bucket=bucket,
                        split=split,
                        baseline_score=float(record["score"]),
                        baseline_visible_f1=float(record["mean_visible_f1"]),
                        baseline_hidden_f1=float(record["mean_hidden_f1"]),
                        selected_candidate=str(record["selected_candidate"]),
                        hidden_feature_count=int(record["hidden_feature_count"]),
                        total_polylines=int(record["total_polylines"]),
                    )
                )

        selected.sort(key=lambda item: (item.split, BUCKET_ORDER.index(item.bucket), -item.baseline_score))
        return selected

    @staticmethod
    def _evenly_spaced(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        """Pick evenly spaced records from a sorted list."""
        if count == 1:
            return [records[len(records) // 2]]
        if len(records) == count:
            return records

        last_index = len(records) - 1
        picked: list[dict[str, Any]] = []
        seen: set[int] = set()
        for step in range(count):
            position = round(step * last_index / (count - 1))
            while position in seen and position < last_index:
                position += 1
            seen.add(position)
            picked.append(records[position])
        return picked

    def _write_selected_cases(self, selected_cases: list[SelectedCase]) -> None:
        save_yaml(
            {
                "dataset": self.dataset,
                "cases": [asdict(case) for case in selected_cases],
            },
            self.report_dir / "selected_cases.yaml",
        )

    def _render_contact_sheet(self, selected_cases: list[SelectedCase]) -> Path:
        rows = []
        for case in selected_cases:
            triplet = self.png_archive.load_triplet(case.case_id)
            rows.append((case, triplet))

        thumb_w = 180
        thumb_h = 180
        pad = 16
        label_h = 40
        canvas_width = pad + 3 * (thumb_w + pad)
        canvas_height = pad + len(rows) * (thumb_h + label_h + pad)
        sheet = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(sheet)

        for row_index, (case, triplet) in enumerate(rows):
            y = pad + row_index * (thumb_h + label_h + pad)
            for col_index, suffix in enumerate(("f", "r", "t")):
                view = triplet.views[suffix]
                image = Image.open(BytesIO(view.image_bytes)).convert("RGB")
                thumb = ImageOps.contain(image, (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
                x = pad + col_index * (thumb_w + pad)
                sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y + (thumb_h - thumb.height) // 2))
                draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline="black", width=1)
                draw.text(
                    (x, y + thumb_h + 4),
                    f"{case.case_id}_{suffix} [{case.bucket}/{case.split[0]}]",
                    fill="black",
                )

        output_path = self.report_dir / "selected_cases_contact_sheet.png"
        sheet.save(output_path)
        return output_path

    def _run_prompt_search(
        self,
        *,
        mode: str,
        cases: list[SelectedCase],
        prompt_settings: dict[str, Any],
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for round_index, round_cfg in enumerate(prompt_settings["rounds"], start=1):
            for candidate_cfg in round_cfg["candidates"]:
                candidate = PromptCandidateConfig(**candidate_cfg)
                logger.info(
                    "gemma4_prompt_candidate_start",
                    mode=mode,
                    round=round_index,
                    prompt_id=candidate.id,
                )
                run_result = self._evaluate_mode(
                    mode=mode,
                    cases=cases,
                    prompt_id=candidate.id,
                    prompt_path=candidate.path,
                    stage=f"calibration_round_{round_index}",
                )
                run_result["round_name"] = round_cfg.get("name", f"round_{round_index}")
                results.append(run_result)
                if best is None or float(run_result["aggregate"]["mean_score"]) > float(best["aggregate"]["mean_score"]):
                    best = run_result

        if best is None:
            raise RuntimeError(f"No prompt candidates evaluated for mode {mode}")

        summary = {
            "mode": mode,
            "best_prompt_id": best["prompt_id"],
            "best_prompt_path": best["prompt_path"],
            "candidates": results,
        }
        save_json(summary, self.report_dir / f"{mode}_prompt_search.json")
        return summary

    def _evaluate_mode(
        self,
        *,
        mode: str,
        cases: list[SelectedCase],
        prompt_id: str,
        prompt_path: str,
        stage: str = "evaluation",
    ) -> dict[str, Any]:
        system_prompt = load_prompt(prompt_path)
        run_dir = ensure_dir(self.scratch_dir / stage / mode / prompt_id)
        case_results: list[dict[str, Any]] = []
        for case in cases:
            if mode == "raw":
                result = self._run_raw_case(case, system_prompt, run_dir / case.case_id)
            elif mode == "tool":
                result = self._run_tool_case(case, system_prompt, run_dir / case.case_id)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            result["prompt_id"] = prompt_id
            result["prompt_path"] = prompt_path
            case_results.append(result)

        aggregate = self._aggregate_case_results(case_results)
        return {
            "mode": mode,
            "prompt_id": prompt_id,
            "prompt_path": prompt_path,
            "cases": case_results,
            "aggregate": aggregate,
        }

    def _evaluate_tools_only_subset(
        self,
        cases: list[SelectedCase],
        pool: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pool_by_case = {record["case_id"]: record for record in pool}
        case_results = []
        for case in cases:
            record = dict(pool_by_case[case.case_id])
            record["mode"] = "tools_only"
            record["success"] = bool(record.get("success", True))
            record["score"] = float(record["score"])
            record["mean_visible_f1"] = float(record["mean_visible_f1"])
            record["mean_hidden_f1"] = float(record["mean_hidden_f1"])
            case_results.append(record)

        return {
            "mode": "tools_only",
            "prompt_id": None,
            "prompt_path": None,
            "cases": case_results,
            "aggregate": self._aggregate_case_results(case_results),
        }

    def _run_raw_case(
        self,
        case: SelectedCase,
        system_prompt: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        ensure_dir(output_dir)
        images = self._case_images(case.case_id)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": self._raw_user_prompt(case.case_id),
                "images": images,
            },
        ]

        started = time.perf_counter()
        response = self._ollama_chat(messages=messages)
        elapsed = time.perf_counter() - started
        text = response["message"].get("content", "")
        code = extract_code(text)
        result = self._evaluate_generated_code(
            case_id=case.case_id,
            mode="gemma4_raw",
            code=code,
            output_dir=output_dir,
        )
        result.update(
            {
                "case_id": case.case_id,
                "bucket": case.bucket,
                "split": case.split,
                "duration_seconds": elapsed,
                "usage": {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                },
                "response_preview": text[:4000],
            }
        )
        save_json(
            {"messages": messages, "response": response, "result": result},
            output_dir / "raw_run.json",
        )
        return result

    def _run_tool_case(
        self,
        case: SelectedCase,
        system_prompt: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        ensure_dir(output_dir)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": self._tool_user_prompt(case.case_id),
                "images": self._case_images(case.case_id),
            },
        ]
        tool_calls_made = 0
        usage_prompt = 0
        usage_completion = 0
        best_seen: dict[str, Any] | None = None
        final_text = ""

        started = time.perf_counter()
        for round_index in range(self.max_tool_rounds):
            response = self._ollama_chat(messages=messages, tools=self._tool_schemas())
            usage_prompt += int(response.get("prompt_eval_count", 0))
            usage_completion += int(response.get("eval_count", 0))
            assistant_message = response["message"]
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", []) or []
            if not tool_calls:
                final_text = assistant_message.get("content", "")
                break

            for tool_call in tool_calls:
                tool_calls_made += 1
                tool_result = self._dispatch_tool_call(
                    case_id=case.case_id,
                    tool_call=tool_call,
                    output_dir=output_dir,
                )
                if tool_result.get("score") is not None:
                    if best_seen is None or float(tool_result["score"]) > float(best_seen["score"]):
                        best_seen = tool_result
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_call["function"]["name"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        elapsed = time.perf_counter() - started
        code = extract_code(final_text)
        result = self._evaluate_generated_code(
            case_id=case.case_id,
            mode="gemma4_with_tools",
            code=code,
            output_dir=output_dir,
        )
        if not result["success"] and best_seen is not None:
            result = {
                **best_seen,
                "mode": "gemma4_with_tools",
                "used_verified_fallback": True,
                "fallback_from_failed_final": True,
            }
        result.update(
            {
                "case_id": case.case_id,
                "bucket": case.bucket,
                "split": case.split,
                "duration_seconds": elapsed,
                "tool_calls": tool_calls_made,
                "usage": {
                    "prompt_tokens": usage_prompt,
                    "completion_tokens": usage_completion,
                },
                "response_preview": final_text[:4000],
            }
        )
        save_json(
            {"messages": messages, "result": result},
            output_dir / "tool_run.json",
        )
        return result

    def _run_tools_only_case(self, case_id: str, output_dir: Path) -> dict[str, Any]:
        ensure_dir(output_dir)
        triplet = self._get_triplet(case_id)
        candidate_records = []
        best: dict[str, Any] | None = None
        for candidate in self._get_candidates(case_id):
            evaluation = self._evaluate_candidate_code(
                case_id=case_id,
                candidate_name=candidate.name,
                code=candidate.result.code,
                output_dir=output_dir,
            )
            candidate_records.append(evaluation)
            if evaluation["success"] and (
                best is None or float(evaluation["score"]) > float(best["score"])
            ):
                best = evaluation

        if best is None:
            best = {
                "success": False,
                "score": 0.0,
                "mean_visible_f1": 0.0,
                "mean_hidden_f1": 0.0,
                "selected_candidate": None,
                "hidden_feature_count": 0,
                "selected_step_path": None,
            }

        candidate_list = self._get_candidates(case_id)
        has_axisymmetric = any(
            candidate.name.startswith("axisymmetric")
            for candidate in candidate_list
        )
        max_hidden_features = max(
            len(candidate.result.hidden_cylinders)
            for candidate in candidate_list
        )
        if has_axisymmetric:
            bucket = "axisymmetric"
        elif max_hidden_features > 0:
            bucket = "prismatic_hidden"
        else:
            bucket = "prismatic"

        total_polylines = sum(len(view.polylines) for view in triplet.views.values())
        record = {
            "case_id": case_id,
            "mode": "tools_only",
            "success": bool(best["success"]),
            "score": float(best["score"]),
            "mean_visible_f1": float(best["mean_visible_f1"]),
            "mean_hidden_f1": float(best["mean_hidden_f1"]),
            "selected_candidate": best.get("source"),
            "selected_step_path": best.get("step_path"),
            "hidden_feature_count": int(best.get("hidden_feature_count", 0)),
            "bucket": bucket,
            "total_polylines": total_polylines,
            "candidates": candidate_records,
        }
        save_json(record, output_dir / "tools_only.json")
        return record

    def _evaluate_candidate_code(
        self,
        *,
        case_id: str,
        candidate_name: str,
        code: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        evaluation = self._evaluate_generated_code(
            case_id=case_id,
            mode="tools_only",
            code=code,
            output_dir=output_dir,
            filename_prefix=candidate_name,
        )
        hidden_feature_count = 0
        for candidate in self._get_candidates(case_id):
            if candidate.name == candidate_name:
                hidden_feature_count = len(candidate.result.hidden_cylinders)
                break
        evaluation["source"] = candidate_name
        evaluation["hidden_feature_count"] = hidden_feature_count
        return evaluation

    def _evaluate_generated_code(
        self,
        *,
        case_id: str,
        mode: str,
        code: str,
        output_dir: Path,
        filename_prefix: str = "generated",
    ) -> dict[str, Any]:
        ensure_dir(output_dir)
        code_path = output_dir / f"{filename_prefix}.py"
        step_path = output_dir / f"{filename_prefix}.step"

        if not code:
            return {
                "mode": mode,
                "success": False,
                "score": 0.0,
                "mean_visible_f1": 0.0,
                "mean_hidden_f1": 0.0,
                "code_path": None,
                "step_path": None,
                "error_category": "no_code",
                "stderr": "",
            }

        code_path.write_text(code, encoding="utf-8")
        execution = execute_build123d(
            code,
            output_path=str(step_path),
            timeout=self.pipeline_config.pipeline.execution_timeout,
        )
        if not execution.success:
            return {
                "mode": mode,
                "success": False,
                "score": 0.0,
                "mean_visible_f1": 0.0,
                "mean_hidden_f1": 0.0,
                "code_path": str(code_path),
                "step_path": None,
                "error_category": execution.error_category.value,
                "stderr": execution.stderr,
            }

        triplet = self._get_triplet(case_id)
        score, _ = evaluate_step_against_triplet(
            step_path,
            triplet,
            config=self.pipeline_config.reprojection,
            view_suffixes=tuple(self.pipeline_config.total_view_data.preferred_views),  # type: ignore[arg-type]
        )
        mean_visible_f1 = sum(view.visible.f1 for view in score.views.values()) / len(score.views)
        mean_hidden_f1 = sum(view.hidden.f1 for view in score.views.values()) / len(score.views)
        return {
            "mode": mode,
            "success": True,
            "score": score.score,
            "mean_visible_f1": mean_visible_f1,
            "mean_hidden_f1": mean_hidden_f1,
            "code_path": str(code_path),
            "step_path": str(step_path),
            "error_category": "success",
            "stderr": execution.stderr,
        }

    def _dispatch_tool_call(
        self,
        *,
        case_id: str,
        tool_call: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        if name == "list_candidate_programs":
            return self._tool_list_candidate_programs(case_id)
        if name == "get_candidate_program":
            return self._tool_get_candidate_program(case_id, str(arguments["name"]))
        if name == "evaluate_candidate_program":
            return self._tool_evaluate_candidate_program(
                case_id,
                str(arguments["name"]),
                output_dir,
            )
        if name == "evaluate_code":
            return self._tool_evaluate_code(case_id, str(arguments["code"]), output_dir)
        return {"error": f"Unknown tool: {name}"}

    def _tool_list_candidate_programs(self, case_id: str) -> dict[str, Any]:
        candidates = self._get_candidates(case_id)
        return {
            "case_id": case_id,
            "candidates": [
                {
                    "name": candidate.name,
                    "strategy": _strategy_description(candidate.name),
                    "consensus_extents": candidate.result.consensus_extents,
                    "hidden_feature_count": len(candidate.result.hidden_cylinders),
                }
                for candidate in candidates
            ],
        }

    def _tool_get_candidate_program(self, case_id: str, name: str) -> dict[str, Any]:
        for candidate in self._get_candidates(case_id):
            if candidate.name == name:
                return {
                    "case_id": case_id,
                    "name": name,
                    "strategy": _strategy_description(name),
                    "consensus_extents": candidate.result.consensus_extents,
                    "hidden_feature_count": len(candidate.result.hidden_cylinders),
                    "code": candidate.result.code,
                }
        return {"error": f"Unknown candidate {name}"}

    def _tool_evaluate_candidate_program(
        self,
        case_id: str,
        name: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        for candidate in self._get_candidates(case_id):
            if candidate.name == name:
                result = self._evaluate_candidate_code(
                    case_id=case_id,
                    candidate_name=name,
                    code=candidate.result.code,
                    output_dir=output_dir / "tool_evaluations",
                )
                return result
        return {"error": f"Unknown candidate {name}"}

    def _tool_evaluate_code(
        self,
        case_id: str,
        code: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        result = self._evaluate_generated_code(
            case_id=case_id,
            mode="tool_custom_code",
            code=code,
            output_dir=output_dir / "tool_custom_code",
            filename_prefix=f"custom_{int(time.time() * 1000)}",
        )
        result["source"] = "custom_code"
        return result

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_candidate_programs",
                    "description": "List the deterministic CAD candidate programs available for the current case.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_candidate_program",
                    "description": "Fetch the full build123d code for one deterministic candidate program.",
                    "parameters": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The candidate name returned by list_candidate_programs.",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_candidate_program",
                    "description": "Execute and score one deterministic candidate program against the current case.",
                    "parameters": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The candidate name returned by list_candidate_programs.",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_code",
                    "description": "Execute custom build123d code and score the generated STEP file against the current case.",
                    "parameters": {
                        "type": "object",
                        "required": ["code"],
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "A complete build123d Python program that exports output.step.",
                            }
                        },
                    },
                },
            },
        ]

    def _aggregate_case_results(self, case_results: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(case_results)
        if total == 0:
            return {
                "total_cases": 0,
                "successful_cases": 0,
                "mean_score": 0.0,
                "mean_visible_f1": 0.0,
                "mean_hidden_f1": 0.0,
                "mean_duration_seconds": 0.0,
                "mean_tool_calls": 0.0,
            }

        successful = sum(1 for result in case_results if result.get("success"))
        mean = lambda key: sum(float(result.get(key, 0.0)) for result in case_results) / total
        return {
            "total_cases": total,
            "successful_cases": successful,
            "success_rate": successful / total,
            "mean_score": mean("score"),
            "mean_visible_f1": mean("mean_visible_f1"),
            "mean_hidden_f1": mean("mean_hidden_f1"),
            "mean_duration_seconds": mean("duration_seconds"),
            "mean_tool_calls": mean("tool_calls"),
        }

    def _write_report(self, result: dict[str, Any]) -> str:
        selected_cases = result["selected_cases"]
        prompt_search = result["prompt_search"]
        evaluation = result["evaluation"]
        raw_best = next(
            candidate
            for candidate in prompt_search["raw"]["candidates"]
            if candidate["prompt_id"] == prompt_search["raw"]["best_prompt_id"]
        )
        tool_best = next(
            candidate
            for candidate in prompt_search["tool"]["candidates"]
            if candidate["prompt_id"] == prompt_search["tool"]["best_prompt_id"]
        )

        lines = [
            "# Gemma 4 Total_view_data Experiment",
            "",
            f"- Dataset: `{result['dataset']}`",
            f"- Model: `{result['model']}`",
            f"- Selected cases: `{', '.join(case['case_id'] for case in selected_cases)}`",
            "",
            "## Case Selection",
            "",
            "| Case | Bucket | Split | Tools-only score | Candidate | Hidden cuts |",
            "| --- | --- | --- | ---: | --- | ---: |",
        ]
        for case in selected_cases:
            lines.append(
                f"| `{case['case_id']}` | `{case['bucket']}` | `{case['split']}` | "
                f"{case['baseline_score']:.3f} | `{case['selected_candidate']}` | "
                f"`{case['hidden_feature_count']}` |"
            )

        lines.extend(
            [
                "",
                "## Prompt Search",
                "",
                "| Mode | Best prompt | Mean score | Success rate |",
                "| --- | --- | ---: | ---: |",
                f"| Raw | `{prompt_search['raw']['best_prompt_id']}` | "
                f"{raw_best['aggregate']['mean_score']:.3f} | "
                f"{raw_best['aggregate']['success_rate']:.2f} |",
                f"| With tools | `{prompt_search['tool']['best_prompt_id']}` | "
                f"{tool_best['aggregate']['mean_score']:.3f} | "
                f"{tool_best['aggregate']['success_rate']:.2f} |",
                "",
                "## Held-Out Evaluation",
                "",
                "| Mode | Mean score | Mean visible F1 | Mean hidden F1 | Success rate | Mean tool calls |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        for key in ("gemma4_raw", "gemma4_with_tools", "tools_only"):
            aggregate = evaluation[key]["aggregate"]
            lines.append(
                f"| `{key}` | {aggregate['mean_score']:.3f} | {aggregate['mean_visible_f1']:.3f} | "
                f"{aggregate['mean_hidden_f1']:.3f} | {aggregate['success_rate']:.2f} | "
                f"{aggregate['mean_tool_calls']:.2f} |"
            )

        lines.extend(
            [
                "",
                "## Per-Case Held-Out Scores",
                "",
                "| Case | Bucket | Raw | With tools | Tools only |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        raw_cases = {case["case_id"]: case for case in evaluation["gemma4_raw"]["cases"]}
        tool_cases = {case["case_id"]: case for case in evaluation["gemma4_with_tools"]["cases"]}
        tools_only_cases = {case["case_id"]: case for case in evaluation["tools_only"]["cases"]}
        for case in [case for case in selected_cases if case["split"] == "evaluation"]:
            case_id = case["case_id"]
            lines.append(
                f"| `{case_id}` | `{case['bucket']}` | "
                f"{float(raw_cases[case_id]['score']):.3f} | "
                f"{float(tool_cases[case_id]['score']):.3f} | "
                f"{float(tools_only_cases[case_id]['score']):.3f} |"
            )

        report = "\n".join(lines) + "\n"
        (self.report_dir / "report.md").write_text(report, encoding="utf-8")
        return report

    def _raw_user_prompt(self, case_id: str) -> str:
        return (
            f"Current case: {case_id}\n"
            "Attached in order are the front, right, and top orthographic views of a single part.\n"
            "Front view = X-Z, right view = Y-Z, top view = X-Y.\n"
            "Use the images to infer a build123d model that best matches the silhouettes and obvious hidden features."
        )

    def _tool_user_prompt(self, case_id: str) -> str:
        return (
            f"Current case: {case_id}\n"
            "Attached in order are the front, right, and top orthographic views of a single part.\n"
            "Front view = X-Z, right view = Y-Z, top view = X-Y.\n"
            "Use tools when they reduce uncertainty or let you verify a candidate before finalizing code."
        )

    def _case_images(self, case_id: str) -> list[str]:
        triplet = self.png_archive.load_triplet(case_id)
        return [
            base64.b64encode(triplet.views[suffix].image_bytes).decode("ascii")
            for suffix in ("f", "r", "t")
        ]

    def _get_triplet(self, case_id: str) -> OrthographicTriplet:
        if case_id not in self._triplets:
            self._triplets[case_id] = self.svg_archive.load_triplet(
                case_id,
                view_suffixes=tuple(self.pipeline_config.total_view_data.preferred_views),  # type: ignore[arg-type]
            )
        return self._triplets[case_id]

    def _get_candidates(self, case_id: str) -> list[ReconstructionCandidate]:
        if case_id not in self._candidates:
            self._candidates[case_id] = self.reconstructor.generate_candidate_programs(
                self._get_triplet(case_id)
            )
        return self._candidates[case_id]

    def _ollama_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        response = self._http.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json()


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    """Convenience wrapper used by the CLI."""
    experiment = Gemma4TotalViewExperiment(config_path)
    try:
        return experiment.run()
    finally:
        experiment.close()
