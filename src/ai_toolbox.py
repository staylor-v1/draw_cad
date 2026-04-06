"""AI-facing toolbox wrappers and manifest generation for drawing-to-cad."""
from __future__ import annotations

import ast
import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

import src.reconstruction.orthographic_solver as orthographic_solver_module
import src.reconstruction.reprojection as reprojection_module
import src.reconstruction.total_view_dataset as total_view_dataset_module
import src.tools.cad as cad_module
from src.reconstruction import (
    OrthographicReconstructionResult,
    OrthographicTriplet,
    OrthographicTripletReconstructor,
    ReconstructionCandidate,
    TotalViewArchive,
    evaluate_step_against_triplet,
)
from src.schemas.pipeline_config import PipelineConfig
from src.tools.cad import execute_build123d


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLBOX_MANIFEST_PATH = REPO_ROOT / "config" / "ai_toolbox_manifest.yaml"


@dataclass(frozen=True)
class ToolParameterSpec:
    """Machine-readable description of one callable parameter."""

    name: str
    annotation: str
    required: bool
    default: str | None = None
    kind: str = "POSITIONAL_OR_KEYWORD"


@dataclass(frozen=True)
class ToolDescriptor:
    """Manifest entry for one function, method, or script entrypoint."""

    name: str
    qualname: str
    module: str
    kind: str
    visibility: str
    source_path: str
    line: int
    signature: str
    description: str
    parameters: list[ToolParameterSpec] = field(default_factory=list)
    returns: str = "Any"
    callable_from_registry: bool = False
    registry_name: str | None = None
    owner: str | None = None


DESCRIPTION_OVERRIDES = {
    "src.reconstruction.total_view_dataset.TotalViewArchive._build_index": (
        "Build the case-id to member-name index from a zipped Total_view_data SVG archive."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor.generate_candidate_programs": (
        "Generate deterministic reconstruction candidates that vary the shape model and hidden-feature carving strategy."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor._extract_outer_contour": (
        "Rasterize visible SVG polylines, fill the silhouette, and recover the outer contour used for reconstruction."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor._infer_axisymmetric_profile": (
        "Detect whether the triplet is consistent with a lathe-like part and derive a revolvable radius-versus-height profile."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor._sample_axisymmetric_profile": (
        "Sample horizontal spans through a contour and convert them into radius samples around an inferred revolution axis."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor._infer_hidden_cylinders": (
        "Infer conservative cylindrical cuts from corroborated red hidden-line circles and side-view support segments."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor._extract_top_view_circles": (
        "Fit circular hidden-feature candidates from red top-view polylines."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor._extract_horizontal_support_segments": (
        "Extract merged horizontal red support segments from side views for hidden-feature depth inference."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor._build_code": (
        "Emit a build123d program for the visual-hull reconstruction candidate."
    ),
    "src.reconstruction.orthographic_solver.OrthographicTripletReconstructor._build_revolve_code": (
        "Emit a build123d program for the axisymmetric revolve reconstruction candidate."
    ),
    "src.reconstruction.reprojection.evaluate_step_against_triplet": (
        "Project a generated STEP solid back into the source orthographic views and score visible and hidden line fidelity."
    ),
    "src.reconstruction.reprojection.project_shape_to_view_polylines": (
        "Project a B-Rep shape into one orthographic view and sample both visible and hidden edges into 2D polylines."
    ),
    "src.reconstruction.reprojection.render_comparison": (
        "Align source and predicted line work in a shared raster frame and build masks plus an optional overlay image."
    ),
    "src.reconstruction.reprojection.compare_line_masks": (
        "Compute tolerance-aware precision, recall, F1, and IoU for source versus predicted line masks."
    ),
    "src.reconstruction.reprojection._shape_to_polylines": (
        "Sample each projected OpenCASCADE edge into a 2D polyline representation."
    ),
    "src.tools.cad.execute_build123d": (
        "Execute generated build123d code in a subprocess and classify import, geometry, export, or runtime failures."
    ),
    "scripts.reconstruct_total_view_data._maybe_reexec_into_project_venv": (
        "CLI helper that re-executes the reconstruction runner inside the repo-local virtualenv when available."
    ),
    "scripts.reconstruct_total_view_data.main": (
        "CLI entrypoint for deterministic Total_view_data reconstruction with optional candidate selection."
    ),
    "scripts.benchmark_total_view_data._maybe_reexec_into_project_venv": (
        "CLI helper that re-executes the benchmark runner inside the repo-local virtualenv when available."
    ),
    "scripts.benchmark_total_view_data.main": (
        "CLI entrypoint for reprojection benchmarking over a slice of the Total_view_data dataset."
    ),
    "scripts.export_ai_toolbox._maybe_reexec_into_project_venv": (
        "CLI helper that re-executes the toolbox manifest exporter inside the repo-local virtualenv when available."
    ),
    "scripts.export_ai_toolbox.main": (
        "CLI entrypoint that writes the AI toolbox manifest for this repository."
    ),
}


def load_pipeline_config(config_path: str | Path = "config/total_view_data.yaml") -> PipelineConfig:
    """Load a pipeline configuration file for toolbox-driven reconstruction work."""
    return PipelineConfig.from_yaml(config_path)


def load_total_view_archive(svg_zip_path: str | Path) -> TotalViewArchive:
    """Open a zipped Total_view_data SVG archive."""
    return TotalViewArchive(svg_zip_path)


def list_total_view_cases(
    svg_zip_path: str | Path,
    required_views: tuple[str, ...] = ("f", "r", "t"),
    require_complete: bool = True,
) -> list[str]:
    """List case ids from a zipped Total_view_data SVG archive."""
    archive = load_total_view_archive(svg_zip_path)
    return archive.case_ids(
        required_views=required_views,
        require_complete=require_complete,
    )


def load_total_view_triplet(
    svg_zip_path: str | Path,
    case_id: str,
    view_suffixes: tuple[str, str, str] = ("f", "r", "t"),
) -> OrthographicTriplet:
    """Load one grouped orthographic triplet from a zipped Total_view_data SVG archive."""
    archive = load_total_view_archive(svg_zip_path)
    return archive.load_triplet(case_id, view_suffixes=view_suffixes)


def build_reconstructor(
    config: PipelineConfig | str | Path = "config/total_view_data.yaml",
    view_suffixes: tuple[str, str, str] | None = None,
) -> OrthographicTripletReconstructor:
    """Build the deterministic orthographic reconstructor from a config object or YAML path."""
    pipeline_config = (
        load_pipeline_config(config)
        if isinstance(config, (str, Path))
        else config
    )
    reconstructor = OrthographicTripletReconstructor.from_pipeline_config(pipeline_config)
    if view_suffixes is None:
        return reconstructor
    return OrthographicTripletReconstructor(
        config=pipeline_config.orthographic_reconstruction,
        view_suffixes=view_suffixes,
    )


def generate_reconstruction_program(
    triplet: OrthographicTriplet,
    config: PipelineConfig | str | Path = "config/total_view_data.yaml",
    view_suffixes: tuple[str, str, str] | None = None,
) -> OrthographicReconstructionResult:
    """Generate the default deterministic reconstruction program for one orthographic triplet."""
    reconstructor = build_reconstructor(config=config, view_suffixes=view_suffixes)
    return reconstructor.generate_program(triplet)


def generate_reconstruction_candidates(
    triplet: OrthographicTriplet,
    config: PipelineConfig | str | Path = "config/total_view_data.yaml",
    view_suffixes: tuple[str, str, str] | None = None,
) -> list[ReconstructionCandidate]:
    """Generate the deterministic reconstruction candidate set for one orthographic triplet."""
    reconstructor = build_reconstructor(config=config, view_suffixes=view_suffixes)
    return reconstructor.generate_candidate_programs(triplet)


def run_build123d_script(
    script_content: str,
    output_path: str = "output.step",
    timeout: int = 60,
):
    """Run generated build123d code and return the execution result."""
    return execute_build123d(
        script_content=script_content,
        output_path=output_path,
        timeout=timeout,
    )


def evaluate_reconstruction_step(
    step_path: str | Path,
    triplet: OrthographicTriplet,
    config: PipelineConfig | str | Path = "config/total_view_data.yaml",
    view_suffixes: tuple[str, str, str] = ("f", "r", "t"),
):
    """Evaluate a generated STEP file against its source orthographic triplet."""
    pipeline_config = (
        load_pipeline_config(config)
        if isinstance(config, (str, Path))
        else config
    )
    return evaluate_step_against_triplet(
        step_path=step_path,
        triplet=triplet,
        config=pipeline_config.reprojection,
        view_suffixes=view_suffixes,
    )


def reconstruct_case_with_candidate_search(
    svg_zip_path: str | Path,
    case_id: str,
    output_dir: str | Path,
    config: PipelineConfig | str | Path = "config/total_view_data.yaml",
    view_suffixes: tuple[str, str, str] = ("f", "r", "t"),
    candidate_search: bool = True,
) -> dict[str, Any]:
    """Reconstruct one dataset case, score the candidates, and return the best result."""
    pipeline_config = (
        load_pipeline_config(config)
        if isinstance(config, (str, Path))
        else config
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    triplet = load_total_view_triplet(svg_zip_path, case_id, view_suffixes=view_suffixes)
    reconstructor = build_reconstructor(
        config=pipeline_config,
        view_suffixes=view_suffixes,
    )
    candidates = (
        reconstructor.generate_candidate_programs(triplet)
        if candidate_search and pipeline_config.orthographic_reconstruction.candidate_search_enabled
        else [ReconstructionCandidate(name="default", result=reconstructor.generate_program(triplet))]
    )

    candidate_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    for candidate in candidates:
        code_path = output_path / f"{case_id}__{candidate.name}.py"
        step_path = output_path / f"{case_id}__{candidate.name}.step"
        code_path.write_text(candidate.result.code, encoding="utf-8")

        execution = run_build123d_script(
            candidate.result.code,
            output_path=str(step_path),
            timeout=pipeline_config.pipeline.execution_timeout,
        )
        record: dict[str, Any] = {
            "candidate": candidate.name,
            "code_path": str(code_path),
            "step_path": str(step_path) if execution.success else None,
            "success": execution.success,
            "error_category": execution.error_category.value,
            "stderr": execution.stderr,
            "consensus_extents": candidate.result.consensus_extents,
            "hidden_feature_count": len(candidate.result.hidden_cylinders),
        }
        if execution.success:
            score, _ = evaluate_reconstruction_step(
                step_path,
                triplet,
                config=pipeline_config,
                view_suffixes=view_suffixes,
            )
            record["score"] = score.score
            record["views"] = {
                suffix: {
                    "score": view_score.score,
                    "visible_f1": view_score.visible.f1,
                    "hidden_f1": view_score.hidden.f1,
                }
                for suffix, view_score in score.views.items()
            }
            if best_record is None or float(record["score"]) > float(best_record["score"]):
                best_record = record
        candidate_records.append(record)

    if best_record is None:
        raise RuntimeError(f"No reconstruction candidate succeeded for case {case_id}")

    return {
        "case_id": case_id,
        "selected_candidate": best_record["candidate"],
        "selected_score": best_record["score"],
        "selected_step_path": best_record["step_path"],
        "selected_code_path": best_record["code_path"],
        "candidates": candidate_records,
    }


def list_tool_names() -> list[str]:
    """List the stable toolbox entrypoints exposed by this module."""
    return sorted(TOOL_REGISTRY)


def get_toolbox_manifest() -> dict[str, Any]:
    """Build the machine-readable AI toolbox manifest for the repo."""
    inventory = _build_symbol_inventory()
    stable_tools = [
        descriptor
        for descriptor in inventory
        if descriptor.callable_from_registry
    ]
    return {
        "manifest_version": 1,
        "repo_root": str(REPO_ROOT),
        "stable_tools": [asdict(entry) for entry in stable_tools],
        "inventory": [asdict(entry) for entry in inventory],
    }


def write_toolbox_manifest(
    output_path: str | Path = DEFAULT_TOOLBOX_MANIFEST_PATH,
) -> Path:
    """Write the AI toolbox manifest to YAML and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            get_toolbox_manifest(),
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    return path


def invoke_tool(tool_name: str, **kwargs: Any) -> Any:
    """Invoke one of the stable toolbox entrypoints by name."""
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown toolbox tool: {tool_name}")
    return TOOL_REGISTRY[tool_name](**kwargs)


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "load_pipeline_config": load_pipeline_config,
    "load_total_view_archive": load_total_view_archive,
    "list_total_view_cases": list_total_view_cases,
    "load_total_view_triplet": load_total_view_triplet,
    "build_reconstructor": build_reconstructor,
    "generate_reconstruction_program": generate_reconstruction_program,
    "generate_reconstruction_candidates": generate_reconstruction_candidates,
    "run_build123d_script": run_build123d_script,
    "evaluate_reconstruction_step": evaluate_reconstruction_step,
    "reconstruct_case_with_candidate_search": reconstruct_case_with_candidate_search,
    "list_tool_names": list_tool_names,
    "get_toolbox_manifest": get_toolbox_manifest,
    "write_toolbox_manifest": write_toolbox_manifest,
}


def _build_symbol_inventory() -> list[ToolDescriptor]:
    descriptors: list[ToolDescriptor] = []
    descriptors.extend(_scan_module_functions(total_view_dataset_module))
    descriptors.extend(_scan_class_methods(total_view_dataset_module.TotalViewArchive))
    descriptors.extend(_scan_module_functions(orthographic_solver_module))
    descriptors.extend(_scan_class_methods(orthographic_solver_module.OrthographicTripletReconstructor))
    descriptors.extend(_scan_module_functions(reprojection_module))
    descriptors.extend(_scan_module_functions(cad_module))
    descriptors.extend(_scan_wrappers())
    descriptors.extend(_scan_script_functions(REPO_ROOT / "scripts" / "reconstruct_total_view_data.py", "scripts.reconstruct_total_view_data"))
    descriptors.extend(_scan_script_functions(REPO_ROOT / "scripts" / "benchmark_total_view_data.py", "scripts.benchmark_total_view_data"))
    descriptors.extend(_scan_script_functions(REPO_ROOT / "scripts" / "export_ai_toolbox.py", "scripts.export_ai_toolbox"))
    return sorted(descriptors, key=lambda item: item.qualname)


def _scan_module_functions(module) -> list[ToolDescriptor]:
    descriptors: list[ToolDescriptor] = []
    for _, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ != module.__name__:
            continue
        descriptors.append(
            _descriptor_from_callable(
                obj=obj,
                qualname=f"{module.__name__}.{obj.__name__}",
                kind="function",
                owner=None,
                registry_name=_registry_name_for_callable(obj),
            )
        )
    return descriptors


def _scan_class_methods(cls) -> list[ToolDescriptor]:
    descriptors: list[ToolDescriptor] = []
    for name, obj in inspect.getmembers(cls):
        if name.startswith("__"):
            continue
        if not (inspect.isfunction(obj) or inspect.ismethod(obj)):
            continue
        qualname = f"{cls.__module__}.{cls.__name__}.{name}"
        if not getattr(obj, "__qualname__", "").startswith(f"{cls.__name__}."):
            continue
        descriptors.append(
            _descriptor_from_callable(
                obj=obj,
                qualname=qualname,
                kind="method",
                owner=cls.__name__,
                registry_name=None,
            )
        )
    return descriptors


def _scan_wrappers() -> list[ToolDescriptor]:
    descriptors: list[ToolDescriptor] = []
    for registry_name, callable_obj in TOOL_REGISTRY.items():
        qualname = f"{__name__}.{callable_obj.__name__}"
        descriptors.append(
            _descriptor_from_callable(
                obj=callable_obj,
                qualname=qualname,
                kind="tool",
                owner=None,
                registry_name=registry_name,
            )
        )
    return descriptors


def _scan_script_functions(script_path: Path, module_name: str) -> list[ToolDescriptor]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    descriptors: list[ToolDescriptor] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        qualname = f"{module_name}.{node.name}"
        description = DESCRIPTION_OVERRIDES.get(
            qualname,
            _fallback_description(node.name, owner=None),
        )
        descriptors.append(
            ToolDescriptor(
                name=node.name,
                qualname=qualname,
                module=module_name,
                kind="script_function",
                visibility="internal" if node.name.startswith("_") else "public",
                source_path=str(script_path.relative_to(REPO_ROOT)),
                line=node.lineno,
                signature=_ast_signature(node),
                description=description,
                parameters=_ast_parameters(node),
                returns=ast.unparse(node.returns) if node.returns is not None else "Any",
                callable_from_registry=False,
                registry_name=None,
                owner=None,
            )
        )
    return descriptors


def _descriptor_from_callable(
    obj: Callable[..., Any],
    qualname: str,
    kind: str,
    owner: str | None,
    registry_name: str | None,
) -> ToolDescriptor:
    source_path = Path(inspect.getsourcefile(obj) or "").resolve()
    line = inspect.getsourcelines(obj)[1]
    params = _parameters_from_signature(inspect.signature(obj))
    returns = _annotation_to_string(inspect.signature(obj).return_annotation)
    description = DESCRIPTION_OVERRIDES.get(
        qualname,
        inspect.getdoc(obj).splitlines()[0] if inspect.getdoc(obj) else _fallback_description(obj.__name__, owner),
    )
    return ToolDescriptor(
        name=obj.__name__,
        qualname=qualname,
        module=obj.__module__,
        kind=kind,
        visibility="internal" if obj.__name__.startswith("_") else "public",
        source_path=str(source_path.relative_to(REPO_ROOT)),
        line=line,
        signature=_signature_string(inspect.signature(obj)),
        description=description,
        parameters=params,
        returns=returns,
        callable_from_registry=registry_name is not None,
        registry_name=registry_name,
        owner=owner,
    )


def _registry_name_for_callable(obj: Callable[..., Any]) -> str | None:
    for registry_name, tool in TOOL_REGISTRY.items():
        if tool is obj:
            return registry_name
    return None


def _parameters_from_signature(signature: inspect.Signature) -> list[ToolParameterSpec]:
    params: list[ToolParameterSpec] = []
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        params.append(
            ToolParameterSpec(
                name=parameter.name,
                annotation=_annotation_to_string(parameter.annotation),
                required=parameter.default is inspect._empty,
                default=None if parameter.default is inspect._empty else repr(parameter.default),
                kind=parameter.kind.name,
            )
        )
    return params


def _signature_string(signature: inspect.Signature) -> str:
    rendered: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        chunk = parameter.name
        annotation = _annotation_to_string(parameter.annotation)
        if annotation != "Any":
            chunk += f": {annotation}"
        if parameter.default is not inspect._empty:
            chunk += f" = {repr(parameter.default)}"
        rendered.append(chunk)
    returns = _annotation_to_string(signature.return_annotation)
    return f"({', '.join(rendered)}) -> {returns}"


def _annotation_to_string(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    if getattr(annotation, "__module__", "") == "builtins":
        return annotation.__name__
    rendered = str(annotation)
    return (
        rendered
        .replace("typing.", "")
        .replace("pathlib.", "")
        .replace("src.reconstruction.", "")
        .replace("src.schemas.pipeline_config.", "")
    )


def _fallback_description(name: str, owner: str | None) -> str:
    normalized = name.lstrip("_")
    words = normalized.split("_")
    verb = words[0] if words else normalized
    target = " ".join(words[1:]) if len(words) > 1 else normalized
    phrase_map = {
        "parse": f"Parse {target}",
        "load": f"Load {target}",
        "build": f"Build {target}",
        "extract": f"Extract {target}",
        "infer": f"Infer {target}",
        "scale": f"Scale {target}",
        "cluster": f"Cluster {target}",
        "dedupe": f"Deduplicate {target}",
        "format": f"Format {target}",
        "render": f"Render {target}",
        "compare": f"Compare {target}",
        "fit": f"Fit {target}",
        "simplify": f"Simplify {target}",
        "evaluate": f"Evaluate {target}",
        "project": f"Project {target}",
        "translate": f"Translate {target}",
        "draw": f"Draw {target}",
        "sample": f"Sample {target}",
        "resolve": f"Resolve {target}",
        "categorize": f"Categorize {target}",
        "generate": f"Generate {target}",
        "write": f"Write {target}",
        "invoke": f"Invoke {target}",
        "list": f"List {target}",
    }
    action = phrase_map.get(verb, normalized.replace("_", " "))
    if name.startswith("_"):
        if owner:
            return f"Internal helper on {owner} to {action.lower()}."
        return f"Internal helper to {action.lower()}."
    if owner:
        return f"Method on {owner} to {action.lower()}."
    return f"Function to {action.lower()}."


def _ast_signature(node: ast.FunctionDef) -> str:
    params = [parameter.name for parameter in _ast_parameters(node)]
    returns = ast.unparse(node.returns) if node.returns is not None else "Any"
    return f"({', '.join(params)}) -> {returns}"


def _ast_parameters(node: ast.FunctionDef) -> list[ToolParameterSpec]:
    args = []
    defaults: list[ast.expr | None] = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)
    for argument, default in zip(node.args.args, defaults):
        if argument.arg in {"self", "cls"}:
            continue
        args.append(
            ToolParameterSpec(
                name=argument.arg,
                annotation=ast.unparse(argument.annotation) if argument.annotation is not None else "Any",
                required=default is None,
                default=None if default is None else ast.unparse(default),
                kind="POSITIONAL_OR_KEYWORD",
            )
        )
    return args
