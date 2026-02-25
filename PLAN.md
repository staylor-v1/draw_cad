# Drawing-to-CAD Agentic System — Master Plan

## System Overview

A three-loop agentic system that converts 2D engineering drawings into 3D CAD (STEP) files:

- **Loop 3 (Pipeline)**: Single drawing-to-CAD conversion. Vision model (llama-3.2-11b-Vision) reads the drawing, reasoning model (gpt-oss-120b) generates build123d Python code, execution stage runs it to produce a STEP file. Includes OCR, reconciliation, validation, and retry sub-stages.
- **Loop 2 (Meta-Optimizer)**: Iteratively improves Loop 3 by tuning prompts, few-shot examples, and parameters. Evaluates candidates against benchmark cases and accepts improvements.
- **Loop 1 (Human)**: Reviews Loop 2 outputs, sets strategic direction, approves deployments.

### Training Data

2,981 paired SVG/STEP files from the Autodesk Fusion 360 Gallery:
- `training_data/drawings_svg/*.svg` — FreeCAD TechDraw 2D orthographic drawings (pure geometry, no dimension text)
- `training_data/shapes_step/*.step` — Corresponding 3D B-Rep CAD models (ISO-10303-21)
- Filename format: `{ProjectID}_{Hash}_{ViewNumber}.{ext}` (each pair is a distinct part)

### Inference Backends

Configured in `config/models.yaml`. Supports Ollama, vLLM, and llama.cpp backends:
- Vision model: `llama-3.2-11b-Vision` (via any backend)
- Reasoning model: `gpt-oss-120b` (via any backend)
- Default backend: Ollama at `http://localhost:11434`

---

## Project Structure

```
/app/
├── src/
│   ├── training/           # Training data infrastructure (NEW)
│   │   ├── ground_truth.py     # StepGroundTruth Pydantic schema
│   │   ├── data_loader.py      # TrainingPair + TrainingDataIndex
│   │   ├── manifest.py         # JSON persistence for index
│   │   ├── svg_renderer.py     # SVG→PNG via cairosvg
│   │   ├── tiering.py          # Difficulty tier classification
│   │   ├── sampler.py          # BenchmarkSampler (subset selection)
│   │   ├── curriculum.py       # CurriculumScheduler (phased difficulty)
│   │   └── fewshot_miner.py    # Mine few-shot examples from good runs
│   ├── pipeline/           # Loop 3 pipeline stages
│   │   ├── runner.py           # PipelineRunner orchestrator
│   │   ├── vision_stage.py     # Vision model + SVG auto-render
│   │   ├── ocr_stage.py        # OCR text extraction
│   │   ├── reconciliation.py   # Merge views into 3D description
│   │   ├── reasoning_stage.py  # Code generation via LLM
│   │   ├── execution_stage.py  # Run build123d code → STEP
│   │   ├── validation_stage.py # Validate output geometry
│   │   └── retry_controller.py # Retry logic
│   ├── evaluation/         # Scoring and benchmarking
│   │   ├── benchmark_runner.py # Run suite or programmatic cases
│   │   ├── comparator.py       # STEP comparison + ground truth
│   │   ├── metrics.py          # Metric computations
│   │   └── report.py           # Report generation
│   ├── optimizer/          # Loop 2 meta-optimizer
│   │   ├── meta_optimizer.py   # Main optimizer loop
│   │   ├── prompt_optimizer.py # Prompt patch generation
│   │   ├── fewshot_selector.py # Few-shot example selection
│   │   ├── parameter_tuner.py  # Parameter search
│   │   ├── convergence.py      # Convergence detection
│   │   ├── experiment_tracker.py # Experiment logging
│   │   └── failure_analyzer.py # Failure pattern analysis
│   ├── inference/          # LLM client backends
│   │   ├── ollama_client.py
│   │   ├── vllm_client.py
│   │   ├── llamacpp_client.py
│   │   └── factory.py
│   ├── schemas/            # Pydantic models
│   │   ├── pipeline_config.py
│   │   ├── evaluation_result.py
│   │   ├── geometry.py
│   │   └── experiment.py
│   ├── tools/              # Utility tools
│   │   ├── step_analyzer.py    # OCP/trimesh STEP analysis
│   │   ├── vision.py
│   │   ├── cad.py
│   │   ├── ocr.py
│   │   └── mesh_validator.py
│   └── utils/              # Shared utilities
│       ├── logging_config.py
│       ├── image_utils.py
│       └── file_utils.py
├── training_data/
│   ├── drawings_svg/       # 2,981 SVG files
│   ├── shapes_step/        # 2,981 STEP files
│   ├── rendered_png/       # 2,981 rasterized PNGs (generated)
│   └── manifest.json       # Precomputed ground truth + tiers (generated)
├── config/
│   ├── default.yaml        # Pipeline + evaluation config
│   ├── models.yaml         # Inference backend config
│   └── optimizer.yaml      # Loop 2 optimizer config
├── prompts/                # LLM prompt templates
├── benchmarks/             # Static benchmark suite (5 cases)
├── scripts/
│   ├── install_environment.sh      # Rootless environment bootstrap
│   ├── validate_install.py         # Dependency validation
│   ├── preprocess_training_data.py # One-time STEP analysis + SVG render
│   ├── run_pipeline.py             # Run Loop 3 on one drawing
│   ├── run_benchmark.py            # Run benchmark suite
│   └── run_optimizer.py            # Run Loop 2 meta-optimizer
├── tests/
│   ├── test_training.py    # 41 tests for training modules
│   ├── test_evaluation.py
│   ├── test_pipeline.py
│   ├── test_optimizer.py
│   └── test_inference.py
└── pyproject.toml
```

---

## What Has Been Done

### Phase A: Training Data Foundation — COMPLETE

All 9 files in `src/training/` have been created and tested:

1. **`ground_truth.py`** — `StepGroundTruth` Pydantic model with volume, surface_area, bounding_box, center_of_mass, face/edge/vertex/solid counts, bbox_extents (sorted), is_valid. Converter from `StepProperties`.

2. **`data_loader.py`** — `TrainingPair` dataclass and `TrainingDataIndex` class. Scans SVG/STEP directories, matches by basename, parses `{ProjectID}_{Hash}_{ViewNumber}` filenames. Lookups by id, tier, tags.

3. **`manifest.py`** — `save_manifest()` / `load_manifest()` for JSON persistence. Stores all pairs with ground truth, tiers, and paths.

4. **`svg_renderer.py`** — `render_svg_to_png()` using cairosvg (lazy import). `render_all_svgs()` with `ProcessPoolExecutor` parallel rendering. Skips already-rendered files.

5. **`tiering.py`** — `DifficultyTier` enum (SIMPLE/MEDIUM/COMPLEX). `classify_tier()` with primary thresholds (face/edge/solid counts) and secondary heuristics (volume-to-bbox fill ratio, edge-to-face ratio). `assign_tiers()` and `compute_tier_distribution()`.

6. **`sampler.py`** — `BenchmarkSampler` with RANDOM, STRATIFIED, CURRICULUM, FAILURE_TARGETED strategies. Auto-picks sentinel pairs per tier for stable convergence tracking.

7. **`curriculum.py`** — `CurriculumScheduler` with 3 default phases: foundation (tier 1, advance at 0.6), intermediate (tiers 1-2, advance at 0.5), advanced (tiers 1-2-3). Phase iteration tracking and advancement logic.

8. **`fewshot_miner.py`** — `FewShotMiner` records successful runs (composite > 0.7, geometry_valid >= 1.0) as markdown few-shot examples. Derives tags from ground truth. Deduplication by pair_id.

### Phase B: Preprocessing — COMPLETE

9. **`scripts/preprocess_training_data.py`** — Created and successfully executed. Processed all 2,981 pairs:
   - STEP analysis: 2,981 analyzed, 0 failures (~30s via OCP)
   - SVG rendering: 2,981 PNGs at 150 DPI (~67s, 4 workers)
   - Tier distribution: 960 Simple (32%), 136 Medium (5%), 1,885 Complex (63%)
   - Output: `training_data/manifest.json` (3.3 MB) + `training_data/rendered_png/` (2,981 files)

### Phase C: Pipeline Integration — COMPLETE

10. **`src/pipeline/vision_stage.py`** — Modified to detect `.svg` inputs, look for pre-rendered PNG in `training_data/rendered_png/`, and fall back to on-the-fly rendering.

11. **`src/evaluation/benchmark_runner.py`** — Added `run_cases()` method for programmatic case lists from `TrainingPair` objects. Added `_run_case_programmatic()` that accepts precomputed ground truth.

12. **`src/evaluation/comparator.py`** — Added `compare_with_ground_truth()` method that analyzes only the generated STEP and compares against precomputed `StepGroundTruth` (avoids loading reference STEP).

13. **`src/evaluation/metrics.py`** — Added `compute_face_count_ratio()` (min/max ratio). Wired into `compute_all_metrics()` with `reference_face_count` parameter.

14. **`src/schemas/evaluation_result.py`** — Added `face_count_ratio: float = 0.0` field to `EvaluationMetrics`. Added to `compute_aggregate` metric fields list.

### Phase D: Optimizer Integration — COMPLETE

15. **`src/optimizer/meta_optimizer.py`** — Added `training_index` and `curriculum_enabled` constructor params. `_run_benchmark_adaptive()` uses curriculum or sampler when training data available, falls back to YAML suite. `_run_benchmark_on_sample()` builds case dicts from TrainingPairs, disables OCR. Few-shot miner records after evaluation. Curriculum advancement with convergence reset.

16. **`src/optimizer/fewshot_selector.py`** — Added `load_mined_examples()` for auto-mined examples from `prompts/fewshot_examples/mined/index.yaml`. Added `"similarity"` selection strategy using tag overlap.

17. **`config/optimizer.yaml`** — Added `training_data` section (root_dir, manifest, sample_size=20, stratified) and `curriculum` section (3 phases with tier progression).

### Phase E: Testing — COMPLETE

18. **`tests/test_training.py`** — 41 tests covering all training modules. All passing.

### Environment Setup — COMPLETE

19. **`scripts/install_environment.sh`** — Rewritten for rootless Debian containers. Uses Node.js curl shim → uv → Python 3.12 → micromamba for system libs (cairo, GL, mesa, X11).

20. **`scripts/validate_install.py`** — Validates all dependency imports including project modules.

21. **`~/activate_cad.sh`** — Environment activation script setting PATH, PYTHONPATH, LD_LIBRARY_PATH.

### Environment State

The container at `/app` has a fully operational environment:
- **Python 3.12.12** at `/home/node/cad-venv/bin/python3`
- **uv 0.10.6** at `/home/node/.local/bin/uv`
- **All pip deps installed**: build123d 0.10.0, cadquery-ocp 7.8.1.1, trimesh 4.11.2, cairosvg 2.8.2, numpy 2.4.2, pydantic 2.12.5, structlog, httpx, optuna, pytest, etc.
- **System libs via micromamba**: cairo 1.18.4, mesalib 25.3.5, libgl 1.7.0, X11 at `/home/node/micromamba/envs/syslibs/lib/`
- **Activation**: `source ~/activate_cad.sh`
- **Preprocessing complete**: manifest.json + 2,981 PNGs generated
- **Tests**: 41/41 passing

---

## What Remains To Do

### 1. Connect to Inference Backends

The pipeline requires running LLM servers. None are currently running in this container. To execute real pipeline runs:

- **Option A (Ollama)**: Install Ollama, pull `llama3.2-vision:11b` and `gpt-oss-120b` models. Start server on port 11434.
- **Option B (vLLM)**: Start vLLM server with the models on port 8000.
- **Option C (llama.cpp)**: Start llama-server with the models on port 8080.

Without inference backends, only `--mock` mode works (returns synthetic scores).

### 2. Run the Optimizer (Loop 2)

Once inference backends are available:

```bash
source ~/activate_cad.sh

# Mock mode (no LLM needed, tests the full optimization loop with synthetic scores):
python3 scripts/run_optimizer.py --mock --max-iterations 3

# Real mode (requires running inference backend):
python3 scripts/run_optimizer.py --max-iterations 20
```

The optimizer with training data will:
1. Load manifest.json (2,981 pairs with precomputed ground truth)
2. Start curriculum phase "foundation" (Tier 1 only, 15 samples)
3. Each iteration: sample pairs → run pipeline → evaluate against ground truth → generate candidates → accept best
4. Advance through curriculum phases as scores improve
5. Mine few-shot examples from successful runs (score > 0.7)
6. Converge when score improvement < 0.01 for 3 consecutive iterations

**To wire training data into the optimizer**, `scripts/run_optimizer.py` needs to be updated to:
- Load the `TrainingDataIndex` from manifest.json
- Pass it as `training_index` to `MetaOptimizer`
- Pass `curriculum_enabled=True`

Current `run_optimizer.py` does not do this yet — it only passes `config`, `llm_client`, `vision_client`, `use_mock`, `experiments_dir`, and `benchmark_suite`.

### 3. Update `run_optimizer.py` to Wire Training Data

Add to `scripts/run_optimizer.py`:
```python
from src.training.manifest import load_manifest

# After config loading:
training_index = None
manifest_path = Path("training_data/manifest.json")
if manifest_path.exists():
    training_index = load_manifest(manifest_path)

optimizer = MetaOptimizer(
    config=config,
    ...,
    training_index=training_index,
    curriculum_enabled=True,
)
```

### 4. Run Single Pipeline on Training Pair

Test the pipeline end-to-end on one training pair:

```bash
source ~/activate_cad.sh

# With mock mode:
python3 scripts/run_pipeline.py training_data/rendered_png/37605_e35cc4df_0007.png --mock

# With real inference:
python3 scripts/run_pipeline.py training_data/rendered_png/37605_e35cc4df_0007.png -o output.step
```

### 5. Create Deployment Artifacts

After optimization converges, the deployable artifacts are:

| Artifact | Description |
|----------|-------------|
| `experiments/best_config_{run_id}.yaml` | Optimized pipeline configuration |
| `prompts/*.md` | Tuned prompt templates |
| `prompts/fewshot_examples/mined/` | Auto-mined few-shot examples |
| `experiments/{run_id}/` | Full experiment history (JSON logs) |
| `training_data/manifest.json` | Ground truth database |

### 6. Convergence Criteria

The optimizer considers itself done when ANY of:
- **Convergence**: Score improvement < 0.01 for 3 consecutive iterations
- **Curriculum complete**: All 3 phases exhausted (5 + 8 + 10 = 23 iterations max)
- **Max iterations**: Hits the `--max-iterations` cap (default 20)

### 7. Known Limitations and Future Work

- **Tier imbalance**: 63% of pairs are Complex (Tier 3), only 5% Medium. The curriculum's intermediate phase may have limited variety. Consider adjusting tier thresholds.
- **No dimension text**: SVGs contain pure geometry with no dimension annotations. OCR is auto-disabled for training data. The vision model must infer all dimensions from shapes alone.
- **Mock mode only**: Without a running LLM backend, the optimizer can only run in mock mode which returns synthetic scores. This tests the optimization loop mechanics but doesn't improve real pipeline performance.
- **Few-shot storage**: Mined few-shot examples are stored at `prompts/fewshot_examples/mined/`. This directory is not created until the first successful mining run.
- **Editable install**: The project is not `pip install -e .` installed due to Windows mount limitations. It relies on `PYTHONPATH=/app`. If deploying to a different environment, do a proper editable install.

---

## Entry Points Summary

| Script | Purpose | Prereqs |
|--------|---------|---------|
| `scripts/install_environment.sh` | Bootstrap environment from scratch | Node.js, write to ~ |
| `scripts/preprocess_training_data.py` | Analyze STEP files, render SVGs, save manifest | Environment installed |
| `scripts/run_pipeline.py IMAGE` | Run Loop 3 on one drawing | Inference backend or `--mock` |
| `scripts/run_benchmark.py` | Run static benchmark suite | Inference backend or `--mock` |
| `scripts/run_optimizer.py` | Run Loop 2 optimization | Inference backend or `--mock` |
| `scripts/validate_install.py` | Verify all dependencies | Environment installed |

## Key Configuration Files

| File | Controls |
|------|----------|
| `config/default.yaml` | Pipeline settings, model selection, evaluation weights, inference URLs |
| `config/models.yaml` | Backend-specific model names and endpoints |
| `config/optimizer.yaml` | Optimizer iterations, parameter search space, training data, curriculum phases |
