# Agentic CAD Design Loop

An intelligent agent system that converts engineering drawings into dimensionally accurate 3D CAD models (`.step`).

## System Architecture

1.  **Input**: PDF or Image of an engineering drawing.
2.  **Vision Agent (Llama 3.2 Vision)**: Extracts views, dimensions, tolerances, and notes.
3.  **Reasoning Agent (gpt-oss-120b)**: Plans the construction strategy and writes `build123d` Python code.
4.  **Execution Tool**: Runs the code to generate the 3D model.

## Requirements

-   **Python**: 3.9, 3.10, 3.11, or 3.12 (Python 3.13+ is NOT yet supported due to `cadquery-ocp` dependencies).
-   **Libraries**: `build123d`

## Setup

1.  Clone the repository.
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    .venv\Scripts\activate     # Windows
    ```
3.  Install dependencies:
    ```bash
    pip install build123d
    ```

## Usage

```bash
python main.py path/to/drawing.png [output.step]
```

## Total_view_data Dataset

For zipped `Total_view_data` orthographic triplets (`*_f.svg`, `*_r.svg`, `*_t.svg`),
use the deterministic reconstructor:

```bash
.venv/bin/python scripts/reconstruct_total_view_data.py \
  --config config/total_view_data.yaml \
  --dataset ABC \
  --case-id 00008252
```

That path now does a small candidate search per case and keeps the best-scoring
deterministic reconstruction by reprojection against the source views.

To benchmark a larger slice of the dataset:

```bash
.venv/bin/python scripts/benchmark_total_view_data.py \
  --config config/total_view_data.yaml \
  --dataset ABC \
  --limit 100
```

The `total_view_data` config also enables conservative hidden-feature carving from
corroborated red SVG primitives and an axisymmetric revolve candidate for parts whose
views are consistent with lathe-like geometry.

## AI Toolbox

For agent-driven use, the repo now exposes a machine-readable toolbox manifest and a
Python dispatcher in `src.ai_toolbox`.

Export or refresh the manifest with:

```bash
.venv/bin/python scripts/export_ai_toolbox.py \
  --output config/ai_toolbox_manifest.yaml
```

The manifest inventories the reconstruction, reprojection, CAD execution, and dataset
helper functions we added, including internal helpers. Stable entrypoints another AI
agent can call directly are also exposed through `src.ai_toolbox.invoke_tool(...)`.

## Gemma 4 Total_view_data Benchmark

The repo now includes a dedicated Gemma 4 experiment runner that compares three modes
on a selected slice of `Total_view_data`:

- `gemma4_raw`: direct multimodal code generation from PNG orthographic views.
- `gemma4_with_tools`: Gemma 4 with access to deterministic candidate-generation and
  evaluation tools, plus a verified-candidate fallback.
- `tools_only`: the deterministic toolbox baseline without Gemma in the loop.

Run it with:

```bash
.venv/bin/python scripts/run_gemma4_total_view_experiment.py \
  --config config/gemma4_total_view.yaml
```

The checked-in experiment artifacts live under `reports/gemma4_total_view/`, including:

- `report.md`: human-readable summary.
- `results.json`: machine-readable aggregates and per-case records.
- `selected_cases.yaml`: the selected calibration and evaluation cases.
- `selected_cases_contact_sheet.png`: the chosen orthographic triplets.

## Structure

-   `agent_loop.py`: Core logic for the agentic loop.
-   `tools/`: Helper modules for Vision and CAD execution.
-   `prompts/`: System prompts for the AI models.
