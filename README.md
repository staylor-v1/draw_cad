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

## Structure

-   `agent_loop.py`: Core logic for the agentic loop.
-   `tools/`: Helper modules for Vision and CAD execution.
-   `prompts/`: System prompts for the AI models.