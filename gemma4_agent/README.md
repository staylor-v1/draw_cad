# gemma4_agent

`gemma4_agent` is a reusable subproject for a local Ollama-hosted Gemma 4 CAD roundtrip agent.

The target loop is:

```text
drawing -> Gemma/tool CAD reconstruction -> STEP part A
STEP part A -> generated orthographic drawing
generated drawing -> Gemma/tool CAD reconstruction -> STEP part B
compare STEP part A and STEP part B
```

The generated drawings can differ from the original drawing. The acceptance condition is that the two STEP parts are geometrically equivalent.

## Requirements

- Ollama running locally at `http://localhost:11434`.
- A Gemma 4 model available in Ollama, defaulting to `gemma4:e4b`.
- The project Python environment with the repo dependencies installed.

Check local Ollama models with:

```bash
ollama list
```

Update `gemma4_agent/config.yaml` if your local model tag differs.

## Run A Roundtrip

```bash
.venv/bin/python -m gemma4_agent.cli roundtrip path/to/drawing.png \
  --output-dir experiments/gemma4_agent/example
```

For a parseable orthographic SVG drawing, the agent can use deterministic candidates from the existing reconstruction code:

```bash
.venv/bin/python -m gemma4_agent.cli roundtrip training_data/drawings_svg/56430_4f35ba2f_0002.svg
```

## Tool Instructions For Gemma

Print the prompt-facing tool instructions and schemas:

```bash
.venv/bin/python -m gemma4_agent.cli tools
```

Print only the Ollama tool schemas:

```bash
.venv/bin/python -m gemma4_agent.cli schemas
```

## Utility Commands

Render an existing STEP file back into front/right/top drawings:

```bash
.venv/bin/python -m gemma4_agent.cli render-step path/to/part.step \
  --output-dir experiments/gemma4_agent/rendered \
  --stem part
```

Compare two STEP files:

```bash
.venv/bin/python -m gemma4_agent.cli compare part_a.step part_b.step
```
