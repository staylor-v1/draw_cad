# gemma4_agent

`gemma4_agent` is a reusable subproject for a Gemma 4 CAD roundtrip agent that can run
against either local Ollama or an OpenAI-compatible chat endpoint.

See [`PLAN.md`](PLAN.md) for the GD&T agentic harness roadmap, including masking,
annotation extraction, optional YOLO/Florence-2/Donut tools, view reasoning, feature
fitting, and CAD verification goals.

The target loop is:

```text
drawing -> Gemma/tool CAD reconstruction -> STEP part A
STEP part A -> generated orthographic drawing
generated drawing -> Gemma/tool CAD reconstruction -> STEP part B
compare STEP part A and STEP part B
```

The generated drawings can differ from the original drawing. The basic stability
condition is that the two STEP parts are geometrically equivalent.

For raster engineering drawings, roundtrip equivalence is only a stability check. It is
not sufficient evidence that the first STEP understood the source drawing. The stricter
GD&T training loop also requires source-drawing fidelity: the original drawing is judged
against the generated STEP contact sheet, and fallback envelope geometry is recorded as a
non-passing baseline.

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

## Run PI-style Iterative Improvement

This runs a plan/implement/evaluate ("PI") loop over multiple drawings and stops once
the target success-rate is reached or max iterations is hit.

```bash
.venv/bin/python -m gemma4_agent.cli improve \
  training_data/gdt/flange1.png training_data/gdt/simple1.webp \
  --max-iterations 4 \
  --target-success-rate 0.85 \
  --source-fidelity-threshold 0.72 \
  --output-dir experiments/gemma4_agent/pi_loop
```

When `--source-fidelity-threshold` is set, each PI-loop case must also pass the
source-drawing fidelity judge that compares the original drawing with the generated
STEP contact sheet. This prevents a self-consistent but wrong first-pass CAD model from
counting as successful.

## OpenAI-Compatible Endpoint Mode

Set `agent.api_compatibility: "openai"` and point `agent.base_url` to an
OpenAI-compatible host (e.g. vLLM server URL ending before `/chat/completions`).
Provide `OPENAI_API_KEY` or a `.openai_api_key` file in the repo root.

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

## GD&T Prompt Training Loop

Run iterative prompt tuning on the local GD&T raster drawings:

```bash
.venv/bin/python scripts/tune_gemma4_gdt_roundtrip.py \
  --input-dir training_data/gdt \
  --output-dir experiments/gemma4_agent/gdt_training_loop \
  --target-source-fidelity-threshold 0.99 \
  --timeout-hours 8
```

The current curated GD&T cases use readable filenames:
`connecting_rod.webp`, `flange1.png`, `flange2.png`, `flange3.webp`, `hub.webp`,
`l_bracket.png`, `simple1.webp`, and `threaded_cap.jpg`. Use `--include`, for
example `--include 'flange*.png'`, for targeted reruns.

Each case passes only when:

- `pass_1.step` and `pass_2.step` are geometrically equivalent.
- Neither pass used fallback envelope geometry.
- Gemma judges the generated contact sheet as faithful to the original drawing.

The loop writes per-iteration profiles under
`experiments/gemma4_agent/gdt_training_loop/profiles/` and feeds failed cases back into
the next profile revision. Add `--promote-best-profile` to overwrite
`gemma4_agent/prompts/agent.md` with the best profile from the run.

By default, the loop starts with a lower source-fidelity curriculum threshold and
ratchets toward the `0.99` target. Use `--source-fidelity-threshold 0.99` to force the
final threshold from the first iteration.

Optional local drawing-evidence extractors can be compared without changing the CAD
agent:

```bash
.venv/bin/python scripts/run_gemma4_gdt_experiments.py \
  --input-dir training_data/gdt \
  --target-source-fidelity-threshold 0.99 \
  --timeout-hours 8
```

Available extractor names are `heuristic`, `gemma4`, `florence2`, and `yolo_donut`.
The Florence-2 and YOLO/Donut paths are local-only experiment hooks:

```bash
.venv/bin/python scripts/run_gemma4_gdt_experiments.py \
  --florence2-model-path /models/florence2-engineering-drawings \
  --yolo-obb-model-path /models/yolov11-obb-engineering.pt \
  --donut-model-path /models/donut-engineering-parser
```
