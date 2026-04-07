# Gemma 4 Total_view_data Experiment

- Dataset: `ABC`
- Model: `gemma4:e4b`
- Selected cases: `00000053, 00000049, 00000068, 00000008, 00000031, 00000024, 00000000, 00000061, 00000006`

## Case Selection

| Case | Bucket | Split | Tools-only score | Candidate | Hidden cuts |
| --- | --- | --- | ---: | --- | ---: |
| `00000053` | `axisymmetric` | `calibration` | 0.318 | `visual_hull_hidden` | `0` |
| `00000049` | `prismatic_hidden` | `calibration` | 0.305 | `visual_hull_hidden` | `3` |
| `00000068` | `prismatic` | `calibration` | 0.443 | `visual_hull_hidden` | `0` |
| `00000008` | `axisymmetric` | `evaluation` | 0.428 | `axisymmetric_hidden` | `4` |
| `00000031` | `axisymmetric` | `evaluation` | 0.281 | `visual_hull_hidden` | `3` |
| `00000024` | `prismatic_hidden` | `evaluation` | 0.408 | `visual_hull_hidden` | `1` |
| `00000000` | `prismatic_hidden` | `evaluation` | 0.252 | `visual_hull_base` | `0` |
| `00000061` | `prismatic` | `evaluation` | 0.510 | `visual_hull_hidden` | `0` |
| `00000006` | `prismatic` | `evaluation` | 0.398 | `visual_hull_hidden` | `0` |

## Prompt Search

| Mode | Best prompt | Mean score | Success rate |
| --- | --- | ---: | ---: |
| Raw | `raw_v1_direct` | 0.000 | 0.00 |
| With tools | `tool_v1_candidates_first` | 0.355 | 1.00 |

## Held-Out Evaluation

| Mode | Mean score | Mean visible F1 | Mean hidden F1 | Success rate | Mean tool calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gemma4_raw` | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| `gemma4_with_tools` | 0.378 | 0.654 | 0.103 | 1.00 | 3.50 |
| `tools_only` | 0.379 | 0.658 | 0.101 | 1.00 | 0.00 |

## Per-Case Held-Out Scores

| Case | Bucket | Raw | With tools | Tools only |
| --- | --- | ---: | ---: | ---: |
| `00000008` | `axisymmetric` | 0.000 | 0.428 | 0.428 |
| `00000031` | `axisymmetric` | 0.000 | 0.275 | 0.281 |
| `00000024` | `prismatic_hidden` | 0.000 | 0.408 | 0.408 |
| `00000000` | `prismatic_hidden` | 0.000 | 0.252 | 0.252 |
| `00000061` | `prismatic` | 0.000 | 0.510 | 0.510 |
| `00000006` | `prismatic` | 0.000 | 0.398 | 0.398 |
