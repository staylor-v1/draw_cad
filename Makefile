.PHONY: install test lint run-dashboard run-titleblock-harness run-gdt-harness run-vector-harness run-simple1-callout-agent run-pipeline run-benchmark run-optimizer clean

install:
	pip install -e ".[all]"

install-core:
	pip install -e .

test:
	pytest tests/ -v

run-dashboard:
	.venv/bin/python web_ui.py --port 12080

run-titleblock-harness:
	.venv/bin/python scripts/run_titleblock_harness.py --output-dir experiments/titleblock_harness

run-gdt-harness:
	.venv/bin/python scripts/run_gdt_harness.py --output-dir experiments/gdt_harness

run-vector-harness:
	.venv/bin/python scripts/run_vector_harness.py --output-dir experiments/vector_harness

run-simple1-callout-agent:
	.venv/bin/python scripts/run_gemma4_callout_agent.py training_data/gdt/simple1.webp

run-pipeline:
	python scripts/run_pipeline.py $(ARGS)

run-benchmark:
	python scripts/run_benchmark.py $(ARGS)

run-optimizer:
	python scripts/run_optimizer.py $(ARGS)

clean:
	rm -rf experiments/*.json
	rm -rf __pycache__ src/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
