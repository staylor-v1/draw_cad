.PHONY: install test lint run-pipeline run-benchmark run-optimizer clean

install:
	pip install -e ".[all]"

install-core:
	pip install -e .

test:
	pytest tests/ -v

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
