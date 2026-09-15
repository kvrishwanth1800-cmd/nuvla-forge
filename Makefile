.PHONY: install test lint bench train-ab report clean
NGPU ?= 2

install:
	pip install -e ".[dev,data]"

test:
	pytest tests/ -q

lint:
	ruff check nuvla_forge tests

bench:
	python -m nuvla_forge.bench.bench_kernels
	python -m nuvla_forge.bench.bench_dataloader
	torchrun --nproc_per_node=$(NGPU) -m nuvla_forge.bench.bench_comms

train-ab:
	torchrun --nproc_per_node=$(NGPU) -m nuvla_forge.train --preset baseline  --steps 200 --profile
	torchrun --nproc_per_node=$(NGPU) -m nuvla_forge.train --preset optimised --steps 200 --profile

report:
	python -m nuvla_forge.bench.report

clean:
	rm -rf reports RESULTS.md .pytest_cache **/__pycache__
