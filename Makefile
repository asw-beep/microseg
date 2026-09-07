# Common workflows. On Windows without make, run the commands directly --
# each recipe is a single readable command line.

PYTHON ?= python
export PYTHONPATH := src

.PHONY: help install data test lint smoke train evaluate baseline benchmarks app docker clean

help:
	@echo "install    install dependencies (CPU torch)"
	@echo "data       download BBBC038 and build the sample cache + splits"
	@echo "test       run the test suite (no dataset required)"
	@echo "lint       ruff check"
	@echo "smoke      train + evaluate a tiny model end to end (~2 min, CPU)"
	@echo "baseline   evaluate the classical CV baseline on the test split"
	@echo "train      train the U-Net (configs/unet_cpu.yaml)"
	@echo "evaluate   evaluate the trained U-Net, with robustness sweep"
	@echo "benchmarks compare classical / U-Net / Cellpose / SAM"
	@echo "app        launch the Gradio interface on :7860"

install:
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/prepare_data.py --config configs/unet_bbbc038.yaml

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check src tests scripts app

smoke:
	$(PYTHON) -m microseg.train --config configs/smoke.yaml
	$(PYTHON) -m microseg.evaluate --config configs/smoke.yaml --backend unet --split test

baseline:
	$(PYTHON) -m microseg.evaluate --config configs/classical_baseline.yaml --backend classical --split test --robustness

train:
	$(PYTHON) -m microseg.train --config configs/unet_cpu.yaml

evaluate:
	$(PYTHON) -m microseg.evaluate --config configs/unet_cpu.yaml --backend unet --split test --robustness

benchmarks:
	$(PYTHON) scripts/run_benchmarks.py --config configs/unet_cpu.yaml

app:
	$(PYTHON) app/gradio_app.py --checkpoint outputs/unet_cpu/best.pt

docker:
	docker build -t microseg .

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
