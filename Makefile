# ─────────────────────────────────────────────────────────────────────────────
# Makefile — quantum-honors
#
# Local (requires Python + pip):
#   make install        install all Python dependencies locally
#   make run            run the full experiment locally
#   make clean          remove generated output files
#
# Docker (requires Docker Desktop):
#   make docker-build   build the container image (installs all deps inside)
#   make docker-run     run the experiment inside the container
#   make docker-clean   remove the container image
#
#   make help           show this message
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_NAME := quantum-honors
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: install run clean docker-build docker-run docker-clean help

# Default target — show help if someone just types `make`
.DEFAULT_GOAL := help

# ── Local ────────────────────────────────────────────────────────────────────

install:
	@echo "Installing dependencies..."
	@$(PYTHON) -c "import sys; major, minor = sys.version_info[:2]; \
	assert major == 3 and 10 <= minor <= 12, (\
	'Unsupported local Python version: ' + sys.version.split()[0] + '. ' \
	'This project\'s local dependency set currently supports Python 3.10-3.12. ' \
	'Use Python 3.11 (recommended), or run make docker-build && make docker-run.'); \
	print(f'Using Python {major}.{minor} for local install.')"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	@echo "Running full experiment..."
	$(PYTHON) -m scripts.main

clean:
	@echo "Cleaning up generated files..."
	rm -f results.png
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	@echo "Building Docker image '$(IMAGE_NAME)'..."
	@echo "(This installs all dependencies inside the container — takes ~2 min first time)"
	docker build -t $(IMAGE_NAME) .
	@echo "Done. Run 'make docker-run' to start the experiment."

docker-run:
	@echo "Running experiment inside Docker container..."
	@echo "(results.png will be copied out to your current directory when done)"
	docker run --rm \
		-v "$(PWD)/results.png:/app/results.png" \
		$(IMAGE_NAME)

docker-rebuild:
	@echo "Rebuilding Docker image from scratch (no cache)..."
	docker build --no-cache -t $(IMAGE_NAME) .
	@echo "Done. Run 'make docker-run' to start the experiment."

docker-clean:
	@echo "Removing Docker image '$(IMAGE_NAME)'..."
	docker rmi $(IMAGE_NAME) 2>/dev/null || echo "Image not found, nothing to remove."

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  quantum-honors — NN vs QNN vs Hybrid Classification"
	@echo ""
	@echo "  Local (requires Python):"
	@echo "    make install        install Python dependencies"
	@echo "    make run            run the full experiment"
	@echo "    make clean          remove results.png and __pycache__"
	@echo ""
	@echo "  Docker (requires Docker Desktop):"
	@echo "    make docker-build   build the container image"
	@echo "    make docker-run     run the experiment in the container"
	@echo "    make docker-rebuild rebuild the image from scratch (no cache)"
	@echo "    make docker-clean   remove the container image"
	@echo ""
