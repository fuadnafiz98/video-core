.PHONY: install sync run test clean help

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies using uv"
	@echo "  make sync     - Sync dependencies"
	@echo "  make run      - Run the CLI tool"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Remove cache and build files"

install:
	uv sync

sync:
	uv sync

run:
	uv run python -m src.cli

test:
	uv run pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .uv
