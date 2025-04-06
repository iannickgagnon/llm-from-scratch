# Makefile for automation

# Check types using mypy
typecheck:
	mypy .

# Format code with black
format:
	black .

# Lint code with ruff (or flake8)
lint:
	ruff .

# Run all checks
check: format lint typecheck

# Clean up __pycache__, .mypy_cache, etc.
clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf .mypy_cache
