.PHONY: lint #         | Run linter
lint:
	  uvx ruff check

.PHONY: format #       | Check formatting
format:
		uvx ruff format --diff

.PHONY: test #         | Run tests
test:
		uv run pytest -v
