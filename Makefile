UV_RUN = uv run --locked
LEGACY_RUN = UV_PROJECT_ENVIRONMENT=.venv-legacy uv run --locked --no-default-groups --group legacy

# -----------------------------
# Core test targets
# -----------------------------

test:
	$(LEGACY_RUN) pytest -q

cpu:
	$(UV_RUN) pytest -c /dev/null -q environments/cage2/test_adapter.py environments/cage2/test_reproduction_support.py environments/cage2/test_uv_management.py

gates:
	$(LEGACY_RUN) pytest -m gates -q

drift:
	$(LEGACY_RUN) pytest -m drift -q

policy:
	$(LEGACY_RUN) pytest -m policy -q

ppo:
	$(LEGACY_RUN) pytest -m ppo -q

infra:
	$(LEGACY_RUN) pytest -m infra -q

fast:
	$(LEGACY_RUN) pytest -m fast -q


# -----------------------------
# Linting & formatting
# -----------------------------

lint:
	$(UV_RUN) ruff check .

lint-fix:
	$(UV_RUN) ruff check --fix .

format:
	$(UV_RUN) ruff format .
	$(UV_RUN) black .

style: lint-fix format


# -----------------------------
# CI target (exactly what CI runs)
# -----------------------------

ci: lint test


# -----------------------------
# Help
# -----------------------------

help:
	@echo "Available targets:"
	@echo "  test        - run all tests"
	@echo "  cpu         - run CAGE2 CPU adapter/reproduction checks"
	@echo "  gates       - run gate-level invariants"
	@echo "  drift       - run drift & stability invariants"
	@echo "  policy      - run policy-level invariants"
	@echo "  ppo         - run PPO invariants"
	@echo "  infra       - run infrastructure invariants"
	@echo "  fast        - run fast smoke tests"
	@echo "  lint        - run Ruff lint checks"
	@echo "  lint-fix    - auto-fix Ruff issues"
	@echo "  format      - run Ruff + Black formatters"
	@echo "  style       - fix lint + format code"
	@echo "  ci          - run CI pipeline locally"