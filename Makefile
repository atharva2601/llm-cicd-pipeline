# ─────────────────────────────────────────────────────────────────
# Makefile — LLM CI/CD Evaluation Pipeline
#
# Prerequisites: Python 3.10+, pip or Poetry, GOOGLE_API_KEY in .env
#
# Usage:
#   make setup          Install all dependencies
#   make smoke          Run smoke tests (fast, ~3 min)
#   make full           Run full regression suite
#   make safety         Run adversarial safety tests
#   make all-tests      Run smoke + full + safety
#   make synthesize     Generate golden dataset from KB
#   make drift          Show drift analysis
#   make report         Print drift run history
#   make demo           Run the CLI demo
#   make clean          Remove generated files
# ─────────────────────────────────────────────────────────────────

.PHONY: setup smoke full safety all-tests contextual \
        synthesize drift report demo lint clean help

PYTHON  ?= python3
PYTEST  ?= $(PYTHON) -m pytest
PIP     ?= pip

# ── Setup ─────────────────────────────────────────────────────────
setup:
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "🔧 Configuring DeepEval judge (Gemini Flash)..."
	@if [ -f .env ]; then \
		export $$(grep -v '^#' .env | xargs) && \
		deepeval set-gemini \
			--model="gemini-1.5-flash" \
			--google-api-key="$$GOOGLE_API_KEY"; \
	else \
		echo "⚠️  .env not found. Copy .env.example → .env first."; \
	fi
	@echo ""
	@echo "✅ Setup complete. Run: make smoke"

# ── Tests ─────────────────────────────────────────────────────────
smoke:
	@echo "🚀 Running smoke tests..."
	$(PYTHON) scripts/run_eval.py --suite smoke

full:
	@echo "🔬 Running full regression suite..."
	$(PYTHON) scripts/run_eval.py --suite full --log-drift

safety:
	@echo "🛡️  Running adversarial safety tests..."
	$(PYTHON) scripts/run_eval.py --suite safety

contextual:
	@echo "📐 Running contextual precision/recall tests..."
	$(PYTEST) tests/test_contextual.py -m full -v --tb=short

all-tests:
	@echo "🧪 Running all tests..."
	$(PYTHON) scripts/run_eval.py --suite all --html --log-drift

# ── Dataset ───────────────────────────────────────────────────────
synthesize:
	@echo "🧬 Synthesizing golden dataset..."
	$(PYTHON) scripts/synthesize_dataset.py \
		--source data/golden_dataset.json \
		--variations 2 \
		--output data/golden_dataset_synthesized.json
	@echo "✅ Done: data/golden_dataset_synthesized.json"

synthesize-dry:
	@echo "🧬 Dry run — synthesizing golden dataset (not saved)..."
	$(PYTHON) scripts/synthesize_dataset.py --variations 1 --dry-run

# ── Drift Detection ───────────────────────────────────────────────
drift:
	@echo "📊 Drift Analysis..."
	$(PYTHON) scripts/drift_detection.py analyse --window 30

report:
	@echo "📋 Run History..."
	$(PYTHON) scripts/drift_detection.py report

# ── Demo ──────────────────────────────────────────────────────────
demo:
	@echo "🤖 Demo: 'How do I get a refund?'"
	$(PYTHON) -m src.app "How do I get a refund?"

demo-all:
	@echo "🤖 Demo: Multiple queries"
	$(PYTHON) -m src.app "How do I get a refund?"
	$(PYTHON) -m src.app "What are your support hours?"
	$(PYTHON) -m src.app "How do I reset my password?"
	$(PYTHON) -m src.app "How long does shipping take?"

# ── Code quality ──────────────────────────────────────────────────
lint:
	@echo "🔍 Linting..."
	$(PYTHON) -m ruff check src/ tests/ scripts/ || true
	$(PYTHON) -m black --check src/ tests/ scripts/ || true

format:
	@echo "✨ Formatting..."
	$(PYTHON) -m black src/ tests/ scripts/
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/

# ── Clean ─────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf reports/
	rm -rf .deepeval/
	rm -rf __pycache__ **/__pycache__ **/**/__pycache__
	rm -rf .pytest_cache
	rm -rf *.html
	rm -f data/golden_dataset_synthesized.json
	@echo "✅ Clean."

# ── Help ──────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  LLM CI/CD Evaluation Pipeline"
	@echo ""
	@echo "  Quick start:"
	@echo "    1. cp .env.example .env   (add your GOOGLE_API_KEY)"
	@echo "    2. make setup"
	@echo "    3. make smoke"
	@echo ""
	@echo "  Commands:"
	@echo "    make setup        Install deps + configure Gemini judge"
	@echo "    make smoke        Fast smoke tests on every push (~3 min)"
	@echo "    make full         Full regression suite (~15 min)"
	@echo "    make safety       Adversarial red-team tests"
	@echo "    make contextual   Retriever precision/recall tests"
	@echo "    make all-tests    Everything + HTML report"
	@echo "    make synthesize   Generate golden dataset from KB"
	@echo "    make drift        Show metric drift analysis"
	@echo "    make report       Print run history table"
	@echo "    make demo         CLI demo query"
	@echo "    make lint         Check code style"
	@echo "    make clean        Remove generated files"
	@echo ""

# ── Validation & Tuning ───────────────────────────────────────────
check:
	@echo "🔍 Pre-flight environment check..."
	$(PYTHON) scripts/setup_check.py

check-fix:
	@echo "🔧 Pre-flight check with auto-fix..."
	$(PYTHON) scripts/setup_check.py --fix

validate-kb:
	@echo "📊 Validating knowledge base coverage..."
	$(PYTHON) scripts/validate_kb.py

validate-kb-strict:
	@echo "📊 Validating knowledge base coverage (strict mode)..."
	$(PYTHON) scripts/validate_kb.py --strict --report

tune-thresholds:
	@echo "🎛️  Running threshold tuning analysis..."
	$(PYTHON) scripts/tune_thresholds.py --percentile 10 --apply

benchmark:
	@echo "⏱️  Running latency benchmark..."
	$(PYTHON) scripts/benchmark.py --suite smoke --report

benchmark-full:
	@echo "⏱️  Running full latency benchmark..."
	$(PYTHON) scripts/benchmark.py --suite full --report

export:
	@echo "📄 Exporting evaluation reports..."
	$(PYTHON) scripts/export_report.py --format all

export-html:
	@echo "🌐 Exporting HTML report..."
	$(PYTHON) scripts/export_report.py --format html

integration:
	@echo "🔗 Running integration tests..."
	$(PYTEST) tests/test_integration.py -v --tb=short -p no:warnings

custom-metrics:
	@echo "🎯 Running custom domain metric tests..."
	$(PYTEST) tests/test_custom_metrics.py -m full -v --tb=short -p no:warnings

docker-smoke:
	@echo "🐳 Running smoke tests in Docker..."
	docker compose up smoke

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t llm-eval .
