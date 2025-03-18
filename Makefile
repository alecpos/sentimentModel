# WITHIN ML Prediction System Makefile
# Provides useful commands for development, testing, and documentation

.PHONY: help test docs check-docs clean lint format check-alignment check-coverage lint-docs gen-docstrings validate-docs gen-docstrings-nlp fix-docstrings verify-examples validate-alignment validate-docstrings document-model detect-drift explain-model evaluate-fairness install demo cli data benchmark visualize monitor fairness shap

# General
help:
	@echo "Available targets:"
	@echo "  make help                  Show this help message"
	@echo "  make test                  Run all tests"
	@echo "  make clean                 Clean build artifacts"
	@echo "  make lint                  Run code linting (flake8)"
	@echo "  make format                Format code with black"
	@echo "  make docs                  Generate documentation"
	@echo "  make check-docs            Run all documentation checks"
	@echo "  make check-alignment       Check alignment between exports and documentation"
	@echo "  make check-coverage        Check documentation coverage"
	@echo "  make lint-docs             Run documentation linting tool"
	@echo "  make gen-docstrings        Generate docstring templates for a file or directory"
	@echo "  make validate-docs         Validate docstrings in a file or directory"
	@echo "  make gen-docstrings-nlp    Generate NLP-enhanced docstring templates"
	@echo "  make fix-docstrings        Validate and fix docstrings in a file or directory"
	@echo "  make verify-examples       Verify docstring examples in a file or directory"
	@echo "  make validate-alignment    Validate semantic alignment between code and docstrings"
	@echo "  make validate-docstrings  Validate alignment between code and docstrings"
	@echo "  make document-model        Generate comprehensive ML model documentation"
	@echo "  make detect-drift          Detect documentation drift in recent code changes"
	@echo "  make explain-model         Generate SHAP-based model explanations"
	@echo "  make evaluate-fairness     Evaluate model fairness"
	@echo "  make install               Install the package in development mode"
	@echo "  make demo                  Run the demonstration script"
	@echo "  make cli                   Run the CLI tool"
	@echo "  make data                  Generate sample data"
	@echo "  make benchmark             Run performance benchmarks"
	@echo "  make visualize             Visualize benchmark results"
	@echo "  make monitor               Monitor model performance"
	@echo "  make fairness             Analyze model fairness"
	@echo "  make shap                  Analyze model explainability with SHAP"

# Testing
test:
	@echo "Running tests..."
	pytest

# Documentation
docs:
	@echo "Generating documentation..."
	cd docs && make html

# Remove all old 'check-coverage', 'check-alignment', and 'lint-docs' targets

# Formatting and linting
lint:
	@echo "Running linters..."
	black --check app/ tests/
	flake8 app/ tests/
	pylint app/ tests/
	mypy app/ tests/

format:
	@echo "Formatting code..."
	black app/ tests/

gen-docstrings:
ifdef FILE
	@if [ -n "$(CLASS)" ]; then \
		python scripts/generate_docstring_templates.py $(FILE) --class $(CLASS) $(if $(THRESHOLD),--threshold $(THRESHOLD),); \
	else \
		python scripts/generate_docstring_templates.py $(FILE) $(if $(THRESHOLD),--threshold $(THRESHOLD),); \
	fi
else ifdef DIR
	@python scripts/generate_docstring_templates.py $(DIR) --recursive $(if $(APPLY),--apply,)
else
	@echo "Usage:"
	@echo "  make gen-docstrings FILE=path/to/file.py            # Generate template for a file"
	@echo "  make gen-docstrings DIR=path/to/directory           # Generate templates for a directory"
	@echo "  make gen-docstrings FILE=... CLASS=ClassName        # Generate template for a specific class"
	@echo "  make gen-docstrings FILE=... APPLY=1                # Apply the generated templates"
endif

validate-docs:
ifdef FILE
	@python scripts/generate_docstring_templates.py $(FILE) --validate
else ifdef DIR
	@python scripts/generate_docstring_templates.py $(DIR) --recursive --validate
else
	@echo "Please specify a file (FILE=path/to/file.py) or directory (DIR=path/to/directory)"
endif

gen-docstrings-nlp:
ifdef FILE
	@if [ -n "$(APPLY)" ]; then \
		python scripts/generate_docstring_templates.py $(FILE) --apply; \
	elif [ -n "$(CLASS)" ]; then \
		python scripts/generate_docstring_templates.py $(FILE) --class $(CLASS); \
	else \
		python scripts/generate_docstring_templates.py $(FILE); \
	fi
	@echo "\nNLP-driven docstring quality enhancements activated"
else ifdef DIR
	@if [ -n "$(APPLY)" ]; then \
		python scripts/generate_docstring_templates.py $(DIR) --recursive --apply; \
	else \
		python scripts/generate_docstring_templates.py $(DIR) --recursive; \
	fi
	@echo "\nNLP-driven docstring quality enhancements activated"
else
	@echo "Please specify a file (FILE=path/to/file.py) or directory (DIR=path/to/directory)"
endif

fix-docstrings:
ifdef FILE
	@python scripts/generate_docstring_templates.py $(FILE) --validate
	@read -p "Do you want to generate and apply templates for missing docstrings? (y/n) " answer; \
	if [ "$$answer" = "y" ]; then \
		python scripts/generate_docstring_templates.py $(FILE) --apply; \
	fi
else ifdef DIR
	@python scripts/generate_docstring_templates.py $(DIR) --recursive --validate
	@read -p "Do you want to generate and apply templates for missing docstrings? (y/n) " answer; \
	if [ "$$answer" = "y" ]; then \
		python scripts/generate_docstring_templates.py $(DIR) --recursive --apply; \
	fi
else
	@echo "Please specify a file (FILE=path/to/file.py) or directory (DIR=path/to/directory)"
endif

check-coverage:
	@python scripts/verify_documentation_coverage.py

check-alignment:
	@python scripts/verify_documentation_alignment.py

lint-docs:
ifdef FILE
	@python scripts/lint_documentation.py $(FILE)
else ifdef DIR
	@python scripts/lint_documentation.py $(DIR) --recursive
else
	@python scripts/lint_documentation.py app/
endif

check-docs: check-coverage check-alignment lint-docs

verify-examples:
ifdef FILE
	python scripts/verify_docstring_examples.py $(FILE) $(if $(FIX),--fix,) $(if $(VERBOSE),--verbose,)
else
ifdef DIR
	python scripts/verify_docstring_examples.py $(DIR) --recursive $(if $(FIX),--fix,) $(if $(VERBOSE),--verbose,)
else
	@echo "Error: You must specify FILE or DIR."
	@echo "Usage:"
	@echo "  make verify-examples FILE=path/to/file.py         # Verify examples in a file"
	@echo "  make verify-examples DIR=path/to/directory        # Verify examples in a directory"
	@echo "  make verify-examples FILE=path/to/file.py FIX=1   # Fix broken examples automatically"
	@echo "  make verify-examples DIR=path/to/directory FIX=1 VERBOSE=1  # Fix with verbose output"
endif
endif

validate-alignment:
ifdef FILE
	python scripts/validate_semantic_alignment.py $(FILE) --threshold $(or $(THRESHOLD),0.5) $(if $(REPORT),--report $(REPORT),)
else
ifdef DIR
	python scripts/validate_semantic_alignment.py $(DIR) $(if $(RECURSIVE),--recursive,) --threshold $(or $(THRESHOLD),0.5) $(if $(REPORT),--report $(REPORT),)
else
	@echo "Error: You must specify FILE or DIR."
	@echo "Usage:"
	@echo "  make validate-alignment FILE=path/to/file.py              # Validate a file"
	@echo "  make validate-alignment DIR=path/to/directory RECURSIVE=1 # Validate recursively"
	@echo "  make validate-alignment FILE=path/to/file.py THRESHOLD=0.7 # Custom threshold"
	@echo "  make validate-alignment DIR=path/to/directory REPORT=report.json # Generate report"
endif
endif

validate-docstrings:
ifdef FILE
	@python scripts/bidirectional_validate.py $(FILE) $(if $(VERBOSE),--verbose,)
else ifdef DIR
	@python scripts/bidirectional_validate.py $(DIR) $(if $(VERBOSE),--verbose,)
else
	@echo "Usage:"
	@echo "  make validate-docstrings FILE=path/to/file.py       # Validate docstrings for a file"
	@echo "  make validate-docstrings DIR=path/to/directory      # Validate docstrings for a directory"
	@echo "  make validate-docstrings FILE=... VERBOSE=1         # Show detailed validation results"
	@echo "  make validate-docstrings THRESHOLD=0.7 FILE=...     # Set custom similarity threshold"
endif

document-model:
ifdef MODULE
ifdef CLASS
	@python scripts/document_ml_model.py --module $(MODULE) --class $(CLASS) $(if $(OUTPUT),--output $(OUTPUT),--output docs/models)
else
	@echo "Error: CLASS parameter is required."
	@echo "Usage: make document-model MODULE=module.path CLASS=ClassName [OUTPUT=docs/models]"
endif
else
	@echo "Error: MODULE parameter is required."
	@echo "Usage: make document-model MODULE=module.path CLASS=ClassName [OUTPUT=docs/models]"
endif

detect-drift:
ifdef SINCE
	@python scripts/detect_documentation_drift.py --since="$(SINCE)" --path=$(or $(PATH),app) $(if $(REPORT),--report=$(REPORT),)
else ifdef COMMITS
	@python scripts/detect_documentation_drift.py --commits=$(COMMITS) --path=$(or $(PATH),app) $(if $(REPORT),--report=$(REPORT),)
else
	@echo "Usage:"
	@echo "  make detect-drift SINCE='2 weeks ago'               # Detect drift in commits from last 2 weeks"
	@echo "  make detect-drift COMMITS=10                        # Detect drift in last 10 commits"
	@echo "  make detect-drift SINCE='1 month ago' PATH=app/models  # Limit to specific path"
	@echo "  make detect-drift COMMITS=20 REPORT=drift_report.json  # Save report to file"
endif

explain-model:
ifdef MODULE
ifdef CLASS
ifdef DATA
	@echo "Generating SHAP-based model explanations..."
	python scripts/generate_model_explanations.py --module $(MODULE) --class $(CLASS) --data $(DATA) $(if $(OUTPUT),--output $(OUTPUT),--output docs/model_explanations/$(shell echo $(CLASS) | tr A-Z a-z)) $(if $(SAMPLE),--sample $(SAMPLE),) $(if $(TARGET),--target $(TARGET),)
else
	@echo "Error: DATA parameter is required."
	@echo "Usage: make explain-model MODULE=module.path CLASS=ClassName DATA=path/to/data.csv [OUTPUT=docs/model_explanations] [SAMPLE=1000] [TARGET=target_column]"
endif
else
	@echo "Error: CLASS parameter is required."
	@echo "Usage: make explain-model MODULE=module.path CLASS=ClassName DATA=path/to/data.csv"
endif
else
ifdef MODEL
ifdef DATA
	@echo "Generating SHAP-based model explanations..."
	python scripts/generate_model_explanations.py --model $(MODEL) --data $(DATA) $(if $(OUTPUT),--output $(OUTPUT),--output docs/model_explanations/model_explanation) $(if $(SAMPLE),--sample $(SAMPLE),) $(if $(TARGET),--target $(TARGET),)
else
	@echo "Error: DATA parameter is required."
	@echo "Usage: make explain-model MODEL=path/to/model.pkl DATA=path/to/data.csv"
endif
else
	@echo "Error: Either MODULE/CLASS or MODEL parameter is required."
	@echo "Usage: make explain-model MODULE=module.path CLASS=ClassName DATA=path/to/data.csv"
	@echo "   or: make explain-model MODEL=path/to/model.pkl DATA=path/to/data.csv"
endif
endif

# Cleaning
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf __pycache__/ .pytest_cache/ .coverage htmlcov/
	rm -rf results/*.csv results/*.png
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Target to evaluate model fairness
evaluate-fairness:
ifdef MODULE
ifdef CLASS
ifdef DATA
ifdef PROTECTED
	@echo "Evaluating model fairness across demographic groups..."
	python scripts/evaluate_model_fairness.py --module $(MODULE) --class $(CLASS) --data $(DATA) --protected $(PROTECTED) $(if $(TARGET),--target $(TARGET),) $(if $(OUTPUT),--output $(OUTPUT),--output fairness_evaluation/$(shell echo $(CLASS) | tr A-Z a-z)) $(if $(THRESHOLD),--threshold $(THRESHOLD),) $(if $(FAIRNESS_THRESHOLD),--fairness-threshold $(FAIRNESS_THRESHOLD),) $(if $(REPORT),--report $(REPORT),) $(if $(NO_PLOTS),--no-plots,)
else
	@echo "Error: PROTECTED parameter is required."
	@echo "Usage: make evaluate-fairness MODULE=module.path CLASS=ClassName DATA=path/to/data.csv PROTECTED=column_name"
endif
else
	@echo "Error: DATA parameter is required."
	@echo "Usage: make evaluate-fairness MODULE=module.path CLASS=ClassName DATA=path/to/data.csv PROTECTED=column_name"
endif
else
	@echo "Error: CLASS parameter is required."
	@echo "Usage: make evaluate-fairness MODULE=module.path CLASS=ClassName DATA=path/to/data.csv PROTECTED=column_name"
endif
else
ifdef MODEL
ifdef DATA
ifdef PROTECTED
	@echo "Evaluating model fairness across demographic groups..."
	python scripts/evaluate_model_fairness.py --model $(MODEL) --data $(DATA) --protected $(PROTECTED) $(if $(TARGET),--target $(TARGET),) $(if $(OUTPUT),--output $(OUTPUT),--output fairness_evaluation/model_fairness) $(if $(THRESHOLD),--threshold $(THRESHOLD),) $(if $(FAIRNESS_THRESHOLD),--fairness-threshold $(FAIRNESS_THRESHOLD),) $(if $(REPORT),--report $(REPORT),) $(if $(NO_PLOTS),--no-plots,)
else
	@echo "Error: PROTECTED parameter is required."
	@echo "Usage: make evaluate-fairness MODEL=path/to/model.pkl DATA=path/to/data.csv PROTECTED=column_name"
endif
else
	@echo "Error: DATA parameter is required."
	@echo "Usage: make evaluate-fairness MODEL=path/to/model.pkl DATA=path/to/data.csv PROTECTED=column_name"
endif
else
	@echo "Error: Either MODULE/CLASS or MODEL parameter is required."
	@echo "Usage: make evaluate-fairness MODULE=module.path CLASS=ClassName DATA=path/to/data.csv PROTECTED=column_name"
	@echo "   or: make evaluate-fairness MODEL=path/to/model.pkl DATA=path/to/data.csv PROTECTED=column_name"
endif
endif

# Run the demonstration script
demo:
	./scripts/run_demo.sh

# Run the CLI tool
cli:
	./scripts/run_cli.sh

# Create a new virtual environment
venv:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip

# Install all dependencies
deps:
	. venv/bin/activate && pip install -r requirements.txt

# Generate sample data
data:
	./scripts/generate_data.sh

# Run performance benchmarks
benchmark:
	./scripts/run_benchmarks.sh

# Visualize benchmark results
visualize:
	./scripts/visualize_benchmarks.sh

# Monitor model performance
monitor:
	./scripts/monitor_performance.sh

# Analyze model fairness
fairness:
	./scripts/analyze_fairness.sh

# Analyze model explainability with SHAP
shap:
	./scripts/analyze_shap.sh

# Run all checks and tests
all: data lint test demo cli benchmark visualize monitor fairness shap 