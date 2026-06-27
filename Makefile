PYTHON ?= .venv/bin/python
PAPER_DIR := research/papers/option_only_markowitz
PAPER_STEM := option_only_markowitz_cashflow_engineering_dhruv_kohli

.PHONY: help install data-plan data-validate data-public data-paid option-paper paper verify test clean

help:
	@echo "Targets:"
	@echo "  install        Install Python requirements into the active environment"
	@echo "  data-plan      Dry-run the full option-paper data pull plan"
	@echo "  data-validate  Validate expected local input files without network calls"
	@echo "  data-public    Execute credential-free/public data pulls"
	@echo "  data-paid      Execute the full option-paper pull plan, including paid Databento jobs"
	@echo "  option-paper   Regenerate option-only paper artifacts"
	@echo "  paper          Regenerate artifacts and compile the option-only paper PDF"
	@echo "  verify         Run the independent option-only paper verifier"
	@echo "  test           Run focused publication/reproducibility tests"
	@echo "  clean          Remove Python and LaTeX local intermediates"

install:
	$(PYTHON) -m pip install -r requirements.txt

data-plan:
	$(PYTHON) -m data_pull.pull --preset option-paper

data-validate:
	$(PYTHON) -m data_pull.pull --preset validate --execute

data-public:
	$(PYTHON) -m data_pull.pull --preset public --execute

data-paid:
	$(PYTHON) -m data_pull.pull --preset option-paper --execute --allow-paid

option-paper:
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.run_empirics --stage all

paper: option-paper
	cd $(PAPER_DIR) && lualatex -interaction=nonstopmode $(PAPER_STEM).tex && bibtex $(PAPER_STEM) && lualatex -interaction=nonstopmode $(PAPER_STEM).tex && lualatex -interaction=nonstopmode $(PAPER_STEM).tex && lualatex -interaction=nonstopmode $(PAPER_STEM).tex

verify:
	$(PYTHON) -m research.papers.option_only_markowitz.verification.verify

test:
	$(PYTHON) -m pytest tests/test_data_pull_cli.py tests/test_option_only_markowitz_model.py tests/test_option_only_markowitz_verification.py tests/test_option_only_publication_upgrade.py tests/test_option_portfolio_production.py -q -p no:cacheprovider

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) -delete
	find research/papers -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.synctex.gz' -o -name '*.bbl' -o -name '*.blg' \) -delete
