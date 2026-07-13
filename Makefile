PYTHON ?= .venv/bin/python
PAPER_DIR := research/papers/option_only_markowitz
PAPER_STEM := option_only_portfolio_optimization_dhruv_kohli

.PHONY: help install data-plan data-validate data-public data-paid cbbo-surface robustness final-results e1-ablation r1-repaired r11-higher-risk r11-event-cbbo-plan option-paper paper verify test clean

help:
	@echo "Targets:"
	@echo "  install        Install Python requirements into the active environment"
	@echo "  data-plan      Dry-run the full option-paper data pull plan"
	@echo "  data-validate  Validate expected local input files without network calls"
	@echo "  data-public    Execute credential-free/public data pulls"
	@echo "  data-paid      Execute the full option-paper pull plan, including paid Databento jobs"
	@echo "  cbbo-surface   Build the derived OPRA CBBO spread cost surface"
	@echo "  robustness     Run distributional-robustness diagnostics for the option-only paper"
	@echo "  final-results  Build final visual scoreboard from breadth robustness artifacts"
	@echo "  e1-ablation    Run the locked E1 structural-channel ablation"
	@echo "  r1-repaired    Run the repaired monthly R1 development pipeline"
	@echo "  r11-higher-risk Run the R1.1 25%/VIX40/EGARCH development pipeline"
	@echo "  r11-event-cbbo-plan Estimate licensed event-date CBBO cost; download nothing"
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

cbbo-surface:
	$(PYTHON) -m data_ingestion.build_cbbo_cost_surface

robustness:
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.run_empirics --stage robustness

final-results:
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.build_final_results_summary
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.build_inference_panel

e1-ablation:
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.breadth_e1_ablation_experiment
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.build_e1_concentration

r1-repaired:
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.r1_repaired_pipeline --configs all

r11-higher-risk:
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.r11_higher_risk_pipeline --configs all

r11-event-cbbo-plan:
	$(PYTHON) -m data_ingestion.market_data.fetch_r11_event_cbbo

option-paper:
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.run_empirics --stage all
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.run_empirics --stage robustness
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.regenerate_from_artifacts
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.build_final_results_summary
	$(PYTHON) -m research.papers.option_only_markowitz.analysis.build_inference_panel

paper: option-paper
	cd $(PAPER_DIR) && lualatex -interaction=nonstopmode $(PAPER_STEM).tex && bibtex $(PAPER_STEM) && lualatex -interaction=nonstopmode $(PAPER_STEM).tex && lualatex -interaction=nonstopmode $(PAPER_STEM).tex && lualatex -interaction=nonstopmode $(PAPER_STEM).tex

verify:
	$(PYTHON) -m research.papers.option_only_markowitz.verification.verify

test:
	$(PYTHON) -m pytest tests/test_data_pull_cli.py tests/test_option_only_markowitz_model.py tests/test_r1_repair.py tests/test_r11_higher_risk.py tests/test_option_only_markowitz_verification.py tests/test_option_only_publication_upgrade.py tests/test_option_portfolio_production.py tests/test_option_portfolio_shadow.py tests/test_cbbo_cost_surface.py tests/test_vix_chain_features.py tests/test_option_only_cross_validation.py tests/test_option_only_resampled_universes.py tests/test_option_only_mc_repricing.py tests/test_option_only_robustness_wiring.py -q -p no:cacheprovider

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) -delete
	find research/papers -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.synctex.gz' -o -name '*.bbl' -o -name '*.blg' \) -delete

databento-audit-plan:
	$(PYTHON) -m data_ingestion.market_data.fetch_r1_r11_databento_audit plan --max-cost 40

databento-audit-execute:
	$(PYTHON) -m data_ingestion.market_data.fetch_r1_r11_databento_audit execute --max-cost 40

databento-audit-resume:
	$(PYTHON) -m data_ingestion.market_data.fetch_r1_r11_databento_audit resume --max-cost 40

databento-audit-verify:
	$(PYTHON) -m data_ingestion.market_data.fetch_r1_r11_databento_audit verify --max-cost 40

databento-audit-test:
	$(PYTHON) -m pytest tests/test_r1_r11_databento_audit.py -q -p no:cacheprovider
