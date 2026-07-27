UV ?= uv
PROJECT := option-portfolio-greek-risk-premia
UV_RUN := $(UV) run --locked --no-editable --no-build-isolation --reinstall-package $(PROJECT)
UV_RUN_DATA := $(UV) run --locked --no-editable --extra data --no-build-isolation --reinstall-package $(PROJECT)
RELEASE_DIR := build/release
UV_CACHE_DIR ?= $(abspath build/uv-cache)
TEX_CACHE := $(abspath build/tex-cache)
RELEASE_EPOCH := 1785110400
TEX_ENV := SOURCE_DATE_EPOCH=$(RELEASE_EPOCH) FORCE_SOURCE_DATE=1 TZ=UTC
TEX_ENV += TEXMFVAR="$(TEX_CACHE)" TEXMFCACHE="$(TEX_CACHE)"
export UV_CACHE_DIR

.PHONY: test lint build paper verify-artifacts verify-full release

test:
	$(UV_RUN) python -m pytest

lint:
	$(UV_RUN) ruff check .

build:
	$(UV_RUN) python -m analysis.release build

paper: build
	cd $(RELEASE_DIR)/paper && $(TEX_ENV) lualatex -halt-on-error -interaction=nonstopmode paper.tex
	cd $(RELEASE_DIR)/paper && $(TEX_ENV) bibtex paper
	cd $(RELEASE_DIR)/paper && $(TEX_ENV) lualatex -halt-on-error -interaction=nonstopmode paper.tex
	cd $(RELEASE_DIR)/paper && $(TEX_ENV) lualatex -halt-on-error -interaction=nonstopmode paper.tex
	cd $(RELEASE_DIR)/paper && $(TEX_ENV) lualatex -halt-on-error -interaction=nonstopmode paper.tex

verify-artifacts:
	$(UV_RUN) python -m analysis.release verify-artifacts

verify-full:
	$(UV_RUN) python -m analysis.release verify-full --inputs-only
	$(UV_RUN_DATA) python -m analysis.release verify-full

release: paper
	$(UV_RUN) python -m analysis.release release
