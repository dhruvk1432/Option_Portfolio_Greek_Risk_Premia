# Option Portfolio Greek Risk Premia

This repository accompanies a working paper on funded option-portfolio construction. The
reader-facing output is [paper/paper.pdf](paper/paper.pdf).

The repository is deliberately small. It contains the pricing, risk, optimization, execution,
and inference code needed to state and test the method; compact portfolio-level evidence used
by the paper; and the manuscript source. It does not contain raw options data, security-level
rows, historical quotes, or licensed feature stores.

The empirical results are retrospective development evidence. Touch prices are used only as
sensitivity scenarios, not as realized fills. The paper does not claim live performance,
production tradability, or a raw-to-paper rebuild from public files.

## Layout

```text
analysis/                  inference, evidence, and release functions
paper/
  paper.pdf                canonical paper
  paper.tex                manuscript root
  evidence/                compact derived evidence
  figures/                 referenced figures only
  sections/                manuscript sections
  tables/                  referenced tables only
src/option_portfolio/      pricing, metrics, model, execution, and risk controls
tests/                     public regression and release tests
```

## Setup

Install the exact tested environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --no-editable
```

The public workflow has six commands:

```bash
make lint               # inspect code style without rewriting files
make test               # run every retained pytest test
make build              # create a clean candidate under build/
make paper              # build and compile the candidate with LuaLaTeX and BibTeX
make verify-artifacts   # verify committed evidence and paper without changing them
make verify-full        # run the ignored private rebuild and verify its candidate
```

`make release` is the only command that replaces tracked paper outputs. It first compiles and
verifies a candidate in `build/`, then promotes the PDF, derived evidence, and release manifest.

## Reproducibility boundary

Public verification recomputes corrected portfolio metrics from the committed aggregate
monthly returns and checks evidence hashes, manuscript asset closure, and PDF sanity. The exact
R1 and R1.1 return ledgers are preserved byte for byte from the historical research release.

A historical data rebuild needs the externally prepared licensed inputs and maintainer hook
listed in [data/README.md](data/README.md). `make verify-full` reports each missing path before
running that ignored private implementation. The public package verifies the mathematical
implementation and published artifacts; by itself, it cannot reconstruct the unavailable
licensed feature stores.

The figure PDFs and LaTeX table fragments are frozen historical-derived assets checked against
compact public evidence. They are not presented as outputs regenerated from the unavailable
market inputs.

The release inventory is [release_manifest.csv](release_manifest.csv). Each row records a
relative path, role, producer, byte size, and SHA-256 digest.
