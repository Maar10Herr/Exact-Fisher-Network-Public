# Exact Fisher Network

[![CI](https://github.com/Maar10Herr/Exact-Fisher-Network-Public/actions/workflows/ci.yml/badge.svg)](https://github.com/Maar10Herr/Exact-Fisher-Network-Public/actions/workflows/ci.yml)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL_v2%2B-blue.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--8721--6588-A6CE39.svg)](https://orcid.org/0009-0005-8721-6588)

Strict cross-slice order for the Mehta-Patel exact Fisher network, with an
auditable C/R implementation, a command-line interface, and an exact-rational
Python reference.

> [!WARNING]
> **Experimental research implementation; not a drop-in replacement for base R
> FEXACT.** Native execution is limited to three-row tables (or three-column
> tables by transposition). Its query bound does not establish a runtime or
> memory advantage over `stats::fisher.test()`.

**[Read the paper](paper/Herrmann_Strict_Cross_Slice_Fisher_Network.pdf)** ·
**[Paper source](paper/strict_cross_slice_fisher_network.tex)** ·
**[Download the latest release](https://github.com/Maar10Herr/Exact-Fisher-Network-Public/releases/latest)**

## Result

For the exact largest branch weight

```math
B(x)=\binom{m}{x_1,\ldots,x_r}L(R-x;C'),
```

for distinct rows \(i,j,k\), when all four daughter allocations are feasible, the elementary rhombus satisfies

```math
B(x+e_i-e_j)B(x+e_k-e_j)
>
B(x)B(x+e_i+e_k-2e_j).
```

Combined with strict within-line log-concavity, this inequality orders the
leftmost daughter modes across adjacent slices. A consecutive sweep therefore
uses amortized `O(m+s)` peak-location work in place of `O(s log m)` independent
searches.

The implementation provides independently searched and monotone-walk peak
location modes, together with line batching, interval aggregation,
outer-interval acceptance, and prefix reuse. The theorem bounds peak queries;
it does not establish an overall runtime or memory improvement.

## Implementations

| Interface | Intended use | Arithmetic and scope |
|---|---|---|
| `fexactnet` R package | Experimental three-row exact testing | Floating log-factorials and the documented FEXACT-compatible threshold convention |
| `fisher-network` CLI | Scriptable access to the shared C core | Same native scope and numerical convention as the R package |
| Python reference | Verification, certificates, and small exact examples | Arbitrary-precision integer weights and exact rational p-values |

The native engines are `baseline`, `accept_only`, `line_batch`,
`line_batch_monge`, and `auto`. The current `auto` rule selects `line_batch`.

## Quick start

### Standalone C program

```sh
make cli
./cli/fisher-network \
  --engine line_batch \
  r-package/fexactnet/inst/extdata/example.csv
```

The program accepts CSV or whitespace-delimited nonnegative integer matrices.
Use `--json` for machine-readable output and `--help` for all options.

### R package

```sh
R CMD INSTALL r-package/fexactnet
Rscript -e 'x <- matrix(c(2,1,3,1,3,2,3,2,1), 3, byrow=TRUE); \
  print(fexactnet::fisher_network(x, engine="line_batch"))'
```

Native execution supports `3 x c` and, by safe transposition, `r x 3` tables.
Other dimensions are delegated explicitly to
`stats::fisher.test(hybrid = FALSE, simulate.p.value = FALSE)`, and the returned
object records that fallback.

### Exact-rational Python reference

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 - <<'PY'
from exact_fisher_network import exact_production_network

table = ((2, 1, 3), (1, 3, 2), (3, 2, 1))
result = exact_production_network(table)
print(result.p_value)
PY
```

## Evidence and performance boundary

The recorded release gate compared the exact-rational oracle, every manual native
mode and base R on 1,198 cases, including exhaustive positive-margin small
tables and 60 seeded random tables; 48 representatives were also compared to
pinned R FEXACT, with absolute and relative tolerance `2e-12`. Separate tests
cover permutations, transposition, workspace and allocation failures, and
interruption. Finite tests do not establish agreement for every input.

The implementation is experimental. The query bound concerns peak location;
performance depends on daughter generation, threshold searches, forwarding,
state reuse, arithmetic and implementation costs. No general speed or memory
advantage over FEXACT follows from the theorem.

## Verification

```sh
make test
make check-r
```

`make test` runs the exact Python suite and native C/CLI tests. `make check-r`
builds and checks the R package when R package tooling is available. The suite
includes independent exact-oracle comparisons, certificate-corruption tests,
permutation and transposition invariance, strict-rhombus checks, and native
interface tests.

## Intellectual lineage

The implementation begins from the network algorithm of Mehta and Patel and
the practical FEXACT refinements of Clarkson, Fan, and Joe. The proof uses
Murota's network-induced discrete-convex framework and relates to subsequent
path-trimming, transportation, and daughter-ordering work:

- Mehta & Patel, *JASA* 78 (1983),
  [doi:10.1080/01621459.1983.10477989](https://doi.org/10.1080/01621459.1983.10477989)
- Mehta & Patel, *ACM TOMS* 12 (1986),
  [doi:10.1145/6497.214326](https://doi.org/10.1145/6497.214326)
- Clarkson, Fan & Joe, *ACM TOMS* 19 (1993),
  [doi:10.1145/168173.168412](https://doi.org/10.1145/168173.168412)
- Murota, *Advances in Mathematics* 124 (1996),
  [doi:10.1006/aima.1996.0084](https://doi.org/10.1006/aima.1996.0084)
- Aoki (2002), Suzuki, Aoki & Murota (2005), and Requena & Martín Ciudad
  (2006), cited in full in the paper.

This project establishes the strict cross-slice rhombus, the resulting
monotone-leftmost-mode and amortized query bounds, and the accompanying
research implementation. The cited works supply the exact-test,
network, FEXACT, discrete-convex, transportation, path-trimming, and
one-dimensional ordering foundations.

## Repository map

```text
paper/                    general manuscript PDF and LaTeX source
r-package/fexactnet/      R interface, shared C core, documentation, tests
cli/                      command-line interface and native core tests
src/exact_fisher_network/ exact-rational reference and certificate tools
tests/                    deterministic Python verification suite
benchmarks/               retained workload definitions
```

## Citation

Release metadata and the preferred paper citation are in
[`CITATION.cff`](CITATION.cff). Author:
[Maarten Linus Herrmann](https://orcid.org/0009-0005-8721-6588), ORCID
[`0009-0005-8721-6588`](https://orcid.org/0009-0005-8721-6588).

## License

The original software is licensed under
[GPL-2.0-or-later](LICENSE). Third-party methods and components retain their
respective terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The
copyleft license keeps distributed modifications to the implementation under
the same source-available terms. The paper is copyright © 2026 Maarten Linus
Herrmann; all rights reserved.
