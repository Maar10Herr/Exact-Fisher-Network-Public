# fexactnet

Author ORCID: https://orcid.org/0009-0005-8721-6588

`fexactnet` computes the Fisher--Freeman--Halton exact test through a
native network engine for `3 x c` tables. An `r x 3` table is transposed
safely. Other dimensions are delegated explicitly to
`stats::fisher.test()`; the returned object records that fallback.

```r
x <- matrix(
  c(2, 1, 3, 1, 3, 2, 3, 2, 1),
  nrow = 3,
  byrow = TRUE
)
fit <- fexactnet::fisher_network(x, engine = "line_batch")
fit$p.value
fit$engine
```

The native engines are:

- `baseline`: longest- and shortest-path state pruning;
- `accept_only`: exact longest-bound acceptance without shortest pruning;
- `line_batch`: daughter-line batching with independent peak searches;
- `line_batch_monge`: the same batching code with cross-slice monotone
  peak walking;
- `auto`: currently selects `line_batch`. No stable adaptive selection rule
  was found, and the result always reports the engine actually used.

The batching optimization includes exact outer-interval acceptance,
interval-mass aggregation, reduced daughter forwarding, and prefix-state
reuse. The strict cross-slice theorem changes only peak location. Larger
batching effects are not attributed to the theorem.

## Numerical scope

The normal path uses floating log-factorials and the documented FEXACT
threshold convention. It is intended for speed and compatibility with
the probability ordering used by `stats::fisher.test()`. It is not the
same numerical guarantee as the independent exact-rational Python reference
implementation in this repository. No Monte Carlo or hybrid approximation is
used.

## Install from a source tarball

```sh
R CMD INSTALL r-package/fexactnet
```

The standalone `fisher-network` executable is built from the same C core
and has no R runtime dependency.

## Scope

Version 0.2.1 remains an experimental three-row implementation. General
`r x c` FEXACT is outside its native scope; other dimensions delegate to
`stats::fisher.test()`. See the repository paper and `?fisher_network` for
supported families, controlled ablations, and limitations.

Licence: GPL-2-or-later.
