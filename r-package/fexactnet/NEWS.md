# fexactnet 0.2.0

- Published the paper, release metadata, and reproducibility-oriented README.
- Documented the current `auto` selection rule and benchmark boundary.
- Replaced the institutional contact address with a GitHub no-reply address
  while retaining the linked ORCID.

# fexactnet 0.1.0

- Added one reentrant C core shared by the R package and standalone CLI.
- Added native `baseline`, `accept_only`, `line_batch`,
  `line_batch_monge`, and recorded `auto` engine modes.
- Added safe `r x 3` transposition and explicit delegation of other
  dimensions to `stats::fisher.test()`.
- Added exact line-interval acceptance and mass aggregation.
- Added a theorem-only ablation: independent slice peak searches versus
  the strict cross-slice monotone walk.
- Kept floating FEXACT-compatible output separate from the independent
  exact-rational verification oracle.
