test_that("all modes agree with stats on representative families", {
  cases <- list(
    sparse = matrix(c(0, 2, 1, 4, 0, 1, 1, 0, 3, 0, 2, 2), 3, byrow = TRUE),
    dense = matrix(c(3, 4, 5, 4, 5, 3, 5, 3, 4), 3, byrow = TRUE),
    unbalanced = matrix(c(15, 2, 1, 5, 8, 2, 4, 2, 6), 3, byrow = TRUE),
    zeros = matrix(c(0, 1, 4, 2, 0, 3, 3, 4, 0), 3, byrow = TRUE),
    repeated_prefixes = matrix(
      c(1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1),
      3,
      byrow = TRUE
    )
  )
  for (x in cases) {
    fits <- expect_all_engines_equal(x, tolerance = 2e-12)
    reference <- stats::fisher.test(
      x,
      hybrid = FALSE,
      simulate.p.value = FALSE,
      workspace = 5e7
    )$p.value
    expect_equal(fits[[1]]$p.value, reference, tolerance = 2e-12)
  }
})

test_that("row, column, and transpose permutations preserve the result", {
  x <- named_table()
  reference <- fisher_network(x, engine = "line_batch_monge")$p.value
  variants <- list(
    x[c(3, 1, 2), ],
    x[, c(2, 3, 1)],
    x[c(2, 3, 1), c(3, 1, 2)],
    t(x)
  )
  for (variant in variants) {
    expect_equal(
      fisher_network(variant, engine = "line_batch_monge")$p.value,
      reference,
      tolerance = 1e-12
    )
  }
})

test_that("the theorem-only ablation changes only peak-location counters", {
  x <- matrix(c(15, 2, 1, 5, 8, 2, 4, 2, 6), 3, byrow = TRUE)
  independent <- fisher_network(x, "line_batch", diagnostic = TRUE)
  monge <- fisher_network(x, "line_batch_monge", diagnostic = TRUE)
  expect_equal(monge$p.value, independent$p.value, tolerance = 1e-12)
  controlled <- c(
    "residual_states",
    "arcs_considered",
    "arcs_forwarded",
    "interval_accepted_arcs",
    "accepted_intervals",
    "line_searches",
    "total_peak_movement"
  )
  expect_identical(
    monge$diagnostics[controlled],
    independent$diagnostics[controlled]
  )
  expect_gt(monge$diagnostics$monotone_peak_reuses, 0)
  expect_lte(
    monge$diagnostics$branch_bound_calls,
    independent$diagnostics$branch_bound_calls
  )
})

test_that("workspace and injected allocation failures are explicit", {
  x <- named_table()
  storage.mode(x) <- "integer"
  expect_error(
    fexactnet:::fisher_network_native(x, "line_batch", 64, FALSE),
    "workspace limit"
  )
  expect_error(
    fexactnet:::fisher_network_native(
      x,
      "line_batch",
      512 * 1024^2,
      FALSE,
      .fail_after = 1
    ),
    "allocation failure"
  )
})
