test_that("the result is htest-compatible and preserves the table", {
  x <- named_table()
  fit <- fisher_network(x, engine = "line_batch", diagnostic = TRUE)
  expect_s3_class(fit, "htest")
  expect_s3_class(fit, "fisher_network_htest")
  expect_equal(fit$observed, x)
  expect_identical(dimnames(fit$observed), dimnames(x))
  expect_identical(fit$engine, "line_batch")
  expect_false(fit$fallback)
  expect_match(fit$numerical.scope, "floating")
  expect_true(is.list(fit$diagnostics))
  expect_output(print(fit), "Engine: line_batch")
})

test_that("auto reports the actual engine", {
  fit <- fisher_network(named_table(), engine = "auto")
  expect_identical(fit$requested.engine, "auto")
  expect_false(identical(fit$engine, "auto"))
})

test_that("r x 3 input is transposed without changing the tail", {
  x <- matrix(
    c(2, 1, 3, 1, 3, 2, 3, 2, 1, 2, 1, 1),
    nrow = 4,
    byrow = TRUE,
    dimnames = list(letters[1:4], c("u", "v", "w"))
  )
  original <- fisher_network(t(x), engine = "line_batch")
  transposed <- fisher_network(x, engine = "line_batch")
  expect_true(transposed$transposed)
  expect_equal(transposed$p.value, original$p.value, tolerance = 1e-12)
  expect_equal(transposed$observed, x)
  expect_identical(dimnames(transposed$observed), dimnames(x))
})

test_that("unsupported dimensions are an explicit exact fallback", {
  x <- matrix(c(1, 2, 3, 4, 4, 3, 2, 1), nrow = 2)
  expect_warning(
    fit <- fisher_network(x, engine = "line_batch", diagnostic = TRUE),
    "delegating explicitly"
  )
  expect_true(fit$fallback)
  expect_identical(fit$engine, "stats::fisher.test")
  expect_match(fit$method, "explicit fexactnet fallback")
  expect_equal(
    fit$p.value,
    stats::fisher.test(x, hybrid = FALSE, simulate.p.value = FALSE)$p.value
  )
})

test_that("zero margins delegate explicitly", {
  x <- rbind(c(0, 0, 0), c(1, 2, 3), c(2, 1, 3))
  expect_warning(fit <- fisher_network(x), "delegating explicitly")
  expect_true(fit$fallback)
})

test_that("invalid input is rejected", {
  expect_error(fisher_network(data.frame(a = 1:3)), "matrix")
  expect_error(fisher_network(matrix(c(1, -1, 2, 3), 2)), "nonnegative")
  expect_error(fisher_network(matrix(c(1, 1.5, 2, 3), 2)), "integer")
  expect_error(fisher_network(matrix(c(1, NA, 2, 3), 2)), "finite")
  expect_error(fisher_network(matrix(1, 1, 3)), "at least two")
  expect_error(fisher_network(named_table(), exact_rational = TRUE), "not bundled")
  expect_error(fisher_network(named_table(), workspace = 0), "positive")
})
