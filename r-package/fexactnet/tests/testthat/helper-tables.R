engines <- c(
  "baseline",
  "accept_only",
  "line_batch",
  "line_batch_monge"
)

named_table <- function() {
  matrix(
    c(2, 1, 3, 1, 3, 2, 3, 2, 1),
    nrow = 3,
    byrow = TRUE,
    dimnames = list(c("a", "b", "c"), c("u", "v", "w"))
  )
}

expect_all_engines_equal <- function(x, tolerance = 1e-12) {
  fits <- lapply(
    engines,
    function(engine) fisher_network(x, engine = engine, diagnostic = TRUE)
  )
  values <- vapply(fits, function(fit) fit$p.value, numeric(1))
  expect_equal(values, rep(values[[1]], length(values)), tolerance = tolerance)
  invisible(fits)
}
