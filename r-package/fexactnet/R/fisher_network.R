#' Exact Fisher Test Through a Native Three-Row Network
#'
#' Computes the Fisher-Freeman-Halton exact test for a nonnegative integer
#' matrix. Native execution supports \eqn{3 \times c} and \eqn{r \times 3}
#' tables. Other dimensions are delegated explicitly to
#' [stats::fisher.test()] with simulation and the hybrid approximation
#' disabled.
#'
#' @param x A matrix of nonnegative integer counts.
#' @param engine One of `"auto"`, `"line_batch"`, `"line_batch_monge"`,
#'   `"accept_only"`, or `"baseline"`.
#' @param workspace Optional native workspace limit in bytes. The default is
#'   512 MiB. On delegated calls this is converted to four-byte workspace
#'   units for [stats::fisher.test()].
#' @param exact_rational Logical. The public package does not implement a
#'   portable exact-rational hot path; `TRUE` therefore gives an error and
#'   points to the independent Python reference in this repository.
#' @param diagnostic Logical; include extended native counters.
#'
#' @return An object inheriting from `"htest"` and
#'   `"fisher_network_htest"`. It includes the requested engine, engine
#'   actually used, original table with dimnames, numerical scope, source
#'   version and commit, and optional diagnostics.
#'
#' @details
#' The native probability ordering uses floating log-factorials and the
#' FEXACT threshold convention. It is intended for compatibility and speed.
#' It is not the same numerical guarantee as the repository's independent
#' exact-rational Python oracle.
#'
#' `line_batch` and `line_batch_monge` have identical batching,
#' interval-acceptance, and aggregation code. They differ only in peak
#' location: independent search versus the strict cross-slice monotone walk.
#'
#' @examples
#' x <- matrix(
#'   c(2, 1, 3, 1, 3, 2, 3, 2, 1),
#'   nrow = 3,
#'   byrow = TRUE,
#'   dimnames = list(c("a", "b", "c"), c("u", "v", "w"))
#' )
#' fit <- fisher_network(x, engine = "line_batch")
#' fit$p.value
#' fit$engine
#'
#' @export
fisher_network <- function(
    x,
    engine = c(
        "auto",
        "line_batch",
        "line_batch_monge",
        "accept_only",
        "baseline"
    ),
    workspace = NULL,
    exact_rational = FALSE,
    diagnostic = FALSE
) {
    data_name <- paste(deparse(substitute(x)), collapse = "")
    engine <- match.arg(engine)
    x <- validate_fisher_matrix(x)
    if (
        length(exact_rational) != 1L ||
        is.na(exact_rational) ||
        !is.logical(exact_rational)
    ) {
        stop("exact_rational must be one non-missing logical value.")
    }
    if (isTRUE(exact_rational)) {
        stop(
            paste(
                "A portable exact-rational user mode is not bundled.",
                "Use the repository's independent Python reference for verification;",
                "native floating output has a different numerical guarantee."
            )
        )
    }
    if (
        length(diagnostic) != 1L ||
        is.na(diagnostic) ||
        !is.logical(diagnostic)
    ) {
        stop("diagnostic must be one non-missing logical value.")
    }
    workspace_bytes <- validate_workspace(workspace)
    native_supported <- (
        nrow(x) == 3L || ncol(x) == 3L
    ) && all(rowSums(x) > 0) && all(colSums(x) > 0)
    if (!native_supported) {
        return(
            delegated_fisher_test(
                x,
                engine,
                workspace,
                diagnostic,
                data_name
            )
        )
    }
    native <- fisher_network_native(
        x,
        engine,
        workspace_bytes,
        diagnostic
    )
    answer <- list(
        p.value = native$p.value,
        method = paste0(
            "Fisher network exact test (",
            native$engine,
            ")"
        ),
        data.name = data_name,
        engine = native$engine,
        requested.engine = native$requested.engine,
        diagnostics = native$diagnostics,
        observed = x,
        transposed = native$transposed,
        fallback = FALSE,
        numerical.scope = native$numerical.scope,
        table.probability = native$table.probability,
        version = native$version,
        commit = native$commit
    )
    class(answer) <- c("fisher_network_htest", "htest")
    answer
}

validate_fisher_matrix <- function(x) {
    if (!is.matrix(x) || length(dim(x)) != 2L) {
        stop("x must be a matrix.")
    }
    if (nrow(x) < 2L || ncol(x) < 2L) {
        stop("x must have at least two rows and two columns.")
    }
    if (!is.numeric(x) && !is.integer(x)) {
        stop("x must contain numeric integer counts.")
    }
    if (
        anyNA(x) || any(!is.finite(x)) || any(x < 0) ||
        any(x != floor(x))
    ) {
        stop("x must contain finite nonnegative integer counts.")
    }
    if (any(x > .Machine$integer.max)) {
        stop("A cell exceeds the native 32-bit integer range.")
    }
    storage.mode(x) <- "integer"
    x
}

validate_workspace <- function(workspace) {
    if (is.null(workspace)) {
        return(512 * 1024^2)
    }
    if (
        length(workspace) != 1L || is.na(workspace) ||
        !is.numeric(workspace) || !is.finite(workspace) ||
        workspace < 1 || workspace > 2^53
    ) {
        stop("workspace must be NULL or one finite positive byte count.")
    }
    as.double(floor(workspace))
}

fisher_network_native <- function(
    x,
    engine,
    workspace_bytes,
    diagnostic,
    .fail_after = 0
) {
    engine_codes <- c(
        baseline = 0L,
        accept_only = 1L,
        line_batch = 2L,
        line_batch_monge = 3L,
        auto = 4L
    )
    .Call(
        C_fexactnet_compute,
        x,
        unname(engine_codes[[engine]]),
        as.double(workspace_bytes),
        diagnostic,
        as.double(.fail_after)
    )
}

delegated_fisher_test <- function(
    x,
    requested_engine,
    workspace,
    diagnostic,
    data_name
) {
    warning(
        paste(
            "Native fexactnet supports only 3 x c or r x 3 tables with",
            "positive margins; delegating explicitly to stats::fisher.test()."
        ),
        call. = FALSE
    )
    arguments <- list(
        x = x,
        hybrid = FALSE,
        simulate.p.value = FALSE
    )
    if (!is.null(workspace)) {
        arguments$workspace <- max(200000, floor(workspace / 4))
    }
    answer <- do.call(stats::fisher.test, arguments)
    answer$data.name <- data_name
    answer$method <- paste0(
        answer$method,
        " [explicit fexactnet fallback]"
    )
    answer$engine <- "stats::fisher.test"
    answer$requested.engine <- requested_engine
    answer$observed <- x
    answer$transposed <- FALSE
    answer$fallback <- TRUE
    answer$numerical.scope <- "R FEXACT floating log-factorials"
    answer$version <- as.character(utils::packageVersion("fexactnet"))
    answer$commit <- NA_character_
    answer$diagnostics <- if (diagnostic) {
        list(
            fallback = TRUE,
            reason = "unsupported native dimensions or zero margin"
        )
    } else {
        NULL
    }
    class(answer) <- c("fisher_network_htest", "htest")
    answer
}
