#' Print an Extended Fisher Network Test
#'
#' @param x An object returned by [fexactnet::fisher_network()].
#' @param ... Further arguments passed to the standard `"htest"` printer.
#'
#' @return `x`, invisibly.
#' @export
print.fisher_network_htest <- function(x, ...) {
    NextMethod("print")
    cat("\nEngine:", x$engine)
    if (!identical(x$requested.engine, x$engine)) {
        cat(" (requested:", x$requested.engine, ")")
    }
    cat("\nNumerical scope:", x$numerical.scope, "\n")
    if (isTRUE(x$fallback)) {
        cat("Execution: explicit stats::fisher.test fallback\n")
    }
    if (!is.null(x$diagnostics) && !isTRUE(x$fallback)) {
        keep <- c(
            "residual_states",
            "arcs_forwarded",
            "interval_accepted_arcs",
            "line_searches",
            "branch_bound_calls",
            "allocation_peak_bytes"
        )
        values <- x$diagnostics[intersect(keep, names(x$diagnostics))]
        if (length(values)) {
            text <- paste(
                paste(names(values), format(unlist(values)), sep = "="),
                collapse = ", "
            )
            cat("Diagnostics:", text, "\n")
        }
    }
    invisible(x)
}
