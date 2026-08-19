/*
 * R .Call interface for fexactnet
 *
 * Copyright (C) 2026 Maarten Linus Herrmann
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "fisher_network_core.h"

#include <R.h>
#include <R_ext/Utils.h>
#include <Rinternals.h>

#include <limits.h>
#include <stdint.h>
#include <string.h>

static void check_interrupt(void *unused)
{
    (void) unused;
    R_CheckUserInterrupt();
}

static int r_interrupt_callback(void *unused)
{
    (void) unused;
    return R_ToplevelExec(check_interrupt, NULL) == FALSE;
}

static SEXP scalar_real_u64(uint64_t value)
{
    return ScalarReal((double) value);
}

static SEXP metrics_list(const fn_metrics *metrics)
{
    const int count = 29;
    SEXP result = PROTECT(allocVector(VECSXP, count));
    SEXP names = PROTECT(allocVector(STRSXP, count));
    int index = 0;
#define ADD_METRIC(label, field) do {                                      \
    SET_VECTOR_ELT(result, index, scalar_real_u64(metrics->field));         \
    SET_STRING_ELT(names, index, mkChar(label));                            \
    ++index;                                                                \
} while (0)
    ADD_METRIC("residual_states", residual_states);
    ADD_METRIC("prefix_pairs", prefix_pairs);
    ADD_METRIC("state_accept_all", state_accept_all);
    ADD_METRIC("state_reject_all", state_reject_all);
    ADD_METRIC("state_expand", state_expand);
    ADD_METRIC("arcs_considered", arcs_considered);
    ADD_METRIC("arcs_forwarded", arcs_forwarded);
    ADD_METRIC("interval_accepted_arcs", interval_accepted_arcs);
    ADD_METRIC("accepted_intervals", accepted_intervals);
    ADD_METRIC("longest_queries", longest_queries);
    ADD_METRIC("shortest_queries", shortest_queries);
    ADD_METRIC("shortest_skipped", shortest_skipped);
    ADD_METRIC("bound_cache_hits", bound_cache_hits);
    ADD_METRIC("bound_cache_misses", bound_cache_misses);
    ADD_METRIC("bound_internal_states", bound_internal_states);
    ADD_METRIC("branch_bound_calls", branch_bound_calls);
    ADD_METRIC("branch_cache_hits", branch_cache_hits);
    ADD_METRIC("branch_cache_misses", branch_cache_misses);
    ADD_METRIC("peak_comparisons", peak_comparisons);
    ADD_METRIC("peak_walk_steps", peak_walk_steps);
    ADD_METRIC("total_peak_movement", total_peak_movement);
    ADD_METRIC("line_searches", line_searches);
    ADD_METRIC("monotone_peak_reuses", monotone_peak_reuses);
    ADD_METRIC("maximum_prefixes_at_state", maximum_prefixes_at_state);
    ADD_METRIC("maximum_frontier_prefixes", maximum_frontier_prefixes);
    ADD_METRIC("prefix_compressions", prefix_compressions);
    ADD_METRIC("prefixes_merged", prefixes_merged);
    ADD_METRIC("allocations", allocations);
    ADD_METRIC("allocation_peak_bytes", allocation_peak_bytes);
#undef ADD_METRIC
    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}

SEXP fexactnet_compute(
    SEXP table,
    SEXP engine_value,
    SEXP workspace_value,
    SEXP diagnostic_value,
    SEXP fail_after_value
)
{
    if (TYPEOF(table) != INTSXP || !isMatrix(table)) {
        error("Native input must be an integer matrix.");
    }
    SEXP dimensions = getAttrib(table, R_DimSymbol);
    int rows = INTEGER(dimensions)[0];
    int columns = INTEGER(dimensions)[1];
    if (rows < 2 || columns < 2) {
        error("Native input must have at least two rows and columns.");
    }
    if (
        XLENGTH(table) < 0 ||
        (uint64_t) XLENGTH(table) > (uint64_t) SIZE_MAX / sizeof(int)
    ) {
        error("Matrix is too large for the native interface.");
    }
    int *row_major = (int *) R_alloc(
        (size_t) XLENGTH(table),
        sizeof(*row_major)
    );
    const int *column_major = INTEGER(table);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            row_major[row * columns + column] =
                column_major[row + column * rows];
        }
    }
    int engine_code = asInteger(engine_value);
    if (
        engine_code < FN_ENGINE_BASELINE ||
        engine_code > FN_ENGINE_AUTO
    ) {
        error("Invalid native engine code.");
    }
    double workspace = asReal(workspace_value);
    if (
        !R_FINITE(workspace) || workspace < 1.0 ||
        workspace > (double) SIZE_MAX
    ) {
        error("workspace must be a finite positive byte count.");
    }
    double fail_after = asReal(fail_after_value);
    if (
        !R_FINITE(fail_after) || fail_after < 0.0 ||
        fail_after > (double) UINT64_MAX
    ) {
        error("Internal allocation-failure control is invalid.");
    }
    fn_options options;
    fn_options_default(&options);
    options.workspace_bytes = (size_t) workspace;
    options.fail_after_allocations = (uint64_t) fail_after;
    fn_result native_result;
    char message[512];
    fn_status status = fn_compute(
        row_major,
        rows,
        columns,
        (fn_engine) engine_code,
        &options,
        r_interrupt_callback,
        NULL,
        &native_result,
        message,
        sizeof(message)
    );
    if (status != FN_OK) {
        error(
            "fexactnet native %s: %s",
            fn_status_name(status),
            message
        );
    }
    const int count = 13;
    SEXP result = PROTECT(allocVector(VECSXP, count));
    SEXP names = PROTECT(allocVector(STRSXP, count));
    int index = 0;
#define ADD_VALUE(label, value) do {                                       \
    SET_VECTOR_ELT(result, index, (value));                                 \
    SET_STRING_ELT(names, index, mkChar(label));                            \
    ++index;                                                                \
} while (0)
    ADD_VALUE("p.value", ScalarReal(native_result.p_value));
    ADD_VALUE(
        "table.probability",
        ScalarReal(native_result.observed_table_probability)
    );
    ADD_VALUE(
        "engine",
        mkString(fn_engine_name(native_result.engine_used))
    );
    ADD_VALUE(
        "requested.engine",
        mkString(fn_engine_name(native_result.requested_engine))
    );
    ADD_VALUE("transposed", ScalarLogical(native_result.transposed));
    ADD_VALUE("input.rows", ScalarInteger(native_result.input_rows));
    ADD_VALUE("input.columns", ScalarInteger(native_result.input_columns));
    ADD_VALUE("native.rows", ScalarInteger(native_result.native_rows));
    ADD_VALUE("native.columns", ScalarInteger(native_result.native_columns));
    ADD_VALUE("version", mkString(fn_version()));
    ADD_VALUE("commit", mkString(fn_commit()));
    ADD_VALUE(
        "numerical.scope",
        mkString("FEXACT-compatible floating log-factorials")
    );
    if (asLogical(diagnostic_value)) {
        ADD_VALUE("diagnostics", metrics_list(&native_result.metrics));
    } else {
        ADD_VALUE("diagnostics", R_NilValue);
    }
#undef ADD_VALUE
    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}
