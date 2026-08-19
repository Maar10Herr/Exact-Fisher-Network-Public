/*
 * fexactnet shared native core
 *
 * Copyright (C) 2026 Maarten Linus Herrmann
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef FEXACTNET_FISHER_NETWORK_CORE_H
#define FEXACTNET_FISHER_NETWORK_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FN_FEXACT_TOLERANCE 3.45254e-7
#define FN_DEFAULT_WORKSPACE_BYTES ((size_t) 512 * 1024 * 1024)

typedef enum {
    FN_ENGINE_BASELINE = 0,
    FN_ENGINE_ACCEPT_ONLY = 1,
    FN_ENGINE_LINE_BATCH = 2,
    FN_ENGINE_LINE_BATCH_MONGE = 3,
    FN_ENGINE_AUTO = 4
} fn_engine;

typedef enum {
    FN_OK = 0,
    FN_ERR_ARGUMENT = 1,
    FN_ERR_UNSUPPORTED_DIMENSION = 2,
    FN_ERR_INTEGER_OVERFLOW = 3,
    FN_ERR_WORKSPACE = 4,
    FN_ERR_ALLOCATION = 5,
    FN_ERR_INTERRUPTED = 6,
    FN_ERR_INTERNAL = 7
} fn_status;

typedef struct {
    size_t workspace_bytes;
    uint64_t interrupt_interval;
    uint64_t fail_after_allocations;
} fn_options;

typedef struct {
    uint64_t residual_states;
    uint64_t prefix_pairs;
    uint64_t state_accept_all;
    uint64_t state_reject_all;
    uint64_t state_expand;
    uint64_t arcs_considered;
    uint64_t arcs_forwarded;
    uint64_t interval_accepted_arcs;
    uint64_t longest_queries;
    uint64_t shortest_queries;
    uint64_t shortest_skipped;
    uint64_t bound_cache_hits;
    uint64_t bound_cache_misses;
    uint64_t bound_internal_states;
    uint64_t branch_bound_calls;
    uint64_t branch_cache_hits;
    uint64_t branch_cache_misses;
    uint64_t peak_comparisons;
    uint64_t peak_walk_steps;
    uint64_t total_peak_movement;
    uint64_t line_searches;
    uint64_t monotone_peak_reuses;
    uint64_t maximum_prefixes_at_state;
    uint64_t maximum_frontier_prefixes;
    uint64_t prefix_compressions;
    uint64_t prefixes_merged;
    uint64_t accepted_intervals;
    uint64_t allocations;
    uint64_t allocation_current_bytes;
    uint64_t allocation_peak_bytes;
} fn_metrics;

typedef struct {
    double p_value;
    double observed_table_probability;
    fn_engine requested_engine;
    fn_engine engine_used;
    int transposed;
    int input_rows;
    int input_columns;
    int native_rows;
    int native_columns;
    fn_metrics metrics;
} fn_result;

typedef int (*fn_interrupt_callback)(void *data);

const char *fn_engine_name(fn_engine engine);
fn_status fn_parse_engine(const char *name, fn_engine *engine);
const char *fn_status_name(fn_status status);
const char *fn_version(void);
const char *fn_commit(void);

void fn_options_default(fn_options *options);

fn_status fn_compute(
    const int *table_row_major,
    int rows,
    int columns,
    fn_engine requested_engine,
    const fn_options *options,
    fn_interrupt_callback interrupt,
    void *interrupt_data,
    fn_result *result,
    char *error_message,
    size_t error_capacity
);

#ifdef __cplusplus
}
#endif

#endif
