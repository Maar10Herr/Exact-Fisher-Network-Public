/*
 * fexactnet shared native core
 *
 * Copyright (C) 2026 Maarten Linus Herrmann
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * This implementation follows the exact Fisher network recurrence used in
 * this repository. It does not copy R's stats/src/fexact.c. Its probability
 * ordering intentionally follows the R FEXACT log-factorial tolerance.
 */

#include "fisher_network_core.h"
#include "version.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FN_PREFIX_COMPRESS_CAPACITY
#define FN_PREFIX_COMPRESS_CAPACITY 65536
#endif

typedef struct {
    double log_weight;
    long double frequency;
} Prefix;

typedef struct {
    Prefix *items;
    size_t length;
    size_t capacity;
} PrefixVector;

typedef struct {
    int x[3];
    int child[3];
    double arc_log_weight;
    long double completion_mass;
    long double prefix_mass;
    long double suffix_mass;
    double branch_value;
    unsigned char value_known;
} LineElement;

typedef struct {
    LineElement *elements;
    size_t length;
    int peak_index;
    int peak_known;
} Line;

typedef struct {
    fn_engine requested_engine;
    fn_engine engine;
    fn_options options;
    fn_status status;
    char error[256];
    fn_interrupt_callback interrupt;
    void *interrupt_data;
    uint64_t work_since_interrupt;
    uint64_t allocation_attempts;
    int input_rows;
    int input_columns;
    int transposed;
    int columns;
    int *table;
    int *column_margins;
    int row_margins[3];
    int total;
    int *remaining_totals;
    double *log_factorial;
    size_t grid_width;
    size_t grid_size;
    size_t cache_count;
    double *longest_values;
    double *shortest_values;
    unsigned char *longest_known;
    unsigned char *shortest_known;
    double observed_log_weight;
    double log_denominator;
    fn_metrics metrics;
} Context;

static void context_fail(Context *context, fn_status status, const char *message)
{
    if (context->status != FN_OK) {
        return;
    }
    context->status = status;
    if (message == NULL) {
        message = fn_status_name(status);
    }
    (void) snprintf(context->error, sizeof(context->error), "%s", message);
}

static int checked_product(size_t left, size_t right, size_t *product)
{
    if (left != 0 && right > SIZE_MAX / left) {
        return 0;
    }
    *product = left * right;
    return 1;
}

static void *context_allocate(Context *context, size_t count, size_t size)
{
    size_t bytes = 0;
    if (!checked_product(count, size, &bytes)) {
        context_fail(
            context,
            FN_ERR_INTEGER_OVERFLOW,
            "Native allocation size overflow."
        );
        return NULL;
    }
    ++context->allocation_attempts;
    if (
        context->options.fail_after_allocations != 0 &&
        context->allocation_attempts
            >= context->options.fail_after_allocations
    ) {
        context_fail(
            context,
            FN_ERR_ALLOCATION,
            "Injected native allocation failure."
        );
        return NULL;
    }
    if (
        bytes > context->options.workspace_bytes ||
        context->metrics.allocation_current_bytes
            > context->options.workspace_bytes - bytes
    ) {
        context_fail(
            context,
            FN_ERR_WORKSPACE,
            "Native workspace limit exceeded."
        );
        return NULL;
    }
    void *value = calloc(count, size);
    if (value == NULL && bytes != 0) {
        context_fail(
            context,
            FN_ERR_ALLOCATION,
            "Native allocation failed."
        );
        return NULL;
    }
    ++context->metrics.allocations;
    context->metrics.allocation_current_bytes += (uint64_t) bytes;
    if (
        context->metrics.allocation_current_bytes
        > context->metrics.allocation_peak_bytes
    ) {
        context->metrics.allocation_peak_bytes =
            context->metrics.allocation_current_bytes;
    }
    return value;
}

static void *context_reallocate(
    Context *context,
    void *pointer,
    size_t old_count,
    size_t new_count,
    size_t size
)
{
    size_t old_bytes = 0;
    size_t new_bytes = 0;
    if (
        !checked_product(old_count, size, &old_bytes) ||
        !checked_product(new_count, size, &new_bytes) ||
        new_bytes < old_bytes
    ) {
        context_fail(
            context,
            FN_ERR_INTEGER_OVERFLOW,
            "Native reallocation size overflow."
        );
        return NULL;
    }
    size_t extra = new_bytes - old_bytes;
    ++context->allocation_attempts;
    if (
        context->options.fail_after_allocations != 0 &&
        context->allocation_attempts
            >= context->options.fail_after_allocations
    ) {
        context_fail(
            context,
            FN_ERR_ALLOCATION,
            "Injected native allocation failure."
        );
        return NULL;
    }
    if (
        extra > context->options.workspace_bytes ||
        context->metrics.allocation_current_bytes
            > context->options.workspace_bytes - extra
    ) {
        context_fail(
            context,
            FN_ERR_WORKSPACE,
            "Native workspace limit exceeded."
        );
        return NULL;
    }
    void *replacement = realloc(pointer, new_bytes);
    if (replacement == NULL && new_bytes != 0) {
        context_fail(
            context,
            FN_ERR_ALLOCATION,
            "Native reallocation failed."
        );
        return NULL;
    }
    ++context->metrics.allocations;
    context->metrics.allocation_current_bytes += (uint64_t) extra;
    if (
        context->metrics.allocation_current_bytes
        > context->metrics.allocation_peak_bytes
    ) {
        context->metrics.allocation_peak_bytes =
            context->metrics.allocation_current_bytes;
    }
    return replacement;
}

static void context_free(Context *context, void *pointer, size_t bytes)
{
    if (pointer == NULL) {
        return;
    }
    free(pointer);
    if (context->metrics.allocation_current_bytes < bytes) {
        context_fail(
            context,
            FN_ERR_INTERNAL,
            "Internal allocation accounting underflow."
        );
        context->metrics.allocation_current_bytes = 0;
        return;
    }
    context->metrics.allocation_current_bytes -= (uint64_t) bytes;
}

static int context_poll_interrupt(Context *context, uint64_t work)
{
    if (context->status != FN_OK || context->interrupt == NULL) {
        return context->status != FN_OK;
    }
    context->work_since_interrupt += work;
    if (
        context->work_since_interrupt
        < context->options.interrupt_interval
    ) {
        return 0;
    }
    context->work_since_interrupt = 0;
    if (context->interrupt(context->interrupt_data)) {
        context_fail(
            context,
            FN_ERR_INTERRUPTED,
            "Calculation interrupted."
        );
        return 1;
    }
    return 0;
}

static int compare_int(const void *left, const void *right)
{
    int a = *(const int *) left;
    int b = *(const int *) right;
    return (a > b) - (a < b);
}

static int compare_prefix(const void *left, const void *right)
{
    double a = ((const Prefix *) left)->log_weight;
    double b = ((const Prefix *) right)->log_weight;
    return (a > b) - (a < b);
}

const char *fn_engine_name(fn_engine engine)
{
    switch (engine) {
    case FN_ENGINE_BASELINE:
        return "baseline";
    case FN_ENGINE_ACCEPT_ONLY:
        return "accept_only";
    case FN_ENGINE_LINE_BATCH:
        return "line_batch";
    case FN_ENGINE_LINE_BATCH_MONGE:
        return "line_batch_monge";
    case FN_ENGINE_AUTO:
        return "auto";
    }
    return "unknown";
}

fn_status fn_parse_engine(const char *name, fn_engine *engine)
{
    if (name == NULL || engine == NULL) {
        return FN_ERR_ARGUMENT;
    }
    for (int value = FN_ENGINE_BASELINE; value <= FN_ENGINE_AUTO; ++value) {
        if (strcmp(name, fn_engine_name((fn_engine) value)) == 0) {
            *engine = (fn_engine) value;
            return FN_OK;
        }
    }
    return FN_ERR_ARGUMENT;
}

const char *fn_status_name(fn_status status)
{
    switch (status) {
    case FN_OK:
        return "ok";
    case FN_ERR_ARGUMENT:
        return "invalid argument";
    case FN_ERR_UNSUPPORTED_DIMENSION:
        return "unsupported dimension";
    case FN_ERR_INTEGER_OVERFLOW:
        return "integer overflow";
    case FN_ERR_WORKSPACE:
        return "workspace limit";
    case FN_ERR_ALLOCATION:
        return "allocation failure";
    case FN_ERR_INTERRUPTED:
        return "interrupted";
    case FN_ERR_INTERNAL:
        return "internal error";
    }
    return "unknown status";
}

const char *fn_version(void)
{
    return FEXACTNET_VERSION;
}

const char *fn_commit(void)
{
    return FEXACTNET_COMMIT;
}

void fn_options_default(fn_options *options)
{
    if (options == NULL) {
        return;
    }
    options->workspace_bytes = FN_DEFAULT_WORKSPACE_BYTES;
    options->interrupt_interval = UINT64_C(4096);
    options->fail_after_allocations = 0;
}

static size_t state_index(
    Context *context,
    int stage,
    int row0,
    int row1
)
{
    if (
        stage < 0 || stage > context->columns ||
        row0 < 0 || row0 > context->row_margins[0] ||
        row1 < 0 || row1 > context->row_margins[1]
    ) {
        context_fail(
            context,
            FN_ERR_INTERNAL,
            "Invalid native state index."
        );
        return 0;
    }
    return (
        (size_t) stage * context->grid_size
        + (size_t) row0 * context->grid_width
        + (size_t) row1
    );
}

static int residual_row2(
    const Context *context,
    int stage,
    int row0,
    int row1
)
{
    return context->remaining_totals[stage] - row0 - row1;
}

static int valid_state(
    const Context *context,
    int stage,
    int row0,
    int row1
)
{
    int row2 = residual_row2(context, stage, row0, row1);
    return (
        row0 >= 0 && row0 <= context->row_margins[0] &&
        row1 >= 0 && row1 <= context->row_margins[1] &&
        row2 >= 0 && row2 <= context->row_margins[2]
    );
}

static double log_multinomial3(
    const Context *context,
    const int values[3]
)
{
    int total = values[0] + values[1] + values[2];
    return (
        context->log_factorial[total]
        - context->log_factorial[values[0]]
        - context->log_factorial[values[1]]
        - context->log_factorial[values[2]]
    );
}

static fn_engine resolve_auto_engine(
    int columns,
    const int row_margins[3],
    const int *column_margins
)
{
    (void) columns;
    (void) row_margins;
    (void) column_margins;
    /*
     * Conservative pre-benchmark rule. The release benchmark may replace
     * this with a small validated rule; it never selects silently at the R
     * level because the result records the engine used.
     */
    return FN_ENGINE_LINE_BATCH;
}

static void context_release(Context *context)
{
    size_t cache_double_bytes = context->cache_count * sizeof(double);
    size_t cache_flag_bytes =
        context->cache_count * sizeof(unsigned char);
    context_free(
        context,
        context->shortest_known,
        context->shortest_known == NULL ? 0 : cache_flag_bytes
    );
    context_free(
        context,
        context->longest_known,
        context->longest_known == NULL ? 0 : cache_flag_bytes
    );
    context_free(
        context,
        context->shortest_values,
        context->shortest_values == NULL ? 0 : cache_double_bytes
    );
    context_free(
        context,
        context->longest_values,
        context->longest_values == NULL ? 0 : cache_double_bytes
    );
    context_free(
        context,
        context->log_factorial,
        context->log_factorial == NULL
            ? 0
            : ((size_t) context->total + 1) * sizeof(double)
    );
    context_free(
        context,
        context->remaining_totals,
        context->remaining_totals == NULL
            ? 0
            : ((size_t) context->columns + 1) * sizeof(int)
    );
    context_free(
        context,
        context->column_margins,
        context->column_margins == NULL
            ? 0
            : (size_t) context->columns * sizeof(int)
    );
    context_free(
        context,
        context->table,
        context->table == NULL
            ? 0
            : (size_t) 3 * (size_t) context->columns * sizeof(int)
    );
    context->shortest_known = NULL;
    context->longest_known = NULL;
    context->shortest_values = NULL;
    context->longest_values = NULL;
    context->log_factorial = NULL;
    context->remaining_totals = NULL;
    context->column_margins = NULL;
    context->table = NULL;
}

static void context_initialize(
    Context *context,
    const int *input,
    int rows,
    int columns,
    fn_engine requested_engine,
    const fn_options *options,
    fn_interrupt_callback interrupt,
    void *interrupt_data
)
{
    memset(context, 0, sizeof(*context));
    context->requested_engine = requested_engine;
    context->engine = requested_engine;
    context->status = FN_OK;
    context->interrupt = interrupt;
    context->interrupt_data = interrupt_data;
    context->input_rows = rows;
    context->input_columns = columns;
    fn_options_default(&context->options);
    if (options != NULL) {
        context->options = *options;
    }
    if (context->options.workspace_bytes == 0) {
        context->options.workspace_bytes = FN_DEFAULT_WORKSPACE_BYTES;
    }
    if (context->options.interrupt_interval == 0) {
        context->options.interrupt_interval = UINT64_C(4096);
    }
    if (
        input == NULL || rows < 2 || columns < 2 ||
        requested_engine < FN_ENGINE_BASELINE ||
        requested_engine > FN_ENGINE_AUTO
    ) {
        context_fail(
            context,
            FN_ERR_ARGUMENT,
            "Expected a non-null matrix with at least two rows and columns."
        );
        return;
    }
    if (rows != 3 && columns != 3) {
        context_fail(
            context,
            FN_ERR_UNSUPPORTED_DIMENSION,
            "Native engine supports only 3 x c or r x 3 tables."
        );
        return;
    }
    context->transposed = rows != 3;
    context->columns = context->transposed ? rows : columns;
    if (context->columns > 2048) {
        context_fail(
            context,
            FN_ERR_WORKSPACE,
            "Column count exceeds the safe native stack depth."
        );
        return;
    }
    size_t table_count = (size_t) 3 * (size_t) context->columns;
    context->table = context_allocate(
        context,
        table_count,
        sizeof(*context->table)
    );
    if (context->status != FN_OK) {
        return;
    }
    int64_t total = 0;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < context->columns; ++column) {
            int value;
            if (context->transposed) {
                value = input[column * columns + row];
            } else {
                value = input[row * columns + column];
            }
            if (value < 0) {
                context_fail(
                    context,
                    FN_ERR_ARGUMENT,
                    "Table cells must be nonnegative integers."
                );
                return;
            }
            context->table[row * context->columns + column] = value;
            if (
                (int64_t) context->row_margins[row] + value > INT_MAX
            ) {
                context_fail(
                    context,
                    FN_ERR_INTEGER_OVERFLOW,
                    "A row margin exceeds the native integer range."
                );
                return;
            }
            context->row_margins[row] += value;
            total += value;
            if (total > INT_MAX) {
                context_fail(
                    context,
                    FN_ERR_INTEGER_OVERFLOW,
                    "The table total exceeds the native integer range."
                );
                return;
            }
        }
        if (context->row_margins[row] == 0) {
            context_fail(
                context,
                FN_ERR_UNSUPPORTED_DIMENSION,
                "Native execution requires positive row margins."
            );
            return;
        }
    }
    context->total = (int) total;
    context->column_margins = context_allocate(
        context,
        (size_t) context->columns,
        sizeof(*context->column_margins)
    );
    if (context->status != FN_OK) {
        return;
    }
    for (int column = 0; column < context->columns; ++column) {
        int64_t margin = 0;
        for (int row = 0; row < 3; ++row) {
            margin += context->table[row * context->columns + column];
        }
        if (margin <= 0) {
            context_fail(
                context,
                FN_ERR_UNSUPPORTED_DIMENSION,
                "Native execution requires positive column margins."
            );
            return;
        }
        if (margin > INT_MAX) {
            context_fail(
                context,
                FN_ERR_INTEGER_OVERFLOW,
                "A column margin exceeds the native integer range."
            );
            return;
        }
        context->column_margins[column] = (int) margin;
    }
    if (requested_engine == FN_ENGINE_AUTO) {
        context->engine = resolve_auto_engine(
            context->columns,
            context->row_margins,
            context->column_margins
        );
    }
    qsort(
        context->column_margins,
        (size_t) context->columns,
        sizeof(*context->column_margins),
        compare_int
    );
    context->remaining_totals = context_allocate(
        context,
        (size_t) context->columns + 1,
        sizeof(*context->remaining_totals)
    );
    context->log_factorial = context_allocate(
        context,
        (size_t) context->total + 1,
        sizeof(*context->log_factorial)
    );
    if (context->status != FN_OK) {
        return;
    }
    for (int stage = context->columns - 1; stage >= 0; --stage) {
        context->remaining_totals[stage] = (
            context->remaining_totals[stage + 1]
            + context->column_margins[stage]
        );
    }
    for (int value = 2; value <= context->total; ++value) {
        context->log_factorial[value] = (
            context->log_factorial[value - 1] + log((double) value)
        );
    }
    for (int column = 0; column < context->columns; ++column) {
        int cells[3] = {
            context->table[column],
            context->table[context->columns + column],
            context->table[2 * context->columns + column]
        };
        context->observed_log_weight += log_multinomial3(
            context,
            cells
        );
    }
    context->log_denominator = log_multinomial3(
        context,
        context->row_margins
    );
    context->grid_width = (size_t) context->row_margins[1] + 1;
    size_t first_width = (size_t) context->row_margins[0] + 1;
    if (
        !checked_product(
            first_width,
            context->grid_width,
            &context->grid_size
        ) ||
        !checked_product(
            (size_t) context->columns + 1,
            context->grid_size,
            &context->cache_count
        )
    ) {
        context_fail(
            context,
            FN_ERR_INTEGER_OVERFLOW,
            "Native state grid size overflow."
        );
        return;
    }
    context->longest_values = context_allocate(
        context,
        context->cache_count,
        sizeof(*context->longest_values)
    );
    context->longest_known = context_allocate(
        context,
        context->cache_count,
        sizeof(*context->longest_known)
    );
    if (context->engine == FN_ENGINE_BASELINE) {
        context->shortest_values = context_allocate(
            context,
            context->cache_count,
            sizeof(*context->shortest_values)
        );
        context->shortest_known = context_allocate(
            context,
            context->cache_count,
            sizeof(*context->shortest_known)
        );
    }
}

static double get_bound(
    Context *context,
    int stage,
    int row0,
    int row1,
    int maximize
)
{
    if (context->status != FN_OK) {
        return maximize ? -INFINITY : INFINITY;
    }
    if (!valid_state(context, stage, row0, row1)) {
        return maximize ? -INFINITY : INFINITY;
    }
    size_t index = state_index(context, stage, row0, row1);
    double *values = maximize
        ? context->longest_values
        : context->shortest_values;
    unsigned char *known = maximize
        ? context->longest_known
        : context->shortest_known;
    if (values == NULL || known == NULL) {
        context_fail(
            context,
            FN_ERR_INTERNAL,
            "Requested an unavailable bound cache."
        );
        return maximize ? -INFINITY : INFINITY;
    }
    if (known[index]) {
        ++context->metrics.bound_cache_hits;
        return values[index];
    }
    ++context->metrics.bound_cache_misses;
    known[index] = 1;
    ++context->metrics.bound_internal_states;
    int row2 = residual_row2(context, stage, row0, row1);
    if (stage == context->columns) {
        if (row0 != 0 || row1 != 0 || row2 != 0) {
            context_fail(
                context,
                FN_ERR_INTERNAL,
                "Nonzero terminal residual state."
            );
        }
        values[index] = 0.0;
        return values[index];
    }
    int column = context->column_margins[stage];
    double best = maximize ? -INFINITY : INFINITY;
    int x0_min = column - row1 - row2;
    if (x0_min < 0) {
        x0_min = 0;
    }
    int x0_max = column < row0 ? column : row0;
    for (int x0 = x0_min; x0 <= x0_max; ++x0) {
        int left = column - x0;
        int x1_min = left - row2;
        if (x1_min < 0) {
            x1_min = 0;
        }
        int x1_max = left < row1 ? left : row1;
        for (int x1 = x1_min; x1 <= x1_max; ++x1) {
            if (context_poll_interrupt(context, 1)) {
                return maximize ? -INFINITY : INFINITY;
            }
            int x[3] = {x0, x1, left - x1};
            double suffix = get_bound(
                context,
                stage + 1,
                row0 - x[0],
                row1 - x[1],
                maximize
            );
            if (context->status != FN_OK) {
                return maximize ? -INFINITY : INFINITY;
            }
            if (!isfinite(suffix)) {
                continue;
            }
            double candidate = log_multinomial3(context, x) + suffix;
            if (
                (maximize && candidate > best) ||
                (!maximize && candidate < best)
            ) {
                best = candidate;
            }
        }
    }
    if (!isfinite(best)) {
        context_fail(
            context,
            FN_ERR_INTERNAL,
            "Feasible state has no feasible daughter."
        );
        return maximize ? -INFINITY : INFINITY;
    }
    values[index] = best;
    return best;
}

static void prefix_compress(Context *context, PrefixVector *vector);

static int prefix_append(
    Context *context,
    PrefixVector *vector,
    double log_weight,
    long double frequency
)
{
    if (!(frequency > 0.0L) || !isfinite(log_weight)) {
        context_fail(
            context,
            FN_ERR_INTERNAL,
            "Invalid prefix weight or frequency."
        );
        return 0;
    }
    if (vector->length == vector->capacity) {
        size_t old_capacity = vector->capacity;
        if (old_capacity >= FN_PREFIX_COMPRESS_CAPACITY) {
            prefix_compress(context, vector);
            if (context->status != FN_OK) {
                return 0;
            }
            if (vector->length < vector->capacity) {
                old_capacity = vector->capacity;
            }
        }
        if (vector->length < vector->capacity) {
            vector->items[vector->length].log_weight = log_weight;
            vector->items[vector->length].frequency = frequency;
            ++vector->length;
            return 1;
        }
        size_t new_capacity = old_capacity == 0 ? 4 : old_capacity * 2;
        if (new_capacity < old_capacity) {
            context_fail(
                context,
                FN_ERR_INTEGER_OVERFLOW,
                "Prefix capacity overflow."
            );
            return 0;
        }
        Prefix *replacement = context_reallocate(
            context,
            vector->items,
            old_capacity,
            new_capacity,
            sizeof(*replacement)
        );
        if (replacement == NULL) {
            return 0;
        }
        vector->items = replacement;
        vector->capacity = new_capacity;
    }
    vector->items[vector->length].log_weight = log_weight;
    vector->items[vector->length].frequency = frequency;
    ++vector->length;
    return 1;
}

static void prefix_compress(Context *context, PrefixVector *vector)
{
    if (vector->length < 2 || context->status != FN_OK) {
        return;
    }
    qsort(
        vector->items,
        vector->length,
        sizeof(*vector->items),
        compare_prefix
    );
    size_t input_length = vector->length;
    ++context->metrics.prefix_compressions;
    size_t output = 0;
    for (size_t index = 0; index < vector->length; ++index) {
        Prefix current = vector->items[index];
        if (
            output > 0 &&
            fabs(
                current.log_weight
                - vector->items[output - 1].log_weight
            ) <= FN_FEXACT_TOLERANCE
        ) {
            vector->items[output - 1].frequency += current.frequency;
            if (!isfinite(vector->items[output - 1].frequency)) {
                context_fail(
                    context,
                    FN_ERR_INTEGER_OVERFLOW,
                    "Prefix multiplicity exceeds long-double range."
                );
                return;
            }
        } else {
            vector->items[output++] = current;
        }
    }
    vector->length = output;
    context->metrics.prefixes_merged += input_length - output;
}

static void prefix_release(Context *context, PrefixVector *vector)
{
    context_free(
        context,
        vector->items,
        vector->capacity * sizeof(*vector->items)
    );
    vector->items = NULL;
    vector->length = 0;
    vector->capacity = 0;
}

static double line_value(
    Context *context,
    int stage,
    LineElement *element
)
{
    if (element->value_known || context->status != FN_OK) {
        return element->branch_value;
    }
    ++context->metrics.branch_bound_calls;
    size_t index = state_index(
        context,
        stage + 1,
        element->child[0],
        element->child[1]
    );
    if (context->longest_known[index]) {
        ++context->metrics.branch_cache_hits;
    } else {
        ++context->metrics.branch_cache_misses;
    }
    element->branch_value = element->arc_log_weight + get_bound(
        context,
        stage + 1,
        element->child[0],
        element->child[1],
        1
    );
    element->value_known = context->status == FN_OK;
    return element->branch_value;
}

static int independent_peak(Context *context, int stage, Line *line)
{
    if (line->peak_known || context->status != FN_OK) {
        return line->peak_index;
    }
    int low = 0;
    int high = (int) line->length - 1;
    while (low < high && context->status == FN_OK) {
        int middle = low + (high - low) / 2;
        ++context->metrics.peak_comparisons;
        if (
            line_value(context, stage, &line->elements[middle])
            < line_value(context, stage, &line->elements[middle + 1])
        ) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    line->peak_index = low;
    line->peak_known = context->status == FN_OK;
    return low;
}

static int monotone_peak(
    Context *context,
    int stage,
    Line *line,
    int previous_peak_first
)
{
    if (line->peak_known || context->status != FN_OK) {
        return line->peak_index;
    }
    int first = line->elements[0].x[0];
    int last = line->elements[line->length - 1].x[0];
    int start = previous_peak_first < last ? previous_peak_first : last;
    if (start < first) {
        context_fail(
            context,
            FN_ERR_INTERNAL,
            "Cross-slice peak bound lies left of the line domain."
        );
        return 0;
    }
    int index = start - first;
    while (index > 0 && context->status == FN_OK) {
        ++context->metrics.peak_comparisons;
        ++context->metrics.peak_walk_steps;
        if (
            line_value(context, stage, &line->elements[index - 1])
            >= line_value(context, stage, &line->elements[index])
        ) {
            --index;
        } else {
            break;
        }
    }
    line->peak_index = index;
    line->peak_known = context->status == FN_OK;
    return index;
}

static void line_boundaries(
    Context *context,
    int stage,
    Line *line,
    int peak,
    double cutoff,
    int *left_end,
    int *right_start
)
{
    *left_end = -1;
    *right_start = (int) line->length;
    ++context->metrics.peak_comparisons;
    if (line_value(context, stage, &line->elements[peak]) <= cutoff) {
        *left_end = (int) line->length - 1;
        return;
    }
    int low = 0;
    int high = peak;
    while (low < high && context->status == FN_OK) {
        int middle = low + (high - low) / 2;
        ++context->metrics.peak_comparisons;
        if (
            line_value(context, stage, &line->elements[middle])
            <= cutoff
        ) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low > 0) {
        *left_end = low - 1;
    }
    low = peak + 1;
    high = (int) line->length;
    while (low < high && context->status == FN_OK) {
        int middle = low + (high - low) / 2;
        ++context->metrics.peak_comparisons;
        if (
            line_value(context, stage, &line->elements[middle])
            <= cutoff
        ) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    *right_start = low;
}

static Line make_line(
    Context *context,
    const int rows[3],
    int column,
    int fixed_row2
)
{
    Line line;
    memset(&line, 0, sizeof(line));
    int remainder = column - fixed_row2;
    int first = remainder - rows[1];
    if (first < 0) {
        first = 0;
    }
    int last = remainder < rows[0] ? remainder : rows[0];
    if (last < first) {
        return line;
    }
    line.length = (size_t) (last - first + 1);
    line.elements = context_allocate(
        context,
        line.length,
        sizeof(*line.elements)
    );
    if (context->status != FN_OK) {
        line.length = 0;
        return line;
    }
    for (size_t offset = 0; offset < line.length; ++offset) {
        int x0 = first + (int) offset;
        int x1 = remainder - x0;
        LineElement *element = &line.elements[offset];
        element->x[0] = x0;
        element->x[1] = x1;
        element->x[2] = fixed_row2;
        for (int row = 0; row < 3; ++row) {
            element->child[row] = rows[row] - element->x[row];
        }
        element->arc_log_weight = log_multinomial3(
            context,
            element->x
        );
        element->completion_mass = expl(
            (long double) element->arc_log_weight
            + (long double) log_multinomial3(context, element->child)
            - (long double) context->log_denominator
        );
        element->prefix_mass = element->completion_mass;
        if (offset != 0) {
            element->prefix_mass +=
                line.elements[offset - 1].prefix_mass;
        }
    }
    for (size_t offset = line.length; offset-- > 0;) {
        LineElement *element = &line.elements[offset];
        element->suffix_mass = element->completion_mass;
        if (offset + 1 < line.length) {
            element->suffix_mass +=
                line.elements[offset + 1].suffix_mass;
        }
    }
    return line;
}

static void release_line(Context *context, Line *line)
{
    context_free(
        context,
        line->elements,
        line->length * sizeof(*line->elements)
    );
    memset(line, 0, sizeof(*line));
}

static void forward_element(
    Context *context,
    PrefixVector *next,
    const Prefix *prefix,
    const LineElement *element
)
{
    if (
        prefix_append(
            context,
            next,
            prefix->log_weight + element->arc_log_weight,
            prefix->frequency
        )
    ) {
        ++context->metrics.arcs_forwarded;
    }
}

static long double accepted_interval_probability(
    Context *context,
    const Prefix *prefix,
    const Line *line,
    int first,
    int past_last
)
{
    if (first >= past_last) {
        return 0.0L;
    }
    ++context->metrics.accepted_intervals;
    context->metrics.interval_accepted_arcs +=
        (uint64_t) (past_last - first);
    long double line_mass;
    if (first == 0) {
        line_mass = line->elements[past_last - 1].prefix_mass;
    } else if (past_last == (int) line->length) {
        line_mass = line->elements[first].suffix_mass;
    } else {
        line_mass = 0.0L;
        for (int index = first; index < past_last; ++index) {
            line_mass += line->elements[index].completion_mass;
        }
    }
    return prefix->frequency
        * expl((long double) prefix->log_weight)
        * line_mass;
}

static void process_unbatched(
    Context *context,
    int stage,
    const int rows[3],
    const PrefixVector *expanding,
    PrefixVector *next_frontier
)
{
    int column = context->column_margins[stage];
    int x2_min = column - rows[0] - rows[1];
    if (x2_min < 0) {
        x2_min = 0;
    }
    int x2_max = column < rows[2] ? column : rows[2];
    for (
        size_t prefix_index = 0;
        prefix_index < expanding->length &&
            context->status == FN_OK;
        ++prefix_index
    ) {
        const Prefix *prefix = &expanding->items[prefix_index];
        for (
            int x2 = x2_min;
            x2 <= x2_max && context->status == FN_OK;
            ++x2
        ) {
            Line line = make_line(context, rows, column, x2);
            context->metrics.arcs_considered += line.length;
            uint64_t line_work = (uint64_t) line.length;
            for (
                size_t index = 0;
                index < line.length && context->status == FN_OK;
                ++index
            ) {
                LineElement *element = &line.elements[index];
                size_t child_index = (
                    (size_t) element->child[0] * context->grid_width
                    + (size_t) element->child[1]
                );
                forward_element(
                    context,
                    &next_frontier[child_index],
                    prefix,
                    element
                );
            }
            release_line(context, &line);
            (void) context_poll_interrupt(context, line_work);
        }
    }
}

static void process_lines(
    Context *context,
    int stage,
    const int rows[3],
    const PrefixVector *expanding,
    PrefixVector *next_frontier,
    long double *accepted_probability
)
{
    int column = context->column_margins[stage];
    int x2_min = column - rows[0] - rows[1];
    if (x2_min < 0) {
        x2_min = 0;
    }
    int x2_max = column < rows[2] ? column : rows[2];
    int previous_peak_first = -1;
    int have_previous = 0;
    for (
        int x2 = x2_min;
        x2 <= x2_max && context->status == FN_OK;
        ++x2
    ) {
        Line line = make_line(context, rows, column, x2);
        context->metrics.arcs_considered += (
            line.length * expanding->length
        );
        if (context->status != FN_OK || line.length == 0) {
            release_line(context, &line);
            continue;
        }
        int peak;
        if (
            context->engine == FN_ENGINE_LINE_BATCH_MONGE &&
            have_previous
        ) {
            peak = monotone_peak(
                context,
                stage,
                &line,
                previous_peak_first
            );
            ++context->metrics.monotone_peak_reuses;
        } else {
            peak = independent_peak(context, stage, &line);
        }
        if (context->status != FN_OK) {
            release_line(context, &line);
            break;
        }
        int peak_first = line.elements[peak].x[0];
        if (have_previous) {
            int movement = previous_peak_first - peak_first;
            if (movement < 0) {
                context_fail(
                    context,
                    FN_ERR_INTERNAL,
                    "Observed peak movement contradicts strict order."
                );
                release_line(context, &line);
                break;
            }
            context->metrics.total_peak_movement +=
                (uint64_t) movement;
        }
        previous_peak_first = peak_first;
        have_previous = 1;
        ++context->metrics.line_searches;
        for (
            size_t prefix_index = 0;
            prefix_index < expanding->length &&
                context->status == FN_OK;
            ++prefix_index
        ) {
            const Prefix *prefix = &expanding->items[prefix_index];
            int left_end;
            int right_start;
            line_boundaries(
                context,
                stage,
                &line,
                peak,
                context->observed_log_weight
                    + FN_FEXACT_TOLERANCE
                    - prefix->log_weight,
                &left_end,
                &right_start
            );
            if (context->status != FN_OK) {
                break;
            }
            if (left_end == (int) line.length - 1) {
                *accepted_probability +=
                    accepted_interval_probability(
                        context,
                        prefix,
                        &line,
                        0,
                        (int) line.length
                    );
                continue;
            }
            *accepted_probability += accepted_interval_probability(
                context,
                prefix,
                &line,
                0,
                left_end + 1
            );
            *accepted_probability += accepted_interval_probability(
                context,
                prefix,
                &line,
                right_start,
                (int) line.length
            );
            int middle_start = left_end + 1;
            int middle_end = right_start;
            for (
                int index = middle_start;
                index < middle_end && context->status == FN_OK;
                ++index
            ) {
                LineElement *element = &line.elements[index];
                size_t child_index = (
                    (size_t) element->child[0] * context->grid_width
                    + (size_t) element->child[1]
                );
                forward_element(
                    context,
                    &next_frontier[child_index],
                    prefix,
                    element
                );
            }
        }
        uint64_t line_work = (uint64_t) line.length;
        release_line(context, &line);
        (void) context_poll_interrupt(context, line_work);
    }
}

static void release_frontier(
    Context *context,
    PrefixVector *frontier
)
{
    if (frontier == NULL) {
        return;
    }
    for (size_t index = 0; index < context->grid_size; ++index) {
        prefix_release(context, &frontier[index]);
    }
    context_free(
        context,
        frontier,
        context->grid_size * sizeof(*frontier)
    );
}

static void run_network(Context *context, fn_result *result)
{
    PrefixVector *frontier = NULL;
    PrefixVector *next = NULL;
    if (context->status != FN_OK) {
        return;
    }
    frontier = context_allocate(
        context,
        context->grid_size,
        sizeof(*frontier)
    );
    next = context_allocate(
        context,
        context->grid_size,
        sizeof(*next)
    );
    if (context->status != FN_OK) {
        release_frontier(context, next);
        release_frontier(context, frontier);
        return;
    }
    size_t root_index = (
        (size_t) context->row_margins[0] * context->grid_width
        + (size_t) context->row_margins[1]
    );
    (void) prefix_append(
        context,
        &frontier[root_index],
        0.0,
        1.0L
    );
    long double accepted_probability = 0.0L;
    for (
        int stage = 0;
        stage <= context->columns && context->status == FN_OK;
        ++stage
    ) {
        uint64_t frontier_prefixes = 0;
        for (size_t index = 0; index < context->grid_size; ++index) {
            frontier_prefixes += frontier[index].length;
        }
        if (
            frontier_prefixes
            > context->metrics.maximum_frontier_prefixes
        ) {
            context->metrics.maximum_frontier_prefixes =
                frontier_prefixes;
        }
        for (
            int row0 = 0;
            row0 <= context->row_margins[0] &&
                context->status == FN_OK;
            ++row0
        ) {
            for (
                int row1 = 0;
                row1 <= context->row_margins[1] &&
                    context->status == FN_OK;
                ++row1
            ) {
                size_t index = (
                    (size_t) row0 * context->grid_width
                    + (size_t) row1
                );
                PrefixVector *prefixes = &frontier[index];
                if (prefixes->length == 0) {
                    continue;
                }
                if (!valid_state(context, stage, row0, row1)) {
                    context_fail(
                        context,
                        FN_ERR_INTERNAL,
                        "Frontier contains an invalid residual state."
                    );
                    break;
                }
                prefix_compress(context, prefixes);
                if (context->status != FN_OK) {
                    break;
                }
                if (
                    prefixes->length
                    > context->metrics.maximum_prefixes_at_state
                ) {
                    context->metrics.maximum_prefixes_at_state =
                        prefixes->length;
                }
                ++context->metrics.residual_states;
                context->metrics.prefix_pairs += prefixes->length;
                int rows[3] = {
                    row0,
                    row1,
                    residual_row2(context, stage, row0, row1)
                };
                ++context->metrics.longest_queries;
                double longest = get_bound(
                    context,
                    stage,
                    row0,
                    row1,
                    1
                );
                double shortest = 0.0;
                if (context->engine == FN_ENGINE_BASELINE) {
                    ++context->metrics.shortest_queries;
                    shortest = get_bound(
                        context,
                        stage,
                        row0,
                        row1,
                        0
                    );
                } else if (stage < context->columns) {
                    ++context->metrics.shortest_skipped;
                }
                if (context->status != FN_OK) {
                    break;
                }
                PrefixVector expanding;
                memset(&expanding, 0, sizeof(expanding));
                double state_mass_log = log_multinomial3(
                    context,
                    rows
                );
                for (
                    size_t prefix_index = 0;
                    prefix_index < prefixes->length &&
                        context->status == FN_OK;
                    ++prefix_index
                ) {
                    Prefix *prefix = &prefixes->items[prefix_index];
                    double threshold = (
                        context->observed_log_weight
                        + FN_FEXACT_TOLERANCE
                    );
                    if (prefix->log_weight + longest <= threshold) {
                        ++context->metrics.state_accept_all;
                        accepted_probability +=
                            prefix->frequency * expl(
                                (long double) prefix->log_weight
                                + (long double) state_mass_log
                                - (long double) context->log_denominator
                            );
                    } else if (
                        prefix->log_weight + shortest > threshold
                    ) {
                        ++context->metrics.state_reject_all;
                    } else {
                        ++context->metrics.state_expand;
                        (void) prefix_append(
                            context,
                            &expanding,
                            prefix->log_weight,
                            prefix->frequency
                        );
                    }
                }
                if (
                    expanding.length != 0 &&
                    context->status == FN_OK
                ) {
                    if (stage == context->columns) {
                        context_fail(
                            context,
                            FN_ERR_INTERNAL,
                            "Terminal prefix was not classified."
                        );
                    } else if (
                        (
                            context->engine == FN_ENGINE_LINE_BATCH ||
                            context->engine
                                == FN_ENGINE_LINE_BATCH_MONGE
                        ) &&
                        stage + 1 < context->columns
                    ) {
                        process_lines(
                            context,
                            stage,
                            rows,
                            &expanding,
                            next,
                            &accepted_probability
                        );
                    } else {
                        process_unbatched(
                            context,
                            stage,
                            rows,
                            &expanding,
                            next
                        );
                    }
                }
                prefix_release(context, &expanding);
                if (context_poll_interrupt(context, 1)) {
                    break;
                }
            }
        }
        for (size_t index = 0; index < context->grid_size; ++index) {
            prefix_release(context, &frontier[index]);
        }
        PrefixVector *swap = frontier;
        frontier = next;
        next = swap;
    }
    if (context->status == FN_OK) {
        for (size_t index = 0; index < context->grid_size; ++index) {
            if (
                frontier[index].length != 0 ||
                next[index].length != 0
            ) {
                context_fail(
                    context,
                    FN_ERR_INTERNAL,
                    "Network frontier was not exhausted."
                );
                break;
            }
        }
    }
    release_frontier(context, next);
    release_frontier(context, frontier);
    if (context->status != FN_OK) {
        return;
    }
    if (
        accepted_probability < -LDBL_EPSILON ||
        accepted_probability > 1.0L + 1e-10L
    ) {
        context_fail(
            context,
            FN_ERR_INTERNAL,
            "Computed probability lies outside [0, 1]."
        );
        return;
    }
    if (accepted_probability < 0.0L) {
        accepted_probability = 0.0L;
    }
    if (accepted_probability > 1.0L) {
        accepted_probability = 1.0L;
    }
    result->p_value = (double) accepted_probability;
    result->observed_table_probability = exp(
        context->observed_log_weight - context->log_denominator
    );
    result->requested_engine = context->requested_engine;
    result->engine_used = context->engine;
    result->transposed = context->transposed;
    result->input_rows = context->input_rows;
    result->input_columns = context->input_columns;
    result->native_rows = 3;
    result->native_columns = context->columns;
}

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
)
{
    Context context;
    if (result == NULL) {
        if (error_message != NULL && error_capacity != 0) {
            (void) snprintf(
                error_message,
                error_capacity,
                "%s",
                "Result pointer must not be null."
            );
        }
        return FN_ERR_ARGUMENT;
    }
    memset(result, 0, sizeof(*result));
    context_initialize(
        &context,
        table_row_major,
        rows,
        columns,
        requested_engine,
        options,
        interrupt,
        interrupt_data
    );
    run_network(&context, result);
    fn_status status = context.status;
    char saved_error[sizeof(context.error)];
    (void) snprintf(saved_error, sizeof(saved_error), "%s", context.error);
    context_release(&context);
    if (status == FN_OK && context.status != FN_OK) {
        status = context.status;
        (void) snprintf(
            saved_error,
            sizeof(saved_error),
            "%s",
            context.error
        );
    }
    result->metrics = context.metrics;
    if (error_message != NULL && error_capacity != 0) {
        const char *message = saved_error[0] == '\0'
            ? fn_status_name(status)
            : saved_error;
        (void) snprintf(error_message, error_capacity, "%s", message);
    }
    return status;
}
