/*
 * fisher-network command-line interface
 *
 * Copyright (C) 2026 Maarten Linus Herrmann
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "fisher_network_core.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CLI_LINE_CAPACITY 1048576

typedef struct {
    int *cells;
    size_t length;
    size_t capacity;
    int rows;
    int columns;
} Matrix;

static void usage(FILE *stream, const char *program)
{
    fprintf(
        stream,
        "usage: %s [--engine ENGINE] [--json] [--benchmark]"
        " [--repeats N] table.csv\n"
        "       %s --version\n"
        "engines: auto, line_batch, line_batch_monge, accept_only, baseline\n",
        program,
        program
    );
}

static int matrix_append(Matrix *matrix, int value)
{
    if (matrix->length == matrix->capacity) {
        size_t next = matrix->capacity == 0 ? 32 : matrix->capacity * 2;
        if (next < matrix->capacity || next > SIZE_MAX / sizeof(int)) {
            return 0;
        }
        int *replacement = realloc(
            matrix->cells,
            next * sizeof(*replacement)
        );
        if (replacement == NULL) {
            return 0;
        }
        matrix->cells = replacement;
        matrix->capacity = next;
    }
    matrix->cells[matrix->length++] = value;
    return 1;
}

static int parse_line(
    char *line,
    Matrix *matrix,
    int *values_in_row,
    char *error,
    size_t error_capacity
)
{
    char *cursor = line;
    *values_in_row = 0;
    while (*cursor != '\0') {
        while (
            *cursor == ' ' || *cursor == '\t' ||
            *cursor == ',' || *cursor == '\r' || *cursor == '\n'
        ) {
            ++cursor;
        }
        if (*cursor == '\0' || *cursor == '#') {
            break;
        }
        errno = 0;
        char *end = NULL;
        long value = strtol(cursor, &end, 10);
        if (
            errno != 0 || end == cursor || value < 0 ||
            value > INT_MAX
        ) {
            (void) snprintf(
                error,
                error_capacity,
                "Invalid nonnegative integer near: %.32s",
                cursor
            );
            return 0;
        }
        if (
            *end != '\0' && *end != ' ' && *end != '\t' &&
            *end != ',' && *end != '\r' && *end != '\n' &&
            *end != '#'
        ) {
            (void) snprintf(
                error,
                error_capacity,
                "Invalid matrix separator near: %.32s",
                end
            );
            return 0;
        }
        if (!matrix_append(matrix, (int) value)) {
            (void) snprintf(
                error,
                error_capacity,
                "%s",
                "Matrix allocation failed."
            );
            return 0;
        }
        ++*values_in_row;
        cursor = end;
    }
    return 1;
}

static int read_matrix(
    const char *path,
    Matrix *matrix,
    char *error,
    size_t error_capacity
)
{
    memset(matrix, 0, sizeof(*matrix));
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        (void) snprintf(
            error,
            error_capacity,
            "Cannot open %s: %s",
            path,
            strerror(errno)
        );
        return 0;
    }
    char *line = malloc(CLI_LINE_CAPACITY);
    if (line == NULL) {
        fclose(stream);
        (void) snprintf(error, error_capacity, "%s", "Line allocation failed.");
        return 0;
    }
    while (fgets(line, CLI_LINE_CAPACITY, stream) != NULL) {
        size_t length = strlen(line);
        if (
            length == CLI_LINE_CAPACITY - 1 &&
            line[length - 1] != '\n' &&
            !feof(stream)
        ) {
            (void) snprintf(
                error,
                error_capacity,
                "%s",
                "Input row exceeds the CLI line limit."
            );
            free(line);
            fclose(stream);
            free(matrix->cells);
            memset(matrix, 0, sizeof(*matrix));
            return 0;
        }
        int values_in_row = 0;
        if (
            !parse_line(
                line,
                matrix,
                &values_in_row,
                error,
                error_capacity
            )
        ) {
            free(line);
            fclose(stream);
            free(matrix->cells);
            memset(matrix, 0, sizeof(*matrix));
            return 0;
        }
        if (values_in_row == 0) {
            continue;
        }
        if (matrix->columns == 0) {
            matrix->columns = values_in_row;
        } else if (matrix->columns != values_in_row) {
            (void) snprintf(
                error,
                error_capacity,
                "Nonrectangular matrix: expected %d values, found %d.",
                matrix->columns,
                values_in_row
            );
            free(line);
            fclose(stream);
            free(matrix->cells);
            memset(matrix, 0, sizeof(*matrix));
            return 0;
        }
        ++matrix->rows;
    }
    int read_error = ferror(stream);
    free(line);
    fclose(stream);
    if (read_error) {
        (void) snprintf(
            error,
            error_capacity,
            "%s",
            "Input read failed."
        );
        free(matrix->cells);
        memset(matrix, 0, sizeof(*matrix));
        return 0;
    }
    if (
        matrix->rows < 2 || matrix->columns < 2 ||
        matrix->length
            != (size_t) matrix->rows * (size_t) matrix->columns
    ) {
        (void) snprintf(
            error,
            error_capacity,
            "%s",
            "Expected a rectangular matrix with at least 2 x 2 cells."
        );
        free(matrix->cells);
        memset(matrix, 0, sizeof(*matrix));
        return 0;
    }
    return 1;
}

static uint64_t wall_nanoseconds(void)
{
    struct timespec value;
    if (timespec_get(&value, TIME_UTC) != TIME_UTC) {
        return 0;
    }
    return (uint64_t) value.tv_sec * UINT64_C(1000000000)
        + (uint64_t) value.tv_nsec;
}

static int compare_u64(const void *left, const void *right)
{
    uint64_t a = *(const uint64_t *) left;
    uint64_t b = *(const uint64_t *) right;
    return (a > b) - (a < b);
}

static void print_json(
    const fn_result *result,
    const uint64_t *times,
    int repeats
)
{
    const fn_metrics *metrics = &result->metrics;
    printf(
        "{\"p_value\":%.17g,\"engine\":\"%s\","
        "\"requested_engine\":\"%s\",\"version\":\"%s\","
        "\"commit\":\"%s\",\"transposed\":%s,"
        "\"rows\":%d,\"columns\":%d",
        result->p_value,
        fn_engine_name(result->engine_used),
        fn_engine_name(result->requested_engine),
        fn_version(),
        fn_commit(),
        result->transposed ? "true" : "false",
        result->input_rows,
        result->input_columns
    );
    if (times != NULL && repeats > 0) {
        printf(
            ",\"benchmark\":{\"repeats\":%d,\"minimum_ns\":%" PRIu64
            ",\"q1_ns\":%" PRIu64 ",\"median_ns\":%" PRIu64
            ",\"q3_ns\":%" PRIu64 ",\"maximum_ns\":%" PRIu64 "}",
            repeats,
            times[0],
            times[repeats / 4],
            times[repeats / 2],
            times[(3 * repeats) / 4],
            times[repeats - 1]
        );
    }
    printf(
        ",\"diagnostics\":{\"residual_states\":%" PRIu64
        ",\"arcs_considered\":%" PRIu64
        ",\"arcs_forwarded\":%" PRIu64
        ",\"accepted_intervals\":%" PRIu64
        ",\"interval_accepted_arcs\":%" PRIu64
        ",\"line_searches\":%" PRIu64
        ",\"branch_bound_calls\":%" PRIu64
        ",\"branch_cache_hits\":%" PRIu64
        ",\"branch_cache_misses\":%" PRIu64
        ",\"bound_cache_hits\":%" PRIu64
        ",\"bound_cache_misses\":%" PRIu64
        ",\"peak_comparisons\":%" PRIu64
        ",\"peak_walk_steps\":%" PRIu64
        ",\"total_peak_movement\":%" PRIu64
        ",\"monotone_peak_reuses\":%" PRIu64
        ",\"prefix_pairs\":%" PRIu64
        ",\"maximum_frontier_prefixes\":%" PRIu64
        ",\"prefix_compressions\":%" PRIu64
        ",\"prefixes_merged\":%" PRIu64
        ",\"allocation_peak_bytes\":%" PRIu64 "}}\n",
        metrics->residual_states,
        metrics->arcs_considered,
        metrics->arcs_forwarded,
        metrics->accepted_intervals,
        metrics->interval_accepted_arcs,
        metrics->line_searches,
        metrics->branch_bound_calls,
        metrics->branch_cache_hits,
        metrics->branch_cache_misses,
        metrics->bound_cache_hits,
        metrics->bound_cache_misses,
        metrics->peak_comparisons,
        metrics->peak_walk_steps,
        metrics->total_peak_movement,
        metrics->monotone_peak_reuses,
        metrics->prefix_pairs,
        metrics->maximum_frontier_prefixes,
        metrics->prefix_compressions,
        metrics->prefixes_merged,
        metrics->allocation_peak_bytes
    );
}

int main(int argc, char **argv)
{
    fn_engine engine = FN_ENGINE_AUTO;
    int json = 0;
    int benchmark = 0;
    int benchmark_repeats = 7;
    const char *path = NULL;
    for (int index = 1; index < argc; ++index) {
        if (
            strcmp(argv[index], "--help") == 0 ||
            strcmp(argv[index], "-h") == 0
        ) {
            usage(stdout, argv[0]);
            return 0;
        }
        if (strcmp(argv[index], "--version") == 0) {
            printf(
                "fisher-network %s (%s)\n",
                fn_version(),
                fn_commit()
            );
            return 0;
        }
        if (strcmp(argv[index], "--json") == 0) {
            json = 1;
            continue;
        }
        if (strcmp(argv[index], "--benchmark") == 0) {
            benchmark = 1;
            continue;
        }
        if (strcmp(argv[index], "--repeats") == 0) {
            if (index + 1 >= argc) {
                fprintf(stderr, "Missing --repeats value.\n");
                return 64;
            }
            errno = 0;
            char *end = NULL;
            long value = strtol(argv[++index], &end, 10);
            if (
                errno != 0 || end == argv[index] || *end != '\0' ||
                value < 1 || value > 101
            ) {
                fprintf(stderr, "--repeats must be an integer from 1 to 101.\n");
                return 64;
            }
            benchmark_repeats = (int) value;
            continue;
        }
        if (strcmp(argv[index], "--engine") == 0) {
            if (
                index + 1 >= argc ||
                fn_parse_engine(argv[++index], &engine) != FN_OK
            ) {
                fprintf(stderr, "Invalid or missing --engine value.\n");
                usage(stderr, argv[0]);
                return 64;
            }
            continue;
        }
        if (argv[index][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[index]);
            usage(stderr, argv[0]);
            return 64;
        }
        if (path != NULL) {
            fprintf(stderr, "Expected exactly one matrix path.\n");
            usage(stderr, argv[0]);
            return 64;
        }
        path = argv[index];
    }
    if (path == NULL) {
        usage(stderr, argv[0]);
        return 64;
    }
    Matrix matrix;
    char error[512];
    if (!read_matrix(path, &matrix, error, sizeof(error))) {
        fprintf(stderr, "fisher-network: %s\n", error);
        return 66;
    }
    fn_options options;
    fn_options_default(&options);
    fn_result result;
    fn_status status = fn_compute(
        matrix.cells,
        matrix.rows,
        matrix.columns,
        engine,
        &options,
        NULL,
        NULL,
        &result,
        error,
        sizeof(error)
    );
    if (status != FN_OK) {
        fprintf(
            stderr,
            "fisher-network: %s: %s\n",
            fn_status_name(status),
            error
        );
        free(matrix.cells);
        return status == FN_ERR_UNSUPPORTED_DIMENSION ? 65 : 70;
    }
    uint64_t times[101];
    int repeats = benchmark ? benchmark_repeats : 0;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        uint64_t started = wall_nanoseconds();
        fn_result current;
        status = fn_compute(
            matrix.cells,
            matrix.rows,
            matrix.columns,
            engine,
            &options,
            NULL,
            NULL,
            &current,
            error,
            sizeof(error)
        );
        times[iteration] = wall_nanoseconds() - started;
        if (
            status != FN_OK ||
            fabs(current.p_value - result.p_value) > 1e-14
        ) {
            fprintf(stderr, "fisher-network: benchmark repeat failed.\n");
            free(matrix.cells);
            return 70;
        }
    }
    if (repeats != 0) {
        qsort(times, (size_t) repeats, sizeof(*times), compare_u64);
    }
    if (json) {
        print_json(&result, repeats == 0 ? NULL : times, repeats);
    } else {
        printf("p-value: %.17g\n", result.p_value);
        printf("engine: %s\n", fn_engine_name(result.engine_used));
        if (repeats != 0) {
            printf(
                "benchmark: median %.3f ms (min %.3f, max %.3f; n=%d)\n",
                (double) times[repeats / 2] / 1e6,
                (double) times[0] / 1e6,
                (double) times[repeats - 1] / 1e6,
                repeats
            );
        }
    }
    free(matrix.cells);
    return 0;
}
