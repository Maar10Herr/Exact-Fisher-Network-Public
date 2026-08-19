/*
 * Unit tests for the fexactnet shared native core.
 *
 * Copyright (C) 2026 Maarten Linus Herrmann
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "fisher_network_core.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition, message) do {                 \
    if (!(condition)) {                                \
        fprintf(stderr, "FAIL: %s\n", (message));      \
        ++failures;                                    \
    }                                                  \
} while (0)

static int interrupt_now(void *data)
{
    int *calls = (int *) data;
    ++*calls;
    return 1;
}

static fn_status compute(
    const int *table,
    int rows,
    int columns,
    fn_engine engine,
    fn_options *options,
    fn_interrupt_callback interrupt,
    void *interrupt_data,
    fn_result *result,
    char *error
)
{
    return fn_compute(
        table,
        rows,
        columns,
        engine,
        options,
        interrupt,
        interrupt_data,
        result,
        error,
        256
    );
}

int main(void)
{
    const int table[] = {
        15, 2, 1,
        5, 8, 2,
        4, 2, 6
    };
    fn_options options;
    fn_options_default(&options);
    fn_result results[4];
    char error[256];
    for (int engine = FN_ENGINE_BASELINE;
         engine <= FN_ENGINE_LINE_BATCH_MONGE;
         ++engine) {
        memset(&results[engine], 0, sizeof(results[engine]));
        fn_status status = compute(
            table,
            3,
            3,
            (fn_engine) engine,
            &options,
            NULL,
            NULL,
            &results[engine],
            error
        );
        CHECK(status == FN_OK, error);
        CHECK(
            results[engine].metrics.allocation_current_bytes == 0,
            "all native allocations must be released"
        );
        if (engine > FN_ENGINE_BASELINE) {
            CHECK(
                fabs(results[engine].p_value - results[0].p_value) < 1e-12,
                "all engines must agree"
            );
        }
    }
    CHECK(
        results[FN_ENGINE_LINE_BATCH].metrics.arcs_forwarded
            == results[FN_ENGINE_LINE_BATCH_MONGE].metrics.arcs_forwarded,
        "theorem-only ablation must preserve forwarded arcs"
    );
    CHECK(
        results[FN_ENGINE_LINE_BATCH].metrics.branch_bound_calls
            >= results[FN_ENGINE_LINE_BATCH_MONGE].metrics.branch_bound_calls,
        "monotone peak walking must not add branch-bound calls"
    );

    const int four_by_three[] = {
        2, 1, 3,
        1, 3, 2,
        3, 2, 1,
        2, 1, 1
    };
    const int three_by_four[] = {
        2, 1, 3, 2,
        1, 3, 2, 1,
        3, 2, 1, 1
    };
    fn_result transpose_result;
    fn_result direct_result;
    fn_status status = compute(
        four_by_three,
        4,
        3,
        FN_ENGINE_LINE_BATCH,
        &options,
        NULL,
        NULL,
        &transpose_result,
        error
    );
    CHECK(status == FN_OK, error);
    CHECK(transpose_result.transposed, "r x 3 input must report transposition");
    status = compute(
        three_by_four,
        3,
        4,
        FN_ENGINE_LINE_BATCH,
        &options,
        NULL,
        NULL,
        &direct_result,
        error
    );
    CHECK(status == FN_OK, error);
    CHECK(
        fabs(
            transpose_result.p_value
            - direct_result.p_value
        ) < 1e-12,
        "transposed values must preserve the tail"
    );

    fn_options tiny;
    fn_options_default(&tiny);
    tiny.workspace_bytes = 64;
    fn_result failure_result;
    status = compute(
        table,
        3,
        3,
        FN_ENGINE_LINE_BATCH,
        &tiny,
        NULL,
        NULL,
        &failure_result,
        error
    );
    CHECK(status == FN_ERR_WORKSPACE, "tiny workspace must fail explicitly");

    fn_options injected;
    fn_options_default(&injected);
    injected.fail_after_allocations = 1;
    status = compute(
        table,
        3,
        3,
        FN_ENGINE_LINE_BATCH,
        &injected,
        NULL,
        NULL,
        &failure_result,
        error
    );
    CHECK(
        status == FN_ERR_ALLOCATION,
        "injected allocation failure must propagate"
    );

    fn_options interrupted;
    fn_options_default(&interrupted);
    interrupted.interrupt_interval = 1;
    int interrupt_calls = 0;
    status = compute(
        table,
        3,
        3,
        FN_ENGINE_LINE_BATCH,
        &interrupted,
        interrupt_now,
        &interrupt_calls,
        &failure_result,
        error
    );
    CHECK(status == FN_ERR_INTERRUPTED, "interrupt must propagate");
    CHECK(interrupt_calls > 0, "interrupt callback must be polled");

    const int unsupported[] = {
        1, 2, 3, 4,
        4, 3, 2, 1
    };
    status = compute(
        unsupported,
        2,
        4,
        FN_ENGINE_BASELINE,
        &options,
        NULL,
        NULL,
        &failure_result,
        error
    );
    CHECK(
        status == FN_ERR_UNSUPPORTED_DIMENSION,
        "other dimensions must not be disguised as native execution"
    );

    if (failures != 0) {
        fprintf(stderr, "%d native-core test(s) failed\n", failures);
        return 1;
    }
    puts("native core tests passed");
    return 0;
}
