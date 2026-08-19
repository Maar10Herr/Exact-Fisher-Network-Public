/*
 * Native routine registration for fexactnet.
 *
 * Copyright (C) 2026 Maarten Linus Herrmann
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <R.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>
#include <Rinternals.h>

extern SEXP fexactnet_compute(SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef call_methods[] = {
    {
        "fexactnet_compute",
        (DL_FUNC) &fexactnet_compute,
        5
    },
    {NULL, NULL, 0}
};

void attribute_visible
R_init_fexactnet(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}
