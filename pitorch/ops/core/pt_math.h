#ifndef PITORCH_MATH_H
#define PITORCH_MATH_H

/*
 * Software floating-point math for bare-metal Pi.
 * Polynomial approximations — no libm dependency.
 * All target < 1e-5 relative error across their useful ranges.
 */

float pt_expf(float x);
float pt_logf(float x);
float pt_sinf(float x);
float pt_cosf(float x);
float pt_sqrtf(float x);

#endif
