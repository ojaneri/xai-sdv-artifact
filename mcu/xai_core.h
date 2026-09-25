#ifndef XAI_CORE_H
#define XAI_CORE_H
#include <stdint.h>

extern uint32_t xai_passes;      /* model passes since last reset */
extern float bg_expected;        /* mean model margin over the background */
extern void (*xai_tick)(void);   /* optional hook for long loops (timer wrap) */

void xai_init(void);
void xai_seed(uint32_t s);
float predict(const float *x);
float treeshap_expected(void);
void treeshap(const float *x, float *phi);
void kernelshap(const float *x, int K, float *phi);
void lime(const float *x, int N, float *coef);
void occlusion(const float *x, float *phi);

#endif
