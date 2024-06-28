#ifndef MM_H
#define MM_H

#include <stdint.h>
#include "rng.h"
#include "aalloc.h"

typedef uint8_t bool8;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int8_t i8;

typedef struct
{
    float *logits;
    float *probs;
    float *grad;
    u32 width;
    u32 height;
} dmatrix_t;

typedef struct
{
    bool8 *mat;
    u32 width;
    u32 height;
} bmatrix_t;

dmatrix_t dmatrix_init(arena_t *alloc, u32 width, u32 height);
bmatrix_t dmatrix_generate(arena_t *alloc, dmatrix_t *dist, rng_t *rng);
void dmatrix_grad_acc(dmatrix_t *dist, bmatrix_t *sample, float score);
void dmatrix_grad_update(dmatrix_t *dist, float rate, int num_samples);

i8 *bmatrix_mm(arena_t *alloc, bmatrix_t *mat, i8 *input, bool8 act);

#endif //MM_H
