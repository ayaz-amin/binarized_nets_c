#include "mm.h"

typedef struct
{
    dmatrix_t gen_1;
    dmatrix_t gen_2;
    dmatrix_t gen_3;
    dmatrix_t gen_final;
} generator_t;

typedef struct
{
    bmatrix_t layer_1;
    bmatrix_t layer_2;
    bmatrix_t layer_3;
    bmatrix_t layer_final;
} network_t;

generator_t generator_init(arena_t *alloc, u32 num_inputs, u32 num_outputs);
network_t generator_sample(arena_t *alloc, generator_t *generator, rng_t *rng);
void generator_grad_acc(generator_t *generator, network_t *sample_network, float score);
void generator_grad_update(generator_t *generator, float rate, u32 num_samples);

i8 *network_forward(arena_t *alloc, network_t *network, i8 *inputs);
