#include "nn.h"
#include "mm.h"

generator_t generator_init(arena_t *alloc, u32 num_inputs, u32 num_outputs)
{
    generator_t new_generator = {0};
    new_generator.gen_1 = dmatrix_init(alloc, 256, num_inputs);
    new_generator.gen_2 = dmatrix_init(alloc, 512, 256);
    new_generator.gen_3 = dmatrix_init(alloc, 128, 512);
    new_generator.gen_final = dmatrix_init(alloc, num_outputs, 128);
    return new_generator;
}

network_t generator_sample(arena_t *alloc, generator_t *generator, rng_t *rng)
{
    network_t new_network = {0};
    new_network.layer_1 = dmatrix_generate(alloc, &generator->gen_1, rng);
    new_network.layer_2 = dmatrix_generate(alloc, &generator->gen_2, rng);
    new_network.layer_3 = dmatrix_generate(alloc, &generator->gen_3, rng);
    new_network.layer_final = dmatrix_generate(alloc, &generator->gen_final, rng);
    return new_network;
}

void generator_grad_acc(generator_t *generator, network_t *sample_network, float score)
{
    dmatrix_grad_acc(&generator->gen_1, &sample_network->layer_1, score);
    dmatrix_grad_acc(&generator->gen_2, &sample_network->layer_2, score);
    dmatrix_grad_acc(&generator->gen_3, &sample_network->layer_3, score);
    dmatrix_grad_acc(&generator->gen_final, &sample_network->layer_final, score);
}

void generator_grad_update(generator_t *generator, float rate, u32 num_samples)
{
    dmatrix_grad_update(&generator->gen_1, rate, num_samples);
    dmatrix_grad_update(&generator->gen_2, rate, num_samples);
    dmatrix_grad_update(&generator->gen_3, rate, num_samples);
    dmatrix_grad_update(&generator->gen_final, rate, num_samples);
}

i8 *network_forward(arena_t *alloc, network_t *network, i8 *inputs)
{
    i8 *out1 = bmatrix_mm(alloc, &network->layer_1, inputs, 1);
    i8 *out2 = bmatrix_mm(alloc, &network->layer_2, out1, 1);
    i8 *out3 = bmatrix_mm(alloc, &network->layer_3, out2, 1);
    i8 *final = bmatrix_mm(alloc, &network->layer_final, out3, 0);
    return final;
}
