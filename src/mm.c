#include <math.h>
#include "mm.h"

dmatrix_t dmatrix_init(arena_t *alloc, u32 width, u32 height)
{
    float *logits = arena_alloc(alloc, sizeof(float) * width * height);
    float *probs = arena_alloc(alloc, sizeof(float) * width *height);
    float *grad = arena_alloc(alloc, sizeof(float) * width * height);
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
        {
            logits[j * width + i] = 0.;
            probs[j * width + i] = 0.5;
            grad[j * width + i] = 0;
        }
    }
    
    dmatrix_t new_dist = {0};
    new_dist.logits = logits;
    new_dist.probs = probs;
    new_dist.grad = grad;
    new_dist.width = width;
    new_dist.height = height;
    return new_dist;
}

bmatrix_t dmatrix_generate(arena_t *alloc, dmatrix_t *dist, rng_t *rng)
{
    bmatrix_t matrix = {0};
    matrix.width = dist->width;
    matrix.height = dist->height;
    matrix.mat = arena_alloc(alloc, sizeof(bool8) * matrix.width * matrix.height);
    
    for(int i = 0; i < matrix.width; i++)
    {
        for(int j = 0; j < matrix.height; j++)
        {
            float u = rng_generate(rng);
            matrix.mat[j * matrix.width + i] = u < dist->probs[j * matrix.width + i];
        }
    }
    return matrix;
}

void dmatrix_grad_acc(dmatrix_t *dist, bmatrix_t *sample, float score)
{
    for(int i = 0; i < sample->width; i++)
    {
        for(int j = 0; j < sample->height; j++)
        {
            u32 index = j * sample->width + i;
            dist->grad[index] += score * (sample->mat[index] - dist->probs[index]);
        }
    }
}

void dmatrix_grad_update(dmatrix_t *dist, float rate, int num_samples)
{
    for(int i = 0; i < dist->width * dist->height; i++)
    {
        dist->grad[i] /= num_samples * dist->probs[i] * (1 - dist->probs[i]);
        dist->logits[i] -= rate * dist->grad[i];
        dist->probs[i] = 1. / (1. + expf(-dist->logits[i]));
        dist->grad[i] = 0.;
    }
}

i8 *bmatrix_mm(arena_t *alloc, bmatrix_t *mat, i8 *input, bool8 act)
{
    i8 *res = arena_alloc(alloc, sizeof(i8) * mat->width);
    for(int i = 0; i < mat->width; i++)
    {
        i8 sum = 0;
        for(int j = 0; j < mat->height; j++)
        {
            u32 index = j * mat->width + i;
            bool8 w = mat->mat[index];
            bool8 equals = w == input[j];
            sum += equals;
        }
        
        if(act)
        {
            res[i] = (sum << 2) >= mat->width;
            continue;
        }
        res[i] = (sum << 2) - mat->width;
    }
    return res;
}
