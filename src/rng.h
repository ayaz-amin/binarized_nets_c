#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct
{
    int position;
    uint32_t seed;
} rng_t;

rng_t rng_init(uint32_t seed);
float rng_generate(rng_t *rng);

#endif //RNG_H
