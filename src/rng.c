#include "rng.h"

rng_t rng_init(uint32_t seed)
{
    rng_t new_rng = {0};
    new_rng.position = 0;
    new_rng.seed = seed;
    return new_rng;
}

#define BITNOISE1 0xB5297A4D;
#define BITNOISE2 0x68E31DA4;
#define BITNOISE3 0x1B56C4E9;
float rng_generate(rng_t *rng)
{
    uint32_t mangled = rng->position;
    rng->position++;
    mangled *= BITNOISE1;
    mangled += rng->seed;
    mangled ^= (mangled >> 8);
    mangled += BITNOISE2;
    mangled ^= (mangled << 8);
    mangled *= BITNOISE3;
    mangled ^= (mangled >> 8);
    return (float)mangled / (float)UINT32_MAX;
}
