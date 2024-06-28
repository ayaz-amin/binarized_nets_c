#include "aalloc.h"
#include <windows.h>

arena_t create_arena(arena_t *prev, size_t capacity)
{
    char *memory;
    if(prev)
    {
        memory = arena_alloc(prev, capacity);
    } else {
        memory = VirtualAlloc(0, capacity,
                              MEM_COMMIT | MEM_RESERVE,
                              PAGE_READWRITE);
    }
    arena_t arena = {
        .memory = memory,
        .capacity = capacity
    };
    return arena;
}

void destroy_arena(arena_t *arena)
{
    VirtualFree(arena->memory, arena->capacity, MEM_FREE);
    arena->memory = 0;
}

void *arena_alloc(arena_t *arena, size_t size)
{
    size_t alligned = (size + 7) & ~7;
    if (arena->capacity >= arena->offset + alligned)
    {
        char *block = arena->memory + arena->offset;
        arena->offset += alligned;
        return block;
    }
    return 0;
}

void arena_free(arena_t *arena)
{
    arena->offset = 0;
}
