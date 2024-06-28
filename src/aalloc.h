#ifndef AALLOC_H
#define AALLOC_H

#define KB 1024
#define MB 1024 * 1024
#define GB 1024 * 1024 * 1024

typedef struct
{
    char *memory;
    size_t capacity;
    size_t offset;
} arena_t;

arena_t create_arena(arena_t *prev, size_t size);
void destroy_arena(arena_t *arena);

void *arena_alloc(arena_t *arena, size_t size);
void arena_free(arena_t *arena);

#endif //AALLOC_H
