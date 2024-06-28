#include "mm.h"

typedef struct
{
    u8 *train_data;
    u8 *train_labels;
    u8 *test_data;
    u8 *test_labels;
} mnist_dataset_t;

mnist_dataset_t load_mnist(void);
void get_image_batch(mnist_dataset_t *dataset, arena_t *alloc, bool8 **images,
                     size_t batch_size, int batch_num, bool8 train);
void get_labels_batch(mnist_dataset_t *dataset, u8 *labels,
                      size_t batch_size, int batch_num, bool8 train);
