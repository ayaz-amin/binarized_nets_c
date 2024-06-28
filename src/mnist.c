#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <windows.h>
#include "mnist.h"

u8 *open_file(const char *file_path)
{
    HANDLE file = CreateFile(file_path,
        GENERIC_READ, 0, 0,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        0
    );

    unsigned long num_bytes = 0;
    DWORD file_size = GetFileSize(file, 0);
    u8 *buffer = VirtualAlloc(0, file_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    ReadFile(file, buffer, file_size - 1, &num_bytes, 0);
    return buffer;
}

mnist_dataset_t load_mnist(void)
{
    u8 *train_data = open_file("data/train-images.idx3-ubyte");
    u8 *train_labels = open_file("data/train-labels.idx1-ubyte");
    u8 *test_data = open_file("data/t10k-images.idx3-ubyte");
    u8 *test_labels = open_file("data/t10k-labels.idx1-ubyte");

    mnist_dataset_t data = {
        .train_data = train_data,
        .train_labels = train_labels,
        .test_data = test_data,
        .test_labels = test_labels
    };
    
    return data;
}

void get_image_batch(mnist_dataset_t *dataset, arena_t *alloc, bool8 **images,
                     size_t batch_size, int batch_num, bool8 train)
{
    u8 *image_dataset;
    if(train)
    {
        image_dataset = dataset->train_data;
    } else
    {
        image_dataset = dataset->test_data;
    }
    
    u32 start = 16;
    for(int i = batch_num; i < (batch_num + batch_size); i++)
    {
        images[i] = arena_alloc(alloc, sizeof(bool8) * 784);
        for(int j = 0; j < 784; j++)
        {
            bool8 pixel = _byteswap_ushort(image_dataset[784 * i + start + j]) >= 127;
            images[i][j] = pixel;
        }
    }
}

void get_labels_batch(mnist_dataset_t *dataset, u8 *labels,
                      size_t batch_size, int batch_num, bool8 train)
{
    u8 *labels_dataset;
    if(train)
    {
        labels_dataset = dataset->train_labels;
    } else
    {
        labels_dataset = dataset->test_labels;
    }
    
    u32 start = 8;
    for(int i = batch_num; i < (batch_size + batch_num); i++)
    {
        labels[i] = labels_dataset[i + start];
    }
}
