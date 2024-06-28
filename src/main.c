#include <math.h>
#include "m_stdio.h"

#include "rng.h"
#include "nn.h"
#include "aalloc.h"
#include "mnist.h"

float cat_cross_entropy(i8 *outputs, size_t num_classes, u8 label)
{
    i8 max = outputs[0];
    for(int i = 1; i < num_classes; i++)
    {
        if(outputs[i] > max)
        {
            max = outputs[i];
        }
    }

    float numer = 0;
    float denom = 0;
    for(int i = 0; i < num_classes; i++)
    {
        float exp_x = expf(outputs[i] - max);
        denom += exp_x;
        if(i == label) numer = exp_x;
    }

    float prob_y = numer / denom;
    float cross_entropy = -logf(prob_y);
    if(isinf(cross_entropy)) cross_entropy = 100.f;
    return cross_entropy;
}

float step(arena_t *alloc, network_t *network, bool8 **img_batch, u8 *labels_batch, size_t batch_size)
{
    float average_loss = 0.f;
    for(int i = 0; i < batch_size; i++)
    {
        i8 *inputs = (i8 *)img_batch[i];
        i8 *outputs = network_forward(alloc, network, inputs);
        average_loss += cat_cross_entropy(outputs, 10, labels_batch[i]);
    }
    average_loss /= batch_size;
    return average_loss;
}

int main(void)
{
    m_printf("Bernoulli Natural Evolution Strategy\n");
    m_printf("====================================\n");
    mnist_dataset_t dataset = load_mnist();
    
    arena_t global = create_arena(0, 128 * MB);
    arena_t reserved = create_arena(&global, 16 * MB);
    arena_t net_alloc = create_arena(&global, 48 * MB);
    arena_t batch_alloc = create_arena(&global, 64 * 1024);
    arena_t alloc = create_arena(&global, 16 * MB);
    
    rng_t rng = rng_init(10);

    int n_iters = 40;
    int n_samples = 25;
    int batch_sz= 64;
    
    generator_t nn_gen = generator_init(&reserved, 784, 10);
    
    for(int i = 0; i < n_iters; i++)
    {
        bool8 **img_batch = arena_alloc(&batch_alloc, sizeof(bool8 *) * batch_sz);
        u8 *labels_batch = arena_alloc(&batch_alloc, sizeof(u8) * batch_sz);
        get_image_batch(&dataset, &batch_alloc, img_batch, batch_sz, 0, 1);
        get_labels_batch(&dataset, labels_batch, batch_sz, 0, 1);

        float mean = 0;
        float var = 0;
        network_t *nets = arena_alloc(&net_alloc, sizeof(network_t) * n_samples);
        float *scores = arena_alloc(&alloc, sizeof(float) * n_samples);
        for(int j = 0; j < n_samples; j++)
        {
            nets[j] = generator_sample(&alloc, &nn_gen, &rng);
            scores[j] = step(&alloc, &nets[j], img_batch, labels_batch, batch_sz);
            float s = scores[j];
            float old_mean = mean;
            mean += (s - mean) / (j + 1);
            var += (s - mean) * (s - old_mean);
        }
        var /= (n_samples - 1);
        float std = sqrtf(var);

        for(int j = 0; j < n_samples; j++)
        {
            float norm = (scores[j] - mean) / std;
            generator_grad_acc(&nn_gen, &nets[j], norm);
        }
        generator_grad_update(&nn_gen, 0.1, n_samples);
                
        arena_free(&batch_alloc);
        arena_free(&net_alloc);
        arena_free(&alloc);
        m_printf("Iteration: %i | Loss: %f\n", i + 1, mean);
    }

    int num_test = 5;
    bool8 **img = arena_alloc(&batch_alloc, sizeof(bool8 *) * num_test);
    u8 *label = arena_alloc(&batch_alloc, sizeof(u8) * num_test);
    get_image_batch(&dataset, &batch_alloc, img, num_test, 0, 0);
    get_labels_batch(&dataset, label, num_test, 0, 0);

    int select = 4;
    for(int i = 0; i < 784; i++)
    {
        if((i % 28) == 0)
        {
            m_printf("\n");
        }
        m_printf("%i", img[select][i]);
    }

    network_t net = generator_sample(&alloc, &nn_gen, &rng);
    i8 *outputs = network_forward(&alloc, &net, (i8 *)img[select]);

    int idx = 0;
    int max = outputs[0];
    for(int i = 1; i < 10; i++)
    {
        if(outputs[i] > max)
        {
            idx = i;
            max = outputs[i];
        }
    }

    m_printf("\nPredicted class: %i\n", idx);
    m_printf("True class: %i\n", label[select]);
   
    return 0;
}
