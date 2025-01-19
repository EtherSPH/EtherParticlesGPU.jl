/*
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/01/19 17:11:46
  @ license: MIT
  @ language: Julia
  @ declaration: `EtherParticlesGPU.jl` is a particle based simulation framework avialable on multi-backend GPU.
  @ description:
        # * nvcc native_cuda_c.cu -o native_cuda_c.exe -ccbin=clang 
        # * on NVIDIA GeForce RTX 4090
        # * Warm up Start Time: 6.555045 s
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#define N 4096
#define N_THREADS 256
#define K_INNER_LOOP 10000000
#define K_OUTER_LOOP 10

__global__ void device_vadd(const float* a, const float* b, float* c, const int* index) {
    int I = threadIdx.x + blockIdx.x * blockDim.x;
    if (I < N) {
        int J = index[I];
        for (int step = 0; step < K_INNER_LOOP; step++) {
            c[I] += a[J] + b[J];
        }
    }
}

void shuffle(int* array, size_t n) {
    for (size_t i = 0; i < n - 1; i++) {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

int main() {
    float* a = (float*)malloc(N * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* c = (float*)malloc(N * sizeof(float));
    int* ordered_index = (int*)malloc(N * sizeof(int));
    int* disordered_index = (int*)malloc(N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
        c[i] = 0.0f;
        ordered_index[i] = i;
        disordered_index[i] = i;
    }
    shuffle(disordered_index, N);

    float* d_a, *d_b, *d_c;
    int* d_index;

    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));
    cudaMalloc((void**)&d_index, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, disordered_index, N * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (N + N_THREADS - 1) / N_THREADS;
    cudaDeviceSynchronize();

    printf("Warm up\n");
    device_vadd<<<blocks, N_THREADS>>>(d_a, d_b, d_c, d_index);
    cudaDeviceSynchronize();

    printf("Start\n");
    // cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < K_OUTER_LOOP; i++) {
        device_vadd<<<blocks, N_THREADS>>>(d_a, d_b, d_c, d_index);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f s\n", milliseconds / 1000);


    free(a);
    free(b);
    free(c);
    free(ordered_index);
    free(disordered_index);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_index);

    return 0;
}
