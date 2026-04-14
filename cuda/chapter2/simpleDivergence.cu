#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

__global__ void warmingup(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if (tid % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel3(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);
    if (ipred) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);
    printf("Data size %d ", size);

    dim3 block(blocksize, 1);
    dim3 grid((size+block.x-1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    double iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    warmingup<<<grid, block>>> (d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("warmup <<<%u %u>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    // run kernel 1
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<<%4d %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    // run kernel 2
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel2 <<<%4d %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    // run kernel 3
    iStart = cpuSecond();
    mathKernel3<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel3 <<<%4d %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    cudaFree(d_C);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}