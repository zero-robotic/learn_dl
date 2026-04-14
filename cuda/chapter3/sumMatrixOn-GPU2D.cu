#include <cuda_runtime.h>
#include <stdio.h>

#include <sys/time.h>

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; ++i) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY) {
        C[idx] = A[idx] + B[idx];
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    int nx = 1<<14;
    int ny = 1<<14;

    int dimx = 32;
    int dimy = 32;
    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    int nElem = nx * ny;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    float iStart, iElaps;
    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D<<<(%d, %d), (%d, %d)>>> elapsed %f sec\n",
           grid.x, grid.y, block.x, block.y, iElaps);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);

    cudaDeviceReset();
    return 0;
}