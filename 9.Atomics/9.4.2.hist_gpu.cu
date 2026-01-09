#include "../common/book.h"

#define SIZE (100*1024*1024)

__global__ void histo_kernel_race(unsigned char *buffer,
                                 long size,
                                 unsigned int *histo) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        histo[buffer[i]]++;
        i += stride;
    }
}

int main(void) {
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    unsigned char *dev_buffer = nullptr;
    unsigned int  *dev_histo  = nullptr;

    HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));
    HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256 * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(unsigned int)));

    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;

    histo_kernel_race<<<blocks * 2, 256>>>(dev_buffer, SIZE, dev_histo);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    unsigned int histo[256];
    HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime = 0.0f;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    long histoCount = 0;
    for (int i = 0; i < 256; i++) histoCount += histo[i];
    printf("Histogram Sum:  %ld\n", histoCount);

    for (int i = 0; i < SIZE; i++) histo[buffer[i]]--;
    int mismatch = 0;
    for (int i = 0; i < 256; i++) {
        if (histo[i] != 0) {
            if (mismatch < 10) {
                printf("Failure at %d! Off by %d\n", i, histo[i]);
            }
            mismatch++;
        }
    }
    if (mismatch == 0) {
        printf("Unexpected: no mismatch detected (race didn't manifest)\n");
    } else {
        printf("Total mismatched bins: %d / 256\n", mismatch);
    }

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    cudaFree(dev_histo);
    cudaFree(dev_buffer);
    free(buffer);
    return 0;
}
