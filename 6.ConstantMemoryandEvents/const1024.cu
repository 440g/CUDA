#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#include <cstdio>
#include <cstdlib>

#define DIM 1024
#define SPHERES 20
#define WARMUP_ITERS 20

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

struct DataBlock {
    unsigned char *dev_bitmap;
};


struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;
    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

__constant__ Sphere c_s[SPHERES];

__global__ void kernel(unsigned char *ptr) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - DIM/2);
    float oy = (y - DIM/2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;

    #pragma unroll
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = c_s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = c_s[i].r * fscale;
            g = c_s[i].g * fscale;
            b = c_s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

int main() {
    srand(0);

    DataBlock data;
    CPUBitmap bitmap(DIM, DIM, &data);

    unsigned char *dev_bitmap = nullptr;

    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
    data.dev_bitmap = dev_bitmap;

    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }
    HANDLE_ERROR(cudaMemcpyToSymbol(c_s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16, 16);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel<<<grids, threads>>>(dev_bitmap);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    kernel<<<grids, threads>>>(dev_bitmap);
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    HANDLE_ERROR(cudaEventElapsedTime(&kernel_ms, start, stop));

    // CSV 한 줄 출력
    printf("const,%.6f\n", kernel_ms);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    // 측정 제외로 D2H 한번
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_bitmap));
    return 0;
}
