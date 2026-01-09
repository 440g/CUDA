#include "../common/book.h"

#define N 10

__global__ void add (int *a, int *b, int *c){
    //ThreadIdx.x: built-in var
    int tid = threadIdx.x; //data at this thread
    if (tid < N){
        c[tid] = a[tid] +b[tid];
    }
}

int main(void){
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    //allocation
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

    for (int i=0; i<N; i++){
        a[i] = -i;
        b[i] = i*i;
    }
    // cpy a, b >> dev
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
    // add at dev
    add <<<1, N>>>(dev_a, dev_b, dev_c);
    //cpy c >> host
    HANDLE_ERROR(cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice));

    for (int i=0; i<N; i++){
        printf("%d+%d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}