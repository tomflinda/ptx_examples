#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <iostream>

__global__ void test_ldmatrix(half *load, half *store) {
  __shared__ half smem[8 * 8];
  uint32_t reg[4];
  const int start_index = threadIdx.x %8 * 8;
  *(float4 *)(&(smem[start_index])) = *(float4 *)(&(load[start_index]));

  asm("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];\n\t"
      : "=r"(reg[0]): "l"(&smem[start_index]));

  *(float *)(&(store[threadIdx.x * 2])) = *(float *)(&(reg[0]));
  
//  if (threadIdx.x == 0) {
//    store[0] = *(half *)(&(reg[0]));
//    store[1] = *((half *)(&(reg[0])) + 1);
//    printf("T0 R0: %f  %f\n", __half2float(*(half *)(&(reg[0]))), __half2float(*((half *)(&(reg[0])) + 1)));
//  }

   __syncthreads();
}

void init_mem(half *ptr, int N) {
  for (int i = 0; i < N; i++) {
    ptr[i] = i;
  }
}

int main() {
  half h_in[8 * 8];
  half h_out[8 * 8];
  half *d_in, *d_out;

  init_mem(h_in, 8 * 8);

  constexpr int data_size = 8 * 8 * sizeof(half);

  cudaMalloc(&d_in, data_size);
  cudaMalloc(&d_out, data_size);

  cudaMemcpy(d_in, h_in, data_size, cudaMemcpyHostToDevice);
  test_ldmatrix<<<1, 32>>>(d_in, d_out);

  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, data_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < 32; i++) {
    printf("thread %d  holds: ", i);
    printf("%f %f \n", __half2float(h_out[i*2]), __half2float(h_out[i*2 + 1]));
  }

  return 0;
}