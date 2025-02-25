#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <iostream>

__global__ void test_ldmatrix(half *load, half *store) {
  __shared__ half smem[8 * 8 * 4];
  uint32_t reg[4];
  const int start_index = threadIdx.x * 8;
  *(float4 *)(&(smem[start_index])) = *(float4 *)(&(load[start_index]));

  asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n\t"
      : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
      : "l"(&smem[start_index]));

  *(float *)(&(store[start_index + 0])) = *(float *)(&(reg[0]));
  *(float *)(&(store[start_index + 2])) = *(float *)(&(reg[1]));
  *(float *)(&(store[start_index + 4])) = *(float *)(&(reg[2]));
  *(float *)(&(store[start_index + 6])) = *(float *)(&(reg[3]));

  if (threadIdx.x == 0) {

    printf("R0: %f  %f\n", __half2float(*(half *)(&(reg[0]))), __half2float(*((half *)(&(reg[0])) + 1)));
    printf("R1: %f  %f\n", __half2float(*(half *)(&(reg[1]))), __half2float(*((half *)(&(reg[1])) + 1)));
    printf("R2: %f  %f\n", __half2float(*(half *)(&(reg[2]))), __half2float(*((half *)(&(reg[2])) + 1)));
    printf("R3: %f  %f\n", __half2float(*(half *)(&(reg[3]))), __half2float(*((half *)(&(reg[3])) + 1)));

    // printf("R0: %f\n", __half2float(*((half *)(&(reg[0])) + 1)));
    // printf("R1: %f\n", __half2float(*((half *)(&(reg[1])) + 1)));
    // printf("R2: %f\n", __half2float(*((half *)(&(reg[2])) + 1)));
    // printf("R3: %f\n",  __half2float(*((half *)(&(reg[3])) + 1)));
  }
}

void init_mem(half *ptr, int N) {
  for (int i = 0; i < N; i++) {
    ptr[i] = i;
  }
}

int main() {
  half h_in[4 * 8 * 8];
  half h_out[4 * 8 * 8];
  half *d_in, *d_out;

  init_mem(h_in, 4 * 8 * 8);

  constexpr int data_size = 4 * 8 * 8 * sizeof(half);

  cudaMalloc(&d_in, data_size);
  cudaMalloc(&d_out, data_size);

  cudaMemcpy(d_in, h_in, data_size, cudaMemcpyHostToDevice);
  test_ldmatrix<<<1, 32>>>(d_in, d_out);

  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, data_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 32; i++) {
    std::cout << "thread " << i << " holds: ";
    for (int j = 0; j < 8; j += 2) {
      std::cout << "(" << j / 2 << ")"
                << " " << __half2float(h_out[i * 8 + j]) << ", "
                << __half2float(h_out[i * 8 + (j + 1)]) << ", ";
    }
    std::cout << std::endl;
  }

  return 0;
}