#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <iostream>

__global__ void test_ldmatrix(half *load, half *store) {
  __shared__ half smem[8 * 8];
  uint32_t reg[4];
  const int start_index = threadIdx.x %8 * 8;
  *(float4 *)(&(smem[start_index])) = *(float4 *)(&(load[start_index]));

  asm("ldmatrix.sync.aligned.m8n8.x1.trans.b16 {%0}, [%1];\n\t"
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
  //half h_in[8 * 8];
  //init_mem(h_in, 8 * 8);
    
  half h_out[8 * 8];
  half *d_in, *d_out;



  half h_in[64] = {
      0, 1, 2, 3, 4, 5, 6, 7,
      8, 9, 10,11,12,13,14,15,
      16,17,18,19,20,21,22,23,
      24,25,26,27,28,29,30,31,
      32,33,34,35,36,37,38,39,
      40,41,42,43,44,45,46,47,
      48,49,50,51,52,53,54,55,
      56,57,58,59,60,61,62,63
  };
#if 1
  half h_in_trans[64] = {
    0,8, 16,24,32,40,48,56,
    1, 9, 17,25,33,41,49,57,
    2, 10,18,26,34,42,50,58,
    3, 11,19,27,35,43,51,59,
    4, 12,20,28,36,44,52,60,
    5, 13,21,29,37,45,53,61,
    6, 14,22,30,38,46,54,62,
    7,15,23,31,39,47,55,63
  };
#endif


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