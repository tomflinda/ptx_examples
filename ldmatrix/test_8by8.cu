#include <stdio.h>
#include <iostream>

__global__ void helloFromGPU (void)
{
  __shared__ uint32_t aTile[8*2];

  int tidx = threadIdx.x + blockDim.x * threadIdx.y;
  // 下面的代码是把smem中的8*4的矩阵，初始化数值！
  if (tidx == 0) {
    for (int i = 0; i < 8*2; ++i) {
        aTile[i] = i;
    }
  }
  __syncthreads();

  int aTile_index = tidx %8 * 4;
  
  //printf("aTile_index: %d, tidx:%d\n", aTile_index, tidx);
  
  uint32_t a[1];
  uint32_t smem = __cvta_generic_to_shared(aTile+aTile_index);
  asm("ldmatrix.sync.aligned.m8n8.x1.shared.b16 { %0}, [ %1 ];\n"
  : "=r"(a[0]) 
  : "r"(smem)
  );


  if (tidx == 15) 
  {
    printf("%d %d\n", tidx, a[0]);
  }
}

int main(void) {
uint3 block = {32,1,1};
uint3 grid = {1,1,1};
helloFromGPU <<<grid, 32>>>();

cudaDeviceReset();
return 0;
}


