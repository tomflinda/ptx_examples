#include <stdio.h>
#include <iostream>

__global__ void helloFromGPU (void)
{
  __shared__ uint32_t aTile[4*8*4];

  int tidx = threadIdx.x;
  if (tidx == 0) {
    for (int i = 0; i < 4*8*4; ++i) {
        aTile[i] = i;
    }
  }
  __syncthreads();

  int aTile_index = tidx % 16 * 8 + tidx / 16 * 4;
  uint32_t a[4];
  uint32_t smem = __cvta_generic_to_shared(aTile+aTile_index);
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
  : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) 
  : "r"(smem)
  );

  printf("%d %d\n", tidx, aTile_index);
  if (tidx == 1) {
    printf("%d \n", (a[0])); printf("%d \n", (a[1]));
    printf("%d \n", (a[2])); printf("%d \n", (a[3]));
  }
}

int main(void) {
uint3 block = {32,1,1};
uint3 grid = {1,1,1};
helloFromGPU <<<grid, 32>>>();

cudaDeviceReset();
return 0;
}


