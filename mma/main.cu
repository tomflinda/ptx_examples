#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32

#define CHECK_CUDA(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

//mma指令
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

//加载A矩阵(行存储)
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "l"(addr))

//加载B矩阵(行存储),需要转置
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "l"(addr))

//异步加载数据
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

/*
测试一:从dram中加载数据,按mma的layout,放在寄存器中,执行mma执行
*/
__global__ void ptx_mma_global(half* input_A, half* input_B, half* input_C, int M, int N, int K) {
    
    const size_t laneid = threadIdx.x % WARP_SIZE;

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2];
    
    RC[0]=0;
    RC[1]=0;
    
    //A矩阵: M*K 16*16
    /*
    指令文档中要求每一个thread按以下规则存放数据
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =      groupID            for ai where  0 <= i < 2 || 4 <= i < 6
               groupID + 8         Otherwise

    col =  (threadID_in_group * 2) + (i & 0x1)          for ai where i <  4
    (threadID_in_group * 2) + (i & 0x1) + 8      for ai where i >= 4
    */
    
    clock_t begin=clock64();
    int groupID           = laneid /4;
    int threadID_in_group = laneid % 4;

    int row_a0=groupID;
    int col_a0=(threadID_in_group * 2) + (0 & 0x1);
    
    int row_a2=groupID + 8;
    int col_a2=(threadID_in_group * 2) + (2 & 0x1);

    int row_a4=groupID;
    int col_a4=(threadID_in_group * 2) + (4 & 0x1) + 8;
    
    int row_a6=groupID + 8;
    int col_a6=(threadID_in_group * 2) + (6 & 0x1) + 8;
    //A矩阵a0 a1是连续存放的,这里用uint32_t来存放
    RA[0]=*(uint32_t*)&input_A[row_a0*K+col_a0];
    RA[1]=*(uint32_t*)&input_A[row_a2*K+col_a2];
    RA[2]=*(uint32_t*)&input_A[row_a4*K+col_a4];
    RA[3]=*(uint32_t*)&input_A[row_a6*K+col_a6];
    
    //B矩阵 K*N=16*8
    /*
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =  (threadID_in_group * 2) + (i & 0x1)           for bi where i <  2
           (threadID_in_group * 2) + (i & 0x1) + 8       for bi where i >= 2

    col = groupID
    */
    //B矩阵非连续,每个元素单独提取
    int row_b0=(threadID_in_group * 2) + (0 & 0x1);
    int col_b0=groupID;
    
    int row_b1=(threadID_in_group * 2) + (1 & 0x1);
    int col_b1=groupID;
    
    int row_b2=(threadID_in_group * 2) + (2 & 0x1) + 8 ;
    int col_b2=groupID;
    
    int row_b3=(threadID_in_group * 2) + (3 & 0x1) + 8 ;
    int col_b3=groupID;
   
    half *ptr_b=(half*)RB;
    ptr_b[0]=*(half*)&input_B[row_b0*N+col_b0];
    ptr_b[1]=*(half*)&input_B[row_b1*N+col_b1];
    ptr_b[2]=*(half*)&input_B[row_b2*N+col_b2];
    ptr_b[3]=*(half*)&input_B[row_b3*N+col_b3];
    
    //C矩阵 M*N=16*8
    /*
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =      groupID                               for ci where i <  2
             groupID + 8                             for ci where i >= 2

    col =  (threadID_in_group * 2) + (i & 0x1)        for ci where i = {0,..,3}
    */
    
    int row_c0=groupID;
    int col_c0=(threadID_in_group * 2) + (0 & 0x1);
    
    int row_c2=groupID + 8;
    int col_c2=(threadID_in_group * 2) + (2 & 0x1);

    HMMA16816(RC[0], RC[1],
              RA[0], RA[1], RA[2], RA[3], 
              RB[0],RB[1],
              RC[0], RC[1]);    

    *(uint32_t*)&input_C[row_c0*N+col_c0]=RC[0];
    *(uint32_t*)&input_C[row_c2*N+col_c2]=RC[1];  
    clock_t end=clock64();    
    if(laneid==0)
    {
        printf("ptx_mma_global kernel e2e:%ld\n",end-begin);
    }
}

/*
测试二:从dram中加载数据到share memory中,再用ldmatrix指令加载到寄存器中,执行mma执行
*/
__global__ void ptx_mma_shared(half* input_A, half* input_B, half* input_C, int M, int N, int K) {
    
    const size_t laneid = threadIdx.x % WARP_SIZE;
    
    __shared__ half A[16*16];
    __shared__ half B[16*8];
   
    clock_t begin=clock64();

    uint32_t smem_lane_addr = __cvta_generic_to_shared(&A[laneid*8]); 
    CP_ASYNC_CG(smem_lane_addr,&input_A[laneid*8],16);

    if(laneid<16)
    {        
        uint32_t smem_lane_addr = __cvta_generic_to_shared(&B[laneid*8]); 
        CP_ASYNC_CG(smem_lane_addr,&input_B[laneid*8],16);
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2];
    
    RC[0]=0;
    RC[1]=0;
    
    /*
      文档要求:
        When reading 8x8 matrices, a group of four consecutive threads loads 16 bytes. The matrix addresses must be naturally aligned accordingly.
        Each thread in a warp loads fragments of a row
      因此:
        1.对于A矩阵(16*16),需要分成4个8*8的矩阵,二行二列,由32个线程一起完成,一行8个元素,half类型,16字节,由连续的4个thread负责
          一个8*8的矩阵,需要32个线程协同完成(一个warp)
        2.ldmatrix要求传入的地址为一行的首地址,将laneid转成每行的首地址
          lanid%16->生成0-15的行号(因为需要每行首地址)->每一行的步进单位为16(2列)
          lanid/16->生成0-1的列号(因为有二列)->每一列的步进单位为8
          首行地址=laneid % 16 * 16 + laneid / 16 * 8;
          print([(x%16,x//16) for x in range(32)])
          [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), 
           (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1)]
           从第0列到第1列
        3.对于B矩阵(16*8),只有一列,首行地址=laneid*8 
    */
    
    int aTile_index = laneid % 16 * 16 + laneid / 16 * 8;
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], __cvta_generic_to_shared(&A[aTile_index]));
    
    int bTile_index = laneid * 8;
    LDMATRIX_X2(RB[0], RB[1], __cvta_generic_to_shared(&B[bTile_index]));
    
    //执行mma执行
    HMMA16816(RC[0], RC[1],
              RA[0], RA[1], RA[2], RA[3],
              RB[0], RB[1],
              RC[0], RC[1]);   
               
    //C矩阵 M*N=16*8
    /*
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =      groupID                               for ci where i <  2
             groupID + 8                             for ci where i >= 2

    col =  (threadID_in_group * 2) + (i & 0x1)        for ci where i = {0,..,3}
    */

    int groupID           = laneid /4;
    int threadID_in_group = laneid % 4;
    
    int row_c0 = groupID;
    int col_c0 = (threadID_in_group * 2) + (0 & 0x1);
    
    int row_c2 = groupID + 8;
    int col_c2 = (threadID_in_group * 2) + (2 & 0x1);              
              
    //写回到DRAM
    *(uint32_t*)&input_C[row_c0*N+col_c0]=RC[0];
    *(uint32_t*)&input_C[row_c2*N+col_c2]=RC[1];    

    clock_t end=clock64();
    if(laneid==0)
    {
        printf("ptx_mma_shared kernel e2e:%ld\n",end-begin);
    }    
}

int M=16;
int N=8;    
int K=16;

void dump(half *host_c)
{
    for(int r=0;r<M;r++)
    {
       for(int c=0;c<N;c++)
       {
        printf("%8.3f ",__half2float(host_c[r*N+c]));
       }
       printf("\n");
    }
}

int main() {
    half *host_a = new half[M*K];
    half *host_b = new half[K*N];
    half *host_c = new half[M*N];
    
    half *dev_a;
    half *dev_b;
    half *dev_c;
    
    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(half)*M*K));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(half)*K*N));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(half)*M*N));
    
    for(int i = 0; i < M*K; ++i) host_a[i] = __float2half(i*0.01);
    for(int i = 0; i < K*N; ++i) host_b[i] = __float2half(i*0.01);
    
    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(half)*M*K,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(half)*K*N,cudaMemcpyHostToDevice));
    for(int i = 0; i < M*N; ++i) host_c[i] = 0;
    CHECK_CUDA(cudaMemcpy(dev_c, host_c, sizeof(half)*K*N,cudaMemcpyHostToDevice));
    
    
    ptx_mma_global<<<1, 32>>>(dev_a, dev_b,dev_c,M,N,K);cudaDeviceSynchronize();
    cudaMemcpy(host_c, dev_c, sizeof(half)*M*N, cudaMemcpyDeviceToHost);
    dump(host_c);
    printf("------------------------------------------------------------\n");
    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(half)*M*K,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(half)*K*N,cudaMemcpyHostToDevice));
    for(int i = 0; i < M*N; ++i) host_c[i] = 0;
    CHECK_CUDA(cudaMemcpy(dev_c, host_c, sizeof(half)*K*N,cudaMemcpyHostToDevice));    
    ptx_mma_shared<<<1, 32>>>(dev_a, dev_b,dev_c,M,N,K);cudaDeviceSynchronize();
    cudaMemcpy(host_c, dev_c, sizeof(half)*M*N, cudaMemcpyDeviceToHost);
    dump(host_c);
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(host_a);
    free(host_b);
    free(host_c);
    return 0;
}