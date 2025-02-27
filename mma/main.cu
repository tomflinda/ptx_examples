#include <cuda_fp16.h>
#include <iostream>
#include <vector>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

const int M = 16;
const int N = 16;
const int K = 8;

__host__ void initialize_matrices(half* A, half* B, float* C) {
    for (int i = 0; i < M * K; i++) {
        A[i] = __float2half(1.0f); // Initialize A with 1.0
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = __float2half(1.0f); // Initialize B with 1.0
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f; // Initialize C with 0.0
    }
}


#define WARP_SIZE 32

template <const int BM, const int BN>
__global__ void mma_m16n8k16_ptx(half* A, half* B, int m, int n, int k, float* C) {
  //块按warp划分后的warp的col数 
  const int WARP_N = BN / 8;
  //当前warp在块中的序号
  const int WRAP_ID = threadIdx.x / WARP_SIZE;
  //当前线程在warp中的序号0~31
  const int LANE_ID = threadIdx.x % WARP_SIZE;
  //当前warp的row序号
  const int WARP_ROW = WRAP_ID / WARP_N;
  //当前warp的col序号
  const int WARP_COL = WRAP_ID % WARP_N;
  //当前线程在warp中的对于A/B/C的row和col。
  const int A_THREAD_ROW = LANE_ID / 4;
  const int A_THREAD_COL = LANE_ID % 4;
  const int B_THREAD_ROW = LANE_ID % 4;
  const int B_THREAD_COL = LANE_ID / 4;
  const int C_THREAD_ROW = LANE_ID / 4;
  const int C_THREAD_COL = LANE_ID % 4;

  // C = A*B + C,这里先初始化C全部初始化为0.
  float c[4] = {0.0, 0.0, 0.0, 0.0};
  // 从A中读取8个fp16。
  half a[8];
  // 从B中读取4个fp16
  half b[4];
  // 转换为mma指令中的A的4个32位输入数据。
  half2 ra[4];
  // 转换为mma指令中的B的2个32位输入数据。
  half2 rb[2];

  //每个warp计算16*8的结果矩阵。
  const int WARP_ROW_OFFSET = WARP_ROW * 16 + blockIdx.y * BM;
  const int WARP_COL_OFFSET = WARP_COL * 8 + blockIdx.x * BN;

  for (int i = 0; i < K; i += 16) {
    // 从全局内存读取A的8个fp16，这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
    a[0] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i, K)];
    a[1] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 1, K)];
    a[2] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i, K)];
    a[3] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 1, K)];
    a[4] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 8, K)];
    a[5] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 9, K)];
    a[6] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 8, K)];
    a[7] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 9, K)];
   
    // 从全局内存读取B的4个fp16，这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
    b[0] = B[OFFSET(i + B_THREAD_ROW * 2, WARP_COL_OFFSET + B_THREAD_COL, N)];
    b[1] = B[OFFSET(i + B_THREAD_ROW * 2 + 1, WARP_COL_OFFSET + B_THREAD_COL, N)];
    b[2] = B[OFFSET(i + B_THREAD_ROW * 2 + 8, WARP_COL_OFFSET + B_THREAD_COL, N)];
    b[3] = B[OFFSET(i + B_THREAD_ROW * 2 + 9, WARP_COL_OFFSET + B_THREAD_COL, N)];
    
    // mma指令中的A的4个32位数据
    ra[0] = __halves2half2(a[0], a[1]);
    ra[1] = __halves2half2(a[2], a[3]);
    ra[2] = __halves2half2(a[4], a[5]);
    ra[3] = __halves2half2(a[6], a[7]);
    
    // mma指令中的B的2个32位数据
    rb[0] = __halves2half2(b[0], b[1]);
    rb[1] = __halves2half2(b[2], b[3]);
    
    // mma指令计算 C=A*B+C
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(*(reinterpret_cast<int*>(&ra[0]))),
          "r"(*(reinterpret_cast<int*>(&ra[1]))),
          "r"(*(reinterpret_cast<int*>(&ra[2]))),
          "r"(*(reinterpret_cast<int*>(&ra[3]))),
          "r"(*(reinterpret_cast<int*>(&rb[0]))),
          "r"(*(reinterpret_cast<int*>(&rb[1]))));

    __syncthreads();
  }
  // 将mma的结果c写回到结果矩阵C中，这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2, N)] = c[0];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1, N)] = c[1];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2, N)] = c[2];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1, N)] = c[3];
}

#define FETCH_FLOAT4(pointer) *(reinterpret_cast<float4*>(&(pointer)))

template <const int BM, const int BN>
__global__ void mma_m16n8k16_ptx2(half* A, half* B, int M, int N, int K, float* C) {
  //块按warp划分后的warp的col数 
  const int WARP_N = BN / 8;
  //当前warp在块中的序号
  const int WRAP_ID = threadIdx.x / WARP_SIZE;
  //当前线程在warp中的序号0~31
  const int LANE_ID = threadIdx.x % WARP_SIZE;
  //当前warp的row序号
  const int WARP_ROW = WRAP_ID / WARP_N;
  //当前warp的col序号
  const int WARP_COL = WRAP_ID % WARP_N;

  const int C_THREAD_ROW = LANE_ID / 4;
  const int C_THREAD_COL = LANE_ID % 4;

  // C=A*B + C，这里先初始化C全部初始化为0。
  float c[4] = {0.0, 0.0, 0.0, 0.0};
  // A的tile大小为BM*16
  __shared__ half s_a[BM][16];
  // B的tile大小为16*BN
  __shared__ half s_b[16][BN];

  // mma指令中的A的4个32位输入数据。
  uint32_t ra[4];
  // mma指令中的B的2个32位输入数据。
  uint32_t rb[2];

  const int WARP_ROW_OFFSET_A = WARP_ROW * 16;
  const int WARP_COL_OFFSET_B = WARP_COL * 8;
  const int WARP_ROW_OFFSET_C = WARP_ROW * 16 + blockIdx.y * BM;
  const int WARP_COL_OFFSET_C = WARP_COL * 8 + blockIdx.x * BN;

  const int BLOCK_ROW_OFFSET = blockIdx.y * BM;
  const int BLOCK_COL_OFFSET = blockIdx.x * BN;

  const int NUM_ELEMENT_PER_THREAD = sizeof(float4) / sizeof(half);
  const int ROW_OFFSET_A = threadIdx.x * NUM_ELEMENT_PER_THREAD / 16;
  const int COL_OFFSET_A = (threadIdx.x * NUM_ELEMENT_PER_THREAD) % 16;
  const int ROW_STRIDE_A = blockDim.x * NUM_ELEMENT_PER_THREAD / 16;
  const int ROW_OFFSET_B = threadIdx.x * NUM_ELEMENT_PER_THREAD / BN;
  const int COL_OFFSET_B = (threadIdx.x * NUM_ELEMENT_PER_THREAD) % BN;
  const int ROW_STRIDE_B = blockDim.x * NUM_ELEMENT_PER_THREAD / BN;

  for (int i = 0; i < K; i += 16) {
    for (int offset = 0; offset < BM; offset += ROW_STRIDE_A) {
      if (offset + ROW_OFFSET_A < BM) {
       // 这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
        FETCH_FLOAT4(s_a[offset + ROW_OFFSET_A][COL_OFFSET_A]) =
            FETCH_FLOAT4(A[OFFSET(BLOCK_ROW_OFFSET + offset + ROW_OFFSET_A, COL_OFFSET_A + i, K)]);
      }
    }

    for (int offset = 0; offset < 16; offset += ROW_STRIDE_B) {
      if (offset + ROW_OFFSET_B < 16) {
      // 这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
        FETCH_FLOAT4(s_b[offset + ROW_OFFSET_B][COL_OFFSET_B]) =
            FETCH_FLOAT4(B[OFFSET(i + offset + ROW_OFFSET_B, BLOCK_COL_OFFSET + COL_OFFSET_B, N)]);
      }
    }
    __syncthreads();

    uint32_t addr_a =
        __cvta_generic_to_shared(&s_a[WARP_ROW_OFFSET_A + LANE_ID % 16][(LANE_ID / 16) * 8]);
    uint32_t addr_b = __cvta_generic_to_shared(&s_b[LANE_ID % 16][WARP_COL_OFFSET_B]);

    // 对于A矩阵，需要加载4个8*8的矩阵，warp内的32个线程参与。
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
                 : "r"(addr_a));

    // 对于B矩阵，只需要加载2个8*8的矩阵，warp内的前16个线程参与。
    if (LANE_ID < 16) {
      asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
                   : "=r"(rb[0]), "=r"(rb[1])
                   : "r"(addr_b));
    }
    
    // mma指令计算 C=A*B+C
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]));

    __syncthreads();
  }
  // 将mma的结果c写回到结果矩阵C中, 这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
  C[OFFSET(WARP_ROW_OFFSET_C + C_THREAD_ROW, WARP_COL_OFFSET_C + C_THREAD_COL * 2, N)] = c[0];
  C[OFFSET(WARP_ROW_OFFSET_C + C_THREAD_ROW, WARP_COL_OFFSET_C + C_THREAD_COL * 2 + 1, N)] = c[1];
  C[OFFSET(WARP_ROW_OFFSET_C + C_THREAD_ROW + 8, WARP_COL_OFFSET_C + C_THREAD_COL * 2, N)] = c[2];
  C[OFFSET(WARP_ROW_OFFSET_C + C_THREAD_ROW + 8, WARP_COL_OFFSET_C + C_THREAD_COL * 2 + 1, N)] =
      c[3];
}


int main() {
    half *d_A, *d_B;
    float *d_C;
    half h_A[M * K], h_B[K * N];
    float h_C[M * N];

    initialize_matrices(h_A, h_B, h_C);

    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    const int BM = 16;
    const int BN = 16;
    const int WARP_NUM = (BM * BN) / (16 * 8);
    dim3 block_dim(WARP_NUM * 32);
    dim3 grid_dim(1, 1);

    mma_m16n8k16_ptx2<BM, BN><<<grid_dim, block_dim>>>(d_A, d_B, M, N, K, d_C);
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate results
    float expected = K * 1.0f;
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }

    std::cout << "Test " << (correct ? "PASSED" : "FAILED") << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
