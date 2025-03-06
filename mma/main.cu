#include <cuda_fp16.h>
#include <iostream>
#include <vector>

const int M = 32;
const int N = 32;
const int K = 16;

__host__ void initialize_matrices(half *A, half *B, float *C) {

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      A[i * K + j] = __float2half(i * K + j);
    }
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      B[i * N + j] = __float2half(i * N + j);
    }
  }

  for (int i = 0; i < M * N; i++) {
    C[i] = 0.0f; // Initialize C with 0.0
  }
}

__device__ void mma_m16n8k16_simulation(float &c0, float &c1, float &c2,
                                        float &c3, int a0, int a1, int a2,
                                        int a3, int b0, int b1) {
  // TODO:
  // the implementation of mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
}

#define WARP_SIZE 32
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <const int BM, const int BN>
__global__ void mma_m16n8k16_ptx(half *A, half *B, int m, int n, int k,
                                 float *C) {
  // Number of warp columns after dividing the block by warps
  const int WARP_N = BN / 8;
  // Index of the current warp within the block
  const int WRAP_ID = threadIdx.x / WARP_SIZE;
  // Index of the current thread within the warp (0~31)
  const int LANE_ID = threadIdx.x % WARP_SIZE;
  // Row index of the current warp
  const int WARP_ROW = WRAP_ID / WARP_N;
  // Column index of the current warp
  const int WARP_COL = WRAP_ID % WARP_N;
  // Row and column indices for A/B/C for the current thread within the warp
  const int A_THREAD_ROW = LANE_ID / 4;
  const int A_THREAD_COL = LANE_ID % 4;
  const int B_THREAD_ROW = LANE_ID % 4;
  const int B_THREAD_COL = LANE_ID / 4;
  const int C_THREAD_ROW = LANE_ID / 4;
  const int C_THREAD_COL = LANE_ID % 4;

  // Compute C = A * B + C, initializing C with 0
  float c[4] = {0.0, 0.0, 0.0, 0.0};
  // Load 8 fp16 values from A
  half a[8];
  // Load 4 fp16 values from B
  half b[4];
  // Convert A values into 4 32-bit values for the MMA instruction
  half2 ra[4];
  // Convert B values into 2 32-bit values for the MMA instruction
  half2 rb[2];

  // Each warp computes a 16x8 result matrix
  const int WARP_ROW_OFFSET = WARP_ROW * 16 + blockIdx.y * BM;
  const int WARP_COL_OFFSET = WARP_COL * 8 + blockIdx.x * BN;

  for (int i = 0; i < K; i += 16) {
    // Load 8 fp16 values from global memory for A
    a[0] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i, K)];
    a[1] =
        A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 1, K)];
    a[2] =
        A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i, K)];
    a[3] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8,
                    A_THREAD_COL * 2 + i + 1, K)];
    a[4] =
        A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 8, K)];
    a[5] =
        A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 9, K)];
    a[6] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8,
                    A_THREAD_COL * 2 + i + 8, K)];
    a[7] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8,
                    A_THREAD_COL * 2 + i + 9, K)];

    // Load 4 fp16 values from global memory for B
    b[0] = B[OFFSET(i + B_THREAD_ROW * 2, WARP_COL_OFFSET + B_THREAD_COL, N)];
    b[1] =
        B[OFFSET(i + B_THREAD_ROW * 2 + 1, WARP_COL_OFFSET + B_THREAD_COL, N)];
    b[2] =
        B[OFFSET(i + B_THREAD_ROW * 2 + 8, WARP_COL_OFFSET + B_THREAD_COL, N)];
    b[3] =
        B[OFFSET(i + B_THREAD_ROW * 2 + 9, WARP_COL_OFFSET + B_THREAD_COL, N)];

    // Convert A values to 4 32-bit values for the MMA instruction
    ra[0] = __halves2half2(a[0], a[1]);
    ra[1] = __halves2half2(a[2], a[3]);
    ra[2] = __halves2half2(a[4], a[5]);
    ra[3] = __halves2half2(a[6], a[7]);

    // Convert B values to 2 32-bit values for the MMA instruction
    rb[0] = __halves2half2(b[0], b[1]);
    rb[1] = __halves2half2(b[2], b[3]);

    // Compute C = A * B + C using the MMA instruction
#if 1

    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(*(reinterpret_cast<int *>(&ra[0]))),
          "r"(*(reinterpret_cast<int *>(&ra[1]))),
          "r"(*(reinterpret_cast<int *>(&ra[2]))),
          "r"(*(reinterpret_cast<int *>(&ra[3]))),
          "r"(*(reinterpret_cast<int *>(&rb[0]))),
          "r"(*(reinterpret_cast<int *>(&rb[1]))));
#else
    // C implementation for mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    mma_m16n8k16_simulation(
        c[0], c[1], c[2], c[3], *(reinterpret_cast<int *>(&ra[0])),
        *(reinterpret_cast<int *>(&ra[1])), *(reinterpret_cast<int *>(&ra[2])),
        *(reinterpret_cast<int *>(&ra[3])), *(reinterpret_cast<int *>(&rb[0])),
        *(reinterpret_cast<int *>(&rb[1])));
#endif
  }
  // Store the computed C values back to the output matrix
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2,
           N)] = c[0];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW,
           WARP_COL_OFFSET + C_THREAD_COL * 2 + 1, N)] = c[1];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW + 8,
           WARP_COL_OFFSET + C_THREAD_COL * 2, N)] = c[2];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW + 8,
           WARP_COL_OFFSET + C_THREAD_COL * 2 + 1, N)] = c[3];
}

#define FETCH_FLOAT4(pointer) *(reinterpret_cast<float4 *>(&(pointer)))

template <const int BM, const int BN>
__global__ void mma_m16n8k16_ptx2(half *A, half *B, int M, int N, int K,
                                  float *C) {
  // Number of warp columns after dividing the block by warps
  const int WARP_N = BN / 8;
  // Index of the current warp within the block
  const int WRAP_ID = threadIdx.x / WARP_SIZE;
  // Index of the current thread within the warp (0~31)
  const int LANE_ID = threadIdx.x % WARP_SIZE;
  // Row index of the current warp
  const int WARP_ROW = WRAP_ID / WARP_N;
  // Column index of the current warp
  const int WARP_COL = WRAP_ID % WARP_N;

  const int C_THREAD_ROW = LANE_ID / 4;
  const int C_THREAD_COL = LANE_ID % 4;

  // C = A * B + C, initialize C with all zeros fi
  float c[4] = {0.0, 0.0, 0.0, 0.0};
  // Tile size of A: BM × 16
  __shared__ half s_a[BM][16];
  // Tile size of B: 16 × BN
  __shared__ half s_b[16][BN];

  // Four 32-bit input data for the A matrix in the MMA instruction.
  uint32_t ra[4];
  // Two 32-bit input data for the B matrix in the MMA instruction.
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
        FETCH_FLOAT4(s_a[offset + ROW_OFFSET_A][COL_OFFSET_A]) =
            FETCH_FLOAT4(A[OFFSET(BLOCK_ROW_OFFSET + offset + ROW_OFFSET_A,
                                  COL_OFFSET_A + i, K)]);
      }
    }

    for (int offset = 0; offset < 16; offset += ROW_STRIDE_B) {
      if (offset + ROW_OFFSET_B < 16) {
        FETCH_FLOAT4(s_b[offset + ROW_OFFSET_B][COL_OFFSET_B]) =
            FETCH_FLOAT4(B[OFFSET(i + offset + ROW_OFFSET_B,
                                  BLOCK_COL_OFFSET + COL_OFFSET_B, N)]);
      }
    }
    __syncthreads();

    uint32_t addr_a = __cvta_generic_to_shared(
        &s_a[WARP_ROW_OFFSET_A + LANE_ID % 16][(LANE_ID / 16) * 8]);
    uint32_t addr_b =
        __cvta_generic_to_shared(&s_b[LANE_ID % 16][WARP_COL_OFFSET_B]);

    // For matrix A, load four 8×8 submatrices, with all 32 threads in a warp
    // participating
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
        : "r"(addr_a));

    // For matrix B, only load two 8×8 submatrices, with the first 16 threads in
    // the warp participating
    if (LANE_ID < 16) {
      asm volatile(
          "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];"
          : "=r"(rb[0]), "=r"(rb[1])
          : "r"(addr_b));
    }

    // MMA instruction computes C = A * B + C
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]),
          "r"(rb[1]));

    __syncthreads();
  }
  // Store the MMA results back into the output matrix C
  C[OFFSET(WARP_ROW_OFFSET_C + C_THREAD_ROW,
           WARP_COL_OFFSET_C + C_THREAD_COL * 2, N)] = c[0];
  C[OFFSET(WARP_ROW_OFFSET_C + C_THREAD_ROW,
           WARP_COL_OFFSET_C + C_THREAD_COL * 2 + 1, N)] = c[1];
  C[OFFSET(WARP_ROW_OFFSET_C + C_THREAD_ROW + 8,
           WARP_COL_OFFSET_C + C_THREAD_COL * 2, N)] = c[2];
  C[OFFSET(WARP_ROW_OFFSET_C + C_THREAD_ROW + 8,
           WARP_COL_OFFSET_C + C_THREAD_COL * 2 + 1, N)] = c[3];
}

void matrix_multiplication_cpu(half *A, half *B, float *C, int m, int n,
                               int k) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
      }
      C[i * N + j] = sum;
    }
  }
}

void test_1(void) {

  half *d_A, *d_B;
  float *d_C;
  half h_A[M * K], h_B[K * N];
  float h_C[M * N];
  float h_C_ref[M * N];

  initialize_matrices(h_A, h_B, h_C);

  matrix_multiplication_cpu(h_A, h_B, h_C_ref, M, N, K);

  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

  const int BM = 32;
  const int BN = 32;
  const int WARP_NUM = (BM * BN) / (16 * 8);
  dim3 block_dim(WARP_NUM * 32);
  dim3 grid_dim(1, 1);

  mma_m16n8k16_ptx<BM, BN><<<grid_dim, block_dim>>>(d_A, d_B, M, N, K, d_C);
  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Validate results
  bool correct = true;

  for (int i = 0; i < M * N; i++) {
    if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) {
      correct = false;
      break;
    }
  }

  std::cout << "Test1 " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void test_2(void) {

  half *d_A, *d_B;
  float *d_C;
  half h_A[M * K], h_B[K * N];
  float h_C[M * N];
  float h_C_ref[M * N];

  initialize_matrices(h_A, h_B, h_C);

  matrix_multiplication_cpu(h_A, h_B, h_C_ref, M, N, K);

  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

  const int BM = 32;
  const int BN = 32;
  const int WARP_NUM = (BM * BN) / (16 * 8);
  dim3 block_dim(WARP_NUM * 32);
  dim3 grid_dim(1, 1);

  mma_m16n8k16_ptx2<BM, BN><<<grid_dim, block_dim>>>(d_A, d_B, M, N, K, d_C);
  cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Validate results
  bool correct = true;

  for (int i = 0; i < M * N; i++) {
    if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) {
      correct = false;
      break;
    }
  }

  std::cout << "Test2 " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {

  test_1();
  test_2();
  return 0;
}
