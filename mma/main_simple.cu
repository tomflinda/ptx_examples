#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>

const int M = 16;
const int N = 8;
const int K = 16;

  // std::cout << "Matrix h_C:" << std::endl;
  // for (int i = 0; i < M; ++i) {
  //     for (int j = 0; j < N; ++j) {
  //         std::cout << h_C[i * N + j] << " ";
  //     }
  //     std::cout << std::endl;
  // }

void print_matrix(half *array, int m, int n) {
  printf(" ------------ Matrix ----------:\n");
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << std::setw(4) << __half2float(array[i * n + j]) << " ";
    }
    std::cout << std::endl;
  }
  printf(" ------------ End ----------\n\n");
}


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

  // for (int i = 0; i < M * K; i++) {
  //     A[i] = __float2half(1.0f); // Initialize A with 1.0
  // }
  // for (int i = 0; i < K * N; i++) {
  //     B[i] = __float2half(1.0f); // Initialize B with 1.0
  // }
  for (int i = 0; i < M * N; i++) {
    C[i] = 0.0f; // Initialize C with 0.0
  }
}

__device__ void mma_simulation(float c[4], half2 ra[4], half2 rb[2]) {
  // 提取 ra 中的 half 数据
  half a[4][2] = {{__low2half(ra[0]), __high2half(ra[0])},
                  {__low2half(ra[1]), __high2half(ra[1])},
                  {__low2half(ra[2]), __high2half(ra[2])},
                  {__low2half(ra[3]), __high2half(ra[3])}};

  // 提取 rb 中的 half 数据
  half b[2][2] = {{__low2half(rb[0]), __high2half(rb[0])},
                  {__low2half(rb[1]), __high2half(rb[1])}};

  // 计算 C = A * B + C
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < 2; ++k) {
        sum += __half2float(a[i][k]) * __half2float(b[k][j]);
      }
      c[i * 2 + j] += sum; // 累加到 C
    }
  }
}

#define WARP_SIZE 32
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <const int BM, const int BN>
__global__ void mma_m16n8k16_ptx(half *A, half *B, int m, int n, int k,
                                 float *C) {
  //块按warp划分后的warp的col数
  const int WARP_N = BN / 8;
  //当前线程所属的warp的序号
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
    // 从全局内存读取A的8个fp16
    a[0] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i, K)];
    a[1] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 1, K)];
    a[2] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i, K)];
    a[3] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 1, K)];
    a[4] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 8, K)];
    a[5] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 9, K)];
    a[6] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 8, K)];
    a[7] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 9, K)];

    // 从全局内存读取B的4个fp16
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

#if 0
    // mma指令计算 C=A*B+C
    if(LANE_ID == 0) {
      printf("%f %f %f %f\n", c[0], c[1], c[2], c[3]);
    }

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

    if(LANE_ID == 1) 
    {
      printf("T%d:\n\t%f %f\n\t%f %f\n\t%f %f\n\t%f %f\nB:\n\t%f %f\n\t%f %f\n###: %d ,%d\n###: %d ,%d\n###: %d ,%d\n###: %d ,%d\n",
              LANE_ID, 
              __half2float(ra[0].x), __half2float(ra[0].y),
              __half2float(ra[2].x), __half2float(ra[2].y),
              __half2float(ra[1].x), __half2float(ra[1].y),
              __half2float(ra[3].x), __half2float(ra[3].y),
              __half2float(rb[0].x), __half2float(rb[0].y),
              __half2float(rb[1].x), __half2float(rb[1].y),
              WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2,
              WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1,
              WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2,
              WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1
      );
    }
#else
    // C implementation for mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // 执行矩阵乘法并累加到C矩阵

    // __syncthreads();
    __shared__ half s_a[16 * 16];
    __shared__ half s_b[16 * 8];

    s_a[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i, K)] = a[0];
    s_a[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 1, K)] = a[1];
    s_a[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i, K)] = a[2];
    s_a[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 1, K)] = a[3];
    s_a[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 8, K)] = a[4];
    s_a[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 9, K)] = a[5];
    s_a[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 8, K)] = a[6];
    s_a[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 9, K)] = a[7];

    s_b[OFFSET(i + B_THREAD_ROW * 2, WARP_COL_OFFSET + B_THREAD_COL, N)] = b[0];
    s_b[OFFSET(i + B_THREAD_ROW * 2 + 1, WARP_COL_OFFSET + B_THREAD_COL, N)] = b[1];
    s_b[OFFSET(i + B_THREAD_ROW * 2 + 8, WARP_COL_OFFSET + B_THREAD_COL, N)] = b[2];
    s_b[OFFSET(i + B_THREAD_ROW * 2 + 9, WARP_COL_OFFSET + B_THREAD_COL, N)] = b[3];

    __syncthreads(); // 等待所有线程加载完毕
    int row, col;

    row = WARP_ROW_OFFSET + C_THREAD_ROW;
    col = WARP_COL_OFFSET + C_THREAD_COL * 2;
    for (int i = 0; i < 16; ++i) {
      c[0] += __half2float(s_a[row * 16 + i]) * __half2float(s_b[i * 8 + col]);
    }

    row = WARP_ROW_OFFSET + C_THREAD_ROW;
    col = WARP_COL_OFFSET + C_THREAD_COL * 2 + 1;
    for (int i = 0; i < 16; ++i) {
      c[1] += __half2float(s_a[row * 16 + i]) * __half2float(s_b[i * 8 + col]);
    }

    row = WARP_ROW_OFFSET + C_THREAD_ROW + 8;
    col = WARP_COL_OFFSET + C_THREAD_COL * 2;
    for (int i = 0; i < 16; ++i) {
      c[2] += __half2float(s_a[row * 16 + i]) * __half2float(s_b[i * 8 + col]);
    }

    row = WARP_ROW_OFFSET + C_THREAD_ROW + 8;
    col = WARP_COL_OFFSET + C_THREAD_COL * 2 + 1;
    for (int i = 0; i < 16; ++i) {
      c[3] += __half2float(s_a[row * 16 + i]) * __half2float(s_b[i * 8 + col]);
    }

    if(LANE_ID == 3) {
      printf("T%d:\n\t%f %f\n\t%f %f\n\t%f %f\n\t%f %f\nB:\n\t%f %f\n\t%f %f\n###: %d ,%d\n###: %d ,%d\n###: %d ,%d\n###: %d ,%d\n",
              LANE_ID, 
              __half2float(ra[0].x), __half2float(ra[0].y),
              __half2float(ra[2].x), __half2float(ra[2].y),
              __half2float(ra[1].x), __half2float(ra[1].y),
              __half2float(ra[3].x), __half2float(ra[3].y),
              __half2float(rb[0].x), __half2float(rb[0].y),
              __half2float(rb[1].x), __half2float(rb[1].y),
              WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2,
              WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1,
              WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2,
              WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1
      );
    }

#endif
  }
  // 将mma的结果c写回到结果矩阵C中
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2, N)] = c[0];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1, N)] = c[1];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2, N)] = c[2];
  C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1, N)] = c[3];
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


void test(void) {

  half *d_A, *d_B;
  float *d_C;
  half h_A[M * K], h_B[K * N];
  float h_C[M * N];
  float h_C_ref[M * N];

  initialize_matrices(h_A, h_B, h_C);

  print_matrix(h_A, M, K);

  printf("matrix B: %d %d\n", K, N);
  print_matrix(h_B, K, N);

  matrix_multiplication_cpu(h_A, h_B, h_C_ref, M, N, K);

  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

  const int BM = 16;
  const int BN = 8;
  const int WARP_NUM = (BM * BN) / (16 * 8);
  dim3 block_dim(WARP_NUM * 32);
  dim3 grid_dim(1, 1);

  mma_m16n8k16_ptx<BM, BN><<<grid_dim, block_dim>>>(d_A, d_B, M, N, K, d_C);
  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Validate results
  bool correct = true;

  std::cout << "Matrix h_C:" << std::endl;
  for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
          std::cout << h_C[i * N + j] << " ";
      }
      std::cout << std::endl;
  }



  std::cout << "\nReference Result (CPU):" << std::endl;
  for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
          std::cout << h_C_ref[i * N + j] << " ";
      }
      std::cout << std::endl;
  }



  for (int i = 0; i < M * N; i++) {
    if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) {
      correct = false;
      break;
    }
  }

  std::cout << "Test " << (correct ? "PASSED" : "FAILED") << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

int main() {

  test();
  return 0;
}
