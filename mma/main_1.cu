#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>


#define M 4
#define N 4
#define K 4

__global__ void mm_naive(float* A, float* B, int m, int n, int k, float* C) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (r < M && c < N) {
        float sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sum += A[r * K + i] * B[i * N + c];
        }
        C[r * N + c] = sum;
    }
}

__global__ void mm_naive_ptx(float* A, float* B, int m, int n, int k, float* C) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < M && c < N) {
    // 声明寄存器变量f1,f2,f3,并将f1置为0
    asm(".reg .f32 f1, f2, f3;\n"
        "mov.f32 f1, 0.0;\n" ::);
    for (int i = 0; i < K; ++i) {
      // 从全局内存中读取数据到寄存器中，并进行乘法+加法运算
      asm("ld.global.f32 f2, [%0];\n"
          "ld.global.f32 f3, [%1];\n"
          "fma.rn.f32 f1, f2, f3, f1;\n"
          :
          : "l"(&A[r * K + i]), "l"(&B[i * N + c]));
    }
    //将结果写会到全局内存中
    asm("st.global.f32 [%0], f1;\n" ::"l"(&C[r * N + c]));
  }
}

void matrix_multiplication_cpu(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


void initialize_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand() % 10);
    }
}

void print_matrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}



// #define WARP_SIZE 32
// #define OFFSET(row, col, ld) ((row) * (ld) + (col))

// template <const int BM, const int BN>
// __global__ void mma_m16n8k16_ptx(half* A, half* B, int m, int n, int k, float* C) {
//   //块按warp划分后的warp的col数 
//   const int WARP_N = BN / 8;
//   //当前warp在块中的序号
//   const int WRAP_ID = threadIdx.x / WARP_SIZE;
//   //当前线程在warp中的序号0~31
//   const int LANE_ID = threadIdx.x % WARP_SIZE;
//   //当前warp的row序号
//   const int WARP_ROW = WRAP_ID / WARP_N;
//   //当前warp的col序号
//   const int WARP_COL = WRAP_ID % WARP_N;
//   //当前线程在warp中的对于A/B/C的row和col。
//   const int A_THREAD_ROW = LANE_ID / 4;
//   const int A_THREAD_COL = LANE_ID % 4;
//   const int B_THREAD_ROW = LANE_ID % 4;
//   const int B_THREAD_COL = LANE_ID / 4;
//   const int C_THREAD_ROW = LANE_ID / 4;
//   const int C_THREAD_COL = LANE_ID % 4;

//   // C = A*B + C,这里先初始化C全部初始化为0.
//   float c[4] = {0.0, 0.0, 0.0, 0.0};
//   // 从A中读取8个fp16。
//   half a[8];
//   // 从B中读取4个fp16
//   half b[4];
//   // 转换为mma指令中的A的4个32位输入数据。
//   half2 ra[4];
//   // 转换为mma指令中的B的2个32位输入数据。
//   half2 rb[2];

//   //每个warp计算16*8的结果矩阵。
//   const int WARP_ROW_OFFSET = WARP_ROW * 16 + blockIdx.y * BM;
//   const int WARP_COL_OFFSET = WARP_COL * 8 + blockIdx.x * BN;

//   for (int i = 0; i < K; i += 16) {
//     // 从全局内存读取A的8个fp16，这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
//     a[0] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i, K)];
//     a[1] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 1, K)];
//     a[2] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i, K)];
//     a[3] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 1, K)];
//     a[4] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 8, K)];
//     a[5] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW, A_THREAD_COL * 2 + i + 9, K)];
//     a[6] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 8, K)];
//     a[7] = A[OFFSET(WARP_ROW_OFFSET + A_THREAD_ROW + 8, A_THREAD_COL * 2 + i + 9, K)];
   
//     // 从全局内存读取B的4个fp16，这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
//     b[0] = B[OFFSET(i + B_THREAD_ROW * 2, WARP_COL_OFFSET + B_THREAD_COL, N)];
//     b[1] = B[OFFSET(i + B_THREAD_ROW * 2 + 1, WARP_COL_OFFSET + B_THREAD_COL, N)];
//     b[2] = B[OFFSET(i + B_THREAD_ROW * 2 + 8, WARP_COL_OFFSET + B_THREAD_COL, N)];
//     b[3] = B[OFFSET(i + B_THREAD_ROW * 2 + 9, WARP_COL_OFFSET + B_THREAD_COL, N)];
    
//     // mma指令中的A的4个32位数据
//     ra[0] = __halves2half2(a[0], a[1]);
//     ra[1] = __halves2half2(a[2], a[3]);
//     ra[2] = __halves2half2(a[4], a[5]);
//     ra[3] = __halves2half2(a[6], a[7]);
    
//     // mma指令中的B的2个32位数据
//     rb[0] = __halves2half2(b[0], b[1]);
//     rb[1] = __halves2half2(b[2], b[3]);
    
//     // mma指令计算 C=A*B+C
//     asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
//         " { %0, %1, %2, %3 }, "
//         " { %4, %5, %6, %7 }, "
//         " { %8, %9 }, "
//         " { %0, %1, %2, %3 };"
//         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
//         : "r"(*(reinterpret_cast<int*>(&ra[0]))),
//           "r"(*(reinterpret_cast<int*>(&ra[1]))),
//           "r"(*(reinterpret_cast<int*>(&ra[2]))),
//           "r"(*(reinterpret_cast<int*>(&ra[3]))),
//           "r"(*(reinterpret_cast<int*>(&rb[0]))),
//           "r"(*(reinterpret_cast<int*>(&rb[1]))));

//     __syncthreads();
//   }
//   // 将mma的结果c写回到结果矩阵C中，这里也可以用asm ptx来替换，但为了代码阅读，这里忽略。
//   C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2, N)] = c[0];
//   C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1, N)] = c[1];
//   C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2, N)] = c[2];
//   C[OFFSET(WARP_ROW_OFFSET + C_THREAD_ROW + 8, WARP_COL_OFFSET + C_THREAD_COL * 2 + 1, N)] = c[3];
// }



int main() {
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);
    h_C_ref = (float*)malloc(size_C);
    
    initialize_matrix(h_A, M, K);
    initialize_matrix(h_B, K, N);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 blockSize(2, 2);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    //mm_naive<<<gridSize, blockSize>>>(d_A, d_B, M, N, K, d_C);

    mm_naive_ptx<<<gridSize, blockSize>>>(d_A, d_B, M, N, K, d_C);


    // const int BM = 32;
    // const int BN = 32;
    // const int WARP_NUM = (BM * BN) / (16 * 8);
    // const dim3 block_dim(WARP_NUM * WARP_SIZE);
    // const dim3 grid_dim(N / BN, M / BM);
    // mma_m16n8k16_ptx<BM, BN><<<grid_dim, block_dim>>>(d_A, d_B, M, N, K, d_C);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    matrix_multiplication_cpu(h_A, h_B, h_C_ref, M, N, K);
    
    std::cout << "Matrix A:" << std::endl;
    print_matrix(h_A, M, K);
    std::cout << "Matrix B:" << std::endl;
    print_matrix(h_B, K, N);
    std::cout << "Result from GPU:" << std::endl;
    print_matrix(h_C, M, N);
    std::cout << "Reference Result (CPU):" << std::endl;
    print_matrix(h_C_ref, M, N);
    
    bool match = true;
    for (int i = 0; i < M * N; ++i) {
        if (abs(h_C[i] - h_C_ref[i]) > 1e-4) {
            match = false;
            break;
        }
    }
    
    std::cout << (match ? "Test PASSED" : "Test FAILED") << std::endl;
    
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
