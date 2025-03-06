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
