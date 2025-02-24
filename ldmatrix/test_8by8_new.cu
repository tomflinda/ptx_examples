#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void test_ldmatrix(int *output) {
    // Declare shared memory
    __shared__ uint16_t smem[8 * 8];  // 8x8 matrix in shared memory (b16 format)
    
    // Initialize shared memory (each thread initializes its own part)
    int tid = threadIdx.x;
    if (tid < 64) {
        smem[tid] = tid;  // Fill shared memory with some values
    }
    __syncthreads();  // Ensure all threads initialize before loading

    // Register storage for loaded data
    int a[1];
   
    int aTile_index = tid % 8 * 8;

    uint32_t smem1 = __cvta_generic_to_shared(smem+aTile_index);
    // Load data using inline PTX
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 { %0 }, [%1];\n"
        : "=r"(a[0])         // Output register
        : "r"(smem1)          // Input shared memory pointer
    );

    // Write result to global memory for verification
    output[tid] = a[0];
}

int main() {
    const int size = 32 * sizeof(int);
    int *d_output, *h_output;

    // Allocate device and host memory
    cudaMalloc(&d_output, size);
    h_output = (int*)malloc(size);

    // Launch the kernel with 32 threads (for an 8x8 matrix)
    test_ldmatrix<<<1, 32>>>(d_output);
    cudaDeviceSynchronize();
    // Copy results back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Loaded Matrix Data: \n";
    for (int i = 0; i < 32; i++) {
        printf("0x%x ", h_output[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }

    // Cleanup
    cudaFree(d_output);
    free(h_output);

    return 0;
}




