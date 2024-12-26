#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.cuh"
// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define A_ROWS 1024
#define A_COLS 32
#define B_ROWS 32
#define B_COLS 32
#define C_ROWS 1024
#define C_COLS 32
#define TILE_SIZE 32

__global__ void matrixMultiplyKernelSHMEM(float* A, float* B, float* C) {
  __shared__ float BTileSHMEM[TILE_SIZE][TILE_SIZE];

  int b_row = threadIdx.x;
  int row = blockIdx.y * blockDim.x + blockIdx.x;

  if (b_row < B_ROWS) {
    for (int j = 0; j < B_COLS; j++) {
      BTileSHMEM[b_row][j] = B[b_row * B_COLS + j];
    }
  }

  __syncthreads();

  if (b_row < B_ROWS) {
    float a_element = A[row * A_COLS + b_row];
    for (int j = 0; j < B_COLS; j++) {
      int b_col = (threadIdx.x + j) % B_COLS;
      C[blockIdx.x * C_COLS + b_col] += a_element * BTileSHMEM[b_row][b_col];
    }
  }
            
}


__global__ void matrixMultiplyKernel(float* A, float* B, float* C) {
  // Each thread handles one row of B
  int b_row = threadIdx.x;
  int row = blockIdx.y * blockDim.x + blockIdx.x;
  if (b_row < B_ROWS) {
    float a_element = A[row * A_COLS + b_row];
    for (int j = 0; j < B_COLS; j++) {
      int b_col = (threadIdx.x + j) % B_COLS;
      C[blockIdx.x * C_COLS + b_col] += a_element * B[b_row * B_COLS + b_col];
    }
  }
}

void matrixMultiplyCPU(float* A, float* B, float* C) {
  for (int row = 0; row < A_ROWS; row++) {
    for (int col = 0; col < B_COLS; col++) {
      float sum = 0.0f;
      for (int i = 0; i < A_COLS; i++) {
        sum += A[row * A_COLS + i] * B[i * B_COLS + col];
      }
      C[row * C_COLS + col] = sum;
    }
  }
}

// Verify results
bool verifyResults(float* gpuResult, float* cpuResult, int size, float tolerance = 1e-3) {
  for (int i = 0; i < size; i++) {
    if (fabs(gpuResult[i] - cpuResult[i]) > tolerance) {
      printf("Mismatch at position %d: GPU = %f, CPU = %f\n", 
          i, gpuResult[i], cpuResult[i]);
      return false;
    }
  }
  return true;
}

int main() {
  float *h_A = (float*)malloc(A_ROWS * A_COLS * sizeof(float));
  float *h_B = (float*)malloc(B_ROWS * B_COLS * sizeof(float));
  float *h_C = (float*)malloc(C_ROWS * C_COLS * sizeof(float));
  float *h_C_cpu = (float*)malloc(C_ROWS * C_COLS * sizeof(float));

  randomize_matrix(h_A, A_ROWS, A_COLS);
  randomize_matrix(h_B, B_ROWS, B_COLS);

  float *d_A, *d_B, *d_C;
  cudaCheckError(cudaMalloc(&d_A, A_ROWS * A_COLS * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_B, B_ROWS * B_COLS * sizeof(float)));
  cudaCheckError(cudaMalloc(&d_C, C_ROWS * C_COLS * sizeof(float)));

  cudaCheckError(cudaMemcpy(d_A, h_A, A_ROWS * A_COLS * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_B, h_B, B_ROWS * B_COLS * sizeof(float), cudaMemcpyHostToDevice));

  cudaFree(0);
  cudaMemset(d_C, 0, C_ROWS * C_COLS * sizeof(float));

  dim3 gridDim(1024);
  dim3 blockDim(32);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  matrixMultiplyKernelSHMEM<<<gridDim, blockDim>>>(d_A, d_B, d_C);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaCheckError(cudaMemcpy(h_C, d_C, C_ROWS * C_COLS * sizeof(float), cudaMemcpyDeviceToHost));

  float time = 0.0f;
  cudaEventElapsedTime(&time, start, stop);

  std::cout << "GPU Timing: " << time << " ms" << std::endl;
  matrixMultiplyCPU(h_A, h_B, h_C_cpu);

  bool correct = verifyResults(h_C, h_C_cpu, C_ROWS * C_COLS);
  printf("Matrix multiplication %s\n", correct ? "PASSED" : "FAILED");

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_cpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
