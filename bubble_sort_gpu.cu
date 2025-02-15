#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <chrono>

#define MATRIX_SIZE 128

// CUDA Kernel for Bubble Sort using Global Memory
__global__ void bubbleSort_global(int* d_matrix, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n) {
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                int index1 = row * n + j;
                int index2 = row * n + j + 1;
                
                if (d_matrix[index1] > d_matrix[index2]) {
                    int temp = d_matrix[index1];
                    d_matrix[index1] = d_matrix[index2];
                    d_matrix[index2] = temp;
                }
            }
        }
    }
}

// CUDA Kernel for Bubble Sort using Shared Memory
__global__ void bubbleSort_shared(int* d_matrix, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared_row[MATRIX_SIZE];
    
    if (row < n) {
        int row_start = row * n;
        
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            shared_row[i] = d_matrix[row_start + i];
        }
        __syncthreads();

        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (shared_row[j] > shared_row[j + 1]) {
                    int temp = shared_row[j];
                    shared_row[j] = shared_row[j + 1];
                    shared_row[j + 1] = temp;
                }
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            d_matrix[row_start + i] = shared_row[i];
        }
    }
}

// CUDA Kernel to find min - max values using Global Memory
__global__ void findMinMax_global(int* d_matrix, int* d_min, int* d_max, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n) {
        int minVal = d_matrix[row * n];
        int maxVal = d_matrix[row * n];

        for (int j = 1; j < n; j++) {
            int val = d_matrix[row * n + j];
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
        
        d_min[row] = minVal;
        d_max[row] = maxVal;
    }
}

// CUDA Kernel to find min - max values using Shared Memory
__global__ void findMinMax_shared(int* d_matrix, int* d_min, int* d_max, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared_row[MATRIX_SIZE];
    
    if (row < n) {
        int row_start = row * n;
        
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            shared_row[i] = d_matrix[row_start + i];
        }
        __syncthreads();
        
        int minVal = shared_row[0];
        int maxVal = shared_row[0];
        for (int i = 1; i < n; i++) {
            if (shared_row[i] < minVal) minVal = shared_row[i];
            if (shared_row[i] > maxVal) maxVal = shared_row[i];
        }
        
        d_min[row] = minVal;
        d_max[row] = maxVal;
    }
}

int main() {
    std::ifstream file("random_matrix_128x128.csv");
    
    std::vector<int> host_matrix;
    int value;
    
    while (file >> value) {
        host_matrix.push_back(value);
        if (file.peek() == ',') file.ignore();
    }
    file.close();
    
    int size = MATRIX_SIZE * MATRIX_SIZE;
    int *d_matrix, *d_min, *d_max;

    cudaMalloc(&d_matrix, size * sizeof(int));
    cudaMalloc(&d_min, MATRIX_SIZE * sizeof(int));
    cudaMalloc(&d_max, MATRIX_SIZE * sizeof(int));
    
    cudaMemcpy(d_matrix, host_matrix.data(), size * sizeof(int), cudaMemcpyHostToDevice);

//---------------------------------------------------------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();
    bubbleSort_global<<<MATRIX_SIZE, 1>>>(d_matrix, MATRIX_SIZE);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> global_sort_duration = end - start;
//---------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------
    start = std::chrono::high_resolution_clock::now();
    bubbleSort_shared<<<MATRIX_SIZE, 1>>>(d_matrix, MATRIX_SIZE);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> shared_sort_duration = end - start;
//---------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------
    start = std::chrono::high_resolution_clock::now();
    findMinMax_global<<<MATRIX_SIZE, 1>>>(d_matrix, d_min, d_max, MATRIX_SIZE);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> global_minmax_duration = end - start;
//---------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------
    start = std::chrono::high_resolution_clock::now();
    findMinMax_shared<<<MATRIX_SIZE, 1>>>(d_matrix, d_min, d_max, MATRIX_SIZE);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> shared_minmax_duration = end - start;
//---------------------------------------------------------------------------------------------------------------------

    cudaMemcpy(host_matrix.data(), d_matrix, size * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> host_min(MATRIX_SIZE);
    std::vector<int> host_max(MATRIX_SIZE);
    cudaMemcpy(host_min.data(), d_min, MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_max.data(), d_max, MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    std::ofstream sorted_file("sorted_bubble_gpu.csv");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            sorted_file << host_matrix[i * MATRIX_SIZE + j];
            if (j < MATRIX_SIZE - 1) sorted_file << ",";
        }
        sorted_file << "\n";
    }
    sorted_file.close();

    std::cout << "Row-wise Min/Max values:\n";
    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::cout << "Row " << (i + 1) << ": Min = " << host_min[i] << ", Max = " << host_max[i] << "\n";
    }
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Global Memory Sorting: " << global_sort_duration.count() << " ms\n";
    std::cout << "Shared Memory Sorting: " << shared_sort_duration.count() << " ms\n";
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Global Memory Min/Max: " << global_minmax_duration.count() << " ms\n";
    std::cout << "Shared Memory Min/Max: " << shared_minmax_duration.count() << " ms\n";

    cudaFree(d_matrix);
    cudaFree(d_min);
    cudaFree(d_max);

    return 0;
}
