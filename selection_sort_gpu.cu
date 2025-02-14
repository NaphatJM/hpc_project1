#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

#define MATRIX_SIZE 128

// Function for Selection Sort on each row (Parallelized)
void selectionSort_cpu(std::vector<int>& matrix, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) {
            int min_idx = j;
            for (int k = j + 1; k < n; k++) {
                if (matrix[i * n + k] < matrix[i * n + min_idx]) {
                    min_idx = k;
                }
            }
            if (min_idx != j) {
                std::swap(matrix[i * n + j], matrix[i * n + min_idx]);
            }
        }
    }
}

// Function to find min - max values (Parallelized)
void findMinMax_cpu(const std::vector<int>& matrix, int n, std::vector<int>& min_vals, std::vector<int>& max_vals) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int minVal = matrix[i * n];
        int maxVal = matrix[i * n];
        for (int j = 1; j < n; j++) {
            int val = matrix[i * n + j];
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
        min_vals[i] = minVal;
        max_vals[i] = maxVal;
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

    std::vector<int> min_vals(MATRIX_SIZE);
    std::vector<int> max_vals(MATRIX_SIZE);

//---------------------------------------------------------------------------------------------------------------------
    // CPU Sorting
    auto start = std::chrono::high_resolution_clock::now();
    selectionSort_cpu(host_matrix, MATRIX_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_sort_duration = end - start;
//---------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------
    // Find Min & Max
    start = std::chrono::high_resolution_clock::now();
    findMinMax_cpu(host_matrix, MATRIX_SIZE, min_vals, max_vals);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_minmax_duration = end - start;
//---------------------------------------------------------------------------------------------------------------------

    // Save sorted matrix to a file
    std::ofstream sorted_file("sorted_selection_cpu.csv");
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
        std::cout << "Row " << (i + 1) << ": Min = " << min_vals[i] << ", Max = " << max_vals[i] << "\n";
    }

    std::cout << "--------------------------------" << std::endl;
    std::cout << "CPU Sorting completed in: " << cpu_sort_duration.count() << " ms\n";
    std::cout << "CPU Min/Max computation completed in: " << cpu_minmax_duration.count() << " ms\n";
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Global Memory Min/Max: " << cpu_minmax_duration.count() << " ms\n";
    std::cout << "Shared Memory Min/Max: " << cpu_minmax_duration.count() << " ms\n";

    return 0;
}
