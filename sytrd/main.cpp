#include "householder_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iomanip>
#include <sstream>

// Function to print benchmark results table header
void print_table_header(bool validate, bool comparison_mode) {
    if (comparison_mode) {
        printf("Matrix Size | Block Size | Blocked Time (ms) | Unblocked Time (ms) | Speedup");
        if (validate) {
            printf(" | Blocked Valid | Unblocked Valid\n");
            printf("-------------------------------------------------------------------------------------\n");
        } else {
            printf("\n");
            printf("-----------------------------------------------------------------------\n");
        }
    } else {
        printf("Matrix Size | Algorithm      | Time (ms)");
        if (validate) {
            printf(" | Validation\n");
            printf("-----------------------------------------------\n");
        } else {
            printf("\n");
            printf("------------------------------------\n");
        }
    }
}

// Function to print benchmark results
void print_benchmark_result(const BenchmarkResult& result, const std::string& algorithm_name, bool validate) {
    if (result.block_size > 0) {
        printf("%11d | %-14s | %9.2f", result.matrix_size, algorithm_name.c_str(), result.avg_time_ms);
    } else {
        printf("%11d | %-14s | %9.2f", result.matrix_size, algorithm_name.c_str(), result.avg_time_ms);
    }
    
    if (validate) {
        printf(" | %10s", result.validation_result.c_str());
    }
    printf("\n");
}

// Function to print comparison results
void print_comparison_result(const BenchmarkResult& blocked_result, const BenchmarkResult& unblocked_result, bool validate) {
    double speedup = unblocked_result.avg_time_ms / blocked_result.avg_time_ms;
    
    printf("%11d | %10d | %17.2f | %19.2f | %7.2fx",
           blocked_result.matrix_size,
           blocked_result.block_size,
           blocked_result.avg_time_ms,
           unblocked_result.avg_time_ms,
           speedup);
    
    if (validate) {
        printf(" | %13s | %15s", 
               blocked_result.validation_result.c_str(),
               unblocked_result.validation_result.c_str());
    }
    printf("\n");
}

// Function to print usage information
void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -n size1 size2 ...    Matrix sizes to test\n");
    printf("  -b block_size         Block size for blocked algorithm (default: 32)\n");
    printf("  -a algorithm          Algorithm to run: 'blocked', 'unblocked', or 'both' (default: both)\n");
    printf("  -v                    Enable validation\n");
    printf("  -i iterations         Number of benchmark iterations (default: 10)\n");
    printf("  -w warmup_runs        Number of warmup runs (default: 3)\n");
    printf("  -h                    Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -n 128 256 512 -a both -v\n", program_name);
    printf("  %s -n 1024 -b 64 -a blocked\n", program_name);
    printf("  %s -n 256 512 -a unblocked -i 20\n", program_name);
}

int main(int argc, char* argv[]) {
    std::vector<int> matrix_sizes;
    int block_size = 32;
    std::string algorithm = "both";
    bool validate = false;
    int iterations = 10;
    int warmup = 3;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                matrix_sizes.push_back(atoi(argv[++i]));
            }
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            algorithm = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Validate input arguments
    if (matrix_sizes.empty()) {
        printf("Error: No matrix sizes specified.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (algorithm != "blocked" && algorithm != "unblocked" && algorithm != "both") {
        printf("Error: Invalid algorithm '%s'. Use 'blocked', 'unblocked', or 'both'.\n\n", algorithm.c_str());
        print_usage(argv[0]);
        return 1;
    }

    // Print configuration information
    printf("Householder Tridiagonalization Benchmark\n");
    printf("=========================================\n");
    printf("Algorithm(s): %s\n", algorithm.c_str());
    if (algorithm == "blocked" || algorithm == "both") {
        printf("Block size: %d\n", block_size);
    }
    printf("Iterations: %d\n", iterations);
    printf("Warmup runs: %d\n", warmup);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Determine comparison mode
    bool comparison_mode = (algorithm == "both");
    
    // Print table header
    print_table_header(validate, comparison_mode);

    // Run benchmarks for each matrix size
    for (int n : matrix_sizes) {
        if (comparison_mode) {
            // Run both algorithms and compare
            BenchmarkResult blocked_result = benchmark_blocked_algorithm(n, block_size, validate, iterations, warmup);
            BenchmarkResult unblocked_result = benchmark_unblocked_algorithm(n, validate, iterations, warmup);
            print_comparison_result(blocked_result, unblocked_result, validate);
        } else if (algorithm == "blocked") {
            // Run only blocked algorithm
            BenchmarkResult result = benchmark_blocked_algorithm(n, block_size, validate, iterations, warmup);
            std::stringstream ss;
            ss << "Blocked (b=" << block_size << ")";
            print_benchmark_result(result, ss.str(), validate);
        } else if (algorithm == "unblocked") {
            // Run only unblocked algorithm
            BenchmarkResult result = benchmark_unblocked_algorithm(n, validate, iterations, warmup);
            print_benchmark_result(result, "Unblocked", validate);
        }
    }

    printf("\n");
    
    // Print summary information
    if (comparison_mode) {
        printf("Speedup = Unblocked Time / Blocked Time\n");
        printf("Speedup > 1.0 means blocked algorithm is faster\n");
        printf("Speedup < 1.0 means unblocked algorithm is faster\n");
    }

    return 0;
}
