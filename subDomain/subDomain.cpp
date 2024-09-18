#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mkl.h>

// 定义子域大小
#define SUBDOMAIN_SIZE 256  // 假设划分子域的大小为256

// 函数声明
void matrix_exponential_subdomain(const double* A, double* expAt, int subdomain_size, double delt, int N);
bool read_matrix_from_file(const std::string& filename, std::vector<double>& matrix, int& n);
void matrix_multiply(const double* A, const double* B, double* C, int n);
void matrix_add(double* A, const double* B, int n);
void matrix_square(const double* A, double* C, int n);

int main() {
    // 控制浮点异常
    unsigned int current_control;
    _controlfp_s(&current_control, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW | _EM_ZERODIVIDE | _EM_INVALID, _MCW_EM);

    // 记录程序开始时间
    auto program_start = std::chrono::high_resolution_clock::now();

    // 读取矩阵文件
    std::string matrixAFile = "matrixA_1024.txt"; // 假设这是1024阶的矩阵文件

    // 读取矩阵 A
    std::vector<double> A;
    int nA;

    if (!read_matrix_from_file(matrixAFile, A, nA)) {
        std::cerr << "读取矩阵时出错" << std::endl;
        return -1;
    }

    // 子域处理
    int subdomain_size = SUBDOMAIN_SIZE;
    int n = std::ceil(static_cast<double>(nA) / subdomain_size) * subdomain_size; // 扩展矩阵

    // 扩展矩阵 A 以适应子域法
    std::vector<double> A_extended(n * n, 0.0);
    for (int i = 0; i < nA; i++) {
        for (int j = 0; j < nA; j++) {
            A_extended[i * n + j] = A[i * nA + j];
        }
    }

    // 动态分配指数矩阵 expAt
    double* expAt = new (std::nothrow) double[n * n];
    if (!expAt) {
        std::cerr << "内存分配失败" << std::endl;
        return -1;
    }

    // 时间步长和迭代次数
    double delt = 0.01;
    int N = 10;

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 计算子域内的矩阵指数
    matrix_exponential_subdomain(A_extended.data(), expAt, subdomain_size, delt, N);

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 输出求解结果和运行时间
    std::cout << "子域法矩阵指数计算时间: " << elapsed.count() << " 秒" << std::endl;

    // 释放内存
    delete[] expAt;

    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> program_elapsed = program_end - program_start;
    std::cout << "程序总运行时间: " << program_elapsed.count() << " 秒" << std::endl;

    return 0;
}

// 实现子域矩阵指数的计算，使用精细积分法
void matrix_exponential_subdomain(const double* A, double* expAt, int subdomain_size, double delt, int N) {
    double* T_a = new double[subdomain_size * subdomain_size];
    double* temp = new double[subdomain_size * subdomain_size];

    if (!T_a || !temp) {
        std::cerr << "内存分配失败！" << std::endl;
        return;
    }

    // 初始化 T_a = A * delt * (I + (A * delt) / 2)
    cblas_dcopy(subdomain_size * subdomain_size, A, 1, T_a, 1);  // T_a = A
    cblas_dscal(subdomain_size * subdomain_size, delt, T_a, 1);  // T_a = A * delt

    // temp = A * delt / 2
    cblas_dcopy(subdomain_size * subdomain_size, T_a, 1, temp, 1);  // temp = A * delt
    cblas_dscal(subdomain_size * subdomain_size, 0.5, temp, 1);     // temp = A * delt / 2

    // T_a = A * delt * (I + (A * delt) / 2)
    cblas_daxpy(subdomain_size * subdomain_size, 1.0, temp, 1, T_a, 1);  // T_a = A * delt + (A * delt / 2)

    // 递推计算 T_a
    for (int iter = 0; iter < N; iter++) {
        // temp = T_a^2
        matrix_square(T_a, temp, subdomain_size);

        // T_a = 2 * T_a + T_a^2
        cblas_dscal(subdomain_size * subdomain_size, 2.0, T_a, 1);      // T_a = 2 * T_a
        cblas_daxpy(subdomain_size * subdomain_size, 1.0, temp, 1, T_a, 1);
    }

    // 最终计算 expAt = I + T_a
    cblas_dcopy(subdomain_size * subdomain_size, T_a, 1, expAt, 1);  // expAt = T_a
    double* I = new double[subdomain_size * subdomain_size];
    if (!I) {
        std::cerr << "内存分配失败！" << std::endl;
        return;
    }

    for (int i = 0; i < subdomain_size * subdomain_size; i++) {
        I[i] = (i % (subdomain_size + 1)) == 0 ? 1.0 : 0.0;  // 单位矩阵 I
    }

    matrix_add(expAt, I, subdomain_size);  // expAt = I + T_a

    delete[] T_a;
    delete[] temp;
    delete[] I;
}

// 从文件读取矩阵
bool read_matrix_from_file(const std::string& filename, std::vector<double>& matrix, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::vector<double> temp_matrix;
    int row_count = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double value;
        int col_count = 0;
        while (iss >> value) {
            temp_matrix.push_back(value);
            col_count++;
        }
        if (row_count == 0) {
            n = col_count;  // 确定矩阵的维度
        } else if (col_count != n) {
            std::cerr << "文件格式不正确，列数不匹配" << std::endl;
            return false;
        }
        row_count++;
    }

    matrix = std::move(temp_matrix);  // 将临时矩阵赋值给目标矩阵
    return true;
}

// 矩阵乘法：C = A * B (n x n 矩阵)
void matrix_multiply(const double* A, const double* B, double* C, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);
}

// 矩阵加法：A = A + B (n x n 矩阵)
void matrix_add(double* A, const double* B, int n) {
    cblas_daxpy(n * n, 1.0, B, 1, A, 1);
}

// 矩阵平方：C = A^2 (n x n 矩阵)
void matrix_square(const double* A, double* C, int n) {
    matrix_multiply(A, A, C, n);
}
