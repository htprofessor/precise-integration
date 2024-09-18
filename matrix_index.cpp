#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <cstdlib>
#include <fenv.h> 

// 定义处理512阶矩阵的常量
#define MATRIX_SIZE 512

void matrix_exponential(const double* A, double* expAt, int n, double delt, int N);
bool read_matrix_from_file(const std::string& filename, std::vector<double>& matrix, int& n);
void matrix_multiply(const double* A, const double* B, double* C, int n);
void matrix_add(double* A, const double* B, int n);
void matrix_square(const double* A, double* C, int n);

int main() {
    // 设置多线程，MKL库可以利用多核
    mkl_set_num_threads(4);

    // 启用浮点异常控制（无效操作、除以零、溢出）
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

    // 记录程序开始时间
    auto program_start = std::chrono::high_resolution_clock::now();

    // 读取512阶矩阵文件
    std::string matrixAFile = "matrixA_512.txt";

    // 读取矩阵 A
    std::vector<double> A;
    int nA;

    if (!read_matrix_from_file(matrixAFile, A, nA)) {
        std::cerr << "读取矩阵时出错" << std::endl;
        return -1;
    }

    // 确认矩阵是512阶的
    if (nA != MATRIX_SIZE) {
        std::cerr << "矩阵的维度不是512阶" << std::endl;
        return -1;
    }

    int n = nA;  // 矩阵的维度

    // 动态分配指数矩阵 expAt
    double* expAt = new (std::nothrow) double[n * n];
    if (!expAt) {
        std::cerr << "内存分配失败" << std::endl;
        return -1;
    }

    // 时间步长和迭代次数
    double delt = 0.0000001;   // 时间步长
    int N = 20;               // 精细积分法迭代次数

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 计算矩阵指数
    matrix_exponential(A.data(), expAt, n, delt, N);

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 输出求解结果和运行时间
    std::cout << "矩阵指数计算时间: " << elapsed.count() << " 秒" << std::endl;

    // 输出部分结果以验证正确性
    std::cout << "exp(A * delt) 的前4x4块：\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << expAt[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放内存
    delete[] expAt;

    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> program_elapsed = program_end - program_start;
    std::cout << "程序总运行时间: " << program_elapsed.count() << " 秒" << std::endl;

    return 0;
}

// 实现矩阵指数的计算，使用精细积分法
void matrix_exponential(const double* A, double* expAt, int n, double delt, int N) {
    double* T_a = new double[n * n];
    double* temp = new double[n * n];

    if (!T_a || !temp) {
        std::cerr << "内存分配失败！" << std::endl;
        return;
    }

    // 初始化 T_a = A * delt * (I + (A * delt) / 2)
    cblas_dcopy(n * n, A, 1, T_a, 1);  // T_a = A
    cblas_dscal(n * n, delt, T_a, 1);  // T_a = A * delt

    // temp = A * delt / 2
    cblas_dcopy(n * n, T_a, 1, temp, 1);  // temp = A * delt
    cblas_dscal(n * n, 0.5, temp, 1);     // temp = A * delt / 2

    // T_a = A * delt * (I + (A * delt) / 2)
    cblas_daxpy(n * n, 1.0, temp, 1, T_a, 1);  // T_a = A * delt + (A * delt / 2)

    // 递推计算 T_a
    for (int iter = 0; iter < N; iter++) {
        // temp = T_a^2
        matrix_square(T_a, temp, n);

        // T_a = 2 * T_a + T_a^2
        cblas_dscal(n * n, 2.0, T_a, 1);      // T_a = 2 * T_a
        cblas_daxpy(n * n, 1.0, temp, 1, T_a, 1);
    }

    // 最终计算 expAt = I + T_a
    cblas_dcopy(n * n, T_a, 1, expAt, 1);  // expAt = T_a
    double* I = new double[n * n];
    if (!I) {
        std::cerr << "内存分配失败！" << std::endl;
        return;
    }

    for (int i = 0; i < n * n; i++) I[i] = (i % (n + 1)) == 0 ? 1.0 : 0.0;  // 单位矩阵 I

    matrix_add(expAt, I, n);  // expAt = I + T_a

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
        }
        else if (col_count != n) {
            std::cerr << "文件格式不正确，列数不匹配" << std::endl;
            return false;
        }
        row_count++;
    }

    matrix = std::move(temp_matrix);  // 将临时矩阵赋值给目标矩阵
    return true;
}

void matrix_multiply(const double* A, const double* B, double* C, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);
}

void matrix_add(double* A, const double* B, int n) {
    cblas_daxpy(n * n, 1.0, B, 1, A, 1);
}

void matrix_square(const double* A, double* C, int n) {
    matrix_multiply(A, A, C, n);
}
