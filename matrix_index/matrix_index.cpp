#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <cstdlib>
#include <fenv.h> 

#define MATRIX_SIZE 512

void matrix_exponential(const double* A, double* expAt, int n, double delt, int N);
bool read_matrix_from_file(const std::string& filename, std::vector<double>& matrix, int& n);
void matrix_multiply(const double* A, const double* B, double* C, int n);
void matrix_add(double* A, const double* B, int n);
void matrix_square(const double* A, double* C, int n);

int main() {
    // 设置多线程，MKL库可以利用多核
    mkl_set_num_threads(4);


    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);


    auto program_start = std::chrono::high_resolution_clock::now();

  
    std::string matrixAFile = "matrixA_512.txt";

  
    std::vector<double> A;
    int nA;

    if (!read_matrix_from_file(matrixAFile, A, nA)) {
        std::cerr << "读取矩阵时出错" << std::endl;
        return -1;
    }

  
    if (nA != MATRIX_SIZE) {
        std::cerr << "矩阵的维度不是512阶" << std::endl;
        return -1;
    }

    int n = nA;  


    double* expAt = new (std::nothrow) double[n * n];
    if (!expAt) {
        std::cerr << "内存分配失败" << std::endl;
        return -1;
    }

 
    double delt = 0.0000001;   // 时间步长
    int N = 20;               // 精细积分法迭代次数


    auto start = std::chrono::high_resolution_clock::now();

    matrix_exponential(A.data(), expAt, n, delt, N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "矩阵指数计算时间: " << elapsed.count() << " 秒" << std::endl;

 
    std::cout << "exp(A * delt) 的前4x4块：\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << expAt[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] expAt;

    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> program_elapsed = program_end - program_start;
    std::cout << "程序总运行时间: " << program_elapsed.count() << " 秒" << std::endl;

    return 0;
}


void matrix_exponential(const double* A, double* expAt, int n, double delt, int N) {
    double* T_a = new double[n * n];
    double* temp = new double[n * n];

    if (!T_a || !temp) {
        std::cerr << "内存分配失败！" << std::endl;
        return;
    }

 
    cblas_dcopy(n * n, A, 1, T_a, 1);  
    cblas_dscal(n * n, delt, T_a, 1); 

    // temp = A * delt / 2
    cblas_dcopy(n * n, T_a, 1, temp, 1); 
    cblas_dscal(n * n, 0.5, temp, 1);   
    // T_a = A * delt * (I + (A * delt) / 2)
    cblas_daxpy(n * n, 1.0, temp, 1, T_a, 1); 
    // 递推计算 T_a
    for (int iter = 0; iter < N; iter++) {

        matrix_square(T_a, temp, n);

      
        cblas_dscal(n * n, 2.0, T_a, 1);      // T_a = 2 * T_a
        cblas_daxpy(n * n, 1.0, temp, 1, T_a, 1);
    }


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
            n = col_count;  
        else if (col_count != n) {
            std::cerr << "文件格式不正确，列数不匹配" << std::endl;
            return false;
        }
        row_count++;
    }

    matrix = std::move(temp_matrix);  
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
