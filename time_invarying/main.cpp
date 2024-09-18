#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <mkl.h>
#include <cstdlib>
#include <cmath>  
#ifdef _WIN32
#include <float.h>  
#endif

void matrix_multiply(const double* A, const double* B, double* C, int n);
void matrix_add(double* A, const double* B, int n);
void matrix_square(const double* A, double* C, int n);
void matrix_exponential(const double* A, double* expAt, int n, double t, int N);
void solve_linear_system(const double* A, const double* B, const double* x0, const double* u, double* x, int n, double delt, int N, double t_final);
bool read_matrix_from_file(const std::string& filename, std::vector<double>& matrix, int& n);
bool read_vector_from_file(const std::string& filename, std::vector<double>& vec, int& n);
bool is_singular(const double* A, int n);

int main() {
#ifdef _WIN32
    unsigned int current_control;
    _controlfp_s(&current_control, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW | _EM_ZERODIVIDE | _EM_INVALID, _MCW_EM);
#endif

   
    auto program_start = std::chrono::high_resolution_clock::now();
    std::string matrixAFile = "matrixA_1000.txt";
    std::string matrixBFile = "matrixB_1000.txt";
    std::string vectorX0File = "x0_1000.txt";
    std::string vectorUFile = "u_1000.txt";


    std::vector<double> A, B, x0, u;
    int nA, nB, nX0, nU;

    if (!read_matrix_from_file(matrixAFile, A, nA) ||
        !read_matrix_from_file(matrixBFile, B, nB) ||
        !read_vector_from_file(vectorX0File, x0, nX0) ||
        !read_vector_from_file(vectorUFile, u, nU)) {
        std::cerr << "读取矩阵或向量时出错" << std::endl;
        return -1;
    }

    if (nA != nB || nA != nX0 || nA != nU) {
        std::cerr << "矩阵和向量维度不匹配" << std::endl;
        return -1;
    }

    int n = nA;  

    if (is_singular(A.data(), n)) {
        std::cerr << "矩阵A可能是奇异矩阵，无法继续计算" << std::endl;
        return -1;
    }

 
    double* x = new (std::nothrow) double[n];
    if (!x) {
        std::cerr << "内存分配失败" << std::endl;
        return -1;
    }

    
    double delt = 0.00001;   // 时间步长
    double t_final = 0.01; // 总时间
    int N = 5;           // 精细积分法迭代次数

    auto start = std::chrono::high_resolution_clock::now();

    // 求解时不变线性微分方程
    solve_linear_system(A.data(), B.data(), x0.data(), u.data(), x, n, delt, N, t_final);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

  
    std::cout << "求解时间: " << elapsed.count() << " 秒" << std::endl;

    if (n <= 100) {
        std::cout << "x(t) = \n";
        for (int i = 0; i < n; i++) {
            std::cout << x[i] << " ";
        }
        std::cout << "\n";
    }
    delete[] x;
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

   
    cblas_daxpy(n * n, 1.0, temp, 1, T_a, 1);  
    // 递推计算 T_a
    for (int iter = 0; iter < N; iter++) {
 
        matrix_square(T_a, temp, n);

       /*
        std::cout << "T_a at iteration " << iter << ":\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << T_a[i * n + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        */
       
        cblas_dscal(n * n, 2.0, T_a, 1);     
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

bool is_singular(const double* A, int n) {
    std::vector<int> ipiv(n);
    std::vector<double> A_copy(A, A + n * n);

    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_copy.data(), n, ipiv.data());
    if (info != 0) {
        std::cerr << "LU 分解失败，矩阵可能是奇异的" << std::endl;
        return true;
    }
  
    for (int i = 0; i < n; i++) {
        if (fabs(A_copy[i * n + i]) < 1e-10) {
            std::cerr << "矩阵 A 的对角线元素接近零，可能是奇异矩阵" << std::endl;
            return true;
        }
    }

    return false;
}


void solve_linear_system(const double* A, const double* B, const double* x0, const double* u, double* x, int n, double delt, int N, double t_final) {
    double* expAt = new double[n * n];
    double* temp = new double[n];
    double* x_current = new double[n];

    if (!expAt || !temp || !x_current) {
        std::cerr << "内存分配失败！" << std::endl;
        return;
    }


    matrix_exponential(A, expAt, n, delt, N);

  
    cblas_dcopy(n, x0, 1, x_current, 1);

  
    for (double t = 0; t < t_final; t += delt) {
        // 自由响应：x(t + delt) = exp(A * delt) * x(t)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, expAt, n, x_current, 1, 0.0, x, 1);

        // 强迫响应：x(t + delt) += exp(A * delt) * B * u
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, expAt, n, B, 1, 0.0, temp, 1);
        cblas_daxpy(n, 1.0, temp, 1, x, 1);

        // 更新 x_current 为 x(t + delt)
        cblas_dcopy(n, x, 1, x_current, 1);
    }


    delete[] expAt;
    delete[] temp;
    delete[] x_current;
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
        }
        else if (col_count != n) {
            std::cerr << "文件格式不正确，列数不匹配" << std::endl;
            return false;
        }
        row_count++;
    }

    matrix = std::move(temp_matrix); 
    return true;
}


bool read_vector_from_file(const std::string& filename, std::vector<double>& vec, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    double value;
    std::vector<double> temp_vector;
    while (file >> value) {
        temp_vector.push_back(value);
    }

    n = temp_vector.size();
    vec = std::move(temp_vector);  
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
