#include <iostream>
#include <mkl.h>
#include <vector>
#include <random>
using namespace std;

typedef vector<double> Matrix1D;  // 使用1D数组来表示矩阵

// 生成对角占优的非奇异矩阵
Matrix1D generate_diagonal_dominant_matrix(int rows, int cols, double min_val = -10.0, double max_val = 10.0) {
    Matrix1D matrix(rows * cols);
    random_device rd;  // 随机数设备，用于生成种子
    mt19937 gen(rd());  // 随机数生成器
    uniform_real_distribution<> dis(min_val, max_val);  // 生成[min_val, max_val]范围内的随机数

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == j) {
                matrix[i * cols + j] = dis(gen) + rows;  // 增加对角线元素的值，确保对角占优
            } else {
                matrix[i * cols + j] = dis(gen);
            }
        }
    }
    return matrix;
}
Matrix1D initialize_matrix_1D(int rows, int cols, double min_val = 1e-8, double max_val = 1e-6) {
    Matrix1D matrix(rows * cols);
    random_device rd;  // 随机数设备
    mt19937 gen(rd());  // 随机数生成器
    uniform_real_distribution<> dis(min_val, max_val);  // 在[min_val, max_val]之间生成随机数

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = dis(gen);  // 用小随机值填充矩阵
        }
    }
    return matrix;
}


// 初始化单位矩阵
Matrix1D initialize_identity_matrix_1D(int rows, int cols) {
    Matrix1D matrix(rows * cols, 0.0);
    for (int i = 0; i < rows; ++i) {
        matrix[i * cols + i] = 1.0;  // 初始化为单位矩阵
    }
    return matrix;
}
// 矩阵转置函数，1D 存储，行优先
Matrix1D transpose_matrix_1D(const Matrix1D &matrix, int rows, int cols) {
    Matrix1D transposed(cols * rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }
    return transposed;
}
// 矩阵乘法函数，使用1D数组存储矩阵，行优先存储
void mat_mul_1D(const Matrix1D &A, const Matrix1D &B, Matrix1D &C, int rows, int cols, int K) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, K, 1.0,
                &A[0], K, &B[0], cols, 0.0, &C[0], cols);
}

// 矩阵加法函数
void mat_add_1D(const Matrix1D &A, const Matrix1D &B, Matrix1D &C, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        C[i] = A[i] + B[i];
    }
}

// 矩阵减法函数
void mat_sub_1D(const Matrix1D &A, const Matrix1D &B, Matrix1D &C, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        C[i] = A[i] - B[i];
    }
}

// 标量乘法函数
void mat_scalar_1D(const Matrix1D &A, double scalar, Matrix1D &C, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        C[i] = A[i] * scalar;
    }
}

// 矩阵求逆函数，使用LU分解和求逆
void mat_inv_1D(Matrix1D &A, int N) {
    vector<int> ipiv(N);
    vector<double> work(N);
    int info;

    // LU分解
    dgetrf(&N, &N, &A[0], &N, &ipiv[0], &info);
    if (info != 0) {
        cerr << "Matrix inversion failed at LU decomposition with info = " << info << endl;
        return;
    }

    // 计算逆矩阵
    dgetri(&N, &A[0], &N, &ipiv[0], &work[0], &N, &info);
    if (info != 0) {
        cerr << "Matrix inversion failed at inversion with info = " << info << endl;
    }
}
void compute_matrices_1D(Matrix1D &Q, Matrix1D &G, Matrix1D &F_prime, const Matrix1D &C, const Matrix1D &V, const Matrix1D &B, const Matrix1D &W, const Matrix1D &A, double tau, int rows, int cols) {
   // 计算 e1, g1, f1
Matrix1D e1 = initialize_matrix_1D(rows, cols);

// 计算 e1 = C^T * V^{-1} * C
Matrix1D V_inv = V;  // V 的逆矩阵
mat_inv_1D(V_inv, rows);  // 计算 V 的逆
Matrix1D C_T = transpose_matrix_1D(C, rows, cols);  // C 的转置 C^T
Matrix1D temp = initialize_matrix_1D(rows, cols);

// 计算 C^T * V^{-1}
mat_mul_1D(C_T, V_inv, temp, rows, cols, rows);

// 计算 C^T * V^{-1} * C
mat_mul_1D(temp, C, e1, rows, cols, rows);

// 计算 g1 = B * W * B^T
Matrix1D g1 = initialize_matrix_1D(rows, cols);
Matrix1D B_T = transpose_matrix_1D(B, rows, cols);  // B 的转置 B^T

// 计算 B * W
mat_mul_1D(B, W, temp, rows, cols, rows);

// 计算 B * W * B^T
mat_mul_1D(temp, B_T, g1, rows, cols, rows);

// 计算 f1 = A
Matrix1D f1 = A;  // f1 直接等于 A


    // 计算 e2, g2, f2
    Matrix1D e2 = initialize_matrix_1D(rows, cols);
    Matrix1D g2 = initialize_matrix_1D(rows, cols);
    Matrix1D f2 = initialize_matrix_1D(rows, cols);

    Matrix1D temp1 = initialize_matrix_1D(rows, cols);
    
    // 计算 e2 = (f1^T e1 + e1 f1) / 2
    Matrix1D f1T = transpose_matrix_1D(f1, rows, cols);  // f1^T
    mat_mul_1D(f1T, e1, temp1, rows, cols, rows);  // f1^T * e1
    Matrix1D temp2 = initialize_matrix_1D(rows, cols);
    mat_mul_1D(e1, f1, temp2, rows, cols, rows);  // e1 * f1
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // f1^T * e1 + e1 * f1
    mat_scalar_1D(temp1, 0.5, e2, rows, cols);  // e2 = (f1^T * e1 + e1 * f1) / 2

    // 计算 g2 = (A g1 + g1 A^T) / 2
    Matrix1D AT = transpose_matrix_1D(A, rows, cols);  // A^T
    mat_mul_1D(A, g1, temp1, rows, cols, rows);  // A * g1
    mat_mul_1D(g1, AT, temp2, rows, cols, rows);  // g1 * A^T
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // A * g1 + g1 * A^T
    mat_scalar_1D(temp1, 0.5, g2, rows, cols);  // g2 = (A * g1 + g1 * A^T) / 2

    // 计算 f2 = (A^2 - g1 * e1) / 2
    mat_mul_1D(A, A, f2, rows, cols, rows);  // A^2
    mat_mul_1D(g1, e1, temp1, rows, cols, rows);  // g1 * e1
    mat_sub_1D(f2, temp1, f2, rows, cols);  // f2 = A^2 - g1 * e1
    mat_scalar_1D(f2, 0.5, f2, rows, cols);  // f2 = (A^2 - g1 * e1) / 2

    // 计算 e3, g3, f3
    Matrix1D e3 = initialize_matrix_1D(rows, cols);
    Matrix1D g3 = initialize_matrix_1D(rows, cols);
    Matrix1D f3 = initialize_matrix_1D(rows, cols);

    // e3 = (f2^T e1 + e1 f2 + f1^T e1 f1) / 3
    Matrix1D f2T = transpose_matrix_1D(f2, rows, cols);  // f2^T
    mat_mul_1D(f2T, e1, temp1, rows, cols, rows);  // f2^T * e1
    mat_mul_1D(e1, f2, temp2, rows, cols, rows);  // e1 * f2
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // f2^T * e1 + e1 * f2
    mat_mul_1D(f1T, e1, temp2, rows, cols, rows);  // f1^T * e1
    mat_mul_1D(temp2, f1, temp2, rows, cols, rows);  // f1^T * e1 * f1
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // f2^T * e1 + e1 * f2 + f1^T * e1 * f1
    mat_scalar_1D(temp1, 1.0 / 3.0, e3, rows, cols);  // e3 = (f2^T * e1 + e1 * f2 + f1^T * e1 * f1) / 3

    // g3 = (A g2 + g2 A^T - g1 * e1 * g1) / 3
    mat_mul_1D(A, g2, temp1, rows, cols, rows);  // A * g2
    mat_mul_1D(g2, AT, temp2, rows, cols, rows);  // g2 * A^T
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // A * g2 + g2 * A^T
    mat_mul_1D(g1, e1, temp2, rows, cols, rows);  // g1 * e1
    mat_mul_1D(temp2, g1, temp2, rows, cols, rows);  // g1 * e1 * g1
    mat_sub_1D(temp1, temp2, g3, rows, cols);  // g3 = A * g2 + g2 * A^T - g1 * e1 * g1
    mat_scalar_1D(g3, 1.0 / 3.0, g3, rows, cols);  // g3 = (A * g2 + g2 * A^T - g1 * e1 * g1) / 3

    // f3 = (A f2 - g2 * e1 - g1 * e1 * f1) / 3
    mat_mul_1D(A, f2, temp1, rows, cols, rows);  // A * f2
    mat_mul_1D(g2, e1, temp2, rows, cols, rows);  // g2 * e1
    mat_sub_1D(temp1, temp2, temp1, rows, cols);  // A * f2 - g2 * e1
    mat_mul_1D(g1, e1, temp2, rows, cols, rows);  // g1 * e1
    mat_mul_1D(temp2, f1, temp2, rows, cols, rows);  // g1 * e1 * f1
    mat_sub_1D(temp1, temp2, f3, rows, cols);  // f3 = A * f2 - g2 * e1 - g1 * e1 * f1
    mat_scalar_1D(f3, 1.0 / 3.0, f3, rows, cols);  // f3 = (A * f2 - g2 * e1 - g1 * e1 * f1) / 3

    // 计算 e4, g4, f4
    Matrix1D e4 = initialize_matrix_1D(rows, cols);
    Matrix1D g4 = initialize_matrix_1D(rows, cols);
    Matrix1D f4 = initialize_matrix_1D(rows, cols);

    // e4 = (f3^T e1 + e1 f3 + f2^T e1 f1 + f1^T e1 f1 e2) / 4
    Matrix1D f3T = transpose_matrix_1D(f3, rows, cols);  // f3^T
    mat_mul_1D(f3T, e1, temp1, rows, cols, rows);  // f3^T * e1
    mat_mul_1D(e1, f3, temp2, rows, cols, rows);  // e1 * f3
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // f3^T * e1 + e1 * f3
    mat_mul_1D(f2T, e1, temp2, rows, cols, rows);  // f2^T * e1
    mat_mul_1D(temp2, f1, temp2, rows, cols, rows);  // f2^T * e1 * f1
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // 加入 f2^T * e1 * f1
    mat_mul_1D(f1T, e1, temp2, rows, cols, rows);  // f1^T * e1
    mat_mul_1D(temp2, f1, temp2, rows, cols, rows);  // f1^T * e1 * f1
    mat_mul_1D(temp2, e2, temp2, rows, cols, rows);  // f1^T * e1 * f1 * e2
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // 最终结果加到 e4 中
    mat_scalar_1D(temp1, 1.0 / 4.0, e4, rows, cols);  // e4 = (f3^T e1 + e1 * f3 + f2^T * e1 * f1 + f1^T * e1 * f1 * e2) / 4

    // g4 = (A g3 + g3 A^T - g1 * e1 * g2 - g2 * e1 * g1) / 4
    mat_mul_1D(A, g3, temp1, rows, cols, rows);  // A * g3
    mat_mul_1D(g3, AT, temp2, rows, cols, rows);  // g3 * A^T
    mat_add_1D(temp1, temp2, temp1, rows, cols);  // A * g3 + g3 * A^T
    mat_mul_1D(g1, e1, temp2, rows, cols, rows);  // g1 * e1
    mat_mul_1D(temp2, g2, temp2, rows, cols, rows);  // g1 * e1 * g2
    mat_sub_1D(temp1, temp2, temp1, rows, cols);  // 减去 g1 * e1 * g2
    mat_mul_1D(g2, e1, temp2, rows, cols, rows);  // g2 * e1
    mat_mul_1D(temp2, g1, temp2, rows, cols, rows);  // g2 * e1 * g1
    mat_sub_1D(temp1, temp2, g4, rows, cols);  // 减去 g2 * e1 * g1
    mat_scalar_1D(g4, 1.0 / 4.0, g4, rows, cols);  // g4 = (A * g3 + g3 * A^T - g1 * e1 * g2 - g2 * e1 * g1) / 4

    // f4 = (A f3 - g3 * e1 - g2 * e1 * f1 - g1 * e1 * f2) / 4
    mat_mul_1D(A, f3, temp1, rows, cols, rows);  // A * f3
    mat_mul_1D(g3, e1, temp2, rows, cols, rows);  // g3 * e1
    mat_sub_1D(temp1, temp2, temp1, rows, cols);  // A * f3 - g3 * e1
    mat_mul_1D(g2, e1, temp2, rows, cols, rows);  // g2 * e1
    mat_mul_1D(temp2, f1, temp2, rows, cols, rows);  // g2 * e1 * f1
    mat_sub_1D(temp1, temp2, temp1, rows, cols);  // 减去 g2 * e1 * f1
    mat_mul_1D(g1, e1, temp2, rows, cols, rows);  // g1 * e1
    mat_mul_1D(temp2, f2, temp2, rows, cols, rows);  // g1 * e1 * f2
    mat_sub_1D(temp1, temp2, f4, rows, cols);  // 减去 g1 * e1 * f2
    mat_scalar_1D(f4, 1.0 / 4.0, f4, rows, cols);  // f4 = (A * f3 - g3 * e1 - g2 * e1 * f1 - g1 * e1 * f2) / 4
// 在累加之前先乘以 τ 对应的幂次
mat_scalar_1D(e1, tau, e1, rows, cols);
mat_scalar_1D(e2, tau * tau, e2, rows, cols);
mat_scalar_1D(e3, tau * tau * tau, e3, rows, cols);
mat_scalar_1D(e4, tau * tau * tau * tau, e4, rows, cols);

mat_scalar_1D(g1, tau, g1, rows, cols);
mat_scalar_1D(g2, tau * tau, g2, rows, cols);
mat_scalar_1D(g3, tau * tau * tau, g3, rows, cols);
mat_scalar_1D(g4, tau * tau * tau * tau, g4, rows, cols);

mat_scalar_1D(f1, tau, f1, rows, cols);
mat_scalar_1D(f2, tau * tau, f2, rows, cols);
mat_scalar_1D(f3, tau * tau * tau, f3, rows, cols);
mat_scalar_1D(f4, tau * tau * tau * tau, f4, rows, cols);

// 然后累加到 Q, G, F_prime 中
mat_add_1D(Q, e1, Q, rows, cols);
mat_add_1D(Q, e2, Q, rows, cols);
mat_add_1D(Q, e3, Q, rows, cols);
mat_add_1D(Q, e4, Q, rows, cols);

mat_add_1D(G, g1, G, rows, cols);
mat_add_1D(G, g2, G, rows, cols);
mat_add_1D(G, g3, G, rows, cols);
mat_add_1D(G, g4, G, rows, cols);

mat_add_1D(F_prime, f1, F_prime, rows, cols);
mat_add_1D(F_prime, f2, F_prime, rows, cols);
mat_add_1D(F_prime, f3, F_prime, rows, cols);
mat_add_1D(F_prime, f4, F_prime, rows, cols);

}

void compute_QGF_1D(Matrix1D &Qc, Matrix1D &Gc, Matrix1D &F_prime_c, const Matrix1D &Q, const Matrix1D &G, const Matrix1D &F_prime, int rows, int cols) {
    // 初始化单位矩阵 I
    Matrix1D I = initialize_identity_matrix_1D(rows, cols);

    // 计算 (I + F')
    Matrix1D temp = Matrix1D(rows * cols);
    mat_add_1D(I, F_prime, temp, rows, cols);  // temp = I + F'

    // 计算 (Q^{-1} + G)^{-1}
    Matrix1D Q_inv = Q;
    mat_inv_1D(Q_inv, rows);  // Q^{-1}
    Matrix1D QG_inv = Matrix1D(rows * cols);
    mat_add_1D(Q_inv, G, QG_inv, rows, cols);  // Q^{-1} + G
    mat_inv_1D(QG_inv, rows);  // (Q^{-1} + G)^{-1}

    // 计算 Qc = Q + (I + F')^T * (Q^{-1} + G)^{-1} * (I + F')
    Matrix1D temp_T = transpose_matrix_1D(temp, rows, cols);  // (I + F')^T
    Matrix1D temp_mul = Matrix1D(rows * cols);
    mat_mul_1D(temp_T, QG_inv, temp_mul, rows, cols, rows);  // temp_mul = (I + F')^T * (Q^{-1} + G)^{-1}
    mat_mul_1D(temp_mul, temp, temp_mul, rows, cols, rows);  // temp_mul = (I + F')^T * (Q^{-1} + G)^{-1} * (I + F')
    mat_add_1D(Q, temp_mul, Qc, rows, cols);  // Qc = Q + temp_mul

    // 计算 (G^{-1} + Q)^{-1}
    Matrix1D G_inv = G;
    mat_inv_1D(G_inv, rows);  // G^{-1}
    Matrix1D GQ_inv = Matrix1D(rows * cols);
    mat_add_1D(G_inv, Q, GQ_inv, rows, cols);  // G^{-1} + Q
    mat_inv_1D(GQ_inv, rows);  // (G^{-1} + Q)^{-1}

    // 计算 Gc = G + (I + F') * (G^{-1} + Q)^{-1} * (I + F')^T
    Matrix1D temp_mul2 = Matrix1D(rows * cols);
    mat_mul_1D(temp, GQ_inv, temp_mul2, rows, cols, rows);  // temp_mul2 = (I + F') * (G^{-1} + Q)^{-1}
    mat_mul_1D(temp_mul2, temp_T, temp_mul2, rows, cols, rows);  // temp_mul2 = (I + F') * (G^{-1} + Q)^{-1} * (I + F')^T
    mat_add_1D(G, temp_mul2, Gc, rows, cols);  // Gc = G + temp_mul2

    // 计算 F'_c = (F' - GQ/2) * (I + GQ)^{-1} + (I + GQ)^{-1} * (F' - GQ/2)
    Matrix1D GQ = Matrix1D(rows * cols);
    mat_mul_1D(G, Q, GQ, rows, cols, rows);  // GQ = G * Q
    Matrix1D GQ_half = Matrix1D(rows * cols);
    mat_scalar_1D(GQ, 0.5, GQ_half, rows, cols);  // GQ_half = GQ / 2

    Matrix1D F_prime_minus_GQ = Matrix1D(rows * cols);
    mat_sub_1D(F_prime, GQ_half, F_prime_minus_GQ, rows, cols);  // F'_minus_GQ = F' - GQ / 2

    Matrix1D IGQ = Matrix1D(rows * cols);
    mat_add_1D(I, GQ, IGQ, rows, cols);  // IGQ = I + GQ
    mat_inv_1D(IGQ, rows);  // IGQ_inv = (I + GQ)^{-1}

    Matrix1D F_prime_c_part1 = Matrix1D(rows * cols);
    mat_mul_1D(F_prime_minus_GQ, IGQ, F_prime_c_part1, rows, cols, rows);  // F_prime_c_part1 = (F' - GQ/2) * (I + GQ)^{-1}

    Matrix1D F_prime_c_part2 = Matrix1D(rows * cols);
    mat_mul_1D(IGQ, F_prime_minus_GQ, F_prime_c_part2, rows, cols, rows);  // F_prime_c_part2 = (I + GQ)^{-1} * (F' - GQ/2)

    mat_add_1D(F_prime_c_part1, F_prime_c_part2, F_prime_c, rows, cols);  // F'_c = part1 + part2
}


void compute_QGF_63_65_1D(Matrix1D &Qc, Matrix1D &Gc, Matrix1D &Fc, const Matrix1D &Q1, const Matrix1D &G1, const Matrix1D &F1, const Matrix1D &Q2, const Matrix1D &G2, const Matrix1D &F2, int rows, int cols) {
    // 计算 (Q2^{-1} + G1)^{-1}
    Matrix1D Q2_inv = Q2;
    mat_inv_1D(Q2_inv, rows);  // Q2^{-1}
    Matrix1D QG_inv = initialize_matrix_1D(rows, cols);
    mat_add_1D(Q2_inv, G1, QG_inv, rows, cols);  // Q2^{-1} + G1
    mat_inv_1D(QG_inv, rows);  // (Q2^{-1} + G1)^{-1}

    // 计算 Qc = Q1 + F1^T * (Q2^{-1} + G1)^{-1} * F1
    Matrix1D F1_T = transpose_matrix_1D(F1, rows, cols);  // F1^T
    Matrix1D temp = initialize_matrix_1D(rows, cols);
    mat_mul_1D(F1_T, QG_inv, temp, rows, cols, rows);  // F1^T * (Q2^{-1} + G1)^{-1}
    mat_mul_1D(temp, F1, temp, rows, cols, rows);  // F1^T * (Q2^{-1} + G1)^{-1} * F1
    mat_add_1D(Q1, temp, Qc, rows, cols);  // Qc = Q1 + temp

    // 计算 Gc = G2 + F2 * (G1^{-1} + Q2)^{-1} * F2^T
    Matrix1D G1_inv = G1;
    mat_inv_1D(G1_inv, rows);  // G1^{-1}
    mat_add_1D(G1_inv, Q2, QG_inv, rows, cols);  // G1^{-1} + Q2
    mat_inv_1D(QG_inv, rows);  // (G1^{-1} + Q2)^{-1}
    Matrix1D F2_T = transpose_matrix_1D(F2, rows, cols);  // F2^T
    mat_mul_1D(F2, QG_inv, temp, rows, cols, rows);  // F2 * (G1^{-1} + Q2)^{-1}
    mat_mul_1D(temp, F2_T, temp, rows, cols, rows);  // F2 * (G1^{-1} + Q2)^{-1} * F2^T
    mat_add_1D(G2, temp, Gc, rows, cols);  // Gc = G2 + temp

    // 计算 Fc = F2 * (I_n + G1 * Q2)^{-1} * F1
    Matrix1D GQ = initialize_matrix_1D(rows, cols);
    mat_mul_1D(G1, Q2, GQ, rows, cols, rows);  // GQ = G1 * Q2
    Matrix1D I = initialize_identity_matrix_1D(rows, cols);
    mat_add_1D(I, GQ, GQ, rows, cols);  // I_n + G1 * Q2
    mat_inv_1D(GQ, rows);  // (I_n + G1 * Q2)^{-1}
    mat_mul_1D(F2, GQ, temp, rows, cols, rows);  // F2 * (I_n + G1 * Q2)^{-1}
    mat_mul_1D(temp, F1, Fc, rows, cols, rows);  // Fc = F2 * (I_n + G1 * Q2)^{-1} * F1
}


// PIM求解Riccati方程的函数
void PIM_solve_Riccati_1D(const Matrix1D &A, const Matrix1D &B, const Matrix1D &C, const Matrix1D &W, const Matrix1D &V, const Matrix1D &P0, int N, double tau, int rows, int cols) {
    // 初始化 Q(τ), G(τ), F'(τ)
    Matrix1D Q = initialize_matrix_1D(rows, cols);
    Matrix1D G = initialize_matrix_1D(rows, cols);
    Matrix1D F_prime = initialize_matrix_1D(rows, cols);

    compute_matrices_1D(Q, G, F_prime, C, V, B, W, A, tau, rows, cols);

    // 递归计算 Qc, Gc, F_prime_c
    Matrix1D Qc = initialize_matrix_1D(rows, cols);
    Matrix1D Gc = initialize_matrix_1D(rows, cols);
    Matrix1D F_prime_c = initialize_matrix_1D(rows, cols);

    for (int iter = 0; iter < N; ++iter) {
        compute_QGF_1D(Qc, Gc, F_prime_c, Q, G, F_prime, rows, cols);
        Q = Qc;
        G = Gc;
        F_prime = F_prime_c;
    }

    // 计算 F = I + F_prime
    Matrix1D F = initialize_matrix_1D(rows, cols);
    for (int i = 0; i < rows; ++i) F[i * cols + i] = 1.0;  // 初始化为单位矩阵
    mat_add_1D(F, F_prime, F, rows, cols);

    // 初始化 Q2, G2, F2, G1, Q1, F1
    Matrix1D Q2 = Q, G2 = G, F2 = F, G1 = P0, Q1 = initialize_matrix_1D(rows, cols), F1 = initialize_matrix_1D(rows, cols);
    for (int i = 0; i < rows; ++i) F1[i * cols + i] = 1.0;  // 初始化为单位矩阵

    // 根据 (63)-(65) 计算 Qc, Gc, Fc
    for (int k = 0; k < 100; ++k) {
        compute_QGF_63_65_1D(Qc, Gc, F_prime_c, Q1, G1, F1, Q2, G2, F2, rows, cols);
        Q1 = Qc;
        G1 = Gc;
        F1 = F_prime_c;
    }

    // 输出或保存结果矩阵
    cout << "PIM Riccati Equation Solved" << endl;
}

int main() {
    // 初始化参数
    int N = 20;
    double tau = 1.0 / (1 << N);
    int rows = 1000, cols = 1000;

 // 初始化矩阵，生成对角占优矩阵
    Matrix1D A = generate_diagonal_dominant_matrix(rows, cols);
    Matrix1D B = generate_diagonal_dominant_matrix(rows, cols);
    Matrix1D C = generate_diagonal_dominant_matrix(rows, cols);
    Matrix1D W = generate_diagonal_dominant_matrix(rows, cols);
    Matrix1D V = generate_diagonal_dominant_matrix(rows, cols);
    Matrix1D P0 = generate_diagonal_dominant_matrix(rows, cols);

    // 调用PIM求解函数
    PIM_solve_Riccati_1D(A, B, C, W, V, P0, N, tau, rows, cols);

    return 0;
}
