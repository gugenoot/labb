#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
using namespace std;
using namespace std::chrono;

// Генерация матрицы A
vector<vector<double>> generate_matrix(int N) {
    vector<vector<double>> A(N, vector<double>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                A[i][j] = 100.0;
            }
            else {
                A[i][j] = 1.0 + 0.1 * (i + 1) + 0.2 * (j + 1);
            }
        }
    }
    return A;
}

// Генерация точного решения (все элементы равны 1)
vector<double> generate_exact_solution(int N) {
    return vector<double>(N, 1.0);
}

// Вычисление нормы вектора
double vector_norm(const vector<double>& v) {
    double norm = 0.0;
    for (double val : v) {
        norm += val * val;
    }
    return sqrt(norm);
}

// Вычисление погрешности
double compute_error(const vector<double>& x_num, const vector<double>& x_exact) {
    vector<double> diff(x_num.size());
    for (size_t i = 0; i < x_num.size(); ++i) {
        diff[i] = x_num[i] - x_exact[i];
    }
    return vector_norm(diff) / vector_norm(x_exact);
}

// Умножение матрицы на вектор
vector<double> matrix_vector_mult(const vector<vector<double>>& A, const vector<double>& x) {
    vector<double> res(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            res[i] += A[i][j] * x[j];
        }
    }
    return res;
}

// LU-разложение (Doolittle алгоритм)
void lu_decomposition(const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U) {
    int n = A.size();
    L = vector<vector<double>>(n, vector<double>(n, 0.0));
    U = vector<vector<double>>(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        // Верхняя треугольная матрица U
        for (int k = i; k < n; ++k) {
            double sum = 0.0;
            for (int j = 0; j < i; ++j) {
                sum += L[i][j] * U[j][k];
            }
            U[i][k] = A[i][k] - sum;
        }

        // Нижняя треугольная матрица L
        for (int k = i; k < n; ++k) {
            if (i == k) {
                L[i][i] = 1.0;
            }
            else {
                double sum = 0.0;
                for (int j = 0; j < i; ++j) {
                    sum += L[k][j] * U[j][i];
                }
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

// Решение системы с LU-разложением
vector<double> solve_lu(const vector<vector<double>>& A, const vector<double>& b) {
    int n = A.size();
    vector<vector<double>> L, U;
    lu_decomposition(A, L, U);

    // Решение Ly = b (прямая подстановка)
    vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= L[i][j] * y[j];
        }
        y[i] /= L[i][i];
    }

    // Решение Ux = y (обратная подстановка)
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }

    return x;
}

// QR-разложение методом вращений Гивенса
void qr_decomposition(const vector<vector<double>>& A, vector<vector<double>>& Q, vector<vector<double>>& R) {
    int n = A.size();
    Q = vector<vector<double>>(n, vector<double>(n, 0.0));
    R = A;

    // Инициализация Q как единичной матрицы
    for (int i = 0; i < n; ++i) {
        Q[i][i] = 1.0;
    }

    for (int j = 0; j < n; ++j) {
        for (int i = n - 1; i > j; --i) {
            if (fabs(R[i][j]) > 1e-12) {
                double r = sqrt(R[i - 1][j] * R[i - 1][j] + R[i][j] * R[i][j]);
                double c = R[i - 1][j] / r;
                double s = -R[i][j] / r;

                // Применение вращения к R
                for (int k = j; k < n; ++k) {
                    double temp = c * R[i - 1][k] - s * R[i][k];
                    R[i][k] = s * R[i - 1][k] + c * R[i][k];
                    R[i - 1][k] = temp;
                }

                // Применение вращения к Q
                for (int k = 0; k < n; ++k) {
                    double temp = c * Q[i - 1][k] - s * Q[i][k];
                    Q[i][k] = s * Q[i - 1][k] + c * Q[i][k];
                    Q[i - 1][k] = temp;
                }
            }
        }
    }

    // Транспонирование Q (так как мы накапливали вращения)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            swap(Q[i][j], Q[j][i]);
        }
    }
}

// Решение системы с QR-разложением
vector<double> solve_qr(const vector<vector<double>>& A, const vector<double>& b) {
    int n = A.size();
    vector<vector<double>> Q, R;
    qr_decomposition(A, Q, R);

    // Вычисление Qᵀb
    vector<double> qtb(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            qtb[i] += Q[j][i] * b[j];
        }
    }

    // Обратная подстановка для Rx = Qᵀb
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = qtb[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }

    return x;
}

void solve_system(int N) {
    cout << "\nРешаем систему для N = " << N << endl;

    // Генерация данных
    auto A = generate_matrix(N);
    auto x_exact = generate_exact_solution(N);
    auto b = matrix_vector_mult(A, x_exact);

    // Решение методом LU
    auto start = high_resolution_clock::now();
    auto x_lu = solve_lu(A, b);
    auto end = high_resolution_clock::now();
    double time_lu = duration_cast<duration<double>>(end - start).count();
    double error_lu = compute_error(x_lu, x_exact);

    cout << fixed << setprecision(6);
    cout << "LU метод: время = " << time_lu << " сек, погрешность = " << scientific << error_lu << endl;

    // Решение методом QR
    start = high_resolution_clock::now();
    auto x_qr = solve_qr(A, b);
    end = high_resolution_clock::now();
    double time_qr = duration_cast<duration<double>>(end - start).count();
    double error_qr = compute_error(x_qr, x_exact);

    cout << "QR метод: время = " << fixed << time_qr << " сек, погрешность = " << scientific << error_qr << endl;
}

int main() {
    setlocale(LC_ALL, "Ru");
    // Решение для разных размеров систем
    solve_system(250);
    solve_system(500);
    solve_system(1000);

    return 0;
}