#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

MatrixXd generate_matrix(int N) {
    MatrixXd A(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A(i, j) = 1.0 / (1.0 + 0.8 * (i + 1) + 5.0 * (j + 1));
        }
    }
    return A;
}

tuple<VectorXd, double> solve_lu(const MatrixXd& A, const VectorXd& f) {
    auto start = high_resolution_clock::now();
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(f);
    auto end = high_resolution_clock::now();
    double time = duration_cast<duration<double>>(end - start).count();
    return { x, time };
}

tuple<VectorXd, double> solve_qr(const MatrixXd& A, const VectorXd& f) {
    auto start = high_resolution_clock::now();
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(f);
    auto end = high_resolution_clock::now();
    double time = duration_cast<duration<double>>(end - start).count();
    return { x, time };
}

tuple<VectorXd, double> solve_svd(const MatrixXd& A, const VectorXd& f) {
    auto start = high_resolution_clock::now();
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
    VectorXd x = svd.solve(f);
    auto end = high_resolution_clock::now();
    double time = duration_cast<duration<double>>(end - start).count();
    return { x, time };
}

double relative_error(const VectorXd& x_num, const VectorXd& x_exact) {
    return (x_num - x_exact).norm() / x_exact.norm();
}

void print_results(int N, const MatrixXd& A,
    const tuple<VectorXd, double>& lu,
    const tuple<VectorXd, double>& qr,
    const tuple<VectorXd, double>& svd) {
    cout << "N = " << N << "\n\n";

    // Вывод таблицы методов
    cout << "| Метод       | Время (сек)  | Ошибка       |\n";
    cout << "|-------------|--------------|--------------|\n";
    cout << "| LU          | " << setw(12) << scientific << get<1>(lu) << " | "
        << setw(12) << scientific << relative_error(get<0>(lu), VectorXd::Ones(N)) << " |\n";
    cout << "| QR (Гивенс) | " << setw(12) << scientific << get<1>(qr) << " | "
        << setw(12) << scientific << relative_error(get<0>(qr), VectorXd::Ones(N)) << " |\n";
    cout << "| SVD         | " << setw(12) << scientific << get<1>(svd) << " | "
        << setw(12) << scientific << relative_error(get<0>(svd), VectorXd::Ones(N)) << " |\n";
    cout << "\n";

    
    JacobiSVD<MatrixXd> svd_decomp(A, ComputeThinU | ComputeThinV);
    VectorXd s = svd_decomp.singularValues();
    

    // Число обусловленности
    double cond = s(0) / s(s.size() - 1);
    cout << "Число обусловленности: Cond(A) = " << scientific << cond << "\n\n";
}

int main() {
    setlocale(LC_ALL, "Ru");
    vector<int> sizes = { 5, 10, 20 };

    for (int N : sizes) {
        MatrixXd A = generate_matrix(N);
        VectorXd f = A * VectorXd::Ones(N);

        auto lu = solve_lu(A, f);
        auto qr = solve_qr(A, f);
        auto svd = solve_svd(A, f);

        print_results(N, A, lu, qr, svd);
    }

    return 0;
}