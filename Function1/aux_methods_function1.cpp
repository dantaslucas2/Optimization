#include "aux_methods_function1.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <functional>

using namespace std;

void print_table(const string& method, const vector<vector<double>>& points, const vector<int>& iterations, const vector<int>& armijo_calls, const vector<vector<double>>& opt_points, const vector<double>& opt_values, const vector<double>& errors) {
    cout << "                               Resultados computacionais pelo " << method << endl;
    cout << "|" << setw(2) << "               Ponto inicial               " << setw(2) << "|" << setw(12) << "# de iterações" << setw(1) << "|" << setw(5)  << " # de cham. Armijo" << setw(2) << "|" << "              Ponto ótimo atual            " << "|" << setw(15) << "Valor ótimo" << setw(3) << "|" << setw(7) << "   Erro de aproximação" <<  setw(2) << "|" << endl;
    for (size_t i = 0; i < points.size(); ++i) {
        cout << fixed << setprecision(4);
        cout << "|" << "  (";
        for (size_t j = 0; j < points[i].size(); ++j) {
            cout << points[i][j];
            if (j < points[i].size() - 1) cout << ", ";
        }
        cout << ")  " << "|" << setw(8) << iterations[i] << setw(7) << "|" << setw(8) << armijo_calls[i] << setw(12) << "|" << setw(2) << "(";
        for (size_t j = 0; j < opt_points[i].size(); ++j) {
            cout << opt_points[i][j];
            if (j < opt_points[i].size() - 1) cout << ", ";
        }
        cout << ")  " << "|    " <<  setw(5) << opt_values[i] << setw(5)  << "     |   " <<  setw(5) << errors[i] << setw(5) << "              |" << endl;
    }
}

// Gradiente da função f(x)
vector<double> gradiente_f(const vector<double>& x) {
    vector<double> g(5);
    g[0] = 2 * x[0] * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    g[1] = 2 * x[1] * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    g[2] = x[2] * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    g[3] = 0.4 * x[3] * x[3] * x[3] * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    g[4] = 2 * (x[4] - 1) * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    return g;
}

// Hessiana da função f(x)
vector<vector<double>> hess_f(const vector<double>& x) {
    vector<vector<double>> hess(5, vector<double>(5, 0.0));

    hess[0][0] = 2 * (1 - 2 * x[0] * x[0]) * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    hess[1][1] = 2 * (1 - 2 * x[1] * x[1]) * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    hess[2][2] = (1 - x[2] * x[2]) * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    hess[3][3] = 0.4 * (3 * x[3] * x[3] - x[3] * x[3] * x[3] * x[3]) * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
    hess[4][4] = 2 * (1 - (x[4] - 1) * (x[4] - 1)) * exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));

    return hess;
}
// vector<vector<double>> hessian(const function<double(const vector<double>&)>& f, const vector<double>& x, double epsilon = 1e-5) {
//     int n = x.size();
//     vector<vector<double>> hess(n, vector<double>(n, 0.0));

//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j <= i; ++j) {
//             vector<double> x_ij = x;
//             x_ij[i] += epsilon;
//             x_ij[j] += epsilon;
//             double f_ij = f(x_ij);

//             x_ij[j] -= 2 * epsilon;
//             double f_i_j = f(x_ij);

//             x_ij[i] -= 2 * epsilon;
//             x_ij[j] += 2 * epsilon;
//             double f__ij = f(x_ij);

//             x_ij[j] -= 2 * epsilon;
//             double f__i_j = f(x_ij);

//             double f_x = f(x);

//             hess[i][j] = (f_ij - f_i_j - f__ij + f__i_j) / (4 * epsilon * epsilon);
//             hess[j][i] = hess[i][j]; // A Hessiana é simétrica
//         }
//     }

//     return hess;
// }
vector<double> solve_linear_system(const vector<vector<double>>& H, const vector<double>& grad) {
    int n = H.size();
    vector<double> d(n);
    vector<vector<double>> L(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double diag_val = H[i][i] - sum;
                if (diag_val <= 0.0) {
                    diag_val = 1e-10;
                }
                L[i][j] = sqrt(diag_val);
            } else {
                L[i][j] = (H[i][j] - sum) / L[j][j];
            }
        }
    }

    vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
            sum += L[i][k] * y[k];
        }
        y[i] = (-grad[i] - sum) / L[i][i];
    }

    // Resolução do sistema L^T * d = y
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int k = i + 1; k < n; ++k) {
            sum += L[k][i] * d[k];
        }
        d[i] = (y[i] - sum) / L[i][i];
    }

    return d;
}
