#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>  // Para std::inner_product
#include <algorithm>
#include "aux_methods.h"

using namespace std;

const int max_iterations = 1000;

// Definindo a função f(x)
double f(const vector<double>& x) {
    return -exp(-x[0]*x[0] - x[1]*x[1] - 0.5*x[2]*x[2] - 0.1*x[3]*x[3]*x[3]*x[3] - (x[4]-1)*(x[4]-1));
}

// Método de busca de Armijo
double armijo(const vector<double>& x, const vector<double>& d, const vector<double>& grad, double alpha=1.0, double beta=0.5, double sigma=0.1) {
    double f_x = f(x);
    vector<double> x_new(5);
    while (true) {
        for (int i = 0; i < 5; ++i) {
            x_new[i] = x[i] + alpha * d[i];
        }
        double f_x_new = f(x_new);
        double lhs = f_x_new;
        double rhs = f_x + sigma * alpha * inner_product(grad.begin(), grad.end(), d.begin(), 0.0);
        if (lhs <= rhs) {
            break;
        }
        alpha *= beta;
    }
    return alpha;
}

vector<double> gradient_descent(const vector<double>& x0, int max_iter=max_iterations, double tol=1e-6) {
    vector<double> x = x0;
    vector<vector<double>> points;
    vector<int> iterations;
    vector<int> armijo_calls;
    vector<vector<double>> opt_points;
    vector<double> opt_values;
    vector<double> errors;

    int armijo_count = 0;
    for (int k = 0; k < max_iter; ++k) {
        vector<double> grad = grad_f(x);
        double norm_grad = sqrt(inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
        if (norm_grad < tol) {
            break;
        }
        vector<double> d = grad;
        for (auto& di : d) {
            di = -di;
        }
        double alpha = armijo(x, d, grad);
        armijo_count++;
        for (int i = 0; i < 5; ++i) {
            x[i] += alpha * d[i];
        }

        points.push_back(x0);
        iterations.push_back(k + 1);
        armijo_calls.push_back(armijo_count);
        opt_points.push_back(x);
        opt_values.push_back(f(x));
        errors.push_back(norm_grad);
    }

    print_table("Método do Gradiente", points, iterations, armijo_calls, opt_points, opt_values, errors);
    return x;
}

// Método de Newton
vector<double> newton_method(const vector<double>& x0, int max_iter=max_iterations, double tol=1e-6) {
    vector<double> x = x0;
    vector<vector<double>> points;
    vector<int> iterations;
    vector<int> armijo_calls;
    vector<vector<double>> opt_points;
    vector<double> opt_values;
    vector<double> errors;
    cout << "  init newton method "  << endl;

    int armijo_count = 0;
    for (int k = 0; k < max_iter; ++k) {
        vector<double> grad = grad_f(x);
        double norm_grad = sqrt(inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
        if (norm_grad < tol) {
            break;
        }
        vector<vector<double>> H = hess_f(x);
        vector<double> d = solve_linear_system(H, grad);
        for (auto& di : d) {
            di = -di;
        }
        double alpha = armijo(x, d, grad);
        armijo_count++;
        for (int i = 0; i < 5; ++i) {
            x[i] += alpha * d[i];
        }

        points.push_back(x0);
        iterations.push_back(k + 1);
        armijo_calls.push_back(armijo_count);
        opt_points.push_back(x);
        opt_values.push_back(f(x));
        errors.push_back(norm_grad);
    }

    print_table("Método de Newton", points, iterations, armijo_calls, opt_points, opt_values, errors);
    return x;
}

// Método BFGS
vector<double> bfgs_method(const vector<double>& x0, int max_iter=max_iterations, double tol=1e-6) {
    vector<double> x = x0;
    vector<vector<double>> B(5, vector<double>(5, 0.0));
    for (int i = 0; i < 5; ++i) {
        B[i][i] = 1.0;
    }

    vector<vector<double>> points;
    vector<int> iterations;
    vector<int> armijo_calls;
    vector<vector<double>> opt_points;
    vector<double> opt_values;
    vector<double> errors;

    int armijo_count = 0;
    for (int k = 0; k < max_iter; ++k) {
        vector<double> grad = grad_f(x);
        double norm_grad = sqrt(inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
        if (norm_grad < tol) {
            break;
        }
        vector<double> d(5);
        // Calcular direção d = -B^{-1} * grad
        // (Resolver sistema linear B * d = -grad)
        // Aqui estamos apenas fazendo d = -B * grad pois B é a matriz identidade
        std::transform(grad.begin(), grad.end(), d.begin(), [](double g) { return -g; });

        double alpha = armijo(x, d, grad);
        armijo_count++;
        vector<double> s(5), x_new(5);
        for (int i = 0; i < 5; ++i) {
            s[i] = alpha * d[i];
            x_new[i] = x[i] + s[i];
        }
        vector<double> y = grad_f(x_new);
        for (int i = 0; i < 5; ++i) {
            y[i] -= grad[i];
        }
        // Atualizar B usando a fórmula de atualização BFGS
        vector<vector<double>> Bs(5, vector<double>(5));
        vector<vector<double>> By(5, vector<double>(5));
        double sy = 0.0;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                Bs[i][j] = s[i] * s[j];
                By[i][j] = y[i] * y[j];
                sy += s[i] * y[j];
            }
        }
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                B[i][j] += (1.0 + inner_product(y.begin(), y.end(), y.begin(), 0.0) / sy) * Bs[i][j] / sy - By[i][j] / sy - By[j][i] / sy;
            }
        }
        x = x_new;

        points.push_back(x0);
        iterations.push_back(k + 1);
        armijo_calls.push_back(armijo_count);
        opt_points.push_back(x);
        opt_values.push_back(f(x));
        errors.push_back(norm_grad);
    }

    print_table("Método BFGS", points, iterations, armijo_calls, opt_points, opt_values, errors);
    return x;
}

int main() {
    int awaitInput;
    vector<double> x0 = {1, 1, 1, 1, 0}; // Ponto inicial
    // vector<double> x0 = {-1, -1, 0, 0, 0}; // Ponto inicial
    // vector<double> x0 = {5, 5, 0, 0, 0.5}; // Ponto inicial


    // Método do gradiente
    // vector<double> x_grad = gradient_descent(x0);
    // std::cin >> awaitInput;

    // Método de newton
    vector<double> x_newton = newton_method(x0);
    std::cin >> awaitInput;

    // Método de quase newton BFGS
    // vector<double> x_bfgs = bfgs_method(x0);
    // std::cin >> awaitInput;

    return 0;
}
