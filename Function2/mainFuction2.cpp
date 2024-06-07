#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> 
#include <algorithm>
#include "aux_methods_function2.h"

using namespace std;

const int max_iterations = 100;

double f(const vector<double>& x) {
    return pow(x[0] - x[1], 2) + sin(x[0] + x[1]) + pow(x[0] + x[1], 2) + pow(x[2] - x[4], 4) + sin(x[2] + x[3]) + pow(x[3] + x[4], 4);
}

pair<double, int> armijo(const vector<double>& x, const vector<double>& d, const vector<double>& grad, double alpha=0.5, double beta=0.3, double sigma=0.3) {
    double f_x = f(x);
    vector<double> x_new(5);
    int armijo_iterations = 0; 

    while (true) {
        armijo_iterations++; 
        // cout << "armijo" << endl; // Debug log
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
    // cout << "return armijo with interactions: " << armijo_iterations << endl;
    return make_pair(alpha, armijo_iterations);
}

vector<double> gradient(const vector<double>& x0, int max_iter=max_iterations, double tol=1e-4) {
    vector<double> x = x0;
    vector<vector<double>> points;
    vector<int> iterations;
    vector<int> armijo_calls;
    vector<vector<double>> opt_points;
    vector<double> opt_values;
    vector<double> errors;

    int total_armijo_calls = 0;
    for (int k = 0; k < max_iter; ++k) {
        vector<double> grad = gradiente_f(x);
        double norm_grad = sqrt(inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
        if (norm_grad < tol) {
            break;
        }
        vector<double> d = grad;
        for (auto& di : d) {
            di = -di;
        }
        auto [alpha, armijo_iterations] = armijo(x, d, grad); 
        total_armijo_calls += armijo_iterations; 
        for (int i = 0; i < 5; ++i) {
            x[i] += alpha * d[i];
        }

        points.push_back(x0);
        iterations.push_back(k + 1);
        armijo_calls.push_back(total_armijo_calls);
        opt_points.push_back(x);
        opt_values.push_back(f(x));
        errors.push_back(norm_grad);
    }

    print_table("Método do Gradiente", points, iterations, armijo_calls, opt_points, opt_values, errors);
    return x;
}

vector<double> newton_method(const vector<double>& x0, int max_iter=max_iterations, double tol=1e-4) {
    vector<double> x = x0;
    vector<vector<double>> points;
    vector<int> iterations;
    vector<int> armijo_calls;
    vector<vector<double>> opt_points;
    vector<double> opt_values;
    vector<double> errors;

    int total_armijo_calls = 0;
    for (int k = 0; k < max_iter; ++k) {
        vector<double> grad = gradiente_f(x);
        double norm_grad = sqrt(inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
        if (norm_grad < tol) {
            break;
        }
        vector<vector<double>> H = hess_f(x); 
        double lambda = 1e-4; 
        for (int i = 0; i < H.size(); ++i) {
            H[i][i] += lambda;
        }
        vector<double> d = solve_linear_system(H, grad);
        for (auto& di : d) {
            di = -di;
        }
        auto [alpha, armijo_iterations] = armijo(x, d, grad); 
        total_armijo_calls += armijo_iterations; 
        for (int i = 0; i < 5; ++i) {
            x[i] += alpha * d[i];
        }

        points.push_back(x0);
        iterations.push_back(k + 1);
        armijo_calls.push_back(total_armijo_calls);
        opt_points.push_back(x);
        opt_values.push_back(f(x));
        errors.push_back(norm_grad);
    }

    print_table("Método de Newton", points, iterations, armijo_calls, opt_points, opt_values, errors);
    return x;
}

// Método BFGS
vector<double> bfgs_method(const vector<double>& x0, int max_iter=max_iterations, double tol=1e-4) {
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

    int total_armijo_calls = 0;
    for (int k = 0; k < max_iter; ++k) {
        vector<double> grad = gradiente_f(x);
        double norm_grad = sqrt(inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
        if (norm_grad < tol) {
            break;
        }
        vector<double> d(5);
        std::transform(grad.begin(), grad.end(), d.begin(), [](double g) { return -g; });

        auto [alpha, armijo_iterations] = armijo(x, d, grad); 
        total_armijo_calls += armijo_iterations; 
        vector<double> s(5), x_new(5);
        for (int i = 0; i < 5; ++i) {
            s[i] = alpha * d[i];
            x_new[i] = x[i] + s[i];
        }
        vector<double> y = gradiente_f(x_new);
        for (int i = 0; i < 5; ++i) {
            y[i] -= grad[i];
        }
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
        armijo_calls.push_back(total_armijo_calls);
        opt_points.push_back(x);
        opt_values.push_back(f(x));
        errors.push_back(norm_grad);
    }

    print_table("Método BFGS", points, iterations, armijo_calls, opt_points, opt_values, errors);
    return x;
}

vector<double> dfp_method(const vector<double>& x0, int max_iter=max_iterations, double tol=1e-4) {
    vector<double> x = x0;
    vector<vector<double>> H_inv(5, vector<double>(5, 0.0)); 
    for (int i = 0; i < 5; ++i) {
        H_inv[i][i] = 1.0;
    }

    vector<vector<double>> points;
    vector<int> iterations;
    vector<int> armijo_calls;
    vector<vector<double>> opt_points;
    vector<double> opt_values;
    vector<double> errors;

    int total_armijo_calls = 0;
    for (int k = 0; k < max_iter; ++k) {
        vector<double> grad = gradiente_f(x);
        double norm_grad = sqrt(inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));
        if (norm_grad < tol) {
            break;
        }

        vector<double> d(5, 0.0);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                d[i] -= H_inv[i][j] * grad[j];
            }
        }
        auto [alpha, armijo_iterations] = armijo(x, d, grad); 
        total_armijo_calls += armijo_iterations; 
        vector<double> s(5), x_new(5);
        for (int i = 0; i < 5; ++i) {
            s[i] = alpha * d[i];
            x_new[i] = x[i] + s[i];
        }

        vector<double> y = gradiente_f(x_new);
        for (int i = 0; i < 5; ++i) {
            y[i] -= grad[i];
        }

        double sy = inner_product(s.begin(), s.end(), y.begin(), 0.0);
        if (fabs(sy) < 1e-10) {
            cout << "Warning: sTy is too close to zero, skipping update" << endl;
            x = x_new;
            continue;
        }

        vector<vector<double>> ssT(5, vector<double>(5, 0.0));
        vector<vector<double>> HyHyT(5, vector<double>(5, 0.0));
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                ssT[i][j] = s[i] * s[j];
                HyHyT[i][j] = 0.0;
                for (int k = 0; k < 5; ++k) {
                    HyHyT[i][j] += H_inv[i][k] * y[k] * y[j] * H_inv[j][k];
                }
            }
        }

        double yy = inner_product(y.begin(), y.end(), y.begin(), 0.0);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                H_inv[i][j] += (ssT[i][j] / sy) - (HyHyT[i][j] / yy);
            }
        }

        x = x_new;

        points.push_back(x0);
        iterations.push_back(k + 1);
        armijo_calls.push_back(total_armijo_calls);
        opt_points.push_back(x);
        opt_values.push_back(f(x));
        errors.push_back(norm_grad);
    }

    print_table("Método DFP", points, iterations, armijo_calls, opt_points, opt_values, errors);
    return x;
}

int main() {
    int awaitInput;

    vector<double> x4 = {0, 0, 0, 0, 0};
    cout << endl<< endl<< endl;
    vector<double> x_grad4 = dfp_method(x4);
    cout << endl<< endl<< endl;

    vector<double> x0 = {2, 2, 2, 2, 2};
    cout << endl<< endl<< endl;
    vector<double> x_grad = dfp_method(x0);
    cout << endl<< endl<< endl;

    vector<double> x1 = {1,2,3,4,5};
    cout << endl<< endl<< endl;
    vector<double> x_grad1 = dfp_method(x1);
    cout << endl<< endl<< endl;

    vector<double> x2 = {5,4,3,2,1};
    cout << endl<< endl<< endl;
    vector<double> x_grad2 = dfp_method(x2);
    cout << endl<< endl<< endl;

    vector<double> x3 = {9,9,9,9,9};
    cout << endl<< endl<< endl;
    vector<double> x_grad3 = dfp_method(x3);
    cout << endl<< endl<< endl;

    // // Método de Newton
    // vector<double> x_newton = newton_method(x0);
    // // std::cin >> awaitInput;
    // cout << endl<< endl<< endl;

    // // Método BFGS
    // vector<double> x_bfgs = bfgs_method(x0);
    // // std::cin >> awaitInput;
    // cout << endl<< endl<< endl;

    // // Método DFP
    // vector<double> x_dfp = dfp_method(x0);
    // // std::cin >> awaitInput;
    // cout << endl<< endl<< endl;

    return 0;
}
