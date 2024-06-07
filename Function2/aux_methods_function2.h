#ifndef AUXILIAR_METHODS_H
#define AUXILIAR_METHODS_H

#include <vector>
#include <string>
#include <functional>

void print_table(const std::string& method, const std::vector<std::vector<double>>& points, const std::vector<int>& iterations, const std::vector<int>& armijo_calls, const std::vector<std::vector<double>>& opt_points, const std::vector<double>& opt_values, const std::vector<double>& errors);

std::vector<double> gradiente_f(const std::vector<double>& x);

std::vector<std::vector<double>> hess_f(const std::vector<double>& x);
// std::vector<std::vector<double>> hessian(const std::function<double(const std::vector<double>&)>& f, const std::vector<double>& x, double epsilon = 1e-5);

std::vector<double> solve_linear_system(const std::vector<std::vector<double>>& H, const std::vector<double>& grad);

#endif // AUXILIAR_METHODS_H
