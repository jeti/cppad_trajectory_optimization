#include <iostream>
#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>
#include <Eigen/Dense>

using Eigen::Dynamic;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;

template<typename T>
using Vector = Matrix<T, Dynamic, 1>;

int main() {

    /* Types and sizes */
    using ADdouble = CppAD::AD<double>;
    const int sizes[] = {8, 7, 2, 5, 3};
    const int N = sizeof(sizes) / sizeof(sizes[0]);

    /* Define the independent variable that we will be differentiating with respect to */
    Vector<ADdouble> X = Vector<ADdouble>::Ones(sizes[N - 1]);
    CppAD::Independent(X);

    /* Now define the function. The function is simply a bunch of matrix multiplications */
    Matrix<double, Dynamic, Dynamic> true_jacobian = Matrix<double, Dynamic, Dynamic>::Identity(sizes[0], sizes[0]);
    for (size_t i = 1; i < N; ++i)
        true_jacobian *= Matrix<double, Dynamic, Dynamic>::Random(sizes[i - 1], sizes[i]);
    Vector<ADdouble> Y = true_jacobian.cast<ADdouble>() * X;
    CppAD::ADFun<double> f(X, Y);

    /* Now define the value of X that we want to evaluate the jacobian at */
    Vector<double> x = Vector<double>::Random(sizes[N - 1]);

    /* Compute the jacobian and compare to the know value */
    Vector<double> jac = f.Jacobian(x);

    /* We are going to map the jacobian back to a matrix so that it is easier to visualize.
     * Note that CppAD seems to use row major format, while Eigen defaults to
     * column major */
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> cppad_jacobian(jac.data(), sizes[0], sizes[N - 1]);
    std::cout << "CppAD Jacobian: " << std::endl << cppad_jacobian << std::endl << std::endl;

    /* Now compare to the know jacobian, which is simply "A" */
    std::cout << "True  Jacobian: " << std::endl << true_jacobian << std::endl << std::endl;
    return 0;
}