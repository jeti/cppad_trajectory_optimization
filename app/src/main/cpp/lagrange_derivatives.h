#ifndef LAGRANGE_DERIVATIVES_HEADER
#define LAGRANGE_DERIVATIVES_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "utils.h"

template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w, Index max_derivatives>
class LagrangeDerivatives {
private:

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;

    /** These are the coefficients used to generate the derivatives */
    const Eigen::Matrix<Scalar, n_c, n_c> derivative_coefficients;

    /** These are the derivatives. The i^th column holds the (i+1)^th derivatives */
    Eigen::Matrix<Scalar, Get::n_vars, max_derivatives> derivatives;

public:

    template<typename CP>
    LagrangeDerivatives(const CP &collocation_points)
            : derivative_coefficients(lagrangeDerivativeCoefficients(collocation_points.template cast<Scalar>())),
              derivatives(Eigen::Matrix<Scalar, Get::n_vars, max_derivatives>::Zero()) {
    }

    /**
     * Generate all of the derivatives up to the degree specified by "up_to_derivative".
     * For example, if up_to_derivative = 3, then we compute the 1st, 2nd, and 3rd derivatives.
     * To retrieve the computed values, call the getDerivatives method.
     *
     * Note that the pieces of memory in derivative memory that hold other variables,
     * such as time, will not be modified by this function. */
    template<Index up_to_derivative>
    void generate(const Scalar *x0) {

        static_assert(up_to_derivative <= max_derivatives,
                      "The number of derivatives must be less than or equal to number specified in the LagrangeDerivatives template.");

        const Scalar *x = x0;
        for (Index i = 0; i < up_to_derivative; ++i) {
            Scalar *dx = derivatives.col(i).data();
            for (Index i_w = 0; i_w < n_w; ++i_w)
                Get::varsAtWaypoint(dx, i_w) =
                        Get::varsAtWaypoint(x, i_w) * derivative_coefficients / Get::times(x)(i_w);
            x = dx;
        }
    }

    /** Return the specified derivative degree of the data from the last time that you called
     * `generate`. For instance, specify degree=1 means that this function
     * will return the first derivative, etc. If you did not called generate
     * with a degree >= the order that you specify here, then the output is undefined.
     */
    template<Index degree>
    Scalar *get() {

        static_assert(degree >= 0, "The derivative degree must be a positive number.");
        static_assert(degree <= max_derivatives,
                      "The derivative degree must be less than or equal to number specified in the LagrangeDerivatives template.");
        return derivatives.col(degree - 1).data();
    }
};

#endif /* LAGRANGE_DERIVATIVES_HEADER */