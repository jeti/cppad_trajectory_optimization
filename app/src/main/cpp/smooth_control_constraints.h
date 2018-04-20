#ifndef SMOOTH_CONTROL_CONSTRAINTS_HEADER
#define SMOOTH_CONTROL_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"

/**
 * This set of constraints simply ensures that the first derivatives
 * of the controls are equal at the overlapping collocation points.
 *
 * For example, the points waypoint i, collocation 0 and waypoint i-1, collocation point n_c
 * occur at the same time. However, we can produce estimates of the derivatives of the controls
 * at these points from the left and the right, that is, we can compute u_dot at that time
 * by using all of the control values in waypoint i (collocation points 0, ... n_c) amd we can
 * also produce an estimate of u_dot at that time by using all of the control values corresponding
 * to waypoint i-1 (collocation points 0,...n_c). We want to ensure that the control is smooth,
 * so we enforce the fact that these estimates are equal.
 *
 * This gives us n_u * (n_w-1) constraints.
 */
template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w>
struct SmoothControlConstraints
        : EqualityConstraint<
                SmoothControlConstraints<
                        Scalar, Index, n_x, n_u, n_c, n_w>, Scalar, Index, n_u * (n_w - 1)> {

private:

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<Scalar, n_u, n_w - 1 >>;

public:

    static const Index derivatives = 1;

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    template<typename LD>
    Scalar *operator()(Scalar *g, const Scalar *x, LD &lagrange_derivatives) const {

        /* Get all of the control derivatives at the first and last collocation points */
        const Scalar *dx = lagrange_derivatives.template get<1>();
        auto c_0 = Get::controlsAtCollocationPoint(dx, 0);
        auto c_end = Get::controlsAtCollocationPoint(dx, n_c - 1);

        /* We are going to compute c_0 (of waypoint i) - c_end (of waypoint i-1)
         * This will be a matrix of size n_u x (n_w-1) */
        Map G(g);
        G = c_0.template rightCols<n_w - 1>() - c_end.template leftCols<n_w - 1>();
        return g + this->n_constraints;
    }
};

#endif /* SMOOTH_CONTROL_CONSTRAINTS_HEADER */