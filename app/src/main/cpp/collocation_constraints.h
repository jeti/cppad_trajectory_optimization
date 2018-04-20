#ifndef COLLOCATION_CONSTRAINTS_HEADER
#define COLLOCATION_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"

/**
 * We are redundantly estimating the state and control at the
 * ends of the waypoints since the collocation points go from
 * 0...1 (including the endpoints).
 * This gives us (n_x + n_u) * (n_w-1) conditions that must be checked.
 * Note that we have (n_w-1) here because we don't have redundant
 * estimates at the initial time.
 */
template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w>
struct CollocationConstraints
    : EqualityConstraint<
        CollocationConstraints<
            Scalar, Index, n_x, n_u, n_c, n_w>, Scalar, Index, (n_x + n_u) * (n_w - 1)> {

private:

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<Scalar, n_x + n_u, n_w - 1 >>;

public:

    static const Index derivatives = 0;

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    template<typename LD>
    Scalar *operator()(Scalar *g, const Scalar *x, LD &lagrange_derivatives) const {

        /* Get all of the values at the first and last collocation points */
        auto c_0 = Get::varsAtCollocationPoint(x, 0);
        auto c_end = Get::varsAtCollocationPoint(x, n_c - 1);

        /* We are going to compute c_0 (of waypoint i) - c_end (of waypoint i-1)
         * This will be a matrix of size (n_x + n_u) x (n_w-1) */
        Map G(g);
        G = c_0.template rightCols<n_w - 1>() - c_end.template leftCols<n_w - 1>();
        return g + this->n_constraints;
    }
};

#endif /* COLLOCATION_CONSTRAINTS_HEADER */