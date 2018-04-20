#ifndef CONTROL_RATE_CONSTRAINTS_HEADER
#define CONTROL_RATE_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "inequality_constraint.h"

/**
 * This set of constraints simply ensures that control rates lie within the specified bounds, that is,
 *
 * lower_bound <= u_dot <= upper_bound
 *
 * In general, an inequality bound is true if it is negative. So we reformulate the above to
 * the two bounds
 *
 * u_dot - upper_bound <= 0
 * lower_bound - u_dot <= 0
 *
 * @tparam Scalar: Only used to construct the lower and upper bounds.
 * @tparam n_constraints
*/
template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w>
struct ControlRateConstraints
        : InequalityConstraint<
                ControlRateConstraints<
                        Scalar, Index, n_x, n_u, n_c, n_w>, Scalar, Index, 2 * n_u * n_w * n_c> {

private:

    const Eigen::Matrix<Scalar, n_u, 1> lower_bound;
    const Eigen::Matrix<Scalar, n_u, 1> upper_bound;

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<Scalar, n_u, n_c>>;

public:

    static const Index derivatives = 1;

    template<typename Bound>
    ControlRateConstraints(const Bound &lower_bound,
                           const Bound &upper_bound)
            : lower_bound(lower_bound.template cast<Scalar>()),
              upper_bound(upper_bound.template cast<Scalar>()) {
    }

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    template<typename LD>
    Scalar *operator()(Scalar *g, const Scalar *x, LD &lagrange_derivatives) const {

        const Scalar *dx = lagrange_derivatives.template get<1>();
        for (Index i_w = 0; i_w < n_w; ++i_w) {

            Map upper(g);
            upper = Get::controlsAtWaypoint(dx, i_w).colwise() - upper_bound;
            g += n_u * n_c;

            Map lower(g);
            lower = (-Get::controlsAtWaypoint(dx, i_w)).colwise() + lower_bound;
            g += n_u * n_c;
        }
        return g;
    }
};

#endif /* CONTROL_RATE_CONSTRAINTS_HEADER */