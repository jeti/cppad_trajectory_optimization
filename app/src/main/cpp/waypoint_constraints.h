#ifndef WAYPOINT_CONSTRAINTS_HEADER
#define WAYPOINT_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"

/**
 * This set of constraints simply ensures that the states hit
 * all of the specified waypoints. Since there are n_w waypoints
 * and the state is of size n_x, this yields n_w * n_x conditions.
 */
template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w>
struct WaypointConstraints
        : EqualityConstraint<
                WaypointConstraints<
                        Scalar, Index, n_x, n_u, n_c, n_w>, Scalar, Index, n_x * n_w> {

private:

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<Scalar, n_x, n_w >>;

    const Eigen::Matrix<Scalar, n_x, n_w> waypoints;

public:

    static const Index derivatives = 0;

    template<typename Waypoints>
    WaypointConstraints(const Waypoints &waypoints)
            : waypoints(waypoints.template cast<Scalar>()) {
    }

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    template<typename U, typename V, typename LD>
    U *operator()(U *g, const V *x, LD &lagrange_derivatives) const {
        Map G(g);
        G = Get::statesAtCollocationPoint(x, n_c - 1) - waypoints;
        return g + this->n_constraints;
    }
};

#endif /* COLLOCATION_CONSTRAINTS_HEADER */