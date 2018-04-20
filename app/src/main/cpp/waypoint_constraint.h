#ifndef WAYPOINT_CONSTRAINT_HEADER
#define WAYPOINT_CONSTRAINT_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"

/**
 * This ensures that a single component of the state hits some specified 
 * waypoints. Since there are n_w waypoints, this yields n_w conditions.
 */
template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w, Index state_index>
struct WaypointConstraint
        : EqualityConstraint<
                WaypointConstraint<
                        Scalar, Index, n_x, n_u, n_c, n_w, state_index>, Scalar, Index, n_w> {

private:

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<Scalar, 1, n_w>>;

    const Eigen::Matrix<Scalar, 1, n_w> waypoints;

public:

    static const Index derivatives = 0;

    template<typename Waypoints>
    WaypointConstraint(const Waypoints &waypoints)
            : waypoints(waypoints.template cast<Scalar>()) {
        static_assert(state_index >= 0, "The state index must be nonnegative");
        static_assert(state_index < n_x, "The state index must be less than the size of the state");
    }

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    template<typename LD>
    Scalar *operator()(Scalar *g, const Scalar *x, LD &lagrange_derivatives) const {
        Map G(g);
        G = Get::statesAtCollocationPoint(x, n_c - 1).row(state_index) - waypoints;
        return g + this->n_constraints;
    }
};

#endif /* WAYPOINT_CONSTRAINT_HEADER */