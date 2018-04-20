#ifndef INITIAL_STATE_CONSTRAINTS_HEADER
#define INITIAL_STATE_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"

/**
 * This set of constraints simply ensures that the
 * estimate of the state at collocation point 0, waypoint 0 is
 * equal to the value of the initial state provided during construction of this constraint.
 */
template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w>
struct InitialStateConstraints
        : EqualityConstraint<
                InitialStateConstraints<
                        Scalar, Index, n_x, n_u, n_c, n_w>, Scalar, Index, n_x> {

private:

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<Scalar, n_x, 1>>;

    const Eigen::Matrix<Scalar, n_x, 1> initial_state;

public:

    static const Index derivatives = 0;

    template<typename InitialState>
    InitialStateConstraints(const InitialState &initial_state)
            : initial_state(initial_state.template cast<Scalar>()) {
    }

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    template<typename LD>
    Scalar *operator()(Scalar *g, const Scalar *x, LD &lagrange_derivatives) const {
        Map G(g);
        G = Get::state(x, 0, 0) - initial_state;
        return g + this->n_constraints;
    }
};

#endif /* COLLOCATION_CONSTRAINTS_HEADER */