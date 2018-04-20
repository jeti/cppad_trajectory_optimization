#ifndef DYNAMICS_CONSTRAINTS_HEADER
#define DYNAMICS_CONSTRAINTS_HEADER

#include "variable_getter.h"
#include "Eigen/Dense"
#include "equality_constraint.h"

/**
 * This set of constraints simply ensures that the dynamics are actually satisfied,
 * that is, that the estimated derivatives (using Lagrange interpolation polynomials)
 * equal the actual dynamics. These should hold at every collocation and waypoint for
 * each state variables, yielding n_c * n_x * n_w constraints.
 */
template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w>
struct DynamicsConstraints
    : EqualityConstraint<
        DynamicsConstraints<
            Scalar, Index, n_x, n_u, n_c, n_w>, Scalar, Index, n_c * n_x * n_w> {

private:

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;
    using Map = Eigen::Map<Eigen::Matrix<Scalar, n_x, n_c >>;

    const Scalar mass = 1.0;
    const Scalar gravity = 9.81;
    const Scalar mass_gravity = mass * gravity;

public:

    static const Index derivatives = 1;

    /**
     * The dynamics have the state vector
     *
     * x = [ px, py, pz, vx, vy, vz  ]
     */
    void dynamics(const Scalar *x, Scalar *dx, Index waypoint_index) const {

        static_assert(n_x == 6, "This function is only valid for states of size 6");
        static_assert(n_u == 4, "This function is only valid for controls of size 4");

        Eigen::Ref<const Eigen::Matrix<Scalar, n_u, n_c>>
            u = Get::controlsAtWaypoint(x, waypoint_index);
        auto thrust = u.row(0).array();
        auto phi = u.row(1).array();
        auto theta = u.row(2).array();
        auto psi = u.row(3).array();

        Eigen::Map<Eigen::Matrix<Scalar, n_x, n_c >> dX(dx);
        dX.topRows(3) = Get::statesAtWaypoint(x, waypoint_index).bottomRows(3).eval();
        dX.row(3) = -thrust * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta));
        dX.row(4) = thrust * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta));
        dX.row(5) = -thrust * cos(phi) * cos(theta) + mass_gravity;
    }

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    template<typename LD>
    Scalar *operator()(Scalar *g, const Scalar *x, LD &lagrange_derivatives) const {

        /* For each waypoint, compare the dynamics with the Lagrange interpolation polynomial */
        const Scalar *dx = lagrange_derivatives.template get<1>();
        for (Index i_w = 0; i_w < n_w; ++i_w) {
            dynamics(x, g, i_w);
            Map G(g);
            G -= Get::statesAtWaypoint(dx, i_w);
            g += n_x * n_c;
        }
        return g;
    }
};

#endif /* DYNAMICS_CONSTRAINTS_HEADER */