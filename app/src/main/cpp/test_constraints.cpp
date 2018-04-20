#include <iostream>

using std::cout;
using std::endl;

#include "cppad/example/cppad_eigen.hpp"

/* Other STL libraries that we are using */
#include <chrono>
#include <vector>
#include <tuple>
#include <array>

/* Our classes */
#include "equality_constraint.h"
#include "inequality_constraint.h"
#include "fused_contraint.h"
#include "variable_getter.h"
#include "fg_eval.h"

#include "collocation_constraints.h"
#include "control_rate_constraints.h"
#include "dynamics_constraints.h"
#include "initial_state_constraints.h"
#include "smooth_control_constraints.h"
#include "waypoint_constraint.h"
#include "waypoint_constraints.h"

/* Eigen */
#include "Eigen/Dense"

/* Types */
using Scalar = double;
using ADScalar = CppAD::AD<Scalar>;
using Index = size_t;

template<Index size>
using Array = Eigen::Matrix<Scalar, size, 1>;

template<typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

int main() {

    /* Sizes */
    const Index n_x = 6;
    const Index n_u = 4;
    const Index n_c = 11;
    const Index n_w = 6;

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;
    using GetAD = VariableGetter<ADScalar, Index, n_x, n_u, n_c, n_w>;
    const Index n_vars = Get::n_vars;

    /*
     * ----------------------------------------------
     *
     * Initial State
     *
     * ----------------------------------------------
     */
    Array<n_x> initial_state = Array<n_x>::Random();

    /*
     * ----------------------------------------------
     *
     * Waypoints
     *
     * ----------------------------------------------
     */
    Eigen::Matrix<Scalar, n_x, n_w> waypoints;
    waypoints.setZero();
    waypoints.col(0) << 2.0, 2.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 2)
        waypoints.col(1) << 4.0, 2.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 3)
        waypoints.col(2) << 8.0, 0.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 4)
        waypoints.col(3) << 4.0, -2.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 5)
        waypoints.col(4) << 2.0, -2.0, -1.0, 0.0, 0.0, 0.0;
    if (n_w >= 6)
        waypoints.col(5) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    /*
     * ----------------------------------------------
     *
     * Control Rate Bounds
     *
     * ----------------------------------------------
     */
    const Scalar degrees = M_PI / 180;
    const Scalar max_angular_rate = 30 * degrees;

    Array<n_u> control_rate_upper;
    control_rate_upper << 20, max_angular_rate, max_angular_rate, max_angular_rate;

    Array<n_u> control_rate_lower = -control_rate_upper;

    /*
     * ----------------------------------------------
     *
     * Collocation points
     *
     * ----------------------------------------------
     */
    Array<n_c> collocation_points = generateCollocationPoints<Scalar, Index, n_c>();

    /*
     * ----------------------------------------------
     *
     * Constraints
     *
     * ----------------------------------------------
     */
    auto constraints = std::make_tuple(
            CollocationConstraints<ADScalar, Index, n_x, n_u, n_c, n_w>(),
            ControlRateConstraints<ADScalar, Index, n_x, n_u, n_c, n_w>(control_rate_lower,
                                                                        control_rate_upper),
            DynamicsConstraints<ADScalar, Index, n_x, n_u, n_c, n_w>(),
            InitialStateConstraints<ADScalar, Index, n_x, n_u, n_c, n_w>(initial_state),
            SmoothControlConstraints<ADScalar, Index, n_x, n_u, n_c, n_w>(),
            WaypointConstraints<ADScalar, Index, n_x, n_u, n_c, n_w>(waypoints)
    );


    /* Create the fused constraint */
    FusedConstraint<decltype(constraints), ADScalar, Index, n_x, n_u, n_c, n_w, Array>
            fused_constraints(constraints, collocation_points);

    const Index N = fused_constraints.n_constraints;

    cout << "Number of Constraints : " << N << endl;

    cout << "Lower bound: ";
    for (Index i = 0; i < N; ++i)
        cout << fused_constraints.lower_bound[i] << ", ";
    cout << endl;

    cout << "Upper bound: ";
    for (Index i = 0; i < N; ++i)
        cout << fused_constraints.upper_bound[i] << ", ";
    cout << endl;

    cout << "Evaluate: ";
    std::vector<ADScalar> g(N);
    std::vector<ADScalar> X(Get::n_vars);
    for (Index i = 0; i < X.size(); ++i)
        X[i] = i;
    const std::vector<ADScalar> x = X;
    for (Index i = 0; i < x.size(); ++i)
        cout << x[i] << ", ";
    cout << endl;

    fused_constraints(g, x);
    for (Index i = 0; i < N; ++i)
        cout << g[i] << ", ";
    cout << endl;

    return 0;
}