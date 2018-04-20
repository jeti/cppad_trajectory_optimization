/* CppAD */
#include "cppad/example/cppad_eigen.hpp"
#include "cppad/ipopt/solve.hpp"

/* Properly define cout and endl for your system */
#include "cout.h"

/* Other STL libraries that we are using */
#include <chrono>
#include <vector>
#include <tuple>
#include <array>
#include <string>

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

template<typename T>
std::string to_string(T value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w, typename Logger>
void log_state(Scalar *vars,
               Logger &logger,
               bool waypoints = true,
               const Eigen::IOFormat &format = Eigen::IOFormat(4, 0, " ", "\n", "", "", "", "")) {

    using Get = VariableGetter<Scalar, Index, n_x, n_u, n_c, n_w>;

    /* First, write the times to the string  */
    logger << endl
           << endl;
    logger << "Times: ";
    auto t = Get::times(vars);
    logger << t.format(format);
    logger << endl;

    logger << "----------------------------";
    logger << endl
           << endl;
    logger << "Controls: ";
    logger << endl
           << endl;
    if (waypoints) {
        for (Index i_w = 0; i_w < n_w; ++i_w) {
            logger << "Waypoint " << i_w << endl;
            auto u = Get::controlsAtWaypoint(vars, i_w);
            logger << u.format(format);
            logger << endl
                   << endl;
        }
    } else {
        for (Index i_c = 0; i_c < n_c; ++i_c) {
            logger << "Collocation point " << i_c << endl;
            auto u = Get::controlsAtCollocationPoint(vars, i_c);
            logger << u.format(format);
            logger << endl
                   << endl;
        }
    }
    logger << endl;
    logger << "----------------------------";
    logger << endl
           << endl;
    logger << "States: ";
    logger << endl
           << endl;
    if (waypoints) {
        for (Index i_w = 0; i_w < n_w; ++i_w) {
            logger << "Waypoint " << i_w << endl;
            auto x = Get::statesAtWaypoint(vars, i_w);
            logger << x.format(format);
            logger << endl
                   << endl;
        }
    } else {
        for (Index i_c = 0; i_c < n_c; ++i_c) {
            logger << "Collocation point " << i_c << endl;
            auto x = Get::statesAtCollocationPoint(vars, i_c);
            logger << x.format(format);
            logger << endl
                   << endl;
        }
    }
    logger << endl;
    logger << "----------------------------";
    logger << endl;
}

/* The "main" function that will be called from Java */
#ifdef __ANDROID__

extern "C" JNIEXPORT jstring JNICALL
Java_io_jeti_trajectoryoptimization_MainActivity_stringFromJNI(JNIEnv *env,
                                                               jobject object,
                                                               jint iterations = 100,
                                                               jdouble tolerance = 1e-3,
                                                               jboolean adaptive_mu_strategy = true,
                                                               jboolean hessian_approximation = true,
                                                               jboolean sparse_forward = true,
                                                               jboolean sparse_reverse = true,
                                                               jint print_level = 0) {

#else

    int main() {

#endif

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
     * TODO: Pass from Java
     *
     * ----------------------------------------------
     */
    Array<n_x> initial_state;
    initial_state.setZero();

    /*
     * ----------------------------------------------
     *
     * Waypoints
     * TODO: Pass from Java
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

    /*
     * ----------------------------------------------
     *
     * Initial guess
     *
     * ----------------------------------------------
     */
    Array<n_vars> initial_guess;
    initial_guess.setZero();

    /* Set the times to the specified value. */
    Scalar time = 1;
    Get::times(initial_guess.data()).fill(time);

    /* Now compute the difference between each of the waypoints.
     * The first column is the first waypoint - the initial state.
     * The rest are waypoint i - waypoint (i-1)*/
    Eigen::Matrix<Scalar, n_x, n_w> differences;
    differences << waypoints.col(0) - initial_state,
        waypoints.template rightCols<n_w - 1>() - waypoints.template leftCols<n_w - 1>();

    /* Now we can interpolate the values for the state using the formula
     *
     * interpolated = final - (1 - collocation_point) * ( final - initial )
     */
    Array<n_c> dc = -collocation_points.array() + 1;
    for (Index i_c = 0; i_c < n_c; ++i_c) {
        auto x = Get::statesAtCollocationPoint(initial_guess.data(), i_c);
        x = waypoints - dc(i_c) * differences;
    }

    /* Note that we are leaving the initial control guesses as zeros. */

    /*
     * ----------------------------------------------
     *
     * Upper and lower variable bounds
     *
     * ----------------------------------------------
     */
    Array<n_vars> lower_bound;
    Array<n_vars> upper_bound;

    /* State bounds. We will set the same state bounds for each collocation point and waypoint.  */
    Array<n_x> x_min;
    x_min << -2e19, -2e19, -2e19, -2e19, -2e19, -2e19;

    Array<n_x> x_max;
    x_max << 2e19, 2e19, 0, 2e19, 2e19, 2e19;

    for (Index i_w = 0; i_w < n_w; ++i_w) {
        auto x_lower = Get::statesAtWaypoint(lower_bound.data(), i_w);
        x_lower = x_min.replicate<1, n_c>();
        auto x_upper = Get::statesAtWaypoint(upper_bound.data(), i_w);
        x_upper = x_max.replicate<1, n_c>();
    }

    /* Control bounds */
    Array<n_u> u_min;
    u_min << 0, -30 * degrees, -30 * degrees, -2 * 360 * degrees;

    Array<n_u> u_max;
    u_max << 2 * 9.91, 30 * degrees, 30 * degrees, 2 * 360 * degrees;

    for (Index i_w = 0; i_w < n_w; ++i_w) {
        auto u_lower = Get::controlsAtWaypoint(lower_bound.data(), i_w);
        u_lower = u_min.template replicate<1, n_c>();
        auto u_upper = Get::controlsAtWaypoint(upper_bound.data(), i_w);
        u_upper = u_max.template replicate<1, n_c>();
    }

    /* Time bounds */
    const Scalar time_min = 0;
    const Scalar time_max = 10;
    Get::times(lower_bound.data()).fill(time_min);
    Get::times(upper_bound.data()).fill(time_max);

    /*
     * ----------------------------------------------
     *
     * IPOPT Solve
     *
     * ----------------------------------------------
     */
    std::string options;
    options += "Integer print_level  " + to_string(print_level) + "\n";
    options += "Integer max_iter     " + to_string(iterations) + "\n";
    options += "Numeric tol          " + to_string(tolerance) + "\n";
    if (sparse_forward)
        options += "Sparse  true         forward\n";
    if (sparse_reverse)
        options += "Sparse  true         reverse\n";
    if (adaptive_mu_strategy)
        options += "String  mu_strategy  adaptive\n";
    if (hessian_approximation)
        options += "String  hessian_approximation  limited-memory\n";

    /* Allocate space for the solution */
    CppAD::ipopt::solve_result<Vector<Scalar>> solution;

    /* Create the problem */
    cout << "Initial conditions created" << endl;
    using FG = FG_eval<Vector, Scalar, decltype(fused_constraints), GetAD>;
    FG fg_eval(fused_constraints);

    cout << "Initial conditions created" << endl;

    /* Solve the problem */
    long long elapsed = 0;
    const int timing_iterations = 1;
    for (int timing_iteration = 0; timing_iteration < timing_iterations; ++timing_iteration) {

        /* Time how long a solve iteration takes */
        auto start = std::chrono::high_resolution_clock::now();
        CppAD::ipopt::solve<Vector<Scalar>, FG>(options,
                                                initial_guess,
                                                lower_bound,
                                                upper_bound,
                                                fused_constraints.lower_bound,
                                                fused_constraints.upper_bound,
                                                fg_eval,
                                                solution);
        auto finish = std::chrono::high_resolution_clock::now();

        /* Accumulate the elapsed time */
        elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    }

    // The stringstring that we will return
    std::stringstream output;
    output << "Elapsed seconds for " << timing_iterations << " calls: " << (elapsed / 1e9) << endl;

    /*
     * ----------------------------------------------
     *
     * Print the solution and other diagnostics
     *
     * ----------------------------------------------
     */
    /* Log the solution. */
    output << endl << "Cost = " << solution.obj_value << endl << endl;
    const bool verbose = true;
    if (verbose)
        log_state<Scalar, Index, n_x, n_u, n_c, n_w, decltype(output)>(solution.x.data(), output);

    /* Beep when finished */
    output << '\a';
    std::string out = output.str();
    return env->NewStringUTF(out.c_str());
}