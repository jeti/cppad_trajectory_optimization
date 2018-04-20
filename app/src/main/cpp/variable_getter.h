#ifndef VARIABLE_GETTER_HEADER
#define VARIABLE_GETTER_HEADER

/* iostream is just imported to get the endl operator. */
#include <iostream>
#include "Eigen/Dense"

using std::endl;

/**
 * This class provides some easy accessors when dealing with a raw pointer
 * to variables that sits in memory like this (in column-major format):
 *
 *                      waypoint 1,    waypoint 2, ...,    waypoint n_w
 * collocation 1:   [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 * collocation 2:   [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 *     ...          [      ...    ,       ...    , ...,       ...      ]
 * collocation n_c: [       x     ,        x     , ...,        x       ]
 *                  [       u     ,        u     , ...,        u       ]
 *
 * followed by the vector of times for each waypoint, [ t_1,... t_{n_w} ]
 *
 * @tparam n_x: The size of the state
 * @tparam n_u: The size of the control input
 * @tparam n_c: The number of collocation points
 * @tparam n_w: The number of waypoints
 */
template<typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w>
class VariableGetter {
private:

    /** Return a reference to the states and controls as a matrix.
     * This is private because using this function would make your code unportable.
     * Specifically, it would not be portable because
     * it would directly expose the underlying representation of the data.
     */
    static constexpr auto asMatrix(const Scalar *raw_ptr)
    -> decltype(Eigen::Map<const Eigen::Matrix<Scalar, (n_x + n_u) * n_c, n_w>>(raw_ptr)) {
        return Eigen::Map<const Eigen::Matrix<Scalar, (n_x + n_u) * n_c, n_w>>(raw_ptr);
    }

    static constexpr auto asMatrix(Scalar *raw_ptr)
    -> decltype(Eigen::Map<Eigen::Matrix<Scalar, (n_x + n_u) * n_c, n_w>>(raw_ptr)) {
        return Eigen::Map<Eigen::Matrix<Scalar, (n_x + n_u) * n_c, n_w>>(raw_ptr);
    }

public:

    static const Index n_vars = ((n_x + n_u) * n_c + 1) * n_w;

    /** Set all of the variables to zero */
    static void setZero(Scalar *raw_ptr) {
        Eigen::Map<Eigen::Matrix<Scalar, n_vars, 1>>(raw_ptr).setZero();
    }

    /** Return a reference to the submatrix
     *
     * [ x[i,0], ..., x[i,n_w-1]]
     * [ u[i,0], ..., u[i,n_w-1]]
     *
     * that is, a matrix of shape (n_x+n_u) x n_w,
     * holding all of the states and controls at collocation point i_c.
     */
    static constexpr auto
    varsAtCollocationPoint(const Scalar *raw_ptr, Index i_c)
    -> decltype(asMatrix(raw_ptr).template middleRows<n_x + n_u>((n_x + n_u) * i_c)) {
        return asMatrix(raw_ptr).template middleRows<n_x + n_u>((n_x + n_u) * i_c);
    }

    static constexpr auto
    varsAtCollocationPoint(Scalar *raw_ptr, Index i_c)
    -> decltype(asMatrix(raw_ptr).template middleRows<n_x + n_u>((n_x + n_u) * i_c)) {
        return asMatrix(raw_ptr).template middleRows<n_x + n_u>((n_x + n_u) * i_c);
    }

    /** Return a reference to the submatrix
     *
     * [ x[i,0], ..., x[i,n_w-1]]
     *
     * that is, a matrix of shape n_x x n_w,
     * holding all of the states at collocation point i_c.
     */
    static constexpr auto statesAtCollocationPoint(const Scalar *raw_ptr, Index i_c)
    -> decltype(varsAtCollocationPoint(raw_ptr, i_c).template topRows<n_x>()) {
        return varsAtCollocationPoint(raw_ptr, i_c).template topRows<n_x>();
    }

    static constexpr auto statesAtCollocationPoint(Scalar *raw_ptr, Index i_c)
    -> decltype(varsAtCollocationPoint(raw_ptr, i_c).template topRows<n_x>()) {
        return varsAtCollocationPoint(raw_ptr, i_c).template topRows<n_x>();
    }

    /** Return a reference to
     *
     * x[i,j]
     *
     * that is, a matrix of shape n_x x 1,
     * holding the state at collocation point i_c and waypoint i_w.
     */
    static constexpr auto state(const Scalar *raw_ptr, Index i_c, Index i_w)
    -> decltype(statesAtCollocationPoint(raw_ptr, i_c).col(i_w)) {
        return statesAtCollocationPoint(raw_ptr, i_c).col(i_w);
    }

    static constexpr auto state(Scalar *raw_ptr, Index i_c, Index i_w)
    -> decltype(statesAtCollocationPoint(raw_ptr, i_c).col(i_w)) {
        return statesAtCollocationPoint(raw_ptr, i_c).col(i_w);
    }

    /** Return a reference to the submatrix
     *
     * [ u[i,0], ..., u[i,n_w-1]]
     *
     * that is, a matrix of shape n_u x n_w,
     * holding all of the controls at collocation point i_c.
     */
    static constexpr auto controlsAtCollocationPoint(const Scalar *raw_ptr, Index i_c)
    -> decltype(varsAtCollocationPoint(raw_ptr, i_c).template bottomRows<n_u>()) {
        return varsAtCollocationPoint(raw_ptr, i_c).template bottomRows<n_u>();
    }

    static constexpr auto controlsAtCollocationPoint(Scalar *raw_ptr, Index i_c)
    -> decltype(varsAtCollocationPoint(raw_ptr, i_c).template bottomRows<n_u>()) {
        return varsAtCollocationPoint(raw_ptr, i_c).template bottomRows<n_u>();
    }

    /** Return a reference to
     *
     * u[i,j]
     *
     * that is, a matrix of shape n_u x 1,
     * holding all of the controls at collocation point i_c and waypoint i_w.
     */
    static constexpr auto control(const Scalar *raw_ptr, Index i_c, Index i_w)
    -> decltype(controlsAtCollocationPoint(raw_ptr, i_c).col(i_w)) {
        return controlsAtCollocationPoint(raw_ptr, i_c).col(i_w);
    }

    static constexpr auto control(Scalar *raw_ptr, Index i_c, Index i_w)
    -> decltype(controlsAtCollocationPoint(raw_ptr, i_c).col(i_w)) {
        return controlsAtCollocationPoint(raw_ptr, i_c).col(i_w);
    }

    /** Return a reference to the submatrix
     *
     * [ x[0,i], ..., x[n_c-1,i]]
     * [ u[0,i], ..., u[n_c-1,i]]
     *
     * that is, a matrix of shape (n_x + n_u) x n_c,
     * holding all of the states and controls at waypoint i_w.
     */
    static constexpr auto varsAtWaypoint(const Scalar *raw_ptr, Index i_w)
    -> decltype(Eigen::Map<const Eigen::Matrix<Scalar,
                                               n_x + n_u,
                                               n_c>>(asMatrix(raw_ptr).col(i_w).data())) {
        return Eigen::Map<const Eigen::Matrix<Scalar,
                                              n_x + n_u,
                                              n_c>>(asMatrix(raw_ptr).col(i_w).data());
    }

    static constexpr auto varsAtWaypoint(Scalar *raw_ptr, Index i_w)
    -> decltype(Eigen::Map<Eigen::Matrix<Scalar,
                                         n_x + n_u,
                                         n_c>>(asMatrix(raw_ptr).col(i_w).data())) {
        return Eigen::Map<Eigen::Matrix<Scalar, n_x + n_u, n_c>>(asMatrix(raw_ptr).col(i_w).data());
    }

    /** Return a reference to the submatrix
     *
     * [ x[0,i], ..., x[n_c-1,i]]
     *
     * that is, a matrix of shape n_x x n_c,
     * holding all of the states at waypoint i_w.
     */
    static constexpr auto statesAtWaypoint(const Scalar *raw_ptr, Index i_w)
    -> decltype(varsAtWaypoint(raw_ptr, i_w).template topRows<n_x>()) {
        return varsAtWaypoint(raw_ptr, i_w).template topRows<n_x>();
    }

    static constexpr auto statesAtWaypoint(Scalar *raw_ptr, Index i_w)
    -> decltype(varsAtWaypoint(raw_ptr, i_w).template topRows<n_x>()) {
        return varsAtWaypoint(raw_ptr, i_w).template topRows<n_x>();
    }

    /** Return a reference to the submatrix
     *
     * [ u[0,i], ..., u[n_c-1,i]]
     *
     * that is, a matrix of shape n_u x n_c,
     * holding all of the controls at waypoint i_w.
     */
    static constexpr auto controlsAtWaypoint(const Scalar *raw_ptr, Index i_w)
    -> decltype(varsAtWaypoint(raw_ptr, i_w).template bottomRows<n_u>()) {
        return varsAtWaypoint(raw_ptr, i_w).template bottomRows<n_u>();
    }

    static constexpr auto controlsAtWaypoint(Scalar *raw_ptr, Index i_w)
    -> decltype(varsAtWaypoint(raw_ptr, i_w).template bottomRows<n_u>()) {
        return varsAtWaypoint(raw_ptr, i_w).template bottomRows<n_u>();
    }

    /** Return a reference to the submatrix
     *
     * [ t[0], ..., t[n_w-1]]
     *
     * that is, a matrix of shape 1 x n_w,
     * holding all of the times.
     */
    static constexpr auto times(const Scalar *raw_ptr)
    -> decltype(Eigen::Map<const Eigen::Matrix<Scalar, 1, n_w>>(
        raw_ptr + (n_x + n_u) * n_c * n_w)) {
        return Eigen::Map<const Eigen::Matrix<Scalar, 1, n_w>>(raw_ptr + (n_x + n_u) * n_c * n_w);
    }

    static constexpr auto times(Scalar *raw_ptr)
    -> decltype(Eigen::Map<Eigen::Matrix<Scalar, 1, n_w>>(raw_ptr + (n_x + n_u) * n_c * n_w)) {
        return Eigen::Map<Eigen::Matrix<Scalar, 1, n_w>>(raw_ptr + (n_x + n_u) * n_c * n_w);
    }

    /** Return a nice formatted string of all of the variables */
    static std::string asString(const Scalar *raw_vars) {
        std::stringstream out;

        /* First, write the times to the string  */
        out << endl;
        out << endl;
        out << "Times: " << times(raw_vars) << endl;

        out << "----------------------------" << endl;
        out << endl;
        out << "Controls: " << endl;
        out << endl;
        for (Index i_c = 0; i_c < n_c; ++i_c) {
            out << "Collocation point " << i_c << endl;
            out << controlsAtCollocationPoint(raw_vars, i_c) << endl;
        }
        out << endl;
        out << "----------------------------" << endl;

        out << endl;
        out << "States: " << endl;
        out << endl;
        for (Index i_c = 0; i_c < n_c; ++i_c) {
            out << "Collocation point " << i_c << endl;
            out << statesAtCollocationPoint(raw_vars, i_c) << endl;
        }
        out << endl;
        out << "----------------------------" << endl;
        return out.str();
    }
};

#endif /* VARIABLE_GETTER_HEADER */