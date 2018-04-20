#ifndef FUSED_CONSTRAINT_HEADER
#define FUSED_CONSTRAINT_HEADER

#include <tuple>
#include "lagrange_derivatives.h"
#include "template_integer.h"

/**
 *
 * @tparam Tuple
 * @tparam Scalar
 * @tparam Index
 * @tparam n_x
 * @tparam n_u
 * @tparam n_c
 * @tparam n_w
 * @tparam Array This is the type used for the upper and lower bounds. It must be templated to
 * accept the size of the array, and it must define a type trait called "Scalar":
 *
 * Array<n_constraints> bound = ...
 *
 */
template<typename Tuple, typename Scalar, typename Index, Index n_x, Index n_u, Index n_c, Index n_w, template<Index size> class Array>
struct FusedConstraint {

private:

    /*
     * ------------------------------------
     *
     * Number of Constraints
     *
     * ------------------------------------
     */
    static constexpr Index numberOfConstraints(Integer<0>) {
        return std::tuple_element<0, Tuple>::type::n_constraints;
    }

    template<Index i>
    static constexpr Index numberOfConstraints(Integer<i>) {
        return std::tuple_element<i, Tuple>::type::n_constraints
               + numberOfConstraints(Integer<i - 1>());
    }

    /*
     * ------------------------------------
     *
     * Maximum derivative required by constraints
     *
     * ------------------------------------
     */
    static constexpr Index maxDerivative(Integer<0>) {
        return std::tuple_element<0, Tuple>::type::derivatives;
    }

    template<Index i>
    static constexpr Index maxDerivative(Integer<i>) {
        return std::tuple_element<i, Tuple>::type::derivatives >= maxDerivative(Integer<i - 1>())
               ? std::tuple_element<i, Tuple>::type::derivatives : maxDerivative(Integer<i - 1>());
    }

public:

    /** The number of constraint classes that are fused together */
    static const Index n_constraint_classes = std::tuple_size<Tuple>::value;
    static_assert(n_constraint_classes > 0, "You must specify at least one constraint class");

    /** The size of the constraint vector */
    static const Index n_constraints = numberOfConstraints(Integer<n_constraint_classes - 1>());

    /** The maximum derivative term required by the constraints */
    static const Index max_derivative = maxDerivative(Integer<n_constraint_classes - 1>());

    using ArrayScalar =typename Array<n_constraints>::Scalar;

private:

    /*
     * ------------------------------------
     *
     * Upper Bound
     *
     * ------------------------------------
     */
    static constexpr ArrayScalar *fillUpperBound(ArrayScalar *bound, Integer<n_constraint_classes - 1>) {
        return std::tuple_element<n_constraint_classes - 1, Tuple>::type::template writeUpperBound<ArrayScalar>(bound);
    }

    template<Index i>
    static constexpr ArrayScalar *fillUpperBound(ArrayScalar *bound, Integer<i>) {
        return fillUpperBound(std::tuple_element<i, Tuple>::type::template writeUpperBound<ArrayScalar>(bound),
                              Integer<i + 1>());
    }

    static Array<n_constraints> createUpperBound() {
        Array<n_constraints> bound;
        fillUpperBound(bound.data(), Integer<0>());
        return bound;
    }

    /*
     * ------------------------------------
     *
     * Lower Bound
     *
     * ------------------------------------
     */
    static constexpr ArrayScalar *fillLowerBound(ArrayScalar *bound, Integer<n_constraint_classes - 1>) {
        return std::tuple_element<n_constraint_classes - 1, Tuple>::type::template writeLowerBound<ArrayScalar>(bound);
    }

    template<Index i>
    static constexpr ArrayScalar *fillLowerBound(ArrayScalar *bound, Integer<i>) {
        return fillLowerBound(std::tuple_element<i, Tuple>::type::template writeLowerBound<ArrayScalar>(bound),
                              Integer<i + 1>());
    }

    static Array<n_constraints> createLowerBound() {
        Array<n_constraints> bound;
        fillLowerBound(bound.data(), Integer<0>());
        return bound;
    }

    using LD = LagrangeDerivatives<Scalar, Index, n_x, n_u, n_c, n_w, max_derivative>;
    LD lagrange_derivatives;

public:

    const Array<n_constraints> lower_bound = createLowerBound();

    const Array<n_constraints> upper_bound = createUpperBound();

    /** A tuple of the constraint classes that we fuse together.
     * Note that we are referencing the passed-in constraints to avoid making copies */
    Tuple constraints;

    /* Constructors */
    template<typename CollocationPoints>
    FusedConstraint(const Tuple &constraints,
                    const CollocationPoints &collocation_points)
            : constraints(constraints),
              lagrange_derivatives(LD(collocation_points.template cast<Scalar>())) {
    }

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    Scalar *operator()(Scalar *g, const Scalar *x) {
        lagrange_derivatives.template generate<max_derivative>(x);
        return evaluateConstraintsTuple(g, x, Integer<0>());
    }

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g.data() + n_constraints. */
    template<typename U, typename V>
    auto operator()(U &g, const V &x) -> decltype(g.data()) {
        return (*this)(g.data(), x.data());
    }

private:

    /*
     * ------------------------------------
     *
     * Evaluate
     *
     * ------------------------------------
     */
    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    Scalar *evaluateConstraintsTuple(Scalar *g, const Scalar *x, Integer<n_constraint_classes - 1>) {
        return std::get<n_constraint_classes - 1>(constraints)(g, x, lagrange_derivatives);
    }

    /** Evaluate the constraint at x and store the values in g.
     * Then return a pointer to g + n_constraints. */
    template<Index i>
    Scalar *evaluateConstraintsTuple(Scalar *g, const Scalar *x, Integer<i>) {
        return evaluateConstraintsTuple(std::get<i>(constraints)(g, x, lagrange_derivatives),
                                        x,
                                        Integer<i + 1>());
    }
};

#endif /* FUSED_CONSTRAINT_HEADER */