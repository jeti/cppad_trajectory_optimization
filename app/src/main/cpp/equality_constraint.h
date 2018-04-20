#ifndef EQUALITY_CONSTRAINT_HEADER
#define EQUALITY_CONSTRAINT_HEADER

#include <algorithm>

template<typename T, typename Scalar_, typename Index_, Index_ n_constraints_>
struct EqualityConstraint {

    using Scalar = Scalar_;
    using Index = Index_;
    static const Index n_constraints = n_constraints_;

    template<typename BT>
    static BT *writeLowerBound(BT *bounds) {
        std::fill(bounds, bounds + n_constraints, static_cast<BT>(0.0));
        return bounds + n_constraints;
    }

    template<typename BT>
    static BT *writeUpperBound(BT *bounds) {
        std::fill(bounds, bounds + n_constraints, static_cast<BT>(0.0));
        return bounds + n_constraints;
    }

private:

    EqualityConstraint() {};
    friend T;

};

#endif /* EQUALITY_CONSTRAINT_HEADER */
