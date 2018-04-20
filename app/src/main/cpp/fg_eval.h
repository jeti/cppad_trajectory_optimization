#ifndef FG_EVAL_HEADER
#define FG_EVAL_HEADER

#include "cppad/example/cppad_eigen.hpp"
#include "variable_getter.h"

template<template<typename T> class Vector, typename Scalar, typename FusedConstraints, typename Get>
struct FG_eval {

private:

    FusedConstraints &fused_constraints;

public:

    FG_eval(FusedConstraints &fused_constraints)
        : fused_constraints(fused_constraints) {
    }

    using ADvector = Vector<CppAD::AD<Scalar>>;

    void operator()(ADvector &fg, const ADvector &x) {

        assert(fg.size() == 1 + fused_constraints.n_constraints);
        assert(x.size() == Get::n_vars);

        /* The first entry in fg is the cost function.
         * In our case, this is simply the sum of the times */
        auto times = Get::times(x.data());
        fg[0] = 0;
        for (size_t i = 0; i < times.size(); ++i)
            fg[0] += times(i);

        /* The remaining entries are the constraints */
        fused_constraints(fg.data() + 1, x.data());
    }
};

#endif /* FG_EVAL_HEADER */
