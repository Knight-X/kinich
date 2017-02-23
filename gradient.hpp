#pragma once
#include "nn_datatype.hpp"
#include "lossfunc.hpp"

namespace nn
{
nn_vec_t gradient(mean_square_root* func, const nn_vec_t* y, const nn_vec_t* t)
{
    nn_vec_t grad(y->size());

    const nn_vec_t& _y = *y;
    const nn_vec_t& _t = *t;
    for (nn_size s = 0; s < _y.size(); s++) {
        grad[s] = func->derivative_func(_y[s], _t[s]);
    }
    return grad;
}
}
