#include "predict.hpp"

namespace nn
{
nn_vec_t predict(mean_square_root* func, const nn_vec_t* y, const nn_vec_t* t)
{
    nn_vec_t pred(y->size());

    const nn_vec_t& _y = *y;
    const nn_vec_t& _t = *t;

    for (nn_size s = 0; s < _y.size(); s++) {
        pred[s] = func->func(_y[s], _t[s]);
    }

    return pred;
}
}
