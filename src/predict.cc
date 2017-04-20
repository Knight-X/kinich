#include "predict.hpp"

namespace nn
{
nn_vec_t predict_err(mean_square_root* func, const nn_vec_t* y, const nn_vec_t* t)
{
    nn_vec_t pred(y->size());

    const nn_vec_t& _y = *y;
    const nn_vec_t& _t = *t;

    for (nn_size s = 0; s < _y.size(); s++) {
        pred[s] = func->func(_y[s], _t[s]);
    }

    return pred;
}
bool predict(mean_square_root* func, const nn_vec_t* y, const nn_vec_t* t)
{
    nn_vec_t pred(y->size());

    const nn_vec_t& _y = *y;
    const nn_vec_t& _t = *t;

    for (nn_size s = 0; s < _y.size(); s++) {
        if(clampOutput(_y[s]) !=  _t[s]) {
            return false;
        }
    }

    return true;
}
int clampOutput(double x)
{
    if (x < 0.1) return 0;
    else if (x > 0.9) return 1;
    else return -1;
}
}
