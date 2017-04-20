#pragma once
#include "nn_datatype.hpp"
#include "lossfunc.hpp"


namespace nn
{
nn_vec_t predict_err(mean_square_root* func, const nn_vec_t* y, const nn_vec_t* t);
bool predict(mean_square_root* func, const nn_vec_t* y, const nn_vec_t* t);
int clampOutput(double x);
}
