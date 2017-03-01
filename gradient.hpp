#pragma once
#include "nn_datatype.hpp"
#include "lossfunc.hpp"

namespace nn
{
nn_vec_t gradient(mean_square_root* func, const nn_vec_t* y, const nn_vec_t* t);
}
