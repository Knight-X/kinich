#pragma once
#include "nn_datatype.hpp"

namespace nn {
  template<typename LossFunc>
  nn_vec_t gradient(const nn_vec_t* y, const nn_vec_t* t)
  {
    nn_vec_t grad(y->size());
    
    for (nn_size s = 0; s < y->size(); s++) {
	grad[s] = LossFunc::derivative_func(y[s], t[s]);
    }
    return grad;
  }
}
