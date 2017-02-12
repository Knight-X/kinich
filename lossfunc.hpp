#pragma once
#include "type_nn.hpp"

namespace nn {
  class mean_square_root{
    public:
      static float_t func(float_t y, float_t t) {
	return (y - t) * (y - t) / 2;
      }

      static float_t derivative_func(float_t y, float_t t) {
        return y - t;
      }
  };
}


