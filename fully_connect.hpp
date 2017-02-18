#pragma once
#include "layer.hpp"
#include "activations.hpp"
namespace nn {
template<typename ActivationFunc>
class fully_connected_layer : public Layer<ActivationFunc> {
  public:
    typedef Layer<ActivationFunc> Base;

    fully_connected_layer(nn_size input_dim, nn_size output_dim, 
        bool has_bias = true) :
      Base(input_dim, output_dim, size_t(input_dim) * output_dim,
          has_bias ? output_dim : 0), has_bias(has_bias)
      {}


    const nn_vec_t* forward_prop(const nn_vec_t* inn, nn_size index) override
    {
      const nn_vec_t& in = *inn;
      nn::layer_local_storage& storage = Base::get_local_storage(index);

      nn_vec_t &a = storage._activations;
      nn_vec_t &out = storage._layer_curr_output;

      for (nn_size i = 0; i < Base::out_dim; i++) {
        a[i] = float_t(0.0);
        for (nn_size c = 0; c < Base::in_dim; c++) {
          a[i] += Base::weight_vec[c * Base::out_dim + i] * in[c];
          //a[i] += 1.0 * in[c];
//		  std::cout << a[i] << " "; 
        }
        if (has_bias)
        	a[i] += 0.0;
          //a[i] += Base::bias_vec[i];

//		std::cout << std::endl;
      }

      for (nn_size i = 0; i < Base::out_dim; i++) {
        //out[i] = Base::h_.result(a, index);
		out[i] = a[i];
      }

      return &out;
    }

    const nn_vec_t* backward_prop(const nn_vec_t* curr, nn_size index) override
    {
      const nn_vec_t& curr_delta = *curr;
      nn::layer_local_storage& storage = Base::get_local_storage(index);
      const nn_vec_t& prev_out = Base::_prev_layer->output(static_cast<int>(index));
      const nn::activation::activation_interface& prev_h = Base::_prev_layer->activation_func();
      nn_vec_t& prev_delta = storage._layer_prev_delta;
      nn_vec_t& deltaW = storage._delta_w;
      nn_vec_t& deltab = storage._delta_b;

      for (nn_size j = 0; j < Base::in_dim; j++)
      {
        for (nn_size c = 0; c < Base::out_dim; c++) {
            prev_delta[j] += curr_delta[c] * Base::weight_vec[c * Base::out_dim + j];
        }          
             prev_delta[j] *= prev_h.differential_result(prev_out[j]);
      }
      for (nn_size j = 0; j < Base::in_dim; j++) {
        for (nn_size c = 0; c < Base::out_dim; c++) {
            deltaW[j * Base::out_dim + c] += curr_delta[c] * prev_out[j]; 
        }
      }
      
      if (has_bias) {
        for (nn_size i = 0; i < Base::out_dim; i++)
          deltab[i] += curr_delta[i];
      }

      return &storage._layer_prev_delta;
    }

  protected:
    bool has_bias;
};
}
