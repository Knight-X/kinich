#pragma once
#include "layer.hpp"
namespace nn {
class input_layer : public Layer<activation::entity> 
{
  public:
    typedef Layer<activation::entity> base;

    input_layer() : base(0, 0, 0, 0) {}


    const nn_vec_t* forward_prop(const nn_vec_t* in, nn_size index) override
    {
        nn::layer_local_storage& storage = base::get_local_storage(index);
        storage._layer_curr_output = *in;
        nn_size n = base::out_dim;
        return &storage._layer_curr_output;
    }

    const nn_vec_t* backward_prop(const nn_vec_t* current_delta, nn_size index) override
    {
        return current_delta;
    }

};



}
