#include "layer.hpp"

namespace nn {

void baselayer::init_weight()
{
  _weight_init->fill(&weight_vec, in_dim, out_dim );

  _bias_init->fill(&bias_vec, in_dim, out_dim);

}

nn_size baselayer::input_dim()
{
  return in_dim;
}



const nn_vec_t& baselayer::curr_layer_output(nn_size worker_i)
{
  return layer_storage[worker_i]._layer_curr_output;
}

const nn_vec_t& baselayer::prev_layer_delta(nn_size worker_i)
{
  return layer_storage[worker_i]._layer_prev_delta;
}

nn_vec_t& baselayer::weight()
{
  return weight_vec;
}

nn_vec_t& baselayer::bias()
{
  return bias_vec;
}

nn_vec_t& baselayer::weight_diff(nn_size index)
{
  return layer_storage[index]._delta_w;
}

nn_vec_t& baselayer::bias_diff(nn_size index)
{
  return layer_storage[index]._delta_b;
}

baselayer* baselayer::next()
{
  return _next_layer;
}

baselayer* baselayer::prev()
{
  return _prev_layer;
}

template<typename InitWeightType>
baselayer& baselayer::init_w(const InitWeightType& f)
{
  _weight_init = std::make_shared<InitWeightType>(f);
  return *this;
}

template<typename InitBiasType>
baselayer& baselayer::init_b(const InitBiasType& b)
{
  _bias_init = std::make_shared<InitBiasType>(b);
  return *this;
}

void baselayer::setlayerparam(nn_size input_size, nn_size output_size, nn_size w_dim, nn_size b_dim)
{
  in_dim = input_size;
  out_dim = output_size;
  weight_vec.resize(w_dim);
  bias_vec.resize(b_dim);
  for (auto& wps : layer_storage) {
    wps._layer_curr_output.resize(output_size);
    wps._activations.resize(output_size);
    wps._layer_prev_delta.resize(input_size);
    wps._delta_w.resize(w_dim);
    wps._delta_b.resize(b_dim);
  }
}

void baselayer::setjobscount(nn_size count)
{
  layer_storage.resize(count);
}
}
