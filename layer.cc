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

void baselayer::update(stochastic_gradient_descent* optimizer, nn_size batch)
{
  optimizer->update(layer_storage._delta_w, weight_vec);
  optimizer->update(layer_storage._delta_b, bias_vec);
}

void baselayer::merge_delta(nn_size thread_size, nn_size batch_size)
{
  auto& ls = layer_storage;
  float_t totaldw = 0.0;
  float_t totaldb = 0.0;
  for (nn_size i = 0; i < thread_size; i++) {
    ls[0]._delta_w += ls[i]._delta_w;
    ls[0]._delta_b += ls[i]._delta_b;
  }


  std::transform(ls[0]._delta_w.begin(), ls[0]._delta_w.end(), ls[0]._delta_w.begin(), [&](float_t x) { return x / batch_size; });
  std::transform(ls[0]._delta_b.begin(), ls[0]._delta_b.end(), ls[0]._delta_b.begin(), [&](float_t x) { return x / batch_size; });

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

baselayer* baselayer::next()
{
  return _next_layer;
}

baselayer* baselayer::prev()
{
  return _prev_layer;
}
const nn_vec_t* baselayer::result(){
  return &layer_storage[0]._layer_curr_output;
}
}

