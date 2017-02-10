#include "nn.h"
#include "graph.h"


void NNetwork::add(std::shared_ptr<layer_base> layer)
{
  .add(layer);
}

bool NNetwork::train(const std::vector<vec_t>& in,
    const std::vector<T>& t,
    size_t                  batch_size,
    int                     epoch)
{

    for (int iter = 0; iter < epoch; iter++) {
      runTrainEpoch(in);
    }
}

void NNetwork::runTrainEpoch(const std::vector<vec_t>& in)
{
  init_weight();

  for (int tp = 0; tp < (int)in.size(); tp++)
  {
    runTrainBatch(&in[i], &t[i], 
        static_cast<int>(std::min(batch_size, in.size() - 1)
          ));
  }
    return true;
}

void runTrainBatch(const nn_vec_t &in)
{
  bprop(fprop(in));
  update_weight();
}

nn_vec_t NNetwork::fprop(const nn_vec_t &in)
{
    return _layers->first_node->forward_prop(in);
}

nn_vec_t NNetwork::bprop(const nn_vec_t &in)
{
    return _layers->last_node->backward_prop(in);
}

void NNetwork::update_weight()
{
  for (node in nodes) {
    node->update_weight();
  }
}
