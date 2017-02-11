#pragma once
#include <iostream>
#include <memory>
#include "graph.hpp"
#include "layer.hpp"
#include "nn_datatype.hpp"

namespace nn {

class NNetwork 
{
  private:
    nn::Graph* nngraph;
    //OptimizerMethod optimizer_;

  public:
    //OptimizerMethod optimizer();

    void init_weight();

    void add(nn::baselayer* layer);

    nn_vec_t predict(const nn_vec_t& in);

    bool train(const std::vector<nn_vec_t>& in, 
        const std::vector<nn_vec_t>& t,
        size_t                batch_size,
        int                     epoch);
    void runTrainEpoch(const std::vector<nn_vec_t>& in);

    void runTrainBatch(const nn_vec_t& in);

    nn_vec_t fprop(const nn_vec_t& in);
    nn_vec_t bprop(const nn_vec_t& in);
    void update_weight();
};
}
