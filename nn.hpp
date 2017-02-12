#pragma once
#include <iostream>
#include <memory>
#include "graph.hpp"
#include "layer.hpp"
#include "nn_datatype.hpp"
#include "lossfunc.hpp"
#include "gradient.hpp"
#include "optimizer.hpp"

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
    nn::Graph* getGraph() { return nngraph; }
    NNetwork(nn::Graph* g) : nngraph(g) {}
    

    nn_vec_t predict(const nn_vec_t& in);

    bool train(const std::vector<nn_vec_t>& in, 
        const std::vector<nn_vec_t>& t,
        size_t                batch_size,
        int                     epoch);

    void runTrainBatch(const nn_vec_t* in);

    const nn_vec_t* fprop(const nn_vec_t& in);
    const nn_vec_t* bprop(const nn_vec_t* in);
    void update_weight();
};
}
