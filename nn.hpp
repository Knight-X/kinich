#pragma once
#include <iostream>
#include <memory>
#include "graph.hpp"
#include "layer.hpp"
#include "nn_datatype.hpp"
#include "lossfunc.hpp"
#include "optimizer.hpp"
#include "input.hpp"

namespace nn
{

class NNetwork
{
private:
    nn::Graph* nngraph;
    Optimizer* _optimizer;
    mean_square_root*  _lossfunc;
    std::vector<nn_vec_t> forward_res;
    std::vector<nn_vec_t> backward_res;

public:
    //OptimizerMethod optimizer();

    void init_weight();

    void add(nn::baselayer* layer);
    nn::Graph* getGraph()
    {
        return nngraph;
    }
    NNetwork(nn::Graph* g, nn::Optimizer* o, mean_square_root* f) : nngraph(g), _optimizer(o), _lossfunc(f)
    {
        nn::input_layer l1 = nn::input_layer();


    }



    nn_vec_t predict(const nn_vec_t& in);

    bool train(const std::vector<std::vector<nn_vec_t>>& in,
               const std::vector<std::vector<nn_vec_t>>& t,
               size_t                batch_size,
               int                     epoch);

    void runTrainBatch(const std::vector<nn_vec_t>& in, const std::vector<nn_vec_t>& t, nn_size batch_size);

    const std::vector<nn_vec_t>& fprop(const std::vector<nn_vec_t>& in);
    const std::vector<nn_vec_t>& bprop(const std::vector<nn_vec_t>& in, const std::vector<nn_vec_t>& t);
    void update_weight(nn_size batch_size);
};
}
