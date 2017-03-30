#include "nn.hpp"
#include "graph.hpp"

namespace nn
{
void NNetwork::add(nn::baselayer* layer)
{
    if (nngraph->isEmpty()) {
        nn::Node *n = new nn::Node(layer);
        nngraph->addvertex(n);
    } else {
        nn::Node *c = nngraph->lastNode();
        nn::Node *n = new nn::Node(layer);
        layer->setprevlayer(c->getLayer());
        c->getLayer()->setnextlayer(layer);
        nngraph->addvertex(n);
        nn::Edge *e = new nn::Edge(c, n);
        nngraph->addedge(e);
    }
}


void NNetwork::init_weight()
{
}

bool NNetwork::train(const std::vector<std::vector<nn_vec_t>>& in,
                     const std::vector<std::vector<nn_vec_t>>& target,
                     size_t                  batch_size,
                     int                     epoch)
{
    init_weight();
    for (int iter = 0; iter < epoch; iter++) {
        for (nn::nn_size index = 0; index < in.size(); index = index + batch_size) {
            runTrainBatch(in[index], target[index], batch_size);
        }
    }
    return true;
}


void NNetwork::runTrainBatch(const std::vector<nn_vec_t>& in, const std::vector<nn_vec_t>& t, nn_size batch_size)
{
    bprop(fprop(in), t);
    calculate_result(batch_size, t[0].size());
    std::cout << "mse: " << mse << std::endl;
    update_weight(batch_size);
    mse = 0.0;
}

const std::vector<nn_vec_t>& NNetwork::fprop(const std::vector<nn_vec_t>& in)
{
    if (forward_res.size() > 0)
        forward_res.erase(forward_res.begin(), forward_res.end());

    nn::Node* outNode = nngraph->firstNode();
    const nn_vec_t* input = nullptr;
    for (nn::nn_size i = 0; i < in.size(); i++) {
        input = &in[i];
        input = outNode->getLayer()->forward_prop(input, 0);

        nn::Edge* e = nngraph->firstEdge();

        while(e) {
            outNode = e->output();
            input = outNode->getLayer()->forward_prop(input, 0);
            e = nngraph->nextEdge(outNode);
        }
        forward_res.push_back(*input);
    }
    return forward_res;
}

const std::vector<nn_vec_t>& NNetwork::bprop(const std::vector<nn_vec_t>& in, const std::vector<nn_vec_t>& t)
{
    if (backward_res.size() > 0)
        backward_res.erase(backward_res.begin(), backward_res.end());

    nn::Node* outNode = nngraph->lastNode();
    nn::Node* inNode = nullptr;
    const nn::nn_vec_t* input = nullptr;
    for (nn::nn_size i = 0; i < in.size(); i++) {
        nn_vec_t delta(in[i].size());
        nn_vec_t derivitive_e = nn::gradient(_lossfunc, &in[i], &t[i]);
        nn_vec_t pred_v = nn::predict(_lossfunc, &in[i], &t[i]);
        collect_error(pred_v);
        nn::activation::activation_interface& _h = outNode->getLayer()->activation_func();
        for (nn_size index = 0; index < in[i].size(); index++) {
            nn_vec_t derivative_y = _h.differential_result(in[i], index);
            for (nn_size j = 0; j < derivitive_e.size(); j++) {
                delta[index] += derivitive_e[j]  * derivative_y[j];
            }
        }
        input = &delta;
        output_delta = delta;
//        for (int s = 0; s < delta.size(); s++) {
//            std::cout << "delta" << delta[s] << std::endl;
//            std::cout << "in" << in[0][s] << std::endl;
//        }
        nn::Edge* e = nngraph->lastEdge();
        while(e) {
            outNode = e->output();
            inNode = e->input();
            input = outNode->getLayer()->backward_prop(input, 0);
            e = nngraph->prevEdge(inNode);
        }
        backward_res.push_back(*input);
    }
    return backward_res;
}

void NNetwork::update_weight(nn_size batch_size)
{
    nn::Edge* e = nngraph->firstEdge();
    nn::Node* node = nullptr;
    while (e) {
        node = e->output();
        e->input()->getLayer()->update(_optimizer, batch_size);
        e->output()->getLayer()->update(_optimizer, batch_size);
        e = nngraph->nextEdge(node);
    }
}
void NNetwork::collect_error(nn_vec_t r)
{
    for (nn_size i = 0; i < r.size(); i++) {
        mse += r[i];
    }
}

void NNetwork::calculate_result(nn_size t, nn_size dim)
{
    mse = mse / t * dim;
}
}
