#include "nn.hpp"
#include "graph.hpp"

namespace nn {

void NNetwork::add(nn::baselayer* layer)
{
  nn::Node *n = new nn::Node(layer);  
  nn::Node *c = nngraph->lastNode();
  nngraph->addvertex(n);
  nn::Edge *e = new nn::Edge(n, c);
  nngraph->addedge(e);
}

void NNetwork::init_weight()
{
	}

bool NNetwork::train(const std::vector<nn_vec_t>& in,
    const std::vector<nn_vec_t>& target,
    size_t                  batch_size,
    int                     epoch)
{
   init_weight(); 
   for (int iter = 0; iter < epoch; iter++) {
     for (int index = 0; index < in.size(); index = index + batch_size) {
       runTrainBatch(&in[index]);
     }
   } 
}


void NNetwork::runTrainBatch(const nn_vec_t *in)
{
  bprop(fprop(*in));
  update_weight();
}

const nn_vec_t* NNetwork::fprop(const nn_vec_t &in)
{
  nn::Node* s = nngraph->firstNode();
  nn::Edge* e = nngraph->firstEdge();
  nn::Node* inNode = e->input();
  nn::Node* outNode = e->output();

  const nn_vec_t* input = &in;
  while(e) {
    input = inNode->getLayer()->forward_prop(input, 0);
    e = nngraph->nextEdge(e);
    inNode = e->input();
    outNode = e->output();
  } 
    return input;
}

const nn_vec_t* NNetwork::bprop(const nn_vec_t *in)
{
  nn::Node* s = nngraph->lastNode();
  nn::Edge* e = nngraph->lastEdge();
  nn::Node* inNode = e->input();
  nn::Node* outNode = e->output();
  const nn_vec_t* input = in;
  while(e) {
    input = inNode->getLayer()->backward_prop(input, 0);
    e = nngraph->nextEdge(e);
    inNode = e->input();
    outNode = e->output();
  }
    return input;
}

void NNetwork::update_weight()
{
  nn::Edge* e = nngraph->firstEdge();
  while (e) {
    e->input()->getLayer()->update_weight();
    e->output()->getLayer()->update_weight();
    e = nngraph->nextEdge(e);
  }
}
}
