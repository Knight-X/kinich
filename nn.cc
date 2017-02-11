#include "nn.hpp"
#include "graph.hpp"

namespace nn {

void NNetwork::add(nn::baselayer* layer)
{
  nn::Node *n = new nn::Node(layer);  
  nn::Node *c = nngraph->lastvertex();
  nngraph->addvertex(n);
  nn::Edge *e = new nn::Edge(c, n);
  nngraph->addedge(e);
}

bool NNetwork::train(const std::vector<nn_vec_t>& in,
    const std::vector<nn_vec_t>& target,
    size_t                  batch_size,
    int                     epoch)
{
   init_weight(); 
   std::vector<nn_vec_t> input = getEpochData(batch_size); 
   for (int iter = 0; iter < epoch; iter++) {
    runTrainBatch(input);
   } 
}

void NNetwork::runTrainEpoch(const std::vector<nn_vec_t>& in)
{

  for (int tp = 0; tp < (int)in.size(); tp++)
  {
    runTrainBatch(in[tp]);
  }
}

void NNetwork::runTrainBatch(const nn_vec_t &in)
{
  bprop(fprop(in));
  update_weight();
}

nn_vec_t NNetwork::fprop(const nn_vec_t &in)
{
  nn::Node* s = nngraph->firstNode();
  nn::Edge* e = nngraph->firstEdge();
  nn::Node* inNode = e->Input();
  nn::Node* outNode = e->Output();
  nn_vec_t& input = in;
  while(e) {
    input = inNode->forward_prop(input);
    e = nngraph->getNextEdge(e);
    inNode = e->Input();
    outNode = e->Output();
  } 
    return out;
}

nn_vec_t NNetwork::bprop(const nn_vec_t &in)
{
  nn::Node* s = nngrap->lastNode();
  nn::Node* e = nngraph->lastEdge();
  nn::Node* inNode = e->Input();
  nn::Node* outNode = e->Output();
  nn_vec_t& input = in;
  while(e) {
    input = inNode->backward_prop(input);
    e = nngraph->getNextEdge(e);
    inNode = e->Input();
    outNode = e->Output();
  }
    return out;
}

void NNetwork::update_weight()
{
  nn::Node* s = nngraph->firstNode();
  while (s) {
    s->update_weight();
    s = nngraph->nextNode(s);
  }
}
}
