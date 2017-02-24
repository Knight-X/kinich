#include "graph.hpp"


namespace nn
{
void Graph::addvertex(nn::Node *node)
{
    _node.push_back(node);
}

void Graph::addedge(nn::Edge *e)
{
    _edge.push_back(e);
}

Edge* Graph::nextEdge(const nn::Node* n)
{
    Edge* res;
    nn::nn_size index;

    for (nn::nn_size i = 0; i < _edge.size(); i++) {
        if (n == _edge[i]->input())
            index = i;
    }
    if (index < _edge.size()) {
        res = _edge[index];
    } else {
        res = NULL;
    }
    return res;
}
}


