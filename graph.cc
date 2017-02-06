#include "nngraph.hpp"


namespace nn {
    void Graph::addvertex(nn::Node *node)
    {
      _node.push_back(node);
    }

    void Graph::addedge(nn::Edge *e)
    {
      _edge.push_back(e);
    }
}


