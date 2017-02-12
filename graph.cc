#include "graph.hpp"


namespace nn {
    void Graph::addvertex(nn::Node *node)
    {
      _node.push_back(node);
    }

    void Graph::addedge(nn::Edge *e)
    {
      _edge.push_back(e);
    }

    Edge* Graph::nextEdge(const nn::Edge* e)
    {
      Edge* res;
      for (int i = 0; i < _edge.size(); i++) {
	if (e->output() == _edge[i]->output())
	 res = _edge[i];	
	
     } 
	return res;
    }
}


