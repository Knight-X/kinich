#pragma once
#include<iostream>
#include <cstdio>
#include <vector>
#include "layer.hpp"
#include <map>

namespace nn {
  class Node {
    public:
    explicit Node(nn::baselayer* l) : layer(l) {}
    private:
      nn::baselayer* layer;
  };
  class Edge {
    public:
      explicit Edge(nn::baselayer* p, nn::baselayer* c) : Input(p), Output(c) {}
      nn::baselayer* input() { return Input; }
      nn::baselayer* output() { return Output; }
    private:
      nn::baselayer* Input;
      nn::baselayer* Output;
  };
  class Graph {
    public:
      void addvertex(nn::Node *g);

      void addedge(nn::Edge *e);
    private:
      std::vector<Node *> _node;
      std::vector<Edge*> _edge;
  };
}
