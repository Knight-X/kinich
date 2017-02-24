#pragma once
#include<iostream>
#include <cstdio>
#include <vector>
#include "layer.hpp"
#include <map>

namespace nn
{
class Node
{
public:
    explicit Node(nn::baselayer* l) : layer(l) {}
    nn::baselayer* getLayer()
    {
        return layer;
    }
private:
    nn::baselayer* layer;
};

class Edge
{
public:
    explicit Edge(Node* p, Node* c) : Input(p), Output(c) {}
    Node* input() const
    {
        return Input;
    }
    Node* output() const
    {
        return Output;
    }
private:
    Node* Input;
    Node* Output;
};

class Graph
{
public:
    void addvertex(nn::Node *g);

    void addedge(nn::Edge *e);
    bool isEmpty()
    {
        return _node.size() == 0;
    }
    Node* lastNode()
    {
        return _node.back();
    }
    Node* firstNode()
    {
        return _node.front();
    }
    Edge* firstEdge()
    {
        return _edge.front();
    }
    Edge* lastEdge()
    {
        return _edge.back();
    }
    Edge* nextEdge(const nn::Node* n);
    Node* nextNode(const nn::Node* n);
private:
    std::vector<Node *> _node;
    std::vector<Edge*> _edge;
};

}
