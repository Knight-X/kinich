#pragma once
#include "nn_datatype.hpp"
#include <cmath>
namespace nn
{
namespace activation
{
class activation_interface
{
public:
    virtual float_t result(const nn_vec_t& in, nn_size index) const = 0;

    virtual float_t differential_result(float_t result) const = 0;
    virtual nn_vec_t differential_result(const nn_vec_t& y, nn_size i) const
    {
        nn_vec_t tmp(y.size(), 0);
        tmp[i] = differential_result(y[i]);
        return tmp;
    }
};

class sigmoid : public activation_interface
{
public:
    float_t result(const nn_vec_t& in, nn_size index) const override
    {
        return float_t(1) / (float_t(1) + std::exp(-in[index]));
    }

    float_t differential_result(float_t y) const override
    {
        return y * (float_t(1) - y);
    }
};

class entity : public activation_interface
{
public:
    float_t result(const nn_vec_t& in, nn_size index) const override
    {
        return in[index];
    }

    float_t differential_result(float_t y) const override
    {
        return float_t(1);
    }
};
}
}
