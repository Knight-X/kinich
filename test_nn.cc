#include "gtest/gtest.h"
#include "layer.hpp"
#include "input.hpp"
#include "fully_connect.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "lossfunc.hpp"
#include "graph.hpp"
#include "io.cpp"


using ::testing::TestWithParam;
using ::testing::Values;

class QuickTest : public testing::Test
{
protected:

    virtual void SetUp()
    {
        start_time = time(NULL);
    }

    virtual void TearDown()
    {
        const time_t end_time = time(NULL);

        EXPECT_TRUE(end_time - start_time <= 100) << "The go";
    }

    time_t start_time;
};

class NNnetTest : public QuickTest
{
protected:
    virtual void SetUp()
    {
        QuickTest::SetUp();
    }

    nn::Graph *g = new nn::Graph();
    nn::mean_square_root* r = new nn::mean_square_root();
    nn::Optimizer* o = new nn::stochastic_gradient_descent();
    nn::input_layer l1 = nn::input_layer();
    nn::fully_connected_layer<nn::activation::sigmoid> l2 = nn::fully_connected_layer<nn::activation::sigmoid>(10, 10);
    nn::NNetwork nnet {g, o, r};
};

TEST_F(NNnetTest, DefaultTest)
{
    nn::nn_vec_t in;
    nn::nn_vec_t gy;
    loadFile("letter-recognition-2.csv", 16, 3, &in, &gy);
    EXPECT_EQ(g, nnet.getGraph());
    nn::nn_vec_t ans = nn::nn_vec_t(10, 0);
    nn::nn_vec_t out = nn::nn_vec_t(10, 0);
    nn::nn_vec_t& weight = l2.weight();
    nn::nn_vec_t& bias = l2.bias();
    nn::nn_size out_dim = l2.output_dim();
    nn::nn_size in_dim = l2.input_dim();
    nn::activation::activation_interface& h_ = l2.activation_func();
    for (nn::nn_size i = 0; i < out_dim; i++) {
        for (nn::nn_size c = 0; c < in_dim; c++) {
            ans[i] += weight[c * out_dim + i] * in[c];
        }
        ans[i] += bias[i];
    }
    for (nn::nn_size i = 0; i < out_dim; i++) {
        out[i] = h_.result(ans, 0);
    }
    nnet.add(&l1);
    nnet.add(&l2);
    const nn::nn_vec_t* tmp2 = nnet.fprop(in);
    nn::nn_vec_t res = *tmp2;

    for (nn::nn_size i = 0; i < res.size(); i++) 
        EXPECT_EQ(res[i], out[i]);

}

