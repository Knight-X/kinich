#include "gtest/gtest.h"
#include "layer.hpp"
#include "input.hpp"
#include "fully_connect.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "lossfunc.hpp"
#include "graph.hpp"
#include "io.cpp"
#include "gradient.hpp"


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
    nn::fully_connected_layer<nn::activation::sigmoid> l2 = nn::fully_connected_layer<nn::activation::sigmoid>(16, 3);
    nn::NNetwork nnet {g, o, r};
};

TEST_F(NNnetTest, DefaultTest)
{
    std::vector<std::vector<nn::nn_vec_t>> in;
    std::vector<std::vector<nn::nn_vec_t>> gy;
    loadFile("letter-recognition-2.csv", 16, 3, &in, &gy);
    EXPECT_EQ(g, nnet.getGraph());
    nn::nn_vec_t ans = nn::nn_vec_t(16, 0.0);
    nn::nn_vec_t out = nn::nn_vec_t(16, 0.0);
    nn::nn_vec_t& weight = l2.weight();
    nn::nn_vec_t& bias = l2.bias();
    nn::nn_size out_dim = l2.output_dim();
    nn::nn_size in_dim = l2.input_dim();
    nn::activation::activation_interface& h_ = l2.activation_func();
    std::cout << "fully_connected_layer test start" << std::endl;
    for (nn::nn_size i = 0; i < out_dim; i++) {
        for (nn::nn_size c = 0; c < in_dim; c++) {
            ans[i] += weight[c * out_dim + i] * in[0][0][c];

        }

        ans[i] += bias[i];
    }
    for (nn::nn_size i = 0; i < out_dim; i++) {
        out[i] = h_.result(ans, 0);
    }
    nnet.add(&l1);
    nnet.add(&l2);
    const nn::nn_vec_t* newin = l1.forward_prop(&in[0][0], 0);
    const nn::nn_vec_t* tmp2 = l2.forward_prop(newin, 0);
    const nn::nn_vec_t& res = *tmp2;

    for (nn::nn_size i = 0; i < res.size(); i++)
        EXPECT_EQ(res[i], out[i]);

    std::vector<nn::nn_vec_t> resd;
    resd.push_back(res);
    nn::nn_vec_t delta(resd[0].size());
    nn::mean_square_root* _lossfunc = new nn::mean_square_root();
    nn::nn_vec_t derivitive_e = nn::gradient(_lossfunc, &res, &gy[0][0]);
    nn::activation::activation_interface& _h = l2.activation_func();
    for (nn::nn_size index = 0; index < resd[0].size(); index++) {
        nn::nn_vec_t derivitive_y = _h.differential_result(resd[0], index);
        for (nn::nn_size j = 0; j < derivitive_e.size(); j++) {
            delta[index] += derivitive_e[j] * derivitive_y[j];
            cout << "derivitive_e " << derivitive_e[j] << " ";
            cout << "derivitive_y " << derivitive_y[j] << " ";
        }
        cout << endl;
    }
    for (nn::nn_size i = 0; i < delta.size(); i++)
        cout << "delta " << delta[i] << " ";

    cout << endl;
    const nn::nn_vec_t* input = &delta;
    input  = l2.backward_prop(input, 0);
    const nn::nn_vec_t& res_prev = *input;
    nn::nn_vec_t& delta_w = l2.weight_diff(0);
    nn::nn_vec_t& delta_b = l2.bias_diff(0);

    for (nn::nn_size i = 0; i < delta_b.size(); i++) {
        cout << "b" << delta_b[i] << " ";
    }
    cout << endl;
    for (nn::nn_size i = 0; i < delta_w.size(); i++) {
        cout << "w" << delta_w[i] << " ";
    }
    cout << endl;
    for (nn::nn_size i = 0; i < res_prev.size(); i++)
        cout << "prev_res" << res_prev[i] << " ";

}

