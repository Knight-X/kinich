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
    std::vector<nn::nn_vec_t> resd;
    nn::NNetwork nnet {g, o, r};
};

TEST_F(NNnetTest, DefaultTest)
{
    std::vector<std::vector<nn::nn_vec_t>> in;
    std::vector<std::vector<nn::nn_vec_t>> gy;
    loadFile("letter-recognition-2.csv", 16, 3, &in, &gy);
    EXPECT_EQ(g, nnet.getGraph());
    nnet.add(&l1);
    nnet.add(&l2);
    
    std::vector<nn::nn_vec_t> res_fprop = nnet.fprop(in[0]);
    
    
    std::cout << "fully_connected_layer test start" << std::endl;
    //perform forward propagation
    //const nn::nn_vec_t* tmp2 = l2.forward_prop(newin, 0);
    const nn::nn_vec_t& res = l2.output(0);

    //do forward propagation for test case 
    nn::nn_vec_t tmp_res = nn::nn_vec_t(16, 0.0);
    nn::nn_vec_t target_out = nn::nn_vec_t(16, 0.0);
    nn::nn_vec_t& weight = l2.weight();
    nn::nn_vec_t& bias = l2.bias();
    nn::nn_size out_dim = l2.output_dim();
    nn::nn_size in_dim = l2.input_dim();
    nn::activation::activation_interface& h_ = l2.activation_func();
    for (nn::nn_size i = 0; i < out_dim; i++) {
        for (nn::nn_size c = 0; c < in_dim; c++) {
            tmp_res[i] += weight[c * out_dim + i] * in[0][0][c];
            cout << "target in " << in[0][0][c] << endl;
        }

        tmp_res[i] += bias[i];
    }
    for (nn::nn_size i = 0; i < out_dim; i++) {
        target_out[i] = h_.result(tmp_res, 0);
    }

    for (nn::nn_size i = 0; i < res.size(); i++) {
        EXPECT_EQ(res[i], target_out[i]);
        cout << "res" << res[i] << " ";
        cout << "target_out" << target_out[i] << " ";
    }
    cout << endl;

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
    cout << endl;
    
    //backward propagation
    nn::layer_local_storage& storage = l2.get_local_storage(0);
    const nn::nn_vec_t& prev_out = l2.prev()->output(0);
    const nn::activation::activation_interface& prev_h = l2.prev()->activation_func();
    nn::nn_vec_t& prev_delta = storage._layer_prev_delta;
    nn::nn_vec_t& delta_w = storage._delta_w;
    nn::nn_vec_t& delta_b = storage._delta_b;


    for (nn::nn_size j = 0; j < in_dim; j++) {
        for (nn::nn_size c = 0; c < out_dim; c++) {
            delta_w[j * out_dim + c] += delta[c] * prev_out[j]; 
        }
        prev_delta[j] *= prev_h.differential_result(prev_out[j]);
    }
    for (nn::nn_size j = 0; j < in_dim; j++) {
        for (nn::nn_size c = 0; c < out_dim; c++) {
            delta_w[j * out_dim + c] += delta[c] * prev_out[j];
        }
    }

    
    std::vector<nn::nn_vec_t> res_bprop = nnet.bprop(res_fprop, gy[0]);
    nn::nn_vec_t& dw = l2.weight_diff(0);
    nn::nn_vec_t& db = l2.bias_diff(0);
    nn::nn_vec_t d = nnet.output();
    for (nn::nn_size i = 0; i < delta.size(); i++) {
        cout << "delta " << delta[i] << " ";
        EXPECT_EQ(d[i], delta[i]);
    }

   for (nn::nn_size i = 0; i < out_dim; i++) 
            delta_b[i] += delta[i];

    for (nn::nn_size i = 0; i < delta_w.size(); i++) {
        EXPECT_EQ(delta_w[i], dw[i]);
    }
    cout << endl;
    for (nn::nn_size i = 0; i < delta_b.size(); i++) {
        EXPECT_EQ(delta_b[i], db[i]);
    }
    cout << endl;
    

}
