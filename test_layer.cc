#include "gtest/gtest.h"
#include "layer.hpp"
#include "input.hpp"
#include "fully_connect.hpp"
#include "io.cpp"
#include "activations.hpp"


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

class LayerTest : public QuickTest
{
protected:
    virtual void SetUp()
    {
        QuickTest::SetUp();
    }

    nn::input_layer l1 = nn::input_layer();
    nn::fully_connected_layer<nn::activation::sigmoid> l2 = nn::fully_connected_layer<nn::activation::sigmoid>(10, 10);
};

//<<<<<<< HEAD
TEST_F(LayerTest, DefaultTest) {
  nn::nn_vec_t in; 
  nn::nn_vec_t gy;
  loadFile("letter-recognition-2.csv", 16, 3, &in, &gy);
  nn::nn_vec_t ans = nn::nn_vec_t(10, 0);
  nn::nn_vec_t out = nn::nn_vec_t(10, 0);
  EXPECT_EQ(0, l1.input_dim());
  EXPECT_EQ(10, l2.input_dim());
  const nn::nn_vec_t* tmp = l1.forward_prop(&in, 0);
  nn::nn_vec_t res = *tmp;
  for (int i = 0; i < res.size(); i++) {
    EXPECT_EQ(in[i], res[i]);
  }
  EXPECT_EQ(l1.result(), l1.forward_prop(&in, 0));
  EXPECT_EQ(&in, l1.backward_prop(&in, 0));
  //EXPECT_EQ(l2.result(), l2.forward_prop(&ans, 0));
  const nn::nn_vec_t* tmp2 = l2.forward_prop(&in, 0);
  nn::nn_vec_t res2 = *tmp2;
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

  for (int i = 0; i < res2.size(); i++) {
	//std::cout << in[i] << " ";
  	EXPECT_EQ(out[i], res2[i]);
	//std::cout << res2[i] << " ";
  } 

}

