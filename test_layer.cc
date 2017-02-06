#include "gtest/gtest.h"
#include "layer.hpp"
#include "input.hpp"
#include "fully_connect.hpp"


using ::testing::TestWithParam;
using ::testing::Values;

class QuickTest : public testing::Test {
  protected:

    virtual void SetUp() {
      start_time = time(NULL);
    }

    virtual void TearDown() {
      const time_t end_time = time(NULL);

      EXPECT_TRUE(end_time - start_time <= 100) << "The go";
    }

    time_t start_time;
};

class LayerTest : public QuickTest {
  protected:
    virtual void SetUp() {
      QuickTest::SetUp();
    }

    nn::input_layer l1 = nn::input_layer();
    nn::fully_connected_layer<nn::activation::sigmoid> l2 = nn::fully_connected_layer<nn::activation::sigmoid>(10, 10);
};

TEST_F(LayerTest, DefaultTest) {
  nn::nn_vec_t in = nn::nn_vec_t(10, 0); 
  nn::nn_vec_t gy = nn::nn_vec_t(10, 2);
  nn::nn_vec_t ans = nn::nn_vec_t(10, 0.5);
  EXPECT_EQ(0, l1.input_dim());
  EXPECT_EQ(10, l2.input_dim());
  EXPECT_EQ(in, l1.forward_prop(in, 0));
  EXPECT_EQ(in, l1.backward_prop(in, 0));
  EXPECT_EQ(ans, l2.forward_prop(in, 0));
}
      
