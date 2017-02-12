#include "gtest/gtest.h"
#include "layer.hpp"
#include "input.hpp"
#include "fully_connect.hpp"
#include "nn.hpp"


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

class NNnetTest : public QuickTest {
  protected:
    virtual void SetUp() {
      QuickTest::SetUp();
    }

    nn::Graph *g = new nn::Graph(); 
    nn::NNetwork nnet = nn::NNetwork(g);
};

TEST_F(NNnetTest, DefaultTest) {
  EXPECT_EQ(g, nnet.getGraph());
}
      
