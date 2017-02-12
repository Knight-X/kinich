#pragma once


namespace nn {
  class Optimizer {
    public:
    virtual void update(const nn_vec_t& derivative_w, nn_vec_t& w) = 0;
  };

  class stochastic_gradient_descent : public Optimizer
  { 
    public:
    stochastic_gradient_descent() : learning_rate(float_t(0)) {}
    void update(const nn_vec_t& derivative_w, nn_vec_t& w) override
    {
      for (int i = 0; i < w.size(); i++) 
      {
        w[i] = w[i] -  learning_rate * derivative_w[i];
      }
    }

    private:
      float_t learning_rate;
  };
  
}
