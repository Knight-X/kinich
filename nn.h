#pragma once

class NNetwork 
{
  private:
    Graph* nngraph;
    OptimizerMethod optimizer_;

  public:
    OptimizerMethod optimizer();

    void init_weight();

    void add(std::shared_ptr<layer_base> layer);

    vec_t predict(const vec_t& in);

    bool train(const std::vector<vec_t>& in, 
        const std::vector<T>& t,
        size_t                batch_size,
        int                     epoch);
};
