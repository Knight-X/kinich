#pragma once
#include <iostream>
#include "activations.hpp"
#include "layer_param.hpp"
#include "nn_datatype.hpp"


namespace nn {
    struct layer_local_storage {
      nn_vec_t _activations;

      nn_vec_t _layer_curr_output;

      nn_vec_t _layer_prev_delta;

      nn_vec_t _delta_w;

      nn_vec_t _delta_b;
    };
    class baselayer {
      public:
        baselayer(nn_size input_dim, nn_size output_dim, nn_size weight_dim, nn_size bias_dim)
            : _prev_layer(nullptr), _next_layer(nullptr), _weight_init(std::make_shared<nn::sqrtinit>()), 
            _bias_init(std::make_shared<nn::constinit>(float_t(0)))
        {
            setjobscount(1);
            setlayerparam(input_dim, output_dim, weight_dim, bias_dim);

        }
        virtual ~baselayer() = default;
        
        void init_weight();
        const nn_vec_t& result(nn_size worker_i);
        const nn_vec_t& delta(nn_size worker_i);
        baselayer* prev();
        baselayer* next();
        nn_vec_t& weight();
        nn_vec_t& bias();
        nn_vec_t& weight_diff(nn_size index);
        nn_vec_t& bias_diff(nn_size index);
        const nn_vec_t& curr_layer_output(nn_size worker_i);
        const nn_vec_t& prev_layer_delta(nn_size worker_i);

        const nn_vec_t& output(nn_size index) const { return layer_storage[index]._activations; };
        virtual const nn_vec_t& forward_prop(const nn_vec_t &in, nn_size index) = 0;

        virtual const nn_vec_t& backward_prop(const nn_vec_t& current, nn_size index) = 0;
        virtual activation::activation_interface& activation_func() = 0;
        nn_size input_dim();
        nn_size output_dim(); 
        void setlayerparam(nn_size input_size, nn_size output_size, nn_size w_dim, nn_size b_dim);
        void setjobscount(nn_size count);
        template<typename InitBiasType>
        baselayer& init_b(const InitBiasType& b);

        template<typename InitWeightType>
        baselayer& init_w(const InitWeightType& w);
        layer_local_storage& get_local_storage(nn_size i) {
          return layer_storage[i];
        }
      protected:
        nn_size in_dim;
        nn_size out_dim;
        baselayer *_prev_layer;
        baselayer *_next_layer;
        nn_vec_t weight_vec;
        nn_vec_t bias_vec;
        std::shared_ptr<nn::initfunc_interface> _weight_init;
        std::shared_ptr<nn::initfunc_interface> _bias_init;
      private:
        std::vector<layer_local_storage> layer_storage;
    };

    template<typename ActivationFunc>
      class Layer : public baselayer {
        public:
          Layer(nn_size in_dim, nn_size out_dim,
              nn_size weight_dim, nn_size bias_dim)
          : baselayer(in_dim, out_dim, weight_dim, bias_dim){}
          ~Layer() {}

          nn::activation::activation_interface& activation_func() override {
            return h_;
          }
        protected:
          ActivationFunc h_;
      };

}
