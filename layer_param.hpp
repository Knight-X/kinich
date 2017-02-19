#include <vector>
#include <iostream>
#include <cmath>
#include <random>

namespace nn {
    class initfunc_interface{
      public:
        virtual void fill(nn_vec_t *param, nn_size input, nn_size output) = 0;
    };

    class sqrtinit : public initfunc_interface {
      public:
        sqrtinit() : initfunc_interface() , scalefactor(float_t(6)) {}
        explicit sqrtinit(float_t value) : initfunc_interface(), scalefactor(value) {}
        
        void fill(nn_vec_t *param, nn_size input, nn_size output) override {
          const float_t param_base = std::sqrt(scalefactor / (input + output));
          for(nn_vec_iter Iter = param->begin(); Iter != param->end(); ++Iter) {
            static std::mt19937 gen(time(0));
            std::uniform_real_distribution<double> dis(-param_base, param_base);
            *Iter = dis(gen);
          }

        }
      protected:
        float_t scalefactor;
    };

    class constinit : public initfunc_interface {
        public:
          constinit() : initfunc_interface(), scalefactor(float_t(0)) {}
          explicit constinit(float_t value) : initfunc_interface(), scalefactor(value) {}

          void fill(nn_vec_t *param, nn_size input, nn_size output) override {
            
            std::fill(param->begin(), param->end(), scalefactor);
          }
        protected:
          float_t scalefactor;
    };

}
