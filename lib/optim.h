#include "includes.h"

using namespace std;

class SGD{
    private:
        double learning_rate;
        unordered_map<unsigned, Tensor*> parameters;
    public:
        SGD(unordered_map<unsigned, Tensor*> params, double lr);
        void step(Tensor& loss);
};