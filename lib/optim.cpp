#include "vecops.h"
#include "matrix.h"
#include "tensor.h"
#include "optim.h"

SGD::SGD(unordered_map<unsigned, Tensor*> params, double lr){
    parameters = params;
    learning_rate = lr;
}
void SGD::step(Tensor& loss){
    for(auto& g : loss.get_grad()){
        assert(g.second.row_size() == 1);
        if(parameters.find(g.first) != parameters.end()){
            assert(g.second.col_size() == (*parameters[g.first]).size());
            (*parameters[g.first]).update(VecOps::mult(-1.0 * learning_rate, g.second.get(0)));
        }
    }
}