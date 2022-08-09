#include "matrix.h"
#include "tensor.h"
#include "nn.h"
#include "optim.h"
#include <unordered_map>

using namespace std;

int main(){
    vector<Tensor> inputs;
    vector<Tensor> targets;
    inputs.push_back(Tensor(vector<double>{1.0, 0.0}));
    targets.push_back(Tensor(vector<double>{1.0}));
    inputs.push_back(Tensor(vector<double>{0.0, 1.0}));
    targets.push_back(Tensor(vector<double>{1.0}));
    inputs.push_back(Tensor(vector<double>{0.0, 0.0}));
    targets.push_back(Tensor(vector<double>{0.0}));
    inputs.push_back(Tensor(vector<double>{1.0, 1.0}));
    targets.push_back(Tensor(vector<double>{0.0}));

    LinearLayer fc1(2, 2);
    LinearLayer fc2(2, 1);
    cout << "r" << endl;
    unordered_map<unsigned, Tensor*> params = fc1.parameters();
    
    for(auto& p : fc2.parameters()){
        params[p.first] = p.second;
    }
    
    SGD optim(params, 0.5);
    
    for(int i = 0; i < 3000; i++){
        int sample = rand() % 4;
        Tensor o1 = fc1.feedforward(inputs[sample]);
        Tensor o2 = Sigmoid::apply(o1);
        Tensor o3 = fc2.feedforward(o2);
        Tensor output = Sigmoid::apply(o3);
        Tensor loss = MSELoss::apply(output, targets[sample]);
        loss.print();
        optim.step(loss);
    }
    for(int i = 0; i < 4; i++){
        Tensor o1 = fc1.feedforward(inputs[i]);
        Tensor o2 = Sigmoid::apply(o1);
        Tensor o3 = fc2.feedforward(o2);
        Tensor output = Sigmoid::apply(o3);
        inputs[i].print();
        output.print();
    }
}