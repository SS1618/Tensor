#include "vecops.h"

vector<double> VecOps::add(vector<double>& a, vector<double>& b){
    assert(a.size() == b.size());
    vector<double> output;
    output.reserve(a.size());
    for(int i = 0; i < a.size(); i++){
        output.emplace_back(a[i] + b[i]);
    }
    return output;
}
vector<double> VecOps::hadamard(vector<double>& a, vector<double>& b){
    assert(a.size() == b.size());
    vector<double> output;
    output.reserve(a.size());
    for(int i = 0; i < a.size(); i++){
        output.emplace_back(a[i] * b[i]);
    }
    return output;
}
double VecOps::sum(vector<double>& a){
    double output = 0.0;
    for(int i = 0; i < a.size(); i++){
        output += a[i];
    }
    return output;
}
vector<double> VecOps::randn(unsigned sz){
    vector<double> output;
    for(int i = 0; i < sz; i++){
        output.push_back(rand() / double(RAND_MAX));
    }
    return output;
}
vector<double> VecOps::mult(double v, vector<double> x){
    vector<double> output;
    for(int i = 0; i < x.size(); i++){
        output.push_back(v * x[i]);
    }
    return output;
}
vector<double> VecOps::sigmoid(vector<double>& x){
    vector<double> output;
    for(int i = 0; i < x.size(); i++){
        output.push_back(1.0 / (1.0 + exp(-x[i])));
    }
    return output;
}