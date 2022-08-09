#include "vecops.h"
#include "matrix.h"
#include "tensor.h"

Tensor::Tensor(vector<double> v, unordered_map<unsigned, Matrix> g){
    vals = v;
    gradient = g;
    UUID_COUNT += 1;
    uuid = UUID_COUNT;
    equation = to_string(uuid);
    gradient[uuid] = Matrix::identity(v.size());
}
Tensor::Tensor(vector<double> v, unordered_map<unsigned, Matrix> g, string eq){
    vals = v;
    gradient = g;
    UUID_COUNT += 1;
    uuid = UUID_COUNT;
    equation = eq;
    gradient[uuid] = Matrix::identity(v.size());
}
Tensor::Tensor(vector<double> v){
    vals = v;
    UUID_COUNT += 1;
    uuid = UUID_COUNT;
    equation = to_string(uuid);
    gradient[uuid] = Matrix::identity(v.size());
}
vector<double>& Tensor::get_vals(){
    return vals;
}
unordered_map<unsigned, Matrix>& Tensor::get_grad(){
    return gradient;
}
double Tensor::get(unsigned index){
    assert(index < vals.size());
    return vals[index];
}
unsigned Tensor::size(){
    return vals.size();
}
string Tensor::get_equation(){
    return equation;
}
Tensor Tensor::fill(double val, unsigned sz){
    vector<double> v;
    v.reserve(sz);
    for(int i = 0; i < sz; i++){
        v.emplace_back(val);
    }
    return Tensor(v);
}
Tensor Tensor::randn(unsigned sz){
    return Tensor(VecOps::randn(sz));
}
void Tensor::print(){
    string disp = "ID: " + to_string(uuid) + "\n";
    disp += "[";
    for(int i = 0; i < vals.size(); i++){
        disp += to_string(vals[i]);
        if(i < vals.size() - 1){
            disp += ", ";
        }
    }
    disp += "]";
    cout << disp << endl;
}
void Tensor::update(vector<double> d){
    assert(d.size() == vals.size());
    for(int i = 0; i < vals.size(); i++){
        vals[i] += d[i];
    }
}
unsigned Tensor::getID(){
    return uuid;
}
unsigned Tensor::UUID_COUNT = 0;
