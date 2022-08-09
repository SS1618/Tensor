#include "vecops.h"
#include "matrix.h"
#include "tensor.h"
#include "nn.h"

string Add::equation(Tensor& x, Tensor& y){
    return "Add(" + x.get_equation() + ", " + y.get_equation() + ")";
}
vector<double> Add::calc_output(Tensor& x, Tensor& y){
    assert(x.size() == y.size());
    return VecOps::add(x.get_vals(), y.get_vals());
}
unordered_map<unsigned, Matrix> Add::calc_grad(Tensor& x, Tensor& y){
    assert(x.size() == y.size());
    Matrix h_wrt_x = Matrix::identity(x.size());
    Matrix h_wrt_y = Matrix::identity(y.size());
    unordered_map<unsigned, Matrix> grad;
    for(auto& x_wrt_k : x.get_grad()){
        assert(h_wrt_x.col_size() == x_wrt_k.second.row_size());
        grad[x_wrt_k.first] = h_wrt_x.dot(x_wrt_k.second); //dh/dk = (dh/dx)(dx/dk)
    }
    for(auto& y_wrt_k : y.get_grad()){
        assert(h_wrt_y.col_size() == y_wrt_k.second.row_size());
        if(grad.find(y_wrt_k.first) != grad.end()){
            Matrix dum = h_wrt_y.dot(y_wrt_k.second);
            grad[y_wrt_k.first] = grad[y_wrt_k.first].add(dum); //dh/dk += (dh/dy)(dy/dk)
        }
        else{
            grad[y_wrt_k.first] = h_wrt_y.dot(y_wrt_k.second); //dh/dk = (dh/dy)(dy/dk)
        }
    }
    return grad;
}

string Hadamard::equation(Tensor& x, Tensor& y){
    return "Hadamard(" + x.get_equation() + ", " + y.get_equation() + ")";
}
vector<double> Hadamard::calc_output(Tensor& x, Tensor& y){
    assert(x.size() == y.size());
    return VecOps::hadamard(x.get_vals(), y.get_vals());
}
unordered_map<unsigned, Matrix> Hadamard::calc_grad(Tensor& x, Tensor& y){
    assert(x.size() == y.size());
    Matrix h_wrt_x = Matrix::diag(y.get_vals()); //diag matrix of y
    Matrix h_wrt_y = Matrix::diag(x.get_vals()); //diag matrix of x
    unordered_map<unsigned, Matrix> grad;
    for(auto& x_wrt_k : x.get_grad()){
        assert(h_wrt_x.col_size() == x_wrt_k.second.row_size());
        grad[x_wrt_k.first] = h_wrt_x.dot(x_wrt_k.second); //dh/dk = (dh/dx)(dx/dk)
    }
    for(auto& y_wrt_k : y.get_grad()){
        assert(h_wrt_y.col_size() == y_wrt_k.second.row_size());
        if(grad.find(y_wrt_k.first) != grad.end()){
            Matrix dum = h_wrt_y.dot(y_wrt_k.second);
            grad[y_wrt_k.first] = grad[y_wrt_k.first].add(dum); //dh/dk += (dh/dy)(dy/dk)
        }
        else{
            grad[y_wrt_k.first] = h_wrt_y.dot(y_wrt_k.second); //dh/dk = (dh/dy)(dy/dk)
        }
    }
    return grad;
}

string Sum::equation(Tensor& x){
    return "Sum(" + x.get_equation() + ")";
}
vector<double> Sum::calc_output(Tensor& x){
    vector<double> output;
    output.push_back(VecOps::sum(x.get_vals()));
    return output;
}
unordered_map<unsigned, Matrix> Sum::calc_grad(Tensor& x){
    Matrix h_wrt_x = Matrix::fill(1.0, 1, x.size()); // single row jacobian [1 1 1 ...]
    unordered_map<unsigned, Matrix> grad;
    for(auto& x_wrt_k : x.get_grad()){
        assert(h_wrt_x.col_size() == x_wrt_k.second.row_size());
        grad[x_wrt_k.first] = h_wrt_x.dot(x_wrt_k.second); //dh/dk = (dh/dx)(dx/dk)
    }
    return grad;
}

string Dot::equation(vector<Tensor>& x){
    string eq = "[";
    for(int i = 0; i < x.size(); i++){
        eq += x[i].get_equation();
        if(i < x.size() - 1){
            eq += ", ";
        }
    }
    eq += "]";
    return eq;
}
vector<double> Dot::calc_output(vector<Tensor>& x){
    vector<double> output;
    for(int i = 0; i < x.size(); i++){
        assert(x[i].size() == 1);
        output.push_back(x[i].get_vals()[0]); //unwrap tensor around each element
    }
    return output;
}
unordered_map<unsigned, Matrix> Dot::calc_grad(vector<Tensor>& x){
    unordered_map<unsigned, Matrix> grad;
    for(int i = 0; i < x.size(); i++){
        for(auto& xi_wrt_k : x[i].get_grad()){
            assert(xi_wrt_k.second.row_size() == 1); //check gradient is 1D
            if(grad.find(xi_wrt_k.first) == grad.end()){ 
                grad[xi_wrt_k.first] = Matrix::fill(0.0, x.size(), xi_wrt_k.second.col_size());
            }
            assert(xi_wrt_k.second.col_size() == grad[xi_wrt_k.first].col_size());
            grad[xi_wrt_k.first].insert_row(i, xi_wrt_k.second.get(0)); //insert d(x_i)/dk into ith row of dh/dk
        }
    }
    return grad;
}

string Sigmoid::equation(Tensor& x){
    return "Sigmoid(" + x.get_equation() + ")";
}
vector<double> Sigmoid::calc_output(Tensor& x){
    return VecOps::sigmoid(x.get_vals());
}
unordered_map<unsigned, Matrix> Sigmoid::calc_grad(Tensor& x){
    vector<double> sig_res = calc_output(x);
    vector<double> sig_sq = VecOps::hadamard(sig_res, sig_res);
    sig_sq = VecOps::mult(-1.0, sig_sq);
    sig_res = VecOps::add(sig_res, sig_sq);
    Matrix h_wrt_x = Matrix::diag(sig_res);
    unordered_map<unsigned, Matrix> grad;
    for(auto& x_wrt_k : x.get_grad()){
        grad[x_wrt_k.first] = h_wrt_x.dot(x_wrt_k.second); //dh/dk = (dh/dx)(dx/dk)
    }
    return grad;
}

Tensor Add::apply(Tensor& x, Tensor& y){
    return Tensor(calc_output(x, y), calc_grad(x, y), equation(x, y));
}

Tensor Hadamard::apply(Tensor& x, Tensor& y){
    return Tensor(calc_output(x, y), calc_grad(x, y), equation(x, y));
}

Tensor Sum::apply(Tensor& x){
    return Tensor(calc_output(x), calc_grad(x), equation(x));
}

Tensor Dot::apply(vector<Tensor>& x, Tensor& y){
    vector<Tensor> res;
    for(int i = 0; i < x.size(); i++){
        assert(x[i].size() == y.size());
        res.push_back(Hadamard::apply(x[i], y));
        res[i] = Sum::apply(res[i]);
    }
    return Tensor(calc_output(res), calc_grad(res), equation(res));
}

Tensor MSELoss::apply(Tensor& x, Tensor& y){
    assert(x.size() == y.size());
    Tensor neg = Tensor::fill(-1.0, y.size());
    Tensor neg_y = Hadamard::apply(neg, y);
    Tensor sub = Add::apply(x, neg_y);
    Tensor sqr = Hadamard::apply(sub, sub);
    return Sum::apply(sqr);
}

Tensor Sigmoid::apply(Tensor& x){
    return Tensor(calc_output(x), calc_grad(x), equation(x));
}

LinearLayer::LinearLayer(unsigned input_sz, unsigned output_sz){
    for(int i = 0; i < output_sz; i++){
        W.push_back(Tensor::randn(input_sz));
    }
    bias = Tensor::randn(output_sz);
    input_size = input_sz;
    output_size = output_sz;
}
Tensor LinearLayer::feedforward(Tensor& x){
    assert(x.size() == input_size);
    Tensor v1 = Dot::apply(W, x);
    return Add::apply(v1, bias);
}
unordered_map<unsigned, Tensor*> LinearLayer::parameters(){
    unordered_map<unsigned, Tensor*> params;
    params[bias.getID()] = &bias;
    for(int i = 0; i < output_size; i++){
        params[W[i].getID()] = &(W[i]);
    }
    return params;
}
