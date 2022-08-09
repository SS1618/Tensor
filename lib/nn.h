#include "includes.h"

using namespace std;

class Add{
    private:
        static string equation(Tensor& x, Tensor& y);
        static vector<double> calc_output(Tensor& x, Tensor& y);
        static unordered_map<unsigned, Matrix> calc_grad(Tensor& x, Tensor& y);
    public:
        static Tensor apply(Tensor& x, Tensor& y);
};

class Hadamard{
    private:
        static string equation(Tensor& x, Tensor& y);
        static vector<double> calc_output(Tensor& x, Tensor& y);
        static unordered_map<unsigned, Matrix> calc_grad(Tensor& x, Tensor& y);
    public:
        static Tensor apply(Tensor& x, Tensor& y);
};

class Sum{
    private:
        static string equation(Tensor& x);
        static vector<double> calc_output(Tensor& x);
        static unordered_map<unsigned, Matrix> calc_grad(Tensor& x);
    public:
        static Tensor apply(Tensor& x);
};

class Dot{
    private:
        static string equation(vector<Tensor>& x);
        static vector<double> calc_output(vector<Tensor>& x);
        static unordered_map<unsigned, Matrix> calc_grad(vector<Tensor>& x);
    public:
        static Tensor apply(vector<Tensor>& x, Tensor& y);
};
class MSELoss{
    public:
        static Tensor apply(Tensor& x, Tensor& y);
};

class LinearLayer{
    private:
        vector<Tensor> W;
        Tensor bias;
        unsigned input_size, output_size;
    public:
        LinearLayer(unsigned input_sz, unsigned output_sz);
        Tensor feedforward(Tensor& x);
        unordered_map<unsigned, Tensor*> parameters();
};

class Sigmoid{
    private:
        static string equation(Tensor& x);
        static vector<double> calc_output(Tensor& x);
        static unordered_map<unsigned, Matrix> calc_grad(Tensor& x);
    public:
        static Tensor apply(Tensor& x);
};