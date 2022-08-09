#ifndef TENSOR_H
#define TENSOR_H

using namespace std;

class Tensor{
    private:
        vector<double> vals;
        unordered_map<unsigned, Matrix> gradient;
        static unsigned UUID_COUNT;
        string equation;
        unsigned uuid;
    public:
        Tensor(){}
        Tensor(vector<double> v, unordered_map<unsigned, Matrix> g);
        Tensor(vector<double> v, unordered_map<unsigned, Matrix> g, string eq);
        Tensor(vector<double> v);
        vector<double>& get_vals();
        unordered_map<unsigned, Matrix>& get_grad();
        double get(unsigned index);
        unsigned size();
        string get_equation();
        static Tensor fill(double val, unsigned sz);
        static Tensor randn(unsigned sz);
        void print();
        void update(vector<double> d);
        unsigned getID();
};

#endif