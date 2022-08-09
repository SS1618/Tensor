#ifndef VECOPS_H
#define VECOPS_H
#include "includes.h"
using namespace std;

class VecOps{
    public:
        static vector<double> add(vector<double>& a, vector<double>& b);
        static vector<double> hadamard(vector<double>& a, vector<double>& b);
        static double sum(vector<double>& a);
        static vector<double> randn(unsigned sz);
        static vector<double> mult(double v, vector<double> x);
        static vector<double> sigmoid(vector<double>& x);
};
#endif