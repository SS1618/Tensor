#ifndef MATRIX_H
#define MATRIX_H
#include "includes.h"
using namespace std;

class Matrix{
    private:
        vector<vector<double>> vals;
    public:
        Matrix(){}
        Matrix(vector<vector<double>> v);
        Matrix dot(Matrix& y);
        Matrix add(Matrix& y);
        static Matrix identity(unsigned sz);
        static Matrix diag(vector<double>& vals);
        static Matrix fill(double v, unsigned rows, unsigned cols);
        void insert_row(unsigned row, vector<double> val);
        vector<double> get(unsigned index);
        double get(unsigned r, unsigned c);
        unsigned col_size();
        unsigned row_size();
        void print();
};
#endif