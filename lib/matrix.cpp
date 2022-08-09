#include "matrix.h"

Matrix::Matrix(vector<vector<double>> v){
    vals = v;
}
Matrix Matrix::dot(Matrix& y){
    assert(col_size() == y.row_size());
    vector<vector<double>> prod;
    prod.reserve(row_size());
    for(int r = 0; r < row_size(); r++){
        prod.emplace_back(vector<double>());
        prod[r].reserve(y.col_size());
        for(int yc = 0; yc < y.col_size(); yc++){
            double res = 0.0;
            for(int c = 0; c < col_size(); c++){
                res += get(r, c) * y.get(c, yc);
            }
            prod[r].emplace_back(res);
        }
    }
    return Matrix(prod);
}
Matrix Matrix::add(Matrix& y){
    assert(col_size() == y.col_size() && row_size() == y.row_size());
    vector<vector<double>> sum;
    sum.reserve(row_size());
    for(int r = 0; r < row_size(); r++){
        sum.emplace_back(vector<double>());
        sum[r].reserve(col_size());
        for(int c = 0; c < col_size(); c++){
            sum[r].emplace_back(get(r, c) + y.get(r, c));
        }
    }
    return Matrix(sum);
}
Matrix Matrix::identity(unsigned sz){
    vector<vector<double>> output;
    output.reserve(sz);
    for(int r = 0; r < sz; r++){
        output.emplace_back(vector<double>());
        output[r].reserve(sz);
        for(int c = 0; c < sz; c++){
            if(r == c){
                output[r].emplace_back(1.0);
            }
            else{
                output[r].emplace_back(0.0);
            }
        }
    }
    return Matrix(output);
}
Matrix Matrix::diag(vector<double>& vals){
    vector<vector<double>> output;
    unsigned sz = vals.size();
    for(int r = 0; r < sz; r++){
        output.emplace_back(vector<double>());
        output[r].reserve(sz);
        for(int c = 0; c < sz; c++){
            if(r == c){
                output[r].emplace_back(vals[r]);
            }
            else{
                output[r].emplace_back(0.0);
            }
        }
    }
    return Matrix(output);
}
Matrix Matrix::fill(double v, unsigned rows, unsigned cols){
    vector<vector<double>> output;
    output.reserve(rows);
    for(int r = 0; r < rows; r++){
        output.emplace_back(vector<double>());
        output[r].reserve(cols);
        for(int c = 0; c < cols; c++){
            output[r].emplace_back(v);
        }
    }
    return Matrix(output);
}
void Matrix::insert_row(unsigned row, vector<double> val){
    assert(row < row_size());
    assert(val.size() == col_size());
    for(int c = 0; c < col_size(); c++){
        vals[row][c] = val[c];
    }
}
vector<double> Matrix::get(unsigned index){
    return vals[index];
}
double Matrix::get(unsigned r, unsigned c){
    return vals[r][c];
}
unsigned Matrix::col_size(){
    assert(vals.size() > 0);
    return vals[0].size();
}
unsigned Matrix::row_size(){
    return vals.size();
}
void Matrix::print(){
    for(int i = 0; i < vals.size(); i++){
        for(int j = 0; j < vals[i].size(); j++){
            cout << vals[i][j] << " ";
        }
        cout << endl;
    }
}