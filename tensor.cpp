#include <iostream>
#include <string>
#include <random>
#include <math.h>
#include <fstream>

#include "dais_exc.h"
#include "tensor.h"

#define PI 3.141592654
#define FLT_MAX 3.402823466e+38F /* max value */
#define FLT_MIN 1.175494351e-38F /* min positive value */
#define EPSILON 0.000001f

using namespace std;


/**
 * Class constructor
 * 
 * Parameter-less class constructor 
 */
Tensor::Tensor(){
    r=c=d=0;
    data=nullptr;
}

/**
 * Class constructor
 * 
 * Creates a new tensor of size r*c*d initialized at value v
 * 
 * @param r
 * @param c
 * @param d
 * @param v
 * @return new Tensor
 */
Tensor::Tensor(int r, int c, int d, float v){
    init(r,c,d,v);
}

/**
 * Class distructor
 * 
 * Cleanup the data when deallocated
 */
Tensor::~Tensor(){
    for(int i=0; i<this->r; i++){
        for(int j=0; j<this->c; j++){
            delete [] data[i][j];
        }
    }
    for(int i=0; i<this->r; i++){
        delete [] data[i];
    }
    
    delete [] data;
    data=nullptr;
    r=c=d=0;
}

float Tensor::sum(int depth){
    float res = 0;
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            res = res + data[i][j][depth];
        }
    }
    return res;
}

/**
 * Operator overloading ()
 * 
 * if indexes are out of bound throw index_out_of_bound() exception
 * 
 * @return the value at location [i][j][k]
 */
float Tensor::operator()(int i, int j, int k) const{
    if(i<0 || i>r || j<0 || j>c || k<0 || k>d){
        throw(index_out_of_bound());
    }
    else{
        return data[i][j][k];
    }
}

/**
 * Operator overloading ()
 * 
 * Return the pointer to the location [i][j][k] such that the operator (i,j,k) can be used to 
 * modify tensor data.
 * 
 * If indexes are out of bound throw index_out_of_bound() exception
 * 
 * @return the pointer to the location [i][j][k]
 */
float &Tensor::operator()(int i, int j, int k){
    if(i<0 || i>this->r || j<0 || j>this->c || k<0 || k>this->d){
        throw(index_out_of_bound());
    }
    else{
        return data[i][j][k];
    } 
}

/**
 * Copy constructor
 * 
 * This constructor copies the data from another Tensor
 *      
 * @return the new Tensor
 */
Tensor::Tensor(const Tensor& that){
    init(that.r, that.c, that.d, 0.0);
    for(int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            for(int k=0; k<d; k++){
                data[i][j][k] = that.data[i][j][k];
            }
        }
    }
}

/**
 * Operator overloading ==
 * 
 * It performs the point-wise equality check between two Tensors.
 * 
 * The equality check between floating points cannot be simply performed using the 
 * operator == but it should take care on their approximation.
 * 
 * This approximation is known as rounding (do you remember "Architettura degli Elaboratori"?)
 *  
 * For example, given a=0.1232 and b=0.1233 they are 
 * - the same, if we consider a rounding with 1, 2 and 3 decimals 
 * - different when considering 4 decimal points. In this case b>a
 * 
 * So, given two floating point numbers "a" and "b", how can we check their equivalence? 
 * through this formula:
 * 
 * a ?= b if and only if |a-b|<EPSILON
 * 
 * where EPSILON is fixed constant (defined at the beginning of this header file)
 * 
 * Two tensors A and B are the same if:
 * A[i][j][k] == B[i][j][k] for all i,j,k 
 * where == is the above formula.
 * 
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 * 
 * @return returns true if all their entries are "floating" equal
 */
bool Tensor::operator==(const Tensor& rhs) const{
    bool equal=true;
    if (r == rhs.r && c == rhs.c && d == rhs.d){
        for(int i=0; i<r && equal; i++){
            for(int j=0; j<c && equal; j++){
                for(int k=0; k<d && equal; k++){
                    if(data[i][j][k]-rhs.data[i][j][k] > EPSILON)
                        equal=false;
                }
            }
        }
    }
    else{
        throw dimension_mismatch();
    }
    return equal;
}

/**
 * Operator overloading -
 * 
 * It performs the point-wise difference between two Tensors.
 * 
 * result(i,j,k)=this(i,j,k)-rhs(i,j,k)
 * 
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 * 
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator-(const Tensor &rhs) const{
    Tensor res= Tensor();
    res.init(r,c,d);
    if (r == rhs.r && c == rhs.c && d == rhs.d){
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    res.data[i][j][k] = data[i][j][k] - rhs.data[i][j][k];
                }
            }
        }
    } else {
        throw(dimension_mismatch());
    }
    return res;
}

/**
 * Operator overloading +
 * 
 * It performs the point-wise sum between two Tensors.
 * 
 * result(i,j,k)=this(i,j,k)+rhs(i,j,k)
 * 
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 * 
 * @return returns a new Tensor containing the result of the operation
*/
Tensor Tensor::operator+(const Tensor &rhs) const{
    Tensor res= Tensor();
    res.init(r,c,d);
    if (r == rhs.r && c == rhs.c && d == rhs.d){
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    res.data[i][j][k] = data[i][j][k] + rhs.data[i][j][k];
                }
            }
        }
    } else {
        throw(dimension_mismatch());
    }
    return res;
}

/**
 * Operator overloading *
 * 
 * It performs the point-wise product between two Tensors.
 * 
 * result(i,j,k)=this(i,j,k)*rhs(i,j,k)
 * 
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 * 
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator*(const Tensor &rhs) const{
    Tensor res;
    res.init(r,c,d);
    if (r == rhs.r && c == rhs.c && d == rhs.d){
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    res.data[i][j][k] = data[i][j][k] * rhs.data[i][j][k];
                }
            }
        }
    } else {
        throw(dimension_mismatch());
    }
    return res;
}

/**
 * Operator overloading /
 * 
 * It performs the point-wise division between two Tensors.
 * 
 * result(i,j,k)=this(i,j,k)/rhs(i,j,k)
 * 
 * The two tensors must have the same size otherwise throw a dimension_mismatch()
 * 
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator/(const Tensor &rhs) const{
    Tensor res= Tensor();
    res.init(r,c,d);
    if (r == rhs.r && c == rhs.c && d == rhs.d){
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    if(rhs.data[i][j][k]!=0.0)
                        res.data[i][j][k] = data[i][j][k] / rhs.data[i][j][k];
                    else
                        throw unknown_exception();
                }
            }
        }
    } else {
        throw(dimension_mismatch());
    }
    return res;
}

/**
 * Operator overloading - 
 * 
 * It performs the point-wise difference between a Tensor and a constant
 * 
 * result(i,j,k)=this(i,j,k)-rhs
 * 
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator-(const float &rhs) const{
    Tensor res= Tensor();
    res.init(r,c,d);
    for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    res.data[i][j][k] = data[i][j][k] - rhs;
                }
            }
        }
    return res;
}

/**
 * Operator overloading +
 * 
 * It performs the point-wise sum between a Tensor and a constant
 * 
 * result(i,j,k)=this(i,j,k)+rhs
 * 
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator+(const float &rhs) const{
    Tensor res=Tensor();
    res.init(r, c, d);
    for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    res.data[i][j][k] = data[i][j][k] + rhs;
                }
            }
        }
    return res;
}

/**
 * Operator overloading *
 * 
 * It performs the point-wise product between a Tensor and a constant
 * 
 * result(i,j,k)=this(i,j,k)*rhs
 * 
 * @return returns a new Tensor containing the result of the operation
 */
Tensor Tensor::operator*(const float &rhs) const{
    Tensor res=Tensor();
    res.init(r,c,d);
    for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    res.data[i][j][k] = data[i][j][k] * rhs;
                }
            }
        }
    return res;
 }
/**
 * Operator overloading / between a Tensor and a constant
 * 
 * It performs the point-wise division between a Tensor and a constant
 * 
 * result(i,j,k)=this(i,j,k)/rhs
 * 
 * @return returns a new Tensor containing the result of the operation
 */
  Tensor Tensor::operator/(const float &rhs) const{
    Tensor res= Tensor();
    if(rhs!=0.0){
        res.init(r,c,d);
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    res.data[i][j][k] = data[i][j][k] / rhs;
                }               
            }
        }
    }
    else{
        throw unknown_exception();
    }
    return res;
 }

/**
 * Operator overloading = (assignment) 
 * 
 * Perform the assignment between this object and another
 * 
 * @return a reference to the receiver object
 */
Tensor &Tensor::operator=(const Tensor &other){
    init(other.r, other.c, other.d, 0.0);
    if (this != &other){
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                for(int k=0; k<d; k++){
                    data[i][j][k]=other.data[i][j][k];                    
                }
            }
        }
    } 
    return *this;
}

/**
 * Random Initialization
 * 
 * Perform a random initialization of the tensor
 * 
 * @param mean The mean
 * @param std  Standard deviation
 */
void Tensor::init_random(float mean, float std){
    if(data){
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(mean,std);

        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                for(int k=0;k<d;k++){
                    this->operator()(i,j,k)= distribution(generator);
                }
            }
        }    

    }else{
        throw(tensor_not_initialized());
    }
}

/**
 * Constant Initialization
 * 
 * Perform the initialization of the tensor to a value v
 * 
 * @param r The number of rows
 * @param c The number of columns
 * @param d The depth
 * @param v The initialization value
 */
void Tensor::init(int row, int col, int depth, float v){
    r=row;
    c=col;
    d=depth;
    data=new float**[r];
    for(int i=0; i<r; i++){
        data[i]=new float*[c];
    }
    for(int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            data[i][j]=new float[d];
        }
    }
    for(int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            for(int k=0; k<d; k++){
                data[i][j][k]=v;                    
            }
        }
    }
}

/**
 * Tensor Clamp
 * 
 * Clamp the tensor such that the lower value becomes low and the higher one become high.
 * 
 * @param low Lower value
 * @param high Higher value 
 */
void Tensor::clamp(float low, float high){
    for(int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            for(int k=0; k<d; k++){
                if (data[i][j][k] < low)
                    data[i][j][k] = low;
                if (data[i][j][k] > high)
                    data[i][j][k] = high;
            }
        }
    }
}

/**
 * Tensor Rescaling
 * 
 * Rescale the value of the tensor following this rule:
 * 
 * newvalue(i,j,k) = ((data(i,j,k)-min(k))/(max(k)-min(k)))*new_max
 * 
 * where max(k) and min(k) are the maximum and minimum value in the k-th channel.
 * 
 * new_max is the new maximum value for each channel
 * 
 * - if max(k) and min(k) are the same, then the entire k-th channel is set to new_max.
 * 
 * @param new_max New maximum vale
 */
void Tensor::rescale(float new_max){
    float min, max;
    for (int k=0; k<d; k++){
        min=getMin(k);
        max=getMax(k);
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                if (min!=max){
                    data[i][j][k]=((data[i][j][k]-min)/(max-min))*new_max;
                } else {
                    data[i][j][k]=new_max;
                }
            }
        }
    }
}

/**
 * Tensor padding
 * 
 * Zero pad a tensor in height and width, the new tensor will have the following dimensions:
 * 
 * (rows+2*pad_h) x (cols+2*pad_w) x (depth) 
 * 
 * @param pad_h the height padding
 * @param pad_w the width padding
 * @return the padded tensor
 */
Tensor Tensor::padding(int pad_h, int pad_w)const{
    Tensor res;
    res.init(r+2*pad_h, c+2*pad_w, d);
    for(int i=0; i<res.r; i++){
        for(int j=0; j<res.c; j++){
            for(int k=0; k<res.d; k++){
                if(i>=pad_h && i<res.r-pad_h && j>=pad_w && j<res.c-pad_w){
                    res.data[i][j][k]=data[i-pad_h][j-pad_w][k];
                } 
            }
        }
    }
    return res;
}

/**
 * Subset a tensor
 * 
 * retuns a part of the tensor having the following indices:
 * row_start <= i < row_end  
 * col_start <= j < col_end 
 * depth_start <= k < depth_end
 * 
 * The right extrema is NOT included
 * 
 * @param row_start 
 * @param row_end 
 * @param col_start
 * @param col_end
 * @param depth_start
 * @param depth_end
 * @return the subset of the original tensor
 */
Tensor Tensor::subset(unsigned int row_start, unsigned int row_end, unsigned int col_start, unsigned int col_end, unsigned int depth_start, unsigned int depth_end)const{
    Tensor res;
    res.init(row_end-row_start, col_end-col_start, depth_end-depth_start);
    if(row_start>=0 && row_end<=r && col_start>=0 && col_end<=c && depth_start>=0 && depth_end<=d){
        
        for(int i=row_start; i<row_end; i++){
            for(int j=col_start; j<col_end; j++){
                for(int k=depth_start; k<depth_end; k++){
                    res.data[i-row_start][j-col_start][k-depth_start]=data[i][j][k];
                }
            }
        }

    }
    else{
        throw(dimension_mismatch());
    }
    return res;
}


/** 
 * Concatenate 
 * 
 * The function concatenates two tensors along a give axis
 * 
 * Example: this is of size 10x5x6 and rhs is of 25x5x6
 * 
 * if concat on axis 0 (row) the result will be a new Tensor of size 35x5x6
 * 
 * if concat on axis 1 (columns) the operation will fail because the number 
 * of rows are different (10 and 25).
 * 
 * In order to perform the concatenation is mandatory that all the dimensions 
 * different from the axis should be equal, other wise throw concat_wrong_dimension(). 
 *  
 * @param rhs The tensor to concatenate with
 * @param axis The axis along which perform the concatenation 
 * @return a new Tensor containing the result of the concatenation
 */
Tensor Tensor::concat(const Tensor &rhs, int axis)const{
    Tensor result;
    if(axis==0 && c==rhs.c && d==rhs.d){
        result.init(rhs.r + r, c, d);
        for(int i = 0; i < r; i++){
            for(int j = 0; j < c; j++){
                for(int k = 0; k < d; k++){
                    result.data[i][j][k] = data[i][j][k];
                }
            }
        }
    
        for(int i = r; i < result.r; i++){
            for(int j = 0; j < rhs.c; j++){
                for(int k = 0; k < rhs.d; k++){
                    result.data[i][j][k] = rhs.data[i - r][j][k];
                }
            }
        }
    }

    else if(axis==1 && r==rhs.r && d==rhs.d){
        result.init(r, rhs.c + c, d);
        
        for(int i = 0; i < r; i++){
            for(int j = 0; j < c; j++){
                for(int k = 0; k < d; k++){
                    result.data[i][j][k] = data[i][j][k];
                }
            }
        }
        
        for(int i = 0; i < rhs.r; i++){
            for(int j = c; j < result.c; j++){
                for(int k=0; k < rhs.d; k++){
                    result.data[i][j][k] = rhs.data[i][j - c][k];
                }
            }
        }
    }
    
    else if(axis==2 && r==rhs.r && c==rhs.c){
        result.init(r, c,rhs.d + d);
        for(int i = 0; i < r; i++){
            for(int j=0; j < c; j++){
                for(int k = 0; k < d; k++){
                    result.data[i][j][k] = data[i][j][k];
                }
            }
        }
        for(int i = 0; i < rhs.r; i++){
            for(int j = c; j < rhs.c; j++){
                for(int k = d; k < result.d; k++){
                    result.data[i][j][k] = rhs.data[i][j][k - d];
                }
            }
        }
    }
    else{
        throw concat_wrong_dimension();
    }
    return result;
}

/** 
 * Convolution 
 * 
 * This function performs the convolution of the Tensor with a filter.
 * 
 * The filter f must have odd dimensions and same depth. 
 * 
 * Remember to apply the padding before running the convolution
 *  
 * @param f The filter
 * @return a new Tensor containing the result of the convolution
 */
Tensor Tensor::convolve(const Tensor &f)const {
    Tensor res;
    res.init(r, c, d);
    if (f.d==d && f.r%2!=0 && f.c%2!=0 && f.d%2!=0){
        int pad_w=(f.r-1)/2;
        int pad_h=(f.c-1)/2;
        Tensor temp = padding(pad_w, pad_h);
        Tensor sub;
        sub.init(f.r, f.c, f.d);  
        for (int i = 0; i < r; i++){
            for (int j = 0; j < c; j++){
                sub= temp.subset(i , i+f.r, j, j+f.c, 0, f.d);
                sub=sub*f;
                for (int k = 0; k < d; k++){
                    res(i,j,k)=sub.sum(k);
                }
            }
        }
    }
    else{
        if(f.d!=d)
            throw dimension_mismatch();
        else
            throw filter_odd_dimensions();
    }
    return res;
}



/* UTILITY */

/** 
 * Rows 
 * 
 * @return the number of rows in the tensor
 */
int Tensor::rows() const{
    return r;
}

/** 
 * Cols 
 * 
 * @return the number of columns in the tensor
 */
int Tensor::cols() const{
    return c;
}

/** 
 * Depth 
 * 
 * @return the depth of the tensor
 */
int Tensor::depth() const{
    return d;
}

/** 
 * Get minimum 
 * 
 * Compute the minimum value considering a particular index in the third dimension
 * 
 * @return the minimum of data( , , k)
 */
float Tensor::getMin(int k) const{
    float min=data[0][0][k];
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            if(data[i][j][k] < min)
                min = data[i][j][k]; 
        }
    }
    return min;
}

/** 
 * Get maximum 
 * 
 * Compute the maximum value considering a particular index in the third dimension
 * 
 * @return the maximum of data( , , k)
 */
float Tensor::getMax(int k) const{
    float max=data[0][0][k];
    for(int i=0;i<r;i++){
        for(int j=0;j<c;j++){
            if(data[i][j][k] > max)
                max = data[i][j][k]; 
        }
    }
    return max;
}

/** 
 * showSize
 * 
 * shows the dimensions of the tensor on the standard output.
 * 
 * The format is the following:
 * "rows" x "colums" x "depth"
 * 
 */
void Tensor::showSize() const{
    cout<<rows()<<" x "<<cols()<<" x "<<depth()<<endl;
}

/* IOSTREAM */

/**
 * Operator overloading <<
 * 
 * Use the overaloading of << to show the content of the tensor.
 * 
 * You are free to chose the output format, btw we suggest you to show the tensor by layer.
 * 
 * [..., ..., 0]
 * [..., ..., 1]
 * ...
 * [..., ..., k]
 */
ostream& operator<< (ostream& stream, const Tensor & obj){

    for (int k = 0; k < obj.depth(); k++){
        stream << '[';
        for (int i = 0; i < obj.rows(); i++){
            for (int j = 0; j < obj.cols(); j++){
                stream << obj.data[i][j][k];
                if (i != obj.rows()-1 && j != obj.cols() - 1){
                    stream << ',';
                }
            }
        }
        stream << ']' << endl;
    }
    return stream;
}

/**
 * Reading from file
 * 
 * Load the content of a tensor from a textual file.
 * 
 * The file should have this structure: the first three lines provide the dimensions while 
 * the following lines contains the actual data by channel.
 * 
 * For example, a tensor of size 4x3x2 will have the following structure:
 * 4
 * 3
 * 2
 * data(0,0,0)
 * data(0,1,0)
 * data(0,2,0)
 * data(1,0,0)
 * data(1,1,0)
 * .
 * .
 * .
 * data(3,1,1)
 * data(3,2,1)
 * 
 * if the file is not reachable throw unable_to_read_file()
 * 
 * @param filename the filename where the tensor is stored
 */
void Tensor::read_file(string filename){
    ifstream file=ifstream{filename};
    if (file.good()){
        string s;
        getline(file, s);
        r = stoi(s);
        getline(file, s);
        c = stoi(s);
        getline(file, s);
        d = stoi(s);
        init(r, c, d);

        while(file.good()){
            string str;
            float aux;
            for(int k = 0; k < d; k++){
                for(int i = 0; i < r; i++){
                    for(int j = 0; j < c; j++){
                        getline(file, str);
                        if (str.size()!=0)
                            data[i][j][k] = stof(str);
                    }
                }
            }
        }
    } 
    else {
        throw unable_to_read_file();
    }
}

/**
 * Write the tensor to a file
 * 
 * Write the content of a tensor to a textual file.
 * 
 * The file should have this structure: the first three lines provide the dimensions while 
 * the following lines contains the actual data by channel.
 * 
 * For example, a tensor of size 4x3x2 will have the following structure:
 * 4
 * 3
 * 2
 * data(0,0,0)
 * data(0,1,0)
 * data(0,2,0)
 * data(1,0,0)
 * data(1,1,0)
 * .
 * .
 * .
 * data(3,1,1)
 * data(3,2,1)
 * 
 * @param filename the filename where the tensor should be stored
 */
void Tensor::write_file(string filename){
    ofstream file=ofstream{filename};
    file << r << endl;
    file << c << endl;
    file << d <<endl;
    for(int k=0; k<d; k++){
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                file << data[i][j][k] << endl;     
            } 
        }
    }
}

cdf_value* Tensor::cdf(int k, int &size){
    float* values= new float[r*c];
    float aux;
    cdf_value* res= new cdf_value[r*c];
    //linearizzo il mio tensor in canale k
    for (int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            values[i*c+j]=data[i][j][k];    
        }
    }
    //ordino values in modo crescente

    for (int i=0; i < r*c -1; i++){
        for (int j=r*c -2; j>=i; j--){
            if (values[j] > values[j+1]){
                int supp;
                supp = values[j];
                values[j] = values[j+1];
                values[j+1] = supp;
            }
        }
    }

    aux = values[0];
    int i;
    for (i=1; i < r*c; i++){
        if (aux != values[i]){
            res[size].cdf=i;
            res[size].valore=aux;
            aux=values[i];
            size++;
        }
    }
    res[size].cdf=i;
    res[size].valore=aux;

    delete [] values;
    return res;
}

