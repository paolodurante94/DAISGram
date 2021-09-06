#include <iostream>
#include <string>

#include "dais_exc.h"
#include "tensor.h"
#include "libbmp.h"
#include "DAISGram.h"

using namespace std;


DAISGram::DAISGram(){
}
DAISGram::~DAISGram(){
}

/**
 * Load a bitmap from file
 *
 * @param filename String containing the path of the file
 */
void DAISGram::load_image(string filename){
    BmpImg img = BmpImg();

    img.read(filename.c_str());

    const int h = img.get_height();
    const int w = img.get_width();
    data = Tensor (h, w, 3, 0.0);

    for(int i=0;i<img.get_height();i++){
        for(int j=0;j<img.get_width();j++){ 
            data(i,j,0) = (float) img.red_at(j,i);
            data(i,j,1) = (float) img.green_at(j,i);    
            data(i,j,2) = (float) img.blue_at(j,i);   
        }                
    }
}


/**
 * Save a DAISGram object to a bitmap file.
 * 
 * Data is clamped to 0,255 before saving it.
 *
 * @param filename String containing the path where to store the image.
 */
void DAISGram::save_image(string filename){
    data.clamp(0,255);

    BmpImg img = BmpImg(getCols(), getRows());

    img.init(getCols(), getRows());

    for(int i=0;i<getRows();i++){
        for(int j=0;j<getCols();j++){
            img.set_pixel(j,i,(unsigned char) data(i,j,0),(unsigned char) data(i,j,1),(unsigned char) data(i,j,2));                   
        }                
    }

    img.write(filename);

}
/**
 * Get rows
 *
 * @return returns the number of rows in the image
 */
int DAISGram::getRows(){
    return data.rows();
}

/**
 * Get columns
 *
 * @return returns the number of columns in the image
 */
int DAISGram::getCols(){
    return data.cols();
}


/**
 * Get depth
 *
 * @return returns the number of channels in the image
 */
int DAISGram::getDepth()
{
    return data.depth();
}


/**
 * Brighten the image
 * 
 * It sums the bright variable to all the values in the image.
 * 
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 * 
 * @param bright the amount of bright to add (if negative the image gets darker)
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::brighten(float bright){
    DAISGram res;
    res.data.init(getRows(),getCols(),getDepth());
    res.data=data+(bright);
    res.data.clamp(0, 255);
    return res;
}

/**
 * Create a grayscale version of the object
 * 
 * A grayscale image is produced by substituting each pixel with its average on all the channel
 *  
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::grayscale(){
    DAISGram res;
    float sum, avg;
    res.data.init(getRows(),getCols(),getDepth());
    for(int i=0; i < getRows(); i++){
        for(int j=0; j < getCols(); j++){
            sum=0;
            for (int k=0; k < getDepth(); k++){
                sum += data(i, j, k);
            }
            avg = sum / getDepth();
            for (int k=0; k < getDepth(); k++){
                res.data(i,j,k)=avg;
            }
        }
    }
    return res;
}


/**
 * Create a Warhol effect on the image
 * 
 * This function returns a composition of 4 different images in which the:
 * - top left is the original image
 * - top right is the original image in which the Red and Green channel are swapped
 * - bottom left is the original image in which the Blue and Green channel are swapped
 * - bottom right is the original image in which the Red and Blue channel are swapped
 *  
 * The output image is twice the dimensions of the original one.
 * 
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::warhol(){
    int r, c, d;
    r = getRows();
    c = getCols();
    d = getDepth();
    DAISGram res;
    res.data.init(2*r, 2*c, d);
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            for (int k = 0; k < d; k++){
                res.data(i,j,k)=data(i, j, k); 
            }
        }
    }

    for (int i = 0; i < r; i++){
        for (int j = c; j < 2*c; j++){
            for (int k = 0; k < d; k++){
                if (k == 0) {   
                    res.data(i,j,k)=data(i, j-c, 1); 
                } else if (k == 1) { 
                    res.data(i,j,k)=data(i, j-c, 0); 
                } else {    
                    res.data(i,j,k)=data(i, j-c, 2);
                }
            }
        }
    }
    
    for (int i = r; i < 2*r; i++){
        for (int j = 0; j < c; j++){
            for (int k = 0; k < d; k++){
                if (k == 0) {  
                    res.data(i,j,k)=data(i-r, j, 0); 
                } else if (k == 1) {    
                    res.data(i,j,k)=data(i-r, j, 2);
                } else {    
                    res.data(i,j,k)=data(i-r, j, 1); 
                }
            }
        }
    }
    for (int i = r; i < 2*r; i++){
        for (int j = c; j < 2*c; j++){
            for (int k = 0; k < d; k++){
                if (k == 0) {   
                    res.data(i,j,k)=data(i-r, j-c, 2); 
                } else if (k == 1) {    
                    res.data(i,j,k)=data(i-r, j-c, 1); 
                } else {    
                    res.data(i,j,k)=data(i-r, j-c, 0); 
                }
            }
        }
    }
    return res;
}

/**
 * Sharpen the image
 * 
 * This function makes the image sharper by convolving it with a sharp filter
 * 
 * filter[3][3]
 *    0  -1  0
 *    -1  5 -1
 *    0  -1  0
 *  
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 * 
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::sharpen(){
    DAISGram res;

    float s[]={ 0,-1, 0, -1, 5,-1, 0,-1, 0};
    Tensor filter;
    filter.init(3,3,3);
    for(int i=0; i<filter.rows(); i++){
        for(int j=0; j<filter.cols(); j++){
            for(int k=0; k<filter.depth(); k++){
                filter(i,j,k)=s[i*filter.cols()+j];
            }
        }
    }
    res.data=data.convolve(filter);
    res.data.clamp(0, 255);

    return res;
}

/**
 * Emboss the image
 * 
 * This function makes the image embossed (a light 3D effect) by convolving it with an
 * embossing filter
 * 
 * filter[3][3]
 *    -2 -1  0
 *    -1  1  1
 *     0  1  2
 * 
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 *  
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::emboss(){
    DAISGram res;
    float s[]={-2,-1, 0, -1, 1, 1, 0, 1, 2};
    Tensor filter;
    filter.init(3,3,3);
    for(int i=0; i<filter.rows(); i++){
        for(int j=0; j<filter.cols(); j++){
            for(int k=0; k<filter.depth(); k++){
                filter(i,j,k)=s[i*filter.cols()+j];
            }
        }
    }
    res.data=data.convolve(filter);
    res.data.clamp(0, 255);
    return res;
}

/**
 * Smooth the image
 * 
 * This function remove the noise in an image using convolution and an average filter
 * of size h*h:
 * 
 * c = 1/(h*h)
 * 
 * filter[3][3]
 *    c c c
 *    c c c
 *    c c c
 *  
 * @param h the size of the filter
 * @return returns a new DAISGram containing the modified object
 */
DAISGram DAISGram::smooth(int h){
    DAISGram res=DAISGram();
    float c = 1.0/(h*h);
    cout << c << endl;
    Tensor supp;
    supp.init(h, h, this->data.depth(), c);
    for(int i=0; i<supp.rows(); i++){
        for(int j=0; j<supp.cols(); j++){
            for(int k=0; k<supp.depth(); k++){
                cout << supp(i,j,k) << endl;
            }
        }
    }
    res.data=data.convolve(supp);

    return res;
}


/**
 * Edges of an image
 * 
 * This function extract the edges of an image by using the convolution 
 * operator and the following filter
 * 
 * 
 * filter[3][3]
 * -1  -1  -1
 * -1   8  -1
 * -1  -1  -1
 * 
 * Remeber to convert the image to grayscale before running the convolution.
 * 
 * Before returning the image, the corresponding tensor should be clamped in [0,255]
 *  
 * @return returns a new DAISGram containing the modified object
 */ 
DAISGram DAISGram::edge(){
    DAISGram res;
    DAISGram grayed;
    float s[]={-1,-1,-1,-1, 8,-1,-1,-1,-1};
    Tensor filter;
    filter.init(3,3,3);
    for(int i=0; i<filter.rows(); i++){
        for(int j=0; j<filter.cols(); j++){
            for(int k=0; k<filter.depth(); k++){
                filter(i,j,k)=s[i*filter.cols()+j];
                
            }
        }
    }
    grayed=grayscale();
    res.data=grayed.data.convolve(filter);
    res.data.clamp(0, 255);
    return res;
}

/**
 * Blend with another image
 * 
 * This function generate a new DAISGram which is the composition 
 * of the object and another DAISGram object
 * 
 * The composition follows this convex combination:
 * results = alpha*this + (1-alpha)*rhs 
 * 
 * rhs and this obejct MUST have the same dimensions.
 * 
 * @param rhs The second image involved in the blending
 * @param alpha The parameter of the convex combination  
 * @return returns a new DAISGram containing the blending of the two images.
 */
DAISGram DAISGram::blend(const DAISGram & rhs, float alpha){
    DAISGram result;
    result.data.init(rhs.data.rows(), rhs.data.cols(), rhs.data.depth());
    if(getRows()==rhs.data.rows() && getCols()==rhs.data.cols() && getDepth()==rhs.data.depth()){
        DAISGram a, b;
        a.data=data*(alpha);
        b.data=rhs.data*(1.0-alpha);
        result.data=a.data+b.data;
    }
    else{
        throw(dimension_mismatch());
    }
    return result;
 }


/**filter_odd_dimensions();
 * Green Screen
 * 
 * This function substitutes a pixel with the corresponding one in a background image 
 * if its colors are in the surrounding (+- threshold) of a given color (rgb).
 * 
 * (rgb - threshold) <= pixel <= (rgb + threshold)
 * 
 * 
 * @param bkg The second image used as background
 * @param rgb[] The color to substitute (rgb[0] = RED, rgb[1]=GREEN, rgb[2]=BLUE) 
 * @param threshold[] The threshold to add/remove for each color (threshold[0] = RED, threshold[1]=GREEN, threshold[2]=BLUE) 
 * @return returns a new DAISGram containing the result.
 */  
DAISGram DAISGram::greenscreen(DAISGram & bkg, int rgb[], float threshold[]){
    DAISGram res;
    res.data = data;
    for(int i=0; i<getRows(); i++){
        for(int j=0; j<getCols(); j++){
            if ( data(i,j,0) >= ( rgb[0] - threshold[0] )  &&  data(i,j,0) <= ( rgb[0] + threshold[0] )  &&
                 data(i,j,1) >= ( rgb[1] - threshold[1] )  &&  data(i,j,1) <= ( rgb[1] + threshold[1] )  &&
                 data(i,j,2) >= ( rgb[2] - threshold[2] )  &&  data(i,j,2) <= ( rgb[2] + threshold[2] ) ) {
                for(int k=0; k<getDepth(); k++){
                    res.data(i,j,k) = bkg.data(i,j,k);
                }
            }
        }
    }
    return res;
}

/**
 * Equalize
 * 
 * Stretch the distribution of colors of the image in order to use the full range of intesities.
 * 
 * See https://it.wikipedia.org/wiki/Equalizzazione_dell%27istogramma
 * 
 * @return returns a new DAISGram containing the equalized image.
 */ 
DAISGram DAISGram::equalize(){
    float cdf_min, cdf_v;
    DAISGram res;
    res.data.init(getRows(), getCols(), getDepth());
    for(int k=0; k<getDepth(); k++){
        cdf_value* temp;
        int size=0;
        temp = data.cdf(k, size);
        cdf_min = temp[0].cdf;
        for(int i=0; i<getRows(); i++){
            for(int j=0; j<getCols(); j++){
                int l = 0;
                while (temp[l].valore != data(i,j,k) && l < size){
                    l++;
                }

                cdf_v = temp[l].cdf; 
                res.data(i,j,k) = ( (cdf_v - cdf_min) / ( (getRows() * getCols() ) - 1 ) ) * (256 - 1);
            }
        }
        delete [] temp;
        
    }
    return res;
}
/**
 * Generate Random Image
 * 
 * Generate a random image from nois
 * 
 * @param h height of the image
 * @param w width of the image
 * @param d number of channels
 * @return returns a new DAISGram containing the generated image.
 */  
void DAISGram::generate_random(int h, int w, int d){
    data = Tensor(h,w,d,0.0);
    data.init_random(128,50);
    data.rescale(255);
}
