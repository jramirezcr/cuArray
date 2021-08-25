#include <cuda.h>
#include <vector>
#include <iostream>

#include "cuArray.cuh"

template<typename Tprec>
__global__
void it_over(ArrayDev<Tprec> array){

   int i = blockDim.x*blockIdx.x + threadIdx.x;

   array(i) = array(i)*5.0f; 

}

int main(){

    int size = 1024000;

    std::vector<float> A(size);
    ArrayDev<float>  A_d(size);

    for(int i = 0; i < size; i++){
        A[i] = static_cast<float>(i);
    }

    A_d.copyFromHost(A.data());

    it_over<<<1000,1024>>>(A_d); 

    A_d.copyToHost(A.data());


    for(int i = 0; i < 50; i++){
        std::cout<<A[i]<<std::endl;
    }

    return 0;

}
