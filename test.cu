#include <cuda.h>
#include <vector>
#include <iostream>

#include <chrono>

#include "cuArray.cuh"

template<typename Tprec>
__global__
void it_over(ArrayDev<Tprec> array){

   int i = blockDim.x*blockIdx.x + threadIdx.x;

   array(i) = array(i)*5.0f; 

}


template<typename Tprec>
__global__
void it_normal(Tprec* array){

   int i = blockDim.x*blockIdx.x + threadIdx.x;

   array[i] = array[i]*5.0f; 

}

int main(){


    int size = 10240000;

    std::vector<float> A(size);
    ArrayDev<float>  A_d(size);

    for(int i = 0; i < size; i++){
        A[i] = static_cast<float>(i);
    }

    A_d.copyFromHost(A.data());


    auto start1 = std::chrono::high_resolution_clock::now();
    for(int it = 1; it < 10000; it++)
        it_over<<<10000,1024>>>(A_d); 

    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    std::cout<< "Using cuArray: " << duration.count() << std::endl;

    A_d.copyFromHost(A.data());

    auto start2 = std::chrono::high_resolution_clock::now();
    for(int it = 1; it < 10000; it++)
        it_normal<<<10000,1024>>>(A_d.devPtr()); 

    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    std::cout<< "Using raw Pointers: " << duration2.count() << std::endl;

    A_d.copyToHost(A.data());


    //for(int i = 0; i < 50; i++) std::cout<<A[i]<<std::endl;
    

    return 0;

}
