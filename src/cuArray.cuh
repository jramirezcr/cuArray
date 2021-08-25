#ifndef _CUARRAYDEV_HPP
#define _CUARRAYDEV_HPP

#include<cuda.h>

//todo 

template<typename Tprec>
class ArrayDev{


public:

    Tprec *d_data;
    int dimsize;

    bool isCopy;

public:

    __host__ 
    explicit ArrayDev(): d_data(0), dimsize(0){}

    __host__ 
    explicit ArrayDev(int _dimsize);

    ArrayDev(ArrayDev<Tprec>&);

    virtual ~ArrayDev();


    //Services
     __host__
     void resize(int _dimsize); 
     //Change size array

    __host__ __device__ 
    int getDim() const;         
    //Get array dim 

    __host__ 
    Tprec* devPtr();                    
    //Return dev pointer to device data

    __host__ 
    void copyToHost(Tprec* hostPtr);
    // Copy array data to host array

    __host__ 
    void copyFromHost(Tprec* hostPtr);  
   // Copy data from a host array

    __host__
    void copyDevToDev(Tprec* devPtr);

    //Overloaded 
    //    __host__ 
    //ArrayDev<Tprec>& operator=(ArrayDev<Tprec>& );

    __device__ 
    Tprec& operator()(int i); 
    //Allows notation syle A(i) 

private:

    __host__ 
    void allocate();

    __host__ 
    void deallocate();
};

template<typename Tprec>
class ArrayDev3D
: public ArrayDev<Tprec>{

private:

   int nx;
   int ny;
   int nz;

public:
   explicit ArrayDev3D(): ArrayDev<Tprec>(){}
   explicit ArrayDev3D(int i, int j, int k)
   :ArrayDev<Tprec>(){

      nx = i > 0 ? i : 1;
      ny = j > 0 ? j : 1;
      nz = k > 0 ? k : 1;

      ArrayDev<Tprec>::resize(nx*ny*nz);
   }

   void resize(int i, int j, int k){

      nx = i > 0 ? i : 1;
      ny = j > 0 ? j : 1;
      nz = k > 0 ? k : 1;

      ArrayDev<Tprec>::resize(nx*ny*nz);

   }


   virtual ~ArrayDev3D(){};

    __device__ 
    Tprec& operator()(int i, int j, int k); 
    //Allows notation syle A(i,j,k) 

    __host__ __device__
    int getDim(int _dim){
    if(_dim==1)     {return nx;}
    else if(_dim==2){return ny;}
    else if(_dim==3){return nz;}
    else            {return 0 ;}
    }

};


template<typename Tprec>
__host__ 
ArrayDev<Tprec>::ArrayDev(int _dimsize){

  dimsize = _dimsize > 0? _dimsize: 1;

  allocate();

  isCopy = false;

}

template<typename Tprec>
ArrayDev<Tprec>::ArrayDev(ArrayDev<Tprec> &_orig){

  *this = _orig;
  isCopy = true;

}


template<typename Tprec>
ArrayDev<Tprec>::~ArrayDev(){
     if(!isCopy){
       cudaFree(d_data);
     }

}

    
template<typename Tprec>
__host__
void ArrayDev<Tprec>::resize(int _dimsize){


      dimsize = _dimsize > 0? _dimsize: 1;
      allocate();
      isCopy = false;
}


template<typename Tprec>
__device__ 
Tprec& ArrayDev<Tprec>::operator()(int i) 
{

      return  d_data[i];
}


/*
template<typename Tprec>
__host__
ArrayDev<Tprec>& ArrayDev<Tprec>::operator=(ArrayDev<Tprec>& array){

     if(this != &array){
        dimsize = array.getDim();
        allocate();

        copyDevToDev(array.devPtr());
     }

   return *this;
}
*/

template<typename Tprec>
__host__ __device__ 
int ArrayDev<Tprec>::getDim() const { 
    return dimsize;
} 


template<typename Tprec>
__host__ 
Tprec* ArrayDev<Tprec>::devPtr(){
    return d_data; 
} 

template<typename Tprec>
__host__ void ArrayDev<Tprec>::allocate(){

  cudaError_t retult = 
  cudaMalloc((void**)&d_data, dimsize*sizeof(Tprec));
}


template<typename Tprec>
__host__ 
void ArrayDev<Tprec>::copyToHost(Tprec* hostPtr) {

   cudaMemcpy(hostPtr, d_data, dimsize*sizeof(Tprec), 
              cudaMemcpyDeviceToHost);
   
}

template<typename Tprec>
__host__ 
void ArrayDev<Tprec>::copyFromHost(Tprec* hostPtr){

   cudaMemcpy(d_data, hostPtr, dimsize*sizeof(Tprec), 
              cudaMemcpyHostToDevice);
   
}

template<typename Tprec>
__host__ 
void ArrayDev<Tprec>::copyDevToDev(Tprec* devPtr){

   cudaMemcpy(d_data, devPtr, dimsize*sizeof(Tprec), 
              cudaMemcpyDeviceToDevice);
   
}

template<typename Tprec>
__device__ 
Tprec& ArrayDev3D<Tprec>::operator()(int i, int j, int k) 
{
      return  ArrayDev<Tprec>::d_data[nx*ny*k + nx*j + i];
}


#endif

