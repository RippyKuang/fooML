#include <iostream>
#include <cuda_runtime.h>
#include "ops/vec_marcos.h"

#define CHECK(_call)                                                           \
    {                                                                          \
        const cudaError_t error = _call;                                       \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d,reason : %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

namespace ops
{

    declare_vec_func(vec_add);
    declare_vec_func(vec_sub);
    declare_vec_func(vec_div); 
    declare_vec_func(vec_mul); 

    declare_vec_scalar_func(vec_add_scalar);
    declare_vec_scalar_func(vec_sub_scalar);
    declare_vec_scalar_func(vec_mul_scalar);
    declare_vec_scalar_func(vec_div_scalar);

    void assign(float *, float, const int);
    __global__ void assign_kernel(float *, float, const int);
   
}