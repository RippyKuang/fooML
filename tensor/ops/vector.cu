#include <ops.cuh>
                                                        
namespace ops
{

    vec_func(+,vec_add);
    vec_func(-,vec_sub);
    vec_func(*,vec_mul);
    vec_func(/,vec_div);

    vec_scalar_func(+,vec_add_scalar);
    vec_scalar_func(-,vec_sub_scalar);
    vec_scalar_func(*,vec_mul_scalar);
    vec_scalar_func(/,vec_div_scalar);

    __global__ void assign_kernel(float *A, float var, const int N)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
            A[i] = var;
    }

    void assign(float *A, float var, const int N)
    {
        assign_kernel<<<(N+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(A,var,N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        
    }

}