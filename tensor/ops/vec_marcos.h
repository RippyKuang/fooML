#define MAX_THREADS 1024

#define vec_func(op, name)                                                               \
    void name##_cpu(float *a, float *b, float *c, int len)                               \
    {                                                                                    \
        for (int i = 0; i < len; i++)                                                    \
            c[i] = a[i] op b[i];                                                         \
    }                                                                                    \
    __global__ void name##_kernel(float *A, float *B, float *C, const int N)             \
    {                                                                                    \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
        if (i < N)                                                                       \
            C[i] = A[i] op B[i];                                                         \
    }                                                                                    \
    void name(float *A, float *B, float *C, const int N)                                 \
    {                                                                                    \
        name##_kernel<<<(N + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(A, B, C, N); \
        CHECK(cudaGetLastError());                                                       \
        CHECK(cudaDeviceSynchronize());                                                  \
    }

#define vec_scalar_func(op, name)                                                        \
    void name##_cpu(float *a, float b, float *c, int len)                                \
    {                                                                                    \
        for (int i = 0; i < len; i++)                                                    \
            c[i] = a[i] op b;                                                            \
    }                                                                                    \
    __global__ void name##_kernel(float *A, float B, float *C, const int N)              \
    {                                                                                    \
        const int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
        if (i < N)                                                                       \
            C[i] = A[i] op B;                                                            \
    }                                                                                    \
    void name(float *A, float B, float *C, const int N)                                  \
    {                                                                                    \
        name##_kernel<<<(N + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>(A, B, C, N); \
        CHECK(cudaGetLastError());                                                       \
        CHECK(cudaDeviceSynchronize());                                                  \
    }

#define declare_vec_func(name)                       \
    void name##_cpu(float *, float *, float *, int); \
    void name(float *, float *, float *, const int); \
    __global__ void name##_kernel(float *, float *, float *, const int);

#define declare_vec_scalar_func(name)              \
    void name##_cpu(float *, float, float *, int); \
    void name(float *, float, float *, const int); \
    __global__ void name##_kernel(float *, float, float *, const int);
