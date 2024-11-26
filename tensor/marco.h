#define member_unary_op(op, func)                                      \
    Matrix operator op(const Matrix &b)                                \
    {                                                                  \
        assert(device == b.device);                                    \
        Matrix<Scalar> m(Rows, Cols, device);                          \
        if (!device)                                                   \
            ops::func##_cpu(this->data, b.data, m.data, Rows *Cols);   \
        else                                                           \
            ops::func(this->data, b.data, m.data, Rows *Cols);         \
        return m;                                                      \
    }                                                                  \
    Matrix operator op(const Scalar &b)                                \
    {                                                                  \
        Matrix<Scalar> m(Rows, Cols, device);                          \
        if (!device)                                                   \
            ops::func##_scalar_cpu(this->data, b, m.data, Rows *Cols); \
        else                                                           \
            ops::func##_scalar(data, b, m.data, Rows *Cols);           \
        return m;                                                      \
    }

#define global_scalar_unary_op(op, func)                           \
    template <typename T, typename _T>                             \
    friend Matrix<T> operator op(const _T & a, const Matrix<T> &b) \
    {                                                              \
        int Rows, Cols;                                            \
        std::tie(Rows, Cols) = b.getShape();                       \
        Matrix<T> m(Rows, Cols, b.device);                         \
        if (!b.device)                                             \
            ops::func##_scalar_cpu(b.data, a, m.data, Rows *Cols); \
        else                                                       \
            ops::func##_scalar(b.data, a, m.data, Rows *Cols);     \
        return m;                                                  \
    }

#define PRINT(a) std::cout << a << std::endl
#define TIMER(call)     \
    clock_t start, end; \
    start = clock();    \
    call;               \
    end = clock();      \
    printf("time=%f\n", (double)end - start)

    
