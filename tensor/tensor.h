#pragma once
#include <tuple>
#include <time.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cuda_runtime.h>
#include <ops.cuh>
#include <marco.h>


namespace ATen
{
    template <typename Scalar>
    class Matrix
    {
    protected:
        int Rows;
        int Cols;
        bool device;
        Scalar *data;

    public:
        Matrix(int rows = 1, int cols = 1, bool _device = true) : Rows(rows), Cols(cols), device(_device)
        {
            if (!device)
                data = (Scalar *)malloc(Rows * Cols * sizeof(Scalar));
            else
            {
                CHECK(cudaMalloc((void **)&data, Rows * Cols * sizeof(Scalar)));
            }
        }
        Matrix(const Matrix &m) : Rows(m.Rows), Cols(m.Cols)
        {
            data = (Scalar *)malloc(Rows * Cols * sizeof(Scalar));
            for (int i = 0; i < Rows * Cols; i++)
                data[i] = m[i];
        }
        Matrix(int rows, int cols, Scalar *_data) : Rows(rows), Cols(cols)
        {
            data = _data;
        }

        ~Matrix()
        {
            if (!device)
                free(this->data);
            else
                cudaFree(this->data);
        }

        std::tuple<int, int> getShape() const
        {
            return std::tuple<int, int>(Rows, Cols);
        }

        Matrix<Scalar> T()
        {
            Matrix<Scalar> m(Rows, Cols);
            for (int r = 0; r < Cols; r++)
                for (int c = 0; c < Rows; c++)
                    m[r * Rows + c] = this->data[c * Cols + r];

            return m;
        }

        member_unary_op(+, vec_add);
        member_unary_op(-, vec_sub);
        member_unary_op(*, vec_mul);
        member_unary_op(/, vec_div);

        Matrix &operator=(const Matrix &b)
        {
            if (this->Rows != b.Rows || this->Cols != b.Cols)
            {
                this->data = (Scalar *)malloc(b.Rows * b.Cols * sizeof(Scalar));
                this->Rows = b.Rows;
                this->Cols = b.Cols;
            }
            for (int i = 0; i < Rows * Cols; i++)
                this->data[i] = b[i];
            return *this;
        }

        Matrix &operator=(const Scalar b)
        {
            if (!device)
                for (int i = 0; i < Rows * Cols; i++)
                    this->data[i] = b;
            else
                ops::assign(this->data, b, Rows * Cols);

            return *this;
        }

        Scalar &operator[](int i) const
        {
            return this->data[i];
        }

        Matrix cpu()
        {
            int nBytes = Rows * Cols * sizeof(Scalar);
            Scalar *cpu_ptr = (Scalar *)malloc(nBytes);
            if (device == true)
                cudaMemcpy(cpu_ptr, data, nBytes, cudaMemcpyDeviceToHost);
            return Matrix<Scalar>(Rows, Cols, cpu_ptr);
        }

        friend std::ostream &operator<<(std::ostream &output,
                                        const Matrix &D)
        {
            int _R, _C;

            std::tie(_R, _C) = D.getShape();
            for (int x = 0; x < _R; x++)
            {
                for (int y = 0; y < _C; y++)
                    output << D[x * _C + y] << ",";
                output << std::endl;
            }
            return output;
        }

        global_scalar_unary_op(+, vec_add);
        global_scalar_unary_op(-, vec_sub);
        global_scalar_unary_op(*, vec_mul);
        global_scalar_unary_op(/, vec_div);
    };

}