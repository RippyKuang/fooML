#include <autograd.h>
#include <tensor.h>

int main()
{

    ATen::Matrix<float> a(1000, 10000, true);
    ATen::Matrix<float> b(1000, 10000, true);

    a = 3;
    b = 2;

    // autograd::Variable v1(a,"a");
    // autograd::Variable v2(b,"b");
    // autograd::Variable r = v1*v2;
    // autograd::grad(r,std::vector{v1,v2});
  
    TIMER(1.1 + a / 2 * b / 2);


    return 0;
}