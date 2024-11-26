#include <iostream>
#include <functional>
#include <vector>
#include <tensor.h>
#include <map>

namespace autograd
{
    class Variable;

     typedef struct _TapeEntry
    {
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::function<std::vector<ATen::Matrix<float>>(std::vector<ATen::Matrix<float>>&)> propagate;
    } TapeEntry;

    class Variable
    {
    private:
        ATen::Matrix<float> var;
        std::string name;
        static int _name;
        static std::vector<TapeEntry> gradient_tape;

    public:
        static int fresh_name()
        {
            return _name++;
        }
      
        Variable() {}
        Variable(ATen::Matrix<float> _var, std::string _name = "") : var(_var)
        {
            if (_name.empty())
                name = std::to_string(fresh_name());
            else
                name = _name;
        }

        ATen::Matrix<float> get_var()const
        {
            return var;
        }

        void print()const
        {
            std::cout << name << " " << var << std::endl;
        }

        Variable operator+(const Variable &);
        Variable operator-(const Variable &);
        Variable operator*(const Variable &);
        Variable operator/(const Variable &);

        friend void grad(Variable, std::vector<Variable>);
    };

    

    std::vector< ATen::Matrix<float>> gather_grad(std::map<std::string,ATen::Matrix<float>>&, std::vector<std::string>&);
    std::vector< ATen::Matrix<float>> mul_propagate(const Variable&,const Variable&, std::vector< ATen::Matrix<float>>&);
    std::vector< ATen::Matrix<float>> div_propagate(const Variable&,const Variable&, std::vector< ATen::Matrix<float>>&);
    void grad(Variable, std::vector<Variable>);


}